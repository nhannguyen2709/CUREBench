from abc import ABC, abstractmethod
import importlib
import logging
import os
import re
from typing import Any, Callable, Dict, List, Tuple

from openai import OpenAI


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Suppress HTTP request logs from OpenAI client and related libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# Model Classes
class BaseModel(ABC):
    """Abstract base class for all models"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load(self, **kwargs):
        """Load the model"""
        pass

    @abstractmethod
    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """Run inference on the model

        Returns:
            Tuple of (response, messages) where messages is the complete conversation history
        """
        pass


class ChatGPTModel(BaseModel):
    """ChatGPT/OpenAI model wrapper"""

    def load(self, **kwargs):
        """Load ChatGPT model"""

        api_key = os.getenv("AZURE_OPENAI_API_KEY_O1")
        api_version = "2024-12-01-preview"  # "2025-03-01-preview"

        if not api_key:
            raise ValueError(
                f"API key not found in environment. Please set the appropriate environment variable."
            )

        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        from openai import AzureOpenAI

        print("Initializing AzureOpenAI client with endpoint:", azure_endpoint)
        print("Using API version:", api_version)
        self.model_client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )

    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """ChatGPT inference"""
        messages = [{"role": "user", "content": prompt}]

        responses = self.model_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=8192,
        )
        # print("\033[94m" + str(responses) + "\033[0m")
        response = responses.choices[0].message.content

        # Create complete conversation history
        complete_messages = messages + [{"role": "assistant", "content": response}]

        return response, complete_messages


class LocalModel(BaseModel):
    """Local HuggingFace model wrapper"""

    def load(self, **kwargs):
        """Load local HuggingFace model"""
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                BitsAndBytesConfig,
            )
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=BitsAndBytesConfig(load_in_8bit=True, **kwargs),
            )
            logger.info(f"Loaded local model: {self.model_name}")
        except ImportError as e:
            logger.error(f"Failed to import local model dependencies: {e}")
            raise

    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """Local model inference"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        print("messages:", messages)

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            temperature=0.4,
            top_p=0.9,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
        )

        response = outputs[0][input_ids.shape[-1] :]
        response_text = self.tokenizer.decode(response, skip_special_tokens=True)
        print("response_text:", response_text)
        # Create complete conversation history
        complete_messages = messages + [{"role": "assistant", "content": response_text}]

        return response_text, complete_messages


class CustomModel(BaseModel):
    """Custom model wrapper for user-defined models"""

    def __init__(
        self,
        model_name: str,
        model_instance: Any,
        inference_func: Callable,
    ):
        super().__init__(model_name)
        self.model = model_instance
        self._inference_func = inference_func

    def load(self, **kwargs):
        """Custom models are already loaded"""
        logger.info(f"Using custom model: {self.model_name}")

    def inference(self, prompt: str, prompt_type: str) -> Tuple[str, List[Dict]]:
        """Custom model inference"""
        try:
            # For custom models, we'll create a simple message structure
            messages = [{"role": "user", "content": prompt}]

            response = self._inference_func(self.model, prompt, prompt_type)

            # Create complete conversation history
            complete_messages = messages + [{"role": "assistant", "content": response}]

            return response, complete_messages
        except Exception as e:
            logger.error(f"Custom model inference error: {e}")
            error_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "Error occurred"},
            ]
            return "Error occurred", error_messages


def create_model_instance(model_name: str, base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY"):
    """
    Factory function to create the OpenAI client instance for vLLM endpoint.
    This function will be called when model_instance_factory is specified in JSON config.

    Returns:
        Initialized OpenAI client configured for local vLLM endpoint
    """
    # Create OpenAI client pointed at vLLM endpoint
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    # Return a dictionary containing the client and model info
    return {"client": client, "model_name": model_name, "base_url": base_url}


def inference_function(
    model_instance: Dict[str, Any],
    question: str,
    question_type: str,
    sampling_params: Dict[str, Any],
) -> str:
    """
    Inference function using vLLM endpoint.
    This function will be called for each evaluation example.

    Args:
        model_instance: Dictionary containing the client and model info
        question: The question to answer
        kwargs: Additional keyword arguments

    Returns:
        The response from the model
    """
    client: OpenAI = model_instance["client"]
    model_name: str = model_instance["model_name"]

    instructions = "Please reason step-by-step"
    if "multi_choice" in question_type:
        instructions += ", and put your final answer with only the choice letter within \\boxed{}."
    else:
        instructions += ", and put your final answer within \\boxed{}."

    # Call the vLLM endpoint via OpenAI client
    if "gpt-oss" in model_name:
        response = client.responses.create(
            model=model_name, input=question, instructions=instructions, 
            max_output_tokens=sampling_params["max_tokens"], reasoning=sampling_params["reasoning"],
            temperature=sampling_params["temperature"], top_p=sampling_params["top_p"])

        # Extract the response content
        for output in response.output:
            if output.type == "reasoning":
                response_text = output.content[0].text
                return response_text
        return response.output[-1].content[0].text
    else:
        message = [{"role": "user", "content": instructions + "\n\n" + question}]
        response = client.chat.completions.create(
            model=model_name,
            messages=message,
            temperature=sampling_params["temperature"],
            top_p=sampling_params["top_p"],
            max_completion_tokens=sampling_params["max_tokens"],
            extra_body={"chat_template_kwargs": {"add_generation_prompt": True, "enable_thinking": True}}
        )
        import pdb; pdb.set_trace()


def extract_boxed_text(text, extract_mc_letter=True):
    """
    Extract content from LaTeX \boxed{} command.

    Args:
        text: The text to extract boxed content from
        extract_mc_letter: If True, extract just the letter from multiple choice answers
                          like \boxed{A}, \boxed{A. 123}, \boxed{(A) 123}

    Returns:
        The extracted content or empty string if no match
    """
    # Fix the pattern to include the backslash
    pattern = r"oxed{(.*?)}"
    matches = re.findall(pattern, text)

    if not matches:
        return ""

    for match in matches[::-1]:
        if match == "":
            continue

        if extract_mc_letter:
            # Try to extract just the letter for multiple choice
            # Match patterns like: A, A., (A), A:, A), etc.
            mc_pattern = r"^([A-Z])[\.:\)\s]|^\(([A-Z])\)|^([A-Z])$"
            mc_match = re.search(mc_pattern, match.strip())
            if mc_match:
                # Return the first non-None group
                return next((g for g in mc_match.groups() if g is not None), "")

        return match

    return ""


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    if method == "strict":
        final_answer = extract_boxed_text(solution_str)
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer