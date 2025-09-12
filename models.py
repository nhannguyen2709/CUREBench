from abc import ABC, abstractmethod
import importlib
import logging
import os
import re
from typing import Any, Callable, Dict, List, Tuple

from openai import OpenAI

from search_agent_glm import SearchAgent

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
    async def inference(self, prompt: str, prompt_type: str = None, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """Run inference on the model

        Args:
            prompt: The input prompt
            prompt_type: Type of prompt (multi_choice, open_ended, etc.)
            max_tokens: Maximum number of tokens to generate

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

    async def inference(self, prompt: str, prompt_type: str = None, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
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

    async def inference(self, prompt: str, prompt_type: str = None, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
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
        config: Dict[str, Any] = None,
    ):
        super().__init__(model_name)
        self.model = model_instance
        self._inference_func = inference_func
        self.config = config or {}
        
        self.search_agent = None
        self.use_search_agent = self.config.get('search_agent', {}).get('enabled', False)

    def load(self, **kwargs):
        """Custom models are already loaded"""
        mode = "SearchAgent" if self.use_search_agent else "Direct"
        logger.info(f"Using custom model: {self.model_name} (Mode: {mode})")

    async def inference(self, prompt: str, prompt_type: str):
        """Custom model inference with optional search agent"""
        # # Use search agent for inference
        # logger.debug(f"Using SearchAgent for inference: {prompt[:100]}...")
        # search_agent = SearchAgent(config=self.config)
        # response = search_agent.search(instructions + "\n\n" + prompt, self.config.search_agent.max_turns)
        # complete_messages = search_agent.conversation_history
        # Use regular inference function (now async)
        logger.debug(f"Using direct inference for: {prompt[:100]}...")
        responses = await self._inference_func(self.model, prompt, "")
        return responses


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


def load_instructions(file_path="instruction_variations.txt"):
    """Load instruction variations from txt file."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


async def inference_function(
    model_instance: Dict[str, Any],
    question: str,
    instructions: str,
    sampling_params: Dict[str, Any],
) -> str:
    """
    Inference function using vLLM endpoint with N parallel calls using different seeds.
    This function will be called for each evaluation example.

    Args:
        model_instance: Dictionary containing the client and model info
        question: The question to answer
        instructions: Instructions for the model
        sampling_params: Sampling parameters including temperature, top_p, etc.

    Returns:
        The best response from N parallel calls
    """
    import asyncio
    import random
    import concurrent.futures
    
    client: OpenAI = model_instance["client"]
    model_name: str = model_instance["model_name"]
    
    # Get number of parallel calls from sampling params, default to 8
    n_calls = sampling_params.get("n_parallel_calls", 8)
    
    # Generate different seeds for each call
    base_seed = sampling_params.get("base_seed", random.randint(0, 2**31 - 1))
    seeds = [base_seed + i * 1000 for i in range(n_calls)]
    instructions = random.sample(load_instructions(), n_calls)
    
    def single_inference_call(seed: int, instruction: str) -> str:
        """Make a single inference call with a specific seed"""
        if "gpt-oss" in model_name:
            response = client.responses.create(
                model=model_name, 
                input=question, 
                instructions=instructions, 
                max_output_tokens=sampling_params["max_tokens"], 
                reasoning=sampling_params.get("reasoning", {}),
                temperature=sampling_params["temperature"], 
                top_p=sampling_params["top_p"],
                seed=seed
            )
            
            # Extract the response content
            for output in response.output:
                if output.type == "reasoning":
                    return output.content[0].text
            return response.output[-1].content[0].text
        else:
            # For non-gpt-oss models, use regular OpenAI client
            extra_body = {
                "chat_template_kwargs": {"add_generation_prompt": True, "enable_thinking": True},
                "seed": seed
            }
            if "qwen" in model_name:
                extra_body.update({"top_k": 20, "min_p": 0.0})

            message = [{"role": "user", "content": instruction + "\n\n" + question}]
            response = client.chat.completions.create(
                model=model_name,
                messages=message,
                n=1,  # Single response per call since we're making N parallel calls
                temperature=sampling_params["temperature"],
                top_p=sampling_params["top_p"],
                max_completion_tokens=sampling_params["max_tokens"],
                extra_body=extra_body,
            )
            return response.choices[0].message.content

    # Make N parallel calls with different seeds using ThreadPoolExecutor
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_calls) as executor:
        # Create partial functions with different seeds
        tasks = [
            loop.run_in_executor(executor, single_inference_call, seed, instruction) 
            for seed, instruction in zip(seeds, instructions)
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out any exceptions and get valid responses
    valid_responses = []
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            logger.warning(f"Inference call {i} with seed {seeds[i]} failed: {response}")
        else:
            valid_responses.append(response)
    
    if not valid_responses:
        raise RuntimeError("All parallel inference calls failed")
    
    return valid_responses


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
    # Pattern to match both \boxed{...} and boxed{...} (missing leading backslash)
    patterns = [
        r"\\boxed\{(.*?)\}",  # Standard \boxed{...}
        r"boxed\{(.*?)\}"     # Outlier case boxed{...} (missing leading backslash)
    ]
    
    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text))

    if not matches:
        return ""

    for match in matches[::-1]:
        if match == "":
            continue

        # Handle \text{...} commands within the boxed content
        processed_match = match
        # Handle both complete \text{...} and incomplete \text{... (missing closing brace)
        text_patterns = [
            r"\\text\{([^}]*)\}",  # Complete \text{content}
            r"\\text\{([^}]*)$"    # Incomplete \text{content (missing closing brace at end)
        ]
        
        for text_pattern in text_patterns:
            text_matches = re.findall(text_pattern, processed_match)
            for text_content in text_matches:
                processed_match = re.sub(text_pattern, text_content, processed_match, count=1)

        if extract_mc_letter:
            # Try to extract just the letter for multiple choice
            # Match patterns like: A, A., (A), A:, A), etc.
            mc_pattern = r"^([A-Z])[\.:\)\s]|^\(([A-Z])\)|^([A-Z])$"
            mc_match = re.search(mc_pattern, processed_match.strip())
            if mc_match:
                # Return the first non-None group
                return next((g for g in mc_match.groups() if g is not None), "")

        return processed_match

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