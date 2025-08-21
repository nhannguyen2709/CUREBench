import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from models import BaseModel, extract_solution


@dataclass
class EvaluationResult:
    """Simple container for evaluation results"""

    dataset_name: str
    model_name: str
    accuracy: float
    correct_predictions: int
    total_examples: int
    predictions: List[Dict]  # Changed from List[str] to List[Dict]
    reasoning_traces: List[str] = None  # Add reasoning traces
    details: Optional[Dict] = None


def get_prediction_with_trace(model: BaseModel, example: Dict) -> Tuple[Dict, str]:
    """Get model prediction and reasoning trace for a single example"""
    question = example["question"]
    question_type = example["question_type"]

    # Get model response and messages using the model's inference method
    response, reasoning_trace = model.inference(question, prompt_type=question_type)

    # Initialize prediction dictionary
    prediction = {
        "choice": "",  # Use empty string instead of None
        "open_ended_answer": "",  # Use empty string instead of None
    }

    # Extract answer from response
    if (
        question_type == "multi_choice"
        or question_type == "open_ended_multi_choice"
    ):
        # For multiple choice, extract the letter
        # choice = self._extract_multiple_choice_answer(response)
        choice = extract_solution(response)
        # Ensure choice is never None or NULL
        prediction["choice"] = (
            choice if choice and str(choice).upper() not in ["NONE", "NULL"] else ""
        )
        prediction["open_ended_answer"] = response.strip()  # Keep full response too
    elif question_type == "open_ended":
        # For open-ended, only return response, use N/A for choice to avoid empty string issues
        prediction["choice"] = (
            "NOTAVALUE"  # Use N/A instead of empty string to avoid NULL validation issues
        )
        prediction["open_ended_answer"] = response.strip()

    return prediction, reasoning_trace


async def process_example(i: int, model: BaseModel, example: Dict):
    """Process a single example asynchronously"""
    prediction, reasoning_trace = await asyncio.to_thread(get_prediction_with_trace, model, example)

    question_type = example["question_type"]
    expected_answer = example.get("answer")

    local_accuracy_correct = 0
    local_accuracy_total = 0

    if question_type in ["multi_choice", "open_ended_multi_choice"]:
        if expected_answer != "":
            is_correct = prediction["choice"] == expected_answer
            local_accuracy_correct = 1 if is_correct else 0
        local_accuracy_total = 1
    elif question_type == "open_ended":
        # Open-ended questions don't count toward accuracy
        pass

    return i, prediction, reasoning_trace, local_accuracy_correct, local_accuracy_total