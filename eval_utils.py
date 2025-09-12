import asyncio
import random
from collections import Counter
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


def majority_vote_with_random_selection(responses: List[str]) -> Tuple[str, str]:
    """
    Perform majority voting on responses and randomly select one response with the winning choice.
    
    Args:
        responses: List of response strings from N sampling
        
    Returns:
        Tuple of (final_choice, selected_response)
    """
    if not responses:
        return "", ""
    
    # Extract choices from all responses
    choices = [extract_solution(response) for response in responses]
    
    # Filter out empty/None choices and count frequencies
    valid_choices = [choice for choice in choices if choice and str(choice).strip()]
    
    if not valid_choices:
        # If no valid choices, return first response
        return "", responses[0]
    
    # Find most frequent choice (majority vote)
    choice_counts = Counter(valid_choices)
    most_common_choice = choice_counts.most_common(1)[0][0]
    
    # Find all responses that have the winning choice
    matching_responses = []
    for i, choice in enumerate(choices):
        if choice == most_common_choice:
            matching_responses.append(responses[i])
    
    # Randomly select one response from those with the winning choice
    selected_response = random.choice(matching_responses) if matching_responses else responses[0]
    
    return most_common_choice, selected_response


async def get_prediction_with_trace(model: BaseModel, example: Dict) -> Tuple[Dict, str]:
    """Get model prediction and reasoning trace for a single example"""
    question = example["question"]
    question_type = example["question_type"]

    # Get model response and messages using the model's inference method
    responses = await model.inference(question, prompt_type=question_type)
    responses = [r for r in responses if r is not None]

    # Initialize prediction dictionary
    prediction = {
        "choice": "",  # Use empty string instead of None
        "open_ended_answer": "",  # Use empty string instead of None
    }

    # Use majority voting for multiple choice questions
    final_choice, selected_response = majority_vote_with_random_selection(responses)

    # Extract answer from response
    if (
        question_type == "multi_choice"
        or question_type == "open_ended_multi_choice"
    ):
        # Ensure choice is never None or NULL
        prediction["choice"] = (
            final_choice if final_choice and str(final_choice).upper() not in ["NONE", "NULL"] else ""
        )
        prediction["open_ended_answer"] = selected_response.strip()  # Keep selected response
    elif question_type == "open_ended":
        prediction["choice"] = (
            "NOTAVALUE"  # Use NOTAVALUE instead of empty string to avoid NULL validation issues
        )
        prediction["open_ended_answer"] = selected_response.strip()

    return prediction, selected_response


async def process_example(i: int, model: BaseModel, example: Dict):
    """Process a single example asynchronously"""
    prediction, reasoning_trace = await get_prediction_with_trace(model, example)

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