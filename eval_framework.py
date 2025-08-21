"""
Bio-Medical AI Competition Starter Kit

A simple framework for evaluating models on bio-medical datasets.
Perfect for getting started quickly in the competition.

Key Features:
- Easy model loading (ChatGPT, Local models, Custom models)
- Simple dataset loading
- Automatic evaluation and scoring
- Submission file generation

Usage:
    framework = CompetitionKit()
    framework.load_model("gpt-4o-mini")
    results = framework.evaluate("quick_test")
    framework.sa        elif question_type == "open_ended":
            # For open-ended, only return response, use NOTAVALUE for choice to avoid empty string issues
            prediction["choice"] = "NOTAVALUE"  # Use NOTAVALUE instead of empty string to avoid NULL validation issues
            prediction["open_ended_answer"] = response.strip()ubmission(results, "my_submission.json")
"""

import json
import os
import sys
import logging
import argparse
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from tqdm import tqdm
import concurrent.futures
import threading
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from models import ChatGPTModel, CustomModel, LocalModel, extract_solution


logger = logging.getLogger(__name__)


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


class CompetitionKit:
    """
    Simple competition framework - everything you need in one class!
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the competition kit.
        """
        self.config = config
        self.output_dir = self.config.output.dir
        self.model_name = self.config.model.model_name

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Load dataset configurations from config file or use defaults
        self.datasets = self._load_dataset_configs(self.config)

    def load_model(self):
        """
        Load a model for evaluation.
        """
        config = self.config.model
        model_name = config.model_name
        model_type = config.model_type
        base_url = config.base_url
        api_key = config.api_key
        sampling = config.sampling

        if model_type == "chatgpt":
            self.model = ChatGPTModel(model_name)
        elif model_type == "local":
            self.model = LocalModel(model_name)
        elif model_type == "custom":
            from functools import partial
            from models import create_model_instance, inference_function

            model_instance = create_model_instance(model_name, base_url, api_key)
            self.model = CustomModel(model_name, model_instance, partial(inference_function, sampling=sampling))
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _load_dataset_configs(self, config) -> Dict:
        """
        Load dataset configurations from config file or return defaults

        Args:
            config: Configuration dictionary

        Returns:
            Dictionary of dataset configurations
        """
        if not config:
            print("Not config provided, existing.")
            exit(1)

        # Check if config has a single dataset configuration
        if "dataset" in config:
            dataset_config = config.dataset
            dataset_name = dataset_config.name
            # Create a dictionary with the dataset name as key
            return {dataset_name: dataset_config}
        else:
            # If no dataset in config, return defaults
            print("Not config found, existing.")
            exit(1)

    def evaluate(self, dataset_name: str) -> EvaluationResult:
        """
        Evaluate model on a dataset

        Args:
            dataset_name: Name of dataset to evaluate on

        Returns:
            EvaluationResult object with scores and predictions
        """
        if not self.model:
            raise ValueError("No model loaded. Call load_model() first.")

        if dataset_name not in self.datasets:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Available: {list(self.datasets.keys())}"
            )

        dataset_config = self.datasets[dataset_name]
        logger.info(f"Evaluating on {dataset_name}: {dataset_config['description']}")

        # Load dataset
        dataset = self._load_dataset(dataset_config)
        if self.config.evaluation.subset_size is not None:
            dataset = dataset[:self.config.evaluation.subset_size]

        # Store dataset examples for later use in save_submission
        self._last_dataset_examples = dataset

        # Run evaluation with parallel processing
        total_count = len(dataset)
        logger.info(f"Running evaluation on {total_count} examples...")

        # Get number of workers from environment or use default
        max_workers = self.config.evaluation.max_workers
        logger.info(f"Using {max_workers} parallel workers for evaluation")

        # Process examples in parallel
        predictions, reasoning_traces, accuracy_correct_count, accuracy_total_count = (
            self._evaluate_parallel(dataset, max_workers)
        )

        # Calculate final accuracy (excluding open-ended questions)
        accuracy = (
            accuracy_correct_count / accuracy_total_count
            if accuracy_total_count > 0
            else 0.0
        )

        result = EvaluationResult(
            dataset_name=dataset_name,
            model_name=self.model_name,
            accuracy=accuracy,
            correct_predictions=accuracy_correct_count,  # Use accuracy-specific count
            total_examples=accuracy_total_count,  # Use accuracy-specific count
            predictions=predictions,
            reasoning_traces=reasoning_traces,  # Include reasoning traces
        )

        logger.info(
            f"Evaluation completed: {accuracy:.2%} accuracy ({accuracy_correct_count}/{accuracy_total_count}) - excluding open-ended questions"
        )
        logger.info(
            f"Total examples processed: {total_count} (including {total_count - accuracy_total_count} open-ended questions)"
        )

        return result

    def _get_prediction_with_trace(self, example: Dict) -> Tuple[Dict, str]:
        """Get model prediction and reasoning trace for a single example"""
        question = example["question"]
        question_type = example["question_type"]

        # Get model response and messages using the model's inference method
        response, reasoning_trace = self.model.inference(question, self.config.model.config.max_tokens)

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

    def _evaluate_parallel(
        self, dataset: List[Dict], max_workers: int
    ) -> Tuple[List[Dict], List[str], int, int]:
        """
        Evaluate dataset examples in parallel using ThreadPoolExecutor

        Args:
            dataset: List of examples to evaluate
            max_workers: Maximum number of parallel workers

        Returns:
            Tuple of (predictions, reasoning_traces, accuracy_correct_count, accuracy_total_count)
        """
        total_count = len(dataset)
        predictions = [None] * total_count  # Pre-allocate to maintain order
        reasoning_traces = [None] * total_count

        # Thread-safe counters
        accuracy_correct_count = 0
        accuracy_total_count = 0
        counter_lock = threading.Lock()

        def process_example_with_index(index_example_tuple):
            """Process a single example and return results with index for ordering"""
            i, example = index_example_tuple
            try:
                # Get prediction and reasoning trace
                prediction, reasoning_trace = self._get_prediction_with_trace(example)

                # Check if correct based on question type
                is_correct = False
                question_type = example["question_type"]
                expected_answer = example.get("answer")

                local_accuracy_correct = 0
                local_accuracy_total = 0

                if (
                    question_type == "multi_choice"
                    or question_type == "open_ended_multi_choice"
                ):
                    # For multiple choice, compare the choice field
                    if expected_answer != "":
                        is_correct = prediction["choice"] == expected_answer
                    else:
                        is_correct = False
                    # Count for accuracy calculation (exclude open_ended)
                    local_accuracy_total = 1
                    if is_correct:
                        local_accuracy_correct = 1
                elif question_type == "open_ended":
                    # For open-ended, compare the open_ended_answer field but don't count in accuracy
                    if expected_answer != "":
                        is_correct = prediction["open_ended_answer"] == expected_answer
                    else:
                        is_correct = False

                return (
                    i,
                    prediction,
                    reasoning_trace,
                    local_accuracy_correct,
                    local_accuracy_total,
                    None,
                )

            except Exception as e:
                logger.error(f"Error processing example {i}: {e}")
                error_prediction = {
                    "choice": "NOTAVALUE",  # Use NOTAVALUE instead of empty string
                    "open_ended_answer": "Error",
                }
                return (
                    i,
                    error_prediction,
                    "Error occurred during inference",
                    0,
                    0,
                    str(e),
                )

        # Process examples in parallel with progress tracking
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            indexed_dataset = list(enumerate(dataset))

            # Use tqdm to track progress
            with tqdm(total=total_count, desc="Evaluating") as pbar:
                # Submit all jobs
                future_to_index = {
                    executor.submit(process_example_with_index, item): item[0]
                    for item in indexed_dataset
                }

                # Process completed jobs
                for future in concurrent.futures.as_completed(future_to_index):
                    (
                        index,
                        prediction,
                        reasoning_trace,
                        local_correct,
                        local_total,
                        error,
                    ) = future.result()

                    # Store results in order
                    predictions[index] = prediction
                    reasoning_traces[index] = reasoning_trace

                    # Update counters thread-safely
                    with counter_lock:
                        accuracy_correct_count += local_correct
                        accuracy_total_count += local_total

                        # Update progress bar
                        pbar.update(1)

                        # Log progress every 10 completed examples
                        if pbar.n % 10 == 0:
                            current_acc = (
                                accuracy_correct_count / accuracy_total_count
                                if accuracy_total_count > 0
                                else 0.0
                            )
                            logger.info(
                                f"Progress: {pbar.n}/{total_count}, Accuracy: {current_acc:.2%} (excluding open-ended)"
                            )

        logger.info(f"Parallel evaluation completed. Processed {total_count} examples.")
        return (
            predictions,
            reasoning_traces,
            accuracy_correct_count,
            accuracy_total_count,
        )

    def _load_dataset(self, dataset_config: Dict) -> List[Dict]:
        """Load dataset based on configuration"""
        from dataset_utils import build_dataset
        from torch.utils.data import DataLoader

        # Build dataset
        dataset = build_dataset(dataset_config.path)

        # Convert to list of dictionaries for easier processing
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        dataset_list = []

        for batch in dataloader:
            question_type = batch[0][0]

            if question_type == "multi_choice":
                dataset_list.append(
                    {
                        "question_type": batch[0][0],
                        "id": batch[1][0],
                        "question": batch[2][0],
                        "answer": batch[3][0],
                    }
                )
            elif question_type == "open_ended_multi_choice":
                dataset_list.append(
                    {
                        "question_type": batch[0][0],
                        "id": batch[1][0],
                        "question": batch[2][0],
                        "answer": batch[3][0],
                        "meta_question": batch[4][0],
                    }
                )
            elif question_type == "open_ended":
                dataset_list.append(
                    {
                        "question_type": batch[0][0],
                        "id": batch[1][0],
                        "question": batch[2][0],
                        "answer": batch[3][0],
                    }
                )

        return dataset_list

    def save_submission(
        self,
        results: List[EvaluationResult],
        filename: str = "submission.csv",
        metadata: Dict = None,
        dataset_examples: List[Dict] = None,
        config_path: str = None,
        args: argparse.Namespace = None,
    ):
        """
        Save results in competition submission format as CSV file with metadata JSON and zip package

        Args:
            results: List of evaluation results
            filename: Output CSV filename (will be used for CSV inside zip)
            metadata: User-provided metadata dictionary containing model info, track, etc.
            dataset_examples: Original dataset examples to extract question IDs and reasoning traces
            config_path: Path to configuration file containing metadata
            args: Command line arguments containing metadata
        """
        import pandas as pd
        import zipfile

        # Get metadata from various sources with priority order
        metadata = self.get_metadata(config_path, args, metadata)

        # Create submission data for CSV
        submission_data = []

        # Process each result to create the CSV format
        for result in results:
            # Get the corresponding dataset examples if provided
            examples = dataset_examples if dataset_examples else []

            for i, (prediction, example) in enumerate(
                zip(result.predictions, examples)
            ):
                # Use stored reasoning trace if available, convert to simple text format
                reasoning_trace = json.dumps(result.reasoning_traces[i])
                # if result.reasoning_traces and i < len(result.reasoning_traces):
                #     trace = result.reasoning_traces[i]
                #     if isinstance(trace, list) and len(trace) > 0:
                #         # Convert list of messages to a simple text format
                #         text_parts = []
                #         for msg in trace:
                #             if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                #                 role = msg['role']
                #                 content = msg['content'].replace('\n', ' ').replace('\r', '').replace('"', "'")
                #                 text_parts.append(f"{role}: {content}")
                #         reasoning_trace = " | ".join(text_parts)
                #     else:
                #         # Fallback to string representation
                #         reasoning_trace = str(trace).replace('\n', ' ').replace('\r', '').replace('"', "'")

                # Clean up text fields to avoid CSV formatting issues
                prediction_text = (
                    prediction.get("open_ended_answer", "") or ""
                )  # Ensure not None
                if not prediction_text or prediction_text.strip() == "":
                    prediction_text = "No prediction available"

                # Ensure choice is clean and never NULL
                choice_raw = prediction.get("choice", "")
                if choice_raw is None or str(choice_raw).upper() in [
                    "NULL",
                    "NONE",
                    "NAN",
                ]:
                    choice_clean = "NOTAVALUE"  # Use NOTAVALUE instead of empty string
                elif str(choice_raw).strip() == "":
                    choice_clean = "NOTAVALUE"  # Replace empty strings with NOTAVALUE to avoid NULL validation issues
                else:
                    choice_clean = str(choice_raw).strip()

                # Ensure reasoning trace is not null
                if (
                    not reasoning_trace
                    or reasoning_trace == "null"
                    or reasoning_trace.strip() == ""
                ):
                    reasoning_trace = "No reasoning available"

                # Create CSV row - let pandas handle the escaping
                row = {
                    "id": str(example.get("id", str(i)) or f"unknown_{i}"),
                    "prediction": str(prediction_text),
                    "choice": str(choice_clean),
                    "reasoning": str(reasoning_trace),
                }

                # Debug: Log if choice is NULL-like
                if (
                    str(choice_clean).upper() in ["NULL", "NONE", "NAN"]
                    or str(choice_clean).strip() == ""
                ):
                    logger.warning(
                        f"Found NULL-like or empty choice for row {row['id']}: '{choice_clean}' - replacing with NOTAVALUE"
                    )
                    row["choice"] = "NOTAVALUE"

                submission_data.append(row)

        # Create DataFrame and save CSV with proper quoting and NaN handling
        df = pd.DataFrame(submission_data)

        # Convert all columns to string to avoid type issues
        for col in df.columns:
            df[col] = df[col].astype(str)

        # Aggressive null value cleaning
        null_replacements = {
            "id": "unknown_id",
            "prediction": "No prediction available",
            "choice": "NOTAVALUE",  # Use NOTAVALUE for choice instead of empty string
            "reasoning": "No reasoning available",
        }

        # Replace all possible null-like values
        for col in df.columns:
            # Replace pandas null values
            df[col] = df[col].fillna(null_replacements.get(col, "NOTAVALUE"))

            # Replace string representations of null
            null_like_values = [
                "nan",
                "NaN",
                "None",
                "null",
                "NULL",
                "<NA>",
                "nat",
                "NaT",
            ]
            for null_val in null_like_values:
                df[col] = df[col].replace(
                    null_val, null_replacements.get(col, "NOTAVALUE")
                )

            # Special handling for choice column - ensure it's never empty or null-like
            if col == "choice":
                df[col] = df[col].replace(
                    "NOTAVALUE", "NOTAVALUE"
                )  # Keep NOTAVALUE as is for choice
                # Replace any null-like values with NOTAVALUE
                for null_val in null_like_values:
                    df[col] = df[col].replace(null_val, "NOTAVALUE")
                # Replace empty strings with NOTAVALUE for choice column
                df[col] = df[col].replace("", "NOTAVALUE")
                df[col] = df[col].replace(
                    " ", "NOTAVALUE"
                )  # Also replace whitespace-only

            # Replace empty strings (except for choice column which can be empty)
            if col != "choice" and col in null_replacements:
                df[col] = df[col].replace("", null_replacements[col])
                df[col] = df[col].replace(
                    " ", null_replacements[col]
                )  # Also replace whitespace-only

        csv_path = os.path.join(self.output_dir, filename)

        # Validate DataFrame before saving
        logger.info(f"Creating CSV with {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")

        # Final validation - check for any remaining nulls
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                logger.warning(f"Still found {null_count} nulls in column {col}")

        # Check for any problematic data
        for idx, row in df.head().iterrows():
            logger.debug(
                f"Sample row {idx}: id={row['id']}, choice='{row['choice']}', prediction_len={len(str(row['prediction']))}, reasoning_len={len(str(row['reasoning']))}"
            )

        # Final safety check: ensure choice column has no NULL values or empty strings
        logger.info("Performing final NULL check on choice column...")
        null_patterns = [
            "NULL",
            "null",
            "None",
            "NaN",
            "nan",
            "<NA>",
            "nat",
            "NaT",
            "NOTAVALUE",
        ]
        for pattern in null_patterns:
            count_before = (df["choice"] == pattern).sum()
            if count_before > 0:
                logger.warning(
                    f"Found {count_before} instances of '{pattern}' in choice column, replacing with NOTAVALUE"
                )
                df["choice"] = df["choice"].replace(pattern, "NOTAVALUE")

        # Replace empty strings with NOTAVALUE to avoid NULL validation issues
        empty_count = (df["choice"] == "").sum()
        if empty_count > 0:
            logger.warning(
                f"Found {empty_count} empty strings in choice column, replacing with NOTAVALUE"
            )
            df["choice"] = df["choice"].replace("", "NOTAVALUE")

        # Also replace any remaining pandas nulls in choice column
        null_mask = df["choice"].isnull()
        if null_mask.sum() > 0:
            logger.warning(
                f"Found {null_mask.sum()} pandas null values in choice column, replacing with NOTAVALUE"
            )
            df.loc[null_mask, "choice"] = "NOTAVALUE"

        # Use proper CSV parameters for robust handling of complex data
        df.to_csv(
            csv_path, index=False, na_rep="NOTAVALUE", quoting=1
        )  # index=False to avoid pandas index issues
        logger.info(f"Successfully saved CSV to {csv_path}")

        # Create metadata JSON file
        metadata_filename = "meta_data.json"
        metadata_path = os.path.join(self.output_dir, metadata_filename)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Create ZIP file with CSV and metadata
        zip_filename = filename.replace(".csv", ".zip")
        zip_path = os.path.join(self.output_dir, zip_filename)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Add CSV file to zip
            zipf.write(csv_path, filename)
            # Add metadata JSON to zip
            zipf.write(metadata_path, metadata_filename)

        # Calculate and log overall accuracy
        total_correct = sum(r.correct_predictions for r in results)
        total_examples = sum(r.total_examples for r in results)
        overall_accuracy = total_correct / total_examples if total_examples > 0 else 0.0

        logger.info(f"CSV submission saved to: {csv_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
        logger.info(f"Submission package saved to: {zip_path}")
        logger.info(
            f"Overall accuracy (excluding open-ended questions): {overall_accuracy:.2%} ({total_correct}/{total_examples})"
        )

        return zip_path

    def save_submission_with_metadata(
        self,
        results: List[EvaluationResult],
        metadata: Dict = None,
        filename: str = "submission.csv",
        config_path: str = None,
        args: argparse.Namespace = None,
    ):
        """
        Convenient method to save submission with user-provided metadata as CSV with zip package

        Args:
            results: List of evaluation results
            metadata: User-provided metadata dictionary with fields like:
                - model_name: Name of the model
                - model_type: Type of model wrapper used
                - track: "internal_reasoning" or "agentic_reasoning"
                - base_model_type: "API" or "OpenWeighted"
                - base_model_name: Name of the base model
                - dataset: Dataset name
                - additional_info: Any additional information
            filename: Output CSV filename
            config_path: Path to configuration file containing metadata
            args: Command line arguments containing metadata
        """
        # Use the stored dataset examples from the last evaluation
        dataset_examples = getattr(self, "_last_dataset_examples", [])

        return self.save_submission(
            results, filename, metadata, dataset_examples, config_path, args
        )

    def list_datasets(self):
        """List available datasets"""
        print("Available Datasets:")
        print("-" * 50)
        for name, config in self.datasets.items():
            print(f"  {name}: {config['description']}")

    def load_metadata_from_config(self, config_path: str) -> Dict:
        """
        Load metadata from configuration file

        Args:
            config_path: Path to configuration file (JSON or YAML)

        Returns:
            Metadata dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        _, ext = os.path.splitext(config_path)

        # Use OmegaConf to load both JSON and YAML files
        try:
            config = OmegaConf.load(config_path)
        except Exception as e:
            raise ValueError(f"Failed to load config file {config_path}: {e}")

        # Extract metadata from config
        metadata = OmegaConf.to_container(config.get("metadata", config.get("meta_data", {})), resolve=True)

        # Validate required fields
        required_fields = [
            "model_name",
            "track",
            "base_model_type",
            "base_model_name",
            "dataset",
        ]
        for field in required_fields:
            if field not in metadata:
                logger.warning(f"Required metadata field '{field}' not found in config")

        return metadata

    def parse_metadata_from_args(self, args: argparse.Namespace) -> Dict:
        """
        Parse metadata from command line arguments

        Args:
            args: Parsed command line arguments

        Returns:
            Metadata dictionary
        """
        metadata = {}

        # Map argument names to metadata fields
        arg_mapping = {
            "model_name": "model_name",
            "model_type": "model_type",
            "track": "track",
            "base_model_type": "base_model_type",
            "base_model_name": "base_model_name",
            "dataset": "dataset",
            "additional_info": "additional_info",
        }

        for arg_name, meta_field in arg_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                metadata[meta_field] = getattr(args, arg_name)

        return metadata

    def get_metadata(
        self,
        config_path: str = None,
        args: argparse.Namespace = None,
        fallback_metadata: Dict = None,
    ) -> Dict:
        """
        Get metadata from various sources with priority order:
        1. Command line arguments (highest priority)
        2. Configuration file
        3. Fallback metadata provided
        4. Default metadata (lowest priority)

        Args:
            config_path: Path to configuration file
            args: Parsed command line arguments
            fallback_metadata: Fallback metadata dictionary

        Returns:
            Final metadata dictionary
        """
        # Start with default metadata
        metadata = {
            "model_name": self.model_name or "unknown",
            "model_type": type(self.model).__name__ if self.model else "Unknown",
            "track": "internal_reasoning",
            "base_model_type": "API",
            "base_model_name": self.model_name or "unknown",
            "dataset": "unknown",
            "additional_info": "Generated using eval_framework",
        }

        # Override with fallback metadata if provided
        if fallback_metadata:
            metadata.update(fallback_metadata)

        # Override with config file metadata if provided
        if config_path:
            try:
                config_metadata = self.load_metadata_from_config(config_path)
                metadata.update(config_metadata)
                logger.info(f"Loaded metadata from config file: {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config file {config_path}: {e}")

        # Override with command line arguments if provided (highest priority)
        if args:
            arg_metadata = self.parse_metadata_from_args(args)
            metadata.update(arg_metadata)
            if arg_metadata:
                logger.info(f"Applied metadata from command line arguments")

        return metadata


def load_config_file(config_path):
    """Load configuration from YAML or JSON file using OmegaConf"""
    if not os.path.exists(config_path):
        print(f"❌ Error: Configuration file not found: {config_path}")
        sys.exit(1)

    try:
        config = OmegaConf.load(config_path)
        return OmegaConf.to_container(config, resolve=True)
    except Exception as e:
        print(f"❌ Error loading config file {config_path}: {e}")
        sys.exit(1)
