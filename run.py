#!/usr/bin/env python3
"""
Bio-Medical AI Competition - Hydra-powered Evaluation Script

Advanced evaluation script using Hydra for configuration management.

Usage:
    # Basic usage with defaults
    python run.py

    # Override configuration groups
    python run.py model=ii_medical_7b                  # Use 7B model
    python run.py dataset=cure_bench_test              # Use test dataset
    
    # Override specific nested values
    python run.py model.config.temperature=0.7        # Override model temperature
    python run.py evaluation.max_workers=8            # Override worker count
    
    # Complex combinations
    python run.py model=ii_medical_7b dataset=cure_bench_test evaluation.max_workers=8 output.dir=results
    
    # View configuration
    python run.py --cfg job                           # Show resolved config
    python run.py --help                              # Show all options
"""

import os
from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
from eval_framework import CompetitionKit


load_dotenv()


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig) -> None:
    """Run evaluation with Hydra configuration management"""    
    # Print the resolved configuration
    print("📋 Resolved Configuration:")

    # Initialize the competition kit
    kit = CompetitionKit(config=config)
    
    print(f"Loading model: {config.model.model_name}")
    kit.load_model()
    
    # Show available datasets
    print("Available datasets:")
    kit.list_datasets()
    
    # Run evaluation
    print(f"Running evaluation on dataset: {config.dataset.name}")
    results = kit.evaluate(config.dataset.name)
    
    # Generate submission with metadata from config
    print("Generating submission with metadata...")
    submission_path = kit.save_submission_with_metadata(
        results=[results],
        filename=config.output.file,
    )
    
    print(f"\n✅ Evaluation completed successfully!")
    print(
        f"📊 Accuracy: {results.accuracy:.2%} ({results.correct_predictions}/{results.total_examples})"
    )
    print(f"📄 Submission saved to: {submission_path}")


if __name__ == "__main__":
    main()
