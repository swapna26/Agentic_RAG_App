#!/usr/bin/env python3
"""Simple evaluation without Click - direct function calls."""

import asyncio
import sys
from pathlib import Path

from config import config
from ragas_evaluator import RAGASEvaluator
from dataset_generator import TestDatasetGenerator

def generate_dataset(num_questions=20, output=None):
    """Generate test dataset from indexed documents."""
    async def run_generation():
        generator = TestDatasetGenerator()
        
        if not await generator.initialize():
            print(" Failed to initialize dataset generator")
            return False
            
        print(" Generating test dataset...")
        questions = await generator.generate_questions_from_documents(num_questions)
        
        if questions:
            print(f" Generated {len(questions)} test questions")
            return True
        else:
            print(" Failed to generate test dataset")
            return False
    
    return asyncio.run(run_generation())

def evaluate(dataset=None, metrics=None):
    """Run RAGAs evaluation on the RAG system."""
    async def run_evaluation():
        evaluator = RAGASEvaluator()
        
        if not await evaluator.initialize():
            print(" Failed to initialize evaluator")
            return False
            
        print(" Starting RAGAs evaluation...")
        report_path = await evaluator.run_complete_evaluation(
            custom_questions=None,
            metrics=list(metrics) if metrics else None
        )
        
        if report_path:
            print(f" Evaluation completed! Report: {report_path}")
            return True
        else:
            print(" Evaluation failed")
            return False
    
    return asyncio.run(run_evaluation())

def show_report(report_path):
    """Display evaluation report."""
    try:
        import json
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        print(f"\n Evaluation Report")
        print(f"Timestamp: {report.get('evaluation_metadata', {}).get('timestamp', 'Unknown')}")
        
        scores = report.get('evaluation_results', {}).get('scores', {})
        for metric, data in scores.items():
            mean_score = data.get('mean', 0.0)
            print(f"{metric}: {mean_score:.3f}")
            
    except Exception as e:
        print(f" Failed to display report: {e}")

def main():
    """Main function - simple command handling."""
    if len(sys.argv) < 2:
        print("Usage: python3 simple_main.py <command> [options]")
        print("Commands: generate-dataset, evaluate, show-report")
        return
    
    command = sys.argv[1]
    
    if command == "generate-dataset":
        num_questions = 20
        if len(sys.argv) > 2:
            try:
                num_questions = int(sys.argv[2])
            except ValueError:
                print(" Invalid number of questions")
                return
        generate_dataset(num_questions)
        
    elif command == "evaluate":
        dataset = None
        if len(sys.argv) > 2:
            dataset = sys.argv[2]
        evaluate(dataset)
        
    elif command == "show-report":
        if len(sys.argv) < 3:
            print(" Please provide report path")
            return
        show_report(sys.argv[2])
        
    else:
        print(f" Unknown command: {command}")

if __name__ == "__main__":
    main()
