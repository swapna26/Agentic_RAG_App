"""RAGAs-based evaluation system for RAG implementation."""

import asyncio
import json
import pandas as pd
import structlog
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness
)
from datasets import Dataset

from config import config
from utils.database_client import DatabaseClient
from utils.rag_client import RAGClient

logger = structlog.get_logger()


class RAGASEvaluator:
    """RAGAs-based evaluator for RAG system."""

    def __init__(self):
        self.db_client = DatabaseClient()
        self.rag_client = RAGClient()

        # Initialize models for RAGas 0.3.4 with improved compatibility
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama
        from ragas.llms import LlamaIndexLLMWrapper
        from ragas.embeddings import LlamaIndexEmbeddingsWrapper
        
        # Initialize LlamaIndex Ollama models with optimized settings
        llama_embedding = OllamaEmbedding(
            model_name=config.embedding_model,
            base_url=config.ollama_base_url,
            request_timeout=60.0
        )

        llama_llm = Ollama(
            model=config.llm_model,
            base_url=config.ollama_base_url,
            request_timeout=60.0,
            temperature=0.0,  # Use 0.0 for more consistent results
            additional_kwargs={"num_predict": 512}  # Limit response length
        )

        # Wrap with RAGas wrappers
        self.embedding_model = LlamaIndexEmbeddingsWrapper(llama_embedding)
        self.llm = LlamaIndexLLMWrapper(llama_llm)

        # RAGAs metrics mapping
        self.metrics_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_similarity": answer_similarity,
            "answer_correctness": answer_correctness
        }

    async def initialize(self):
        """Initialize the evaluator components."""
        try:
            await self.db_client.initialize()

            # Check if backend is healthy
            if not await self.rag_client.health_check():
                logger.warning("Backend API is not healthy, some features may not work")

            logger.info("RAGAs evaluator initialized successfully")
            return True

        except Exception as e:
            logger.error("Failed to initialize evaluator", error=str(e))
            return False

    async def load_test_dataset(self, dataset_path: str = None) -> List[Dict[str, Any]]:
        """Load test dataset from JSON file."""
        path = Path(dataset_path or config.test_dataset_path)

        if not path.exists():
            logger.warning("Test dataset not found, generating new one", path=str(path))
            return await self.generate_test_dataset()

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle both formats: direct list or wrapped in {"questions": [...]}
            if isinstance(data, list):
                dataset = data
            elif isinstance(data, dict) and "questions" in data:
                dataset = data["questions"]
            else:
                logger.error("Invalid dataset format", data_type=type(data).__name__)
                return []

            logger.info("Loaded test dataset", path=str(path), count=len(dataset))
            return dataset

        except Exception as e:
            logger.error("Failed to load test dataset", error=str(e), path=str(path))
            return []

    async def generate_test_dataset(self, num_questions: int = None) -> List[Dict[str, Any]]:
        """Generate test questions based on indexed documents."""
        num_questions = num_questions or config.num_test_questions

        try:
            # Get sample documents from the database
            sample_docs = await self.db_client.get_sample_documents(limit=min(num_questions, 20))

            if not sample_docs:
                logger.error("No documents found in database for test generation")
                return []

            test_questions = []

            for i, doc in enumerate(sample_docs[:num_questions]):
                # Generate question based on document content
                question_prompt = f"""Based on the following document content, generate a specific question that can be answered using this information:

Document: {doc['content'][:1000]}

Generate a clear, specific question that would require understanding this document to answer correctly. Return only the question, nothing else."""

                try:
                    question_response = self.llm.complete(question_prompt)
                    question = str(question_response).strip()

                    if question and len(question.split()) >= 5:  # Ensure meaningful question
                        test_questions.append({
                            "question": question,
                            "source_document": doc['source_file'],
                            "expected_context": doc['content'][:500]  # First 500 chars as expected context
                        })

                        logger.info(f"Generated question {i+1}/{num_questions}", question=question[:100])

                except Exception as e:
                    logger.warning(f"Failed to generate question for document {i+1}", error=str(e))
                    continue

            # Save generated dataset
            await self.save_test_dataset(test_questions)

            logger.info("Generated test dataset", count=len(test_questions))
            return test_questions

        except Exception as e:
            logger.error("Failed to generate test dataset", error=str(e))
            return []

    async def save_test_dataset(self, dataset: List[Dict[str, Any]]):
        """Save test dataset to JSON file."""
        try:
            path = Path(config.test_dataset_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)

            logger.info("Saved test dataset", path=str(path), count=len(dataset))

        except Exception as e:
            logger.error("Failed to save test dataset", error=str(e))

    async def collect_rag_responses(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collect RAG responses for evaluation questions."""
        responses = []

        for i, q_data in enumerate(questions):
            try:
                question = q_data["question"]

                # Query the RAG system (use CrewAI agents for evaluation)
                rag_response = await self.rag_client.query(question, use_agents=True)

                if "error" in rag_response:
                    logger.warning(f"RAG query failed for question {i+1}", error=rag_response["error"])
                    continue

                # Debug: Log the full RAG response structure
                logger.info(f"RAG Response structure for question {i+1}", 
                           rag_response_keys=list(rag_response.keys()) if isinstance(rag_response, dict) else "Not a dict",
                           context_key_exists="context" in rag_response if isinstance(rag_response, dict) else False,
                           context_value=rag_response.get("context", "No context key") if isinstance(rag_response, dict) else "Not a dict")

                response_data = {
                    "question": question,
                    "answer": rag_response.get("answer", ""),
                    "contexts": [ctx.get("content", "") for ctx in rag_response.get("context", [])],
                    "ground_truth": q_data.get("ground_truth", q_data.get("expected_context", "")),
                    "source_document": q_data.get("source_document", "")
                }

                responses.append(response_data)

                logger.info(f"Collected response {i+1}/{len(questions)}",
                          question=question[:50],
                          answer_length=len(response_data["answer"]),
                          contexts_count=len(response_data["contexts"]),
                          contexts_sample=response_data["contexts"][:2] if response_data["contexts"] else "No contexts",
                          ground_truth_length=len(response_data["ground_truth"]))

            except Exception as e:
                logger.error(f"Failed to collect response for question {i+1}", error=str(e))
                continue

        return responses


    async def evaluate_with_ragas(self, responses: List[Dict[str, Any]],
                                selected_metrics: List[str] = None) -> Dict[str, Any]:
        """Evaluate responses using RAGAs metrics."""
        try:
            if not responses:
                logger.error("No responses to evaluate")
                return {}

            # Prepare data for RAGAs 0.3.4 with proper format
            eval_data = {
                "question": [r["question"] for r in responses],
                "answer": [r["answer"] for r in responses],
                "contexts": [r["contexts"] for r in responses],
                "ground_truth": [r["ground_truth"] for r in responses]
            }

            # Ensure contexts are properly formatted as list of strings
            for i, contexts in enumerate(eval_data["contexts"]):
                if contexts and len(contexts) > 0:
                    # Convert to list of strings if needed
                    if isinstance(contexts[0], dict):
                        eval_data["contexts"][i] = [ctx.get("content", str(ctx)) for ctx in contexts]
                    else:
                        eval_data["contexts"][i] = [str(ctx) for ctx in contexts]
                else:
                    eval_data["contexts"][i] = [""]  # Ensure non-empty contexts

            # Create dataset
            dataset = Dataset.from_dict(eval_data)
            
            # Log dataset info for debugging
            logger.info("Dataset created successfully", 
                       columns=list(dataset.column_names),
                       num_rows=len(dataset))

            # Select metrics to evaluate with Ollama compatibility
            metrics_to_use = selected_metrics or config.metrics
            active_metrics = []
            
            # Prioritize metrics that work better with Ollama
            ollama_compatible_metrics = ["answer_similarity", "context_recall", "answer_relevancy"]
            
            for metric_name in metrics_to_use:
                if metric_name in self.metrics_map:
                    # Prioritize Ollama-compatible metrics
                    if metric_name in ollama_compatible_metrics:
                        active_metrics.insert(0, self.metrics_map[metric_name])
                    else:
                        active_metrics.append(self.metrics_map[metric_name])
                else:
                    logger.warning(f"Unknown metric: {metric_name}")

            if not active_metrics:
                logger.error("No valid metrics selected")
                return {}
                
            logger.info("Selected metrics for evaluation", 
                       metrics=[m.__name__ if hasattr(m, '__name__') else str(m) for m in active_metrics])

            logger.info("Starting RAGAs evaluation",
                       responses_count=len(responses),
                       metrics=metrics_to_use)

            # Debug: Print dataset info
            logger.info("Dataset info", 
                       num_rows=len(dataset),
                       columns=list(dataset.column_names),
                       sample_question=dataset[0]["question"][:100] if len(dataset) > 0 else "No data")
            
            # Debug: Check each row for problematic fields
            for i, row in enumerate(dataset):
                logger.info(f"Row {i} debug",
                           question_length=len(str(row.get('question', ''))),
                           answer_length=len(str(row.get('answer', ''))),
                           contexts_count=len(row.get('contexts', [])),
                           ground_truth_length=len(str(row.get('ground_truth', ''))),
                           has_contexts=bool(row.get('contexts')),
                           has_ground_truth=bool(row.get('ground_truth')),
                           contexts_sample=str(row.get('contexts', []))[:100] if row.get('contexts') else "No contexts",
                           ground_truth_sample=str(row.get('ground_truth', ''))[:100] if row.get('ground_truth') else "No ground truth")

            # Configure RAGAs with improved error handling and timeout management
            try:
                logger.info("Starting RAGas evaluation with optimized settings")
                
                # Set up evaluation with timeout and error handling
                import asyncio
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("RAGas evaluation timed out")
                
                # Set timeout for evaluation (5 minutes)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(300)
                
                try:
                    result = evaluate(
                        dataset=dataset,
                        metrics=active_metrics,
                        llm=self.llm,
                        embeddings=self.embedding_model
                    )
                    signal.alarm(0)  # Cancel timeout
                    logger.info("RAGAs evaluation completed successfully", result_type=type(result).__name__)
                except TimeoutError:
                    signal.alarm(0)
                    logger.error("RAGas evaluation timed out after 5 minutes")
                    return {"error": "RAGas evaluation timed out"}
                except Exception as eval_error:
                    signal.alarm(0)
                    logger.error("RAGas evaluation failed", error=str(eval_error))
                    return {"error": f"RAGAs evaluation failed: {str(eval_error)}"}
                    
            except Exception as e:
                logger.error("RAGAs evaluation setup failed", error=str(e), error_type=type(e).__name__)
                import traceback
                logger.error("Full traceback", traceback=traceback.format_exc())
                return {"error": f"RAGAs evaluation setup failed: {str(e)}"}

            # Convert results to dict
            evaluation_results = {
                "timestamp": datetime.now().isoformat(),
                "total_questions": len(responses),
                "metrics_evaluated": metrics_to_use,
                "scores": {}
            }

            # Debug: Print result info
            logger.info("RAGAs result info", 
                       result_type=type(result).__name__,
                       result_str=str(result)[:200] if result else "None")

            # Extract metric scores from EvaluationResult object (RAGas 0.3.4)
            try:
                import numpy as np
                import pandas as pd
                
                # RAGas 0.3.4 returns a DataFrame with results
                if hasattr(result, 'to_pandas'):
                    # Convert to pandas DataFrame
                    df = result.to_pandas()
                    logger.info("RAGas result DataFrame", shape=df.shape, columns=list(df.columns))
                    
                    # Process each metric
                    for metric_name in metrics_to_use:
                        try:
                            if metric_name in df.columns:
                                # Get the scores for this metric
                                scores = df[metric_name].values
                                
                                # Handle NaN values
                                valid_scores = scores[~pd.isna(scores)]
                                
                                if len(valid_scores) == 0:
                                    logger.warning(f"Metric {metric_name} returned all NaN values")
                                    evaluation_results["scores"][metric_name] = {
                                        "mean": 0.0,
                                        "std": 0.0,
                                        "min": 0.0,
                                        "max": 0.0,
                                        "individual_scores": [],
                                        "note": "All values were NaN"
                                    }
                                else:
                                    # Calculate statistics
                                    evaluation_results["scores"][metric_name] = {
                                        "mean": float(np.mean(valid_scores)),
                                        "std": float(np.std(valid_scores)),
                                        "min": float(np.min(valid_scores)),
                                        "max": float(np.max(valid_scores)),
                                        "individual_scores": [float(s) for s in valid_scores]
                                    }
                                    logger.info(f"Metric {metric_name} processed successfully", 
                                              mean=float(np.mean(valid_scores)),
                                              count=len(valid_scores))
                            else:
                                logger.warning(f"Metric {metric_name} not found in results")
                                evaluation_results["scores"][metric_name] = {
                                    "mean": 0.0,
                                    "std": 0.0,
                                    "min": 0.0,
                                    "max": 0.0,
                                    "individual_scores": [],
                                    "note": "Metric not found in results"
                                }
                        except Exception as e:
                            logger.error(f"Failed to process metric {metric_name}", error=str(e))
                            evaluation_results["scores"][metric_name] = {
                                "mean": 0.0,
                                "std": 0.0,
                                "min": 0.0,
                                "max": 0.0,
                                "individual_scores": [],
                                "note": f"Error processing metric: {str(e)}"
                            }
                else:
                    # Fallback: try to access scores directly
                    logger.warning("RAGas result does not have to_pandas method, trying direct access")
                    
                    # Try different ways to access the scores
                    scores_dict = {}
                    if hasattr(result, 'scores'):
                        scores_dict = result.scores
                    elif hasattr(result, '_scores_dict'):
                        scores_dict = result._scores_dict
                    elif hasattr(result, '__dict__'):
                        scores_dict = result.__dict__.get('scores', {})
                    
                    logger.info("Scores dict", scores=scores_dict)
                    
                    for metric_name in metrics_to_use:
                        try:
                            if metric_name in scores_dict:
                                score = scores_dict[metric_name]
                                
                                if isinstance(score, (list, tuple, np.ndarray)):
                                    # Convert to numpy array
                                    score_array = np.array(score)
                                    valid_scores = score_array[~np.isnan(score_array)]
                                    
                                    if len(valid_scores) == 0:
                                        evaluation_results["scores"][metric_name] = {
                                            "mean": 0.0,
                                            "std": 0.0,
                                            "min": 0.0,
                                            "max": 0.0,
                                            "individual_scores": [],
                                            "note": "All values were NaN"
                                        }
                                    else:
                                        evaluation_results["scores"][metric_name] = {
                                            "mean": float(np.mean(valid_scores)),
                                            "std": float(np.std(valid_scores)),
                                            "min": float(np.min(valid_scores)),
                                            "max": float(np.max(valid_scores)),
                                            "individual_scores": [float(s) for s in valid_scores]
                                        }
                                else:
                                    # Single value
                                    if pd.isna(score):
                                        evaluation_results["scores"][metric_name] = {
                                            "mean": 0.0,
                                            "std": 0.0,
                                            "min": 0.0,
                                            "max": 0.0,
                                            "individual_scores": [],
                                            "note": "Single NaN value"
                                        }
                                    else:
                                        evaluation_results["scores"][metric_name] = {
                                            "mean": float(score),
                                            "std": 0.0,
                                            "min": float(score),
                                            "max": float(score),
                                            "individual_scores": [float(score)]
                                        }
                            else:
                                logger.warning(f"Metric {metric_name} not found in scores dict")
                                evaluation_results["scores"][metric_name] = {
                                    "mean": 0.0,
                                    "std": 0.0,
                                    "min": 0.0,
                                    "max": 0.0,
                                    "individual_scores": [],
                                    "note": "Metric not found in scores"
                                }
                        except Exception as e:
                            logger.error(f"Failed to process metric {metric_name}", error=str(e))
                            evaluation_results["scores"][metric_name] = {
                                "mean": 0.0,
                                "std": 0.0,
                                "min": 0.0,
                                "max": 0.0,
                                "individual_scores": [],
                                "note": f"Error processing metric: {str(e)}"
                            }
                        
            except Exception as e:
                logger.error("Failed to process evaluation result", error=str(e))
                return {"error": f"Failed to process evaluation result: {str(e)}"}

            logger.info("RAGAs evaluation completed",
                       metrics_count=len(evaluation_results["scores"]))

            return evaluation_results

        except Exception as e:
            logger.error("RAGAs evaluation failed", error=str(e))
            return {"error": str(e)}

    async def generate_evaluation_report(self, results: Dict[str, Any],
                                       output_path: str = None) -> str:
        """Generate comprehensive evaluation report."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"crew_ai_evaluation_report_{timestamp}.json"
            report_path = Path(config.reports_dir) / report_filename

            if output_path:
                report_path = Path(output_path)

            # Add system information
            system_info = await self.rag_client.get_system_info()
            db_stats = {
                "document_count": await self.db_client.get_document_count(),
                "chunk_count": await self.db_client.get_chunk_count(),
                "source_files": await self.db_client.get_all_source_files()
            }

            comprehensive_report = {
                "evaluation_metadata": {
                    "timestamp": results.get("timestamp", datetime.now().isoformat()),
                    "evaluator_version": "1.0.0",
                    "ragas_version": "0.1.0",  # Update with actual version
                    "processing_mode": "crew_ai_agents",
                    "configuration": {
                        "embedding_model": config.embedding_model,
                        "llm_model": config.llm_model,
                        "top_k": config.top_k,
                        "chunk_size": config.chunk_size,
                        "agents_enabled": True,
                        "agent_processing": "CrewAI Multi-Agent System"
                    }
                },
                "system_info": system_info,
                "database_stats": db_stats,
                "evaluation_results": results,
                "summary": self._generate_summary(results)
            }

            # Save comprehensive report
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)

            logger.info("Generated evaluation report", path=str(report_path))
            return str(report_path)

        except Exception as e:
            logger.error("Failed to generate evaluation report", error=str(e))
            return ""

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of evaluation results."""
        summary = {
            "overall_performance": "unknown",
            "best_metric": None,
            "worst_metric": None,
            "recommendations": []
        }

        try:
            if "scores" not in results:
                return summary

            scores = results["scores"]
            metric_averages = {}

            # Calculate average scores
            for metric, score_data in scores.items():
                if isinstance(score_data, dict) and "mean" in score_data:
                    metric_averages[metric] = score_data["mean"]
                else:
                    metric_averages[metric] = float(score_data)

            if metric_averages:
                # Find best and worst metrics
                summary["best_metric"] = max(metric_averages, key=metric_averages.get)
                summary["worst_metric"] = min(metric_averages, key=metric_averages.get)

                # Overall performance assessment
                overall_avg = sum(metric_averages.values()) / len(metric_averages)
                if overall_avg >= 0.8:
                    summary["overall_performance"] = "excellent"
                elif overall_avg >= 0.6:
                    summary["overall_performance"] = "good"
                elif overall_avg >= 0.4:
                    summary["overall_performance"] = "fair"
                else:
                    summary["overall_performance"] = "poor"

                # Generate recommendations
                summary["recommendations"] = self._generate_recommendations(metric_averages)

        except Exception as e:
            logger.error("Failed to generate summary", error=str(e))

        return summary

    def _generate_recommendations(self, metric_scores: Dict[str, float]) -> List[str]:
        """Generate recommendations based on metric scores."""
        recommendations = []

        try:
            for metric, score in metric_scores.items():
                if score < 0.5:
                    if metric == "faithfulness":
                        recommendations.append("Consider improving context relevance and reducing hallucinations in responses")
                    elif metric == "answer_relevancy":
                        recommendations.append("Improve query understanding and response relevance")
                    elif metric == "context_precision":
                        recommendations.append("Enhance retrieval quality to get more relevant context")
                    elif metric == "context_recall":
                        recommendations.append("Increase context retrieval coverage and chunk size")
                    elif metric == "answer_similarity":
                        recommendations.append("Improve answer generation quality and consistency")
                    elif metric == "answer_correctness":
                        recommendations.append("Focus on factual accuracy and completeness of answers")

            if not recommendations:
                recommendations.append("Overall performance is good. Consider fine-tuning for specific use cases.")

        except Exception as e:
            logger.error("Failed to generate recommendations", error=str(e))

        return recommendations

    async def run_complete_evaluation(self, custom_questions: List[str] = None,
                                    metrics: List[str] = None) -> str:
        """Run complete evaluation pipeline."""
        try:
            logger.info("Starting complete RAG evaluation")

            # Load or generate test dataset
            if custom_questions:
                test_data = [{"question": q, "source_document": "custom", "expected_context": ""}
                           for q in custom_questions]
            else:
                test_data = await self.load_test_dataset()

            if not test_data:
                logger.error("No test data available")
                return ""

            # Collect RAG responses
            responses = await self.collect_rag_responses(test_data)

            if not responses:
                logger.error("No responses collected")
                return ""

            # Evaluate with RAGAs
            evaluation_results = await self.evaluate_with_ragas(responses, metrics)

            if not evaluation_results or "error" in evaluation_results:
                logger.error("Evaluation failed", error=evaluation_results.get("error"))
                return ""

            # Generate report
            report_path = await self.generate_evaluation_report(evaluation_results)

            if report_path:
                logger.info("Complete evaluation finished successfully", report=report_path)
                return report_path
            else:
                logger.error("Failed to generate evaluation report")
                return ""

        except Exception as e:
            logger.error("Complete evaluation failed", error=str(e))
            return ""

    async def cleanup(self):
        """Cleanup resources."""
        try:
            await self.db_client.cleanup()
            await self.rag_client.cleanup()
            logger.info("Evaluator cleanup completed")
        except Exception as e:
            logger.error("Error during cleanup", error=str(e))