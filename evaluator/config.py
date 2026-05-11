"""Configuration for RAGAs Evaluator."""

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class EvaluatorConfig:
    """Configuration for the RAGAs evaluator."""

    # Database configuration
    database_url: str = "postgresql://raguser:ragpassword@localhost:5432/agentic_rag"
    vector_table: str = "data_llamaindex_vectors_copy"

    # Ollama configuration
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text:v1.5"
    llm_model: str = "llama3.2:1b"

    # Backend API configuration
    backend_api_url: str = "http://localhost:8000"

    # Evaluation configuration
    chunk_size: int = 1024
    chunk_overlap: int = 128
    top_k: int = 5

    # RAGAs metrics to evaluate
    metrics: List[str] = None

    # Test dataset configuration
    num_test_questions: int = 20
    test_dataset_path: str = str(Path(__file__).parent / "datasets" / "test_questions.json")

    # Output configuration
    reports_dir: str = str(Path(__file__).parent / "reports")
    log_level: str = "INFO"

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
                "answer_similarity",
                "answer_correctness"
            ]

        # Create directories if they don't exist
        Path(self.reports_dir).mkdir(parents=True, exist_ok=True)
        Path(self.test_dataset_path).parent.mkdir(parents=True, exist_ok=True)


# Default configuration instance
config = EvaluatorConfig()