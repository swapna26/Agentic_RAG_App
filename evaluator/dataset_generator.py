"""Test dataset generator for RAG evaluation."""

import asyncio
import json
import structlog
from pathlib import Path
from typing import List, Dict, Any, Optional

from llama_index.llms.ollama import Ollama

from config import config
from utils.database_client import DatabaseClient

logger = structlog.get_logger()


class TestDatasetGenerator:
    """Generates test questions and ground truth for RAG evaluation."""

    def __init__(self):
        self.db_client = DatabaseClient()
        self.llm = Ollama(
            model=config.llm_model,
            base_url=config.ollama_base_url,
            request_timeout=120.0
        )

    async def initialize(self):
        """Initialize the dataset generator."""
        try:
            await self.db_client.initialize()
            logger.info("Dataset generator initialized successfully")
            return True
        except Exception as e:
            logger.error("Failed to initialize dataset generator", error=str(e))
            return False

    async def generate_questions_from_documents(self, num_questions: int = 20) -> List[Dict[str, Any]]:
        """Generate questions based on indexed documents."""
        try:
            # Get all source files
            source_files = await self.db_client.get_all_source_files()

            if not source_files:
                logger.error("No documents found in database")
                return []

            questions = []
            questions_per_doc = max(1, num_questions // len(source_files))

            logger.info("Generating questions from documents",
                       total_docs=len(source_files),
                       questions_per_doc=questions_per_doc)

            for doc_file in source_files:
                doc_questions = await self._generate_questions_for_document(
                    doc_file, questions_per_doc
                )
                questions.extend(doc_questions)

                if len(questions) >= num_questions:
                    break

            # Trim to exact number if needed
            questions = questions[:num_questions]

            logger.info("Generated questions from documents", total_questions=len(questions))
            return questions

        except Exception as e:
            logger.error("Failed to generate questions from documents", error=str(e))
            return []

    async def _generate_questions_for_document(self, source_file: str,
                                             num_questions: int) -> List[Dict[str, Any]]:
        """Generate questions for a specific document."""
        try:
            # Get chunks for the document
            chunks = await self.db_client.get_document_chunks(source_file)

            if not chunks:
                logger.warning("No chunks found for document", source_file=source_file)
                return []

            # Select diverse chunks for question generation
            selected_chunks = self._select_diverse_chunks(chunks, num_questions * 2)

            questions = []
            for i, chunk in enumerate(selected_chunks[:num_questions]):
                try:
                    question_data = await self._generate_question_from_chunk(
                        chunk, source_file, i + 1
                    )

                    if question_data:
                        questions.append(question_data)

                except Exception as e:
                    logger.warning(f"Failed to generate question {i+1} for {source_file}",
                                 error=str(e))
                    continue

            logger.info("Generated questions for document",
                       source_file=source_file,
                       questions_count=len(questions))

            return questions

        except Exception as e:
            logger.error("Failed to generate questions for document",
                        source_file=source_file,
                        error=str(e))
            return []

    def _select_diverse_chunks(self, chunks: List[Dict[str, Any]],
                              num_chunks: int) -> List[Dict[str, Any]]:
        """Select diverse chunks for question generation."""
        if len(chunks) <= num_chunks:
            return chunks

        # Simple strategy: select chunks evenly distributed across the document
        step = len(chunks) // num_chunks
        selected = []

        for i in range(0, len(chunks), step):
            if len(selected) < num_chunks:
                selected.append(chunks[i])

        return selected

    async def _generate_question_from_chunk(self, chunk: Dict[str, Any],
                                          source_file: str,
                                          question_num: int) -> Optional[Dict[str, Any]]:
        """Generate a question from a document chunk."""
        try:
            content = chunk["content"]
            metadata = chunk.get("metadata", {})

            # Skip very short chunks
            if len(content.split()) < 10:
                return None

            # Generate different types of questions
            question_types = [
                "factual",
                "conceptual",
                "analytical",
                "procedural"
            ]

            question_type = question_types[question_num % len(question_types)]

            prompt = self._get_question_generation_prompt(content, question_type)

            # Generate question
            response = self.llm.complete(prompt)
            generated_text = str(response).strip()

            # Parse the generated text to extract question and answer
            question, answer = self._parse_generated_qa(generated_text)

            if not question or len(question.split()) < 5:
                return None

            # Generate ground truth answer if not provided
            if not answer:
                answer = await self._generate_ground_truth_answer(question, content)

            return {
                "question": question,
                "ground_truth": answer,
                "context": content,
                "source_document": source_file,
                "question_type": question_type,
                "chunk_metadata": metadata,
                "chunk_id": metadata.get("chunk_id", 0)
            }

        except Exception as e:
            logger.error("Failed to generate question from chunk", error=str(e))
            return None

    def _get_question_generation_prompt(self, content: str, question_type: str) -> str:
        """Get prompt for question generation based on type."""
        prompts = {
            "factual": f"""Based on the following text, generate a factual question that asks for specific information contained in the text. The question should have a clear, factual answer.

Text: {content}

Generate a question and its answer in the format:
Question: [your question]
Answer: [the answer from the text]

Make the question specific and answerable from the given text.""",

            "conceptual": f"""Based on the following text, generate a conceptual question that asks about the meaning, definition, or explanation of concepts mentioned in the text.

Text: {content}

Generate a question and its answer in the format:
Question: [your question]
Answer: [explanation based on the text]

Focus on understanding and explaining concepts.""",

            "analytical": f"""Based on the following text, generate an analytical question that requires analyzing, comparing, or evaluating information in the text.

Text: {content}

Generate a question and its answer in the format:
Question: [your question]
Answer: [analytical answer based on the text]

The question should require analysis or evaluation of the information.""",

            "procedural": f"""Based on the following text, generate a procedural question about processes, steps, or how-to information mentioned in the text.

Text: {content}

Generate a question and its answer in the format:
Question: [your question]
Answer: [procedural answer based on the text]

Focus on processes, procedures, or step-by-step information."""
        }

        return prompts.get(question_type, prompts["factual"])

    def _parse_generated_qa(self, generated_text: str) -> tuple[Optional[str], Optional[str]]:
        """Parse generated Q&A text to extract question and answer."""
        try:
            lines = generated_text.strip().split('\n')
            question = None
            answer = None

            for line in lines:
                line = line.strip()
                if line.startswith("Question:"):
                    question = line[9:].strip()
                elif line.startswith("Answer:"):
                    answer = line[7:].strip()

            return question, answer

        except Exception as e:
            logger.error("Failed to parse generated Q&A", error=str(e))
            return None, None

    async def _generate_ground_truth_answer(self, question: str, context: str) -> str:
        """Generate ground truth answer for a question based on context."""
        try:
            prompt = f"""Answer the following question based strictly on the provided context. Be concise and accurate.

Context: {context}

Question: {question}

Answer:"""

            response = self.llm.complete(prompt)
            return str(response).strip()

        except Exception as e:
            logger.error("Failed to generate ground truth answer", error=str(e))
            return ""

    async def generate_custom_questions(self, questions_list: List[str]) -> List[Dict[str, Any]]:
        """Generate test data for custom questions."""
        try:
            test_data = []

            for i, question in enumerate(questions_list):
                # For custom questions, we need to find relevant context from the database
                # This is a simplified approach - in practice you might want more sophisticated matching

                # Generate embedding for the question to find relevant context
                # For now, we'll create a basic structure
                test_item = {
                    "question": question,
                    "ground_truth": "",  # Will need to be provided or generated
                    "context": "",  # Will be filled by retrieval
                    "source_document": "custom",
                    "question_type": "custom",
                    "chunk_metadata": {},
                    "chunk_id": i
                }

                test_data.append(test_item)

            logger.info("Generated custom test data", count=len(test_data))
            return test_data

        except Exception as e:
            logger.error("Failed to generate custom questions", error=str(e))
            return []

    async def save_dataset(self, dataset: List[Dict[str, Any]], output_path: str = None):
        """Save generated dataset to JSON file."""
        try:
            path = Path(output_path or config.test_dataset_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Add metadata
            dataset_with_metadata = {
                "metadata": {
                    "generated_at": str(asyncio.get_event_loop().time()),
                    "total_questions": len(dataset),
                    "generator_config": {
                        "llm_model": config.llm_model,
                        "embedding_model": config.embedding_model
                    }
                },
                "questions": dataset
            }

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(dataset_with_metadata, f, indent=2, ensure_ascii=False)

            logger.info("Saved test dataset", path=str(path), questions=len(dataset))

        except Exception as e:
            logger.error("Failed to save dataset", error=str(e))

    async def load_dataset(self, dataset_path: str = None) -> List[Dict[str, Any]]:
        """Load dataset from JSON file."""
        try:
            path = Path(dataset_path or config.test_dataset_path)

            if not path.exists():
                logger.warning("Dataset file not found", path=str(path))
                return []

            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle both old format (list) and new format (dict with metadata)
            if isinstance(data, list):
                questions = data
            elif isinstance(data, dict) and "questions" in data:
                questions = data["questions"]
            else:
                logger.error("Invalid dataset format")
                return []

            logger.info("Loaded test dataset", path=str(path), questions=len(questions))
            return questions

        except Exception as e:
            logger.error("Failed to load dataset", error=str(e), path=dataset_path)
            return []

    async def cleanup(self):
        """Cleanup resources."""
        try:
            await self.db_client.cleanup()
            logger.info("Dataset generator cleanup completed")
        except Exception as e:
            logger.error("Error during cleanup", error=str(e))