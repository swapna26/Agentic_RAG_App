"""RAG client for interacting with the backend API."""

import httpx
import structlog
from typing import Dict, Any, List, Optional
from config import config

logger = structlog.get_logger()


class RAGClient:
    """Client for interacting with the RAG backend API."""

    def __init__(self, base_url: str = None):
        self.base_url = base_url or config.backend_api_url
        self.client = httpx.AsyncClient(timeout=300.0)

    async def query(self, question: str, top_k: int = None, use_agents: bool = True) -> Dict[str, Any]:
        """Query the RAG system using OpenWebUI-compatible chat completions endpoint."""
        try:
            # For evaluation, use CrewAI agents by default
            model_name = "agentic-rag-ollama"

            # Add instruction to use CrewAI agents if enabled
            if use_agents:
                question = f"[CREW_AI] {question}"
            else:
                question = f"[SIMPLE_RAG] {question}"

            # Use OpenWebUI-compatible chat completions format
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": question}
                ],
                "stream": False,
                "temperature": 0.1,
                "max_tokens": 4000
            }

            response = await self.client.post(
                f"{self.base_url}/api/chat/completions",
                json=payload
            )
            response.raise_for_status()

            chat_response = response.json()

            # Extract answer from OpenWebUI chat format
            answer = ""
            sources = []

            if "choices" in chat_response and chat_response["choices"]:
                content = chat_response["choices"][0]["message"]["content"]

                # Split content to separate answer from sources
                # Try multiple source formats
                parts = content.split("**Sources:**")
                if len(parts) == 1:
                    parts = content.split("\n\n Sources")
                if len(parts) == 1:
                    parts = content.split("\n\nSources")
                
                answer = parts[0].strip()

                # Extract sources if present
                if len(parts) > 1:
                    source_lines = parts[1].strip().split('\n')
                    for line in source_lines:
                        line = line.strip()
                        if line and (line.startswith(('1.', '2.', '3.', '4.', '5.')) or 
                                   line.startswith(('•', '-', '*')) or
                                   'relevance:' in line.lower()):
                            # Clean up the source line
                            clean_line = line
                            if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                                clean_line = line[2:].strip()  # Remove numbering
                            
                            sources.append({"content": clean_line})

            # Return in expected format for evaluator
            return {
                "answer": answer,
                "context": sources,  # Use sources as context
                "sources": sources
            }

        except Exception as e:
            logger.error("Failed to query RAG system", error=str(e), question=question)
            return {
                "answer": "",
                "context": [],
                "sources": [],
                "error": str(e)
            }

    async def health_check(self) -> bool:
        """Check if the backend API is healthy."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error("Backend health check failed", error=str(e))
            return False

    async def get_system_info(self) -> Dict[str, Any]:
        """Get system information from the backend."""
        try:
            response = await self.client.get(f"{self.base_url}/info")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Failed to get system info", error=str(e))
            return {}

    async def cleanup(self):
        """Close the HTTP client."""
        await self.client.aclose()