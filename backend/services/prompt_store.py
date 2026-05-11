"""Local prompt storage with versioning for RAG prompt management."""

import json
import structlog
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

logger = structlog.get_logger()


class PromptVersion(BaseModel):
    """A single version of a prompt template."""
    version: int
    template: str
    variables: List[str]
    created_at: str
    created_by: str = "system"
    changelog: str = ""


class PromptEntry(BaseModel):
    """A prompt with all its versions."""
    name: str
    description: str = ""
    tags: List[str] = []
    active_version: int
    versions: List[PromptVersion]


class PromptStore:
    """Manages prompt templates with local JSON storage and versioning."""

    def __init__(self, storage_path: str = None):
        default_path = Path(__file__).parent.parent / "prompts" / "prompts.json"
        self.storage_path = Path(storage_path) if storage_path else default_path
        self.prompts: Dict[str, PromptEntry] = {}

    async def initialize(self):
        """Load prompts from disk. Seed defaults if file doesn't exist."""
        try:
            if self.storage_path.exists():
                self._load_from_disk()
                logger.info("Prompt store loaded", count=len(self.prompts))
            else:
                self._seed_defaults()
                self._save_to_disk()
                logger.info("Prompt store seeded with defaults", count=len(self.prompts))
            return True
        except Exception as e:
            logger.error("Failed to initialize prompt store", error=str(e))
            self._seed_defaults()
            return True

    def _load_from_disk(self):
        """Load prompts from JSON file."""
        with open(self.storage_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.prompts = {}
        for prompt_id, prompt_data in data.get("prompts", {}).items():
            versions = [PromptVersion(**v) for v in prompt_data.get("versions", [])]
            self.prompts[prompt_id] = PromptEntry(
                name=prompt_data["name"],
                description=prompt_data.get("description", ""),
                tags=prompt_data.get("tags", []),
                active_version=prompt_data.get("active_version", 1),
                versions=versions,
            )

    def _save_to_disk(self):
        """Persist prompts to JSON file."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        data = {"prompts": {}}
        for prompt_id, entry in self.prompts.items():
            data["prompts"][prompt_id] = {
                "name": entry.name,
                "description": entry.description,
                "tags": entry.tags,
                "active_version": entry.active_version,
                "versions": [v.model_dump() for v in entry.versions],
            }

        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _seed_defaults(self):
        """Seed the store with default RAG prompts."""
        now = datetime.now(timezone.utc).isoformat()

        defaults = {
            "query_analysis": {
                "name": "Query Analysis",
                "description": "Analyzes user queries for better document retrieval",
                "tags": ["rag", "query-analysis"],
                "template": (
                    "Analyze the following user query and optimize it for document retrieval:\n\n"
                    "Query: {query}\n\n"
                    "Your tasks:\n"
                    "1. Identify key concepts and topics\n"
                    "2. Extract important keywords\n"
                    "3. Determine the query intent and type\n"
                    "4. Suggest improvements for better retrieval\n\n"
                    "Provide clear analysis and an optimized query."
                ),
                "variables": ["query"],
            },
            "document_retrieval": {
                "name": "Document Retrieval",
                "description": "Retrieves and ranks relevant documents",
                "tags": ["rag", "retrieval"],
                "template": (
                    "Find and retrieve the most relevant documents for this query:\n\n"
                    "Query: {query}\n"
                    "Available Documents: {document_list}\n\n"
                    "Your tasks:\n"
                    "1. Identify which documents are most relevant\n"
                    "2. Rank them by relevance to the query\n"
                    "3. Explain why each document is relevant\n"
                    "4. Provide key excerpts from top documents\n\n"
                    "Focus on documents that directly answer the user's question."
                ),
                "variables": ["query", "document_list"],
            },
            "response_generation": {
                "name": "Response Generation",
                "description": "Generates responses from retrieved documents",
                "tags": ["rag", "generation"],
                "template": (
                    "Generate a comprehensive response based on the retrieved documents:\n\n"
                    "Original Query: {query}\n"
                    "Retrieved Documents: {documents}\n\n"
                    "Your tasks:\n"
                    "1. Synthesize information from the documents\n"
                    "2. Create a clear, well-structured answer\n"
                    "3. Include proper source citations\n"
                    "4. Ensure factual accuracy\n"
                    "5. Address the query directly\n\n"
                    "The response should be informative and properly sourced."
                ),
                "variables": ["query", "documents"],
            },
            "response_validation": {
                "name": "Response Validation",
                "description": "Validates and improves generated responses",
                "tags": ["rag", "validation"],
                "template": (
                    "Review and validate this generated response:\n\n"
                    "Original Query: {query}\n"
                    "Generated Response: {response}\n"
                    "Source Documents: {sources}\n\n"
                    "Your tasks:\n"
                    "1. Check factual accuracy against sources\n"
                    "2. Verify proper source citations\n"
                    "3. Ensure the response fully addresses the query\n"
                    "4. Check for clarity and coherence\n"
                    "5. Suggest improvements if needed\n\n"
                    "Provide the final validated response or corrections."
                ),
                "variables": ["query", "response", "sources"],
            },
        }

        for prompt_id, data in defaults.items():
            self.prompts[prompt_id] = PromptEntry(
                name=data["name"],
                description=data["description"],
                tags=data["tags"],
                active_version=1,
                versions=[
                    PromptVersion(
                        version=1,
                        template=data["template"],
                        variables=data["variables"],
                        created_at=now,
                        created_by="system",
                        changelog="Initial version",
                    )
                ],
            )

    # --- CRUD Operations ---

    def list_prompts(self) -> Dict[str, Dict[str, Any]]:
        """List all prompts with their active version info."""
        result = {}
        for prompt_id, entry in self.prompts.items():
            active = self._get_version(entry, entry.active_version)
            result[prompt_id] = {
                "name": entry.name,
                "description": entry.description,
                "tags": entry.tags,
                "active_version": entry.active_version,
                "total_versions": len(entry.versions),
                "template": active.template if active else "",
                "variables": active.variables if active else [],
            }
        return result

    def get_prompt(self, prompt_id: str) -> Optional[PromptEntry]:
        """Get a prompt by ID."""
        return self.prompts.get(prompt_id)

    def get_active_template(self, prompt_id: str) -> Optional[str]:
        """Get the active version's template text for a prompt."""
        entry = self.prompts.get(prompt_id)
        if not entry:
            return None
        active = self._get_version(entry, entry.active_version)
        return active.template if active else None

    def get_active_variables(self, prompt_id: str) -> Optional[List[str]]:
        """Get the active version's variables for a prompt."""
        entry = self.prompts.get(prompt_id)
        if not entry:
            return None
        active = self._get_version(entry, entry.active_version)
        return active.variables if active else None

    def create_prompt(
        self,
        prompt_id: str,
        name: str,
        template: str,
        variables: List[str],
        description: str = "",
        tags: List[str] = None,
        created_by: str = "user",
    ) -> PromptEntry:
        """Create a new prompt. Raises ValueError if it already exists."""
        if prompt_id in self.prompts:
            raise ValueError(f"Prompt '{prompt_id}' already exists. Use update to create a new version.")

        now = datetime.now(timezone.utc).isoformat()
        entry = PromptEntry(
            name=name,
            description=description,
            tags=tags or [],
            active_version=1,
            versions=[
                PromptVersion(
                    version=1,
                    template=template,
                    variables=variables,
                    created_at=now,
                    created_by=created_by,
                    changelog="Initial version",
                )
            ],
        )
        self.prompts[prompt_id] = entry
        self._save_to_disk()
        logger.info("Prompt created", prompt_id=prompt_id)
        return entry

    def update_prompt(
        self,
        prompt_id: str,
        template: str,
        variables: List[str] = None,
        changelog: str = "",
        created_by: str = "user",
        name: str = None,
        description: str = None,
        tags: List[str] = None,
    ) -> PromptEntry:
        """Update a prompt by creating a new version. Raises KeyError if not found."""
        entry = self.prompts.get(prompt_id)
        if not entry:
            raise KeyError(f"Prompt '{prompt_id}' not found")

        # Update metadata if provided
        if name is not None:
            entry.name = name
        if description is not None:
            entry.description = description
        if tags is not None:
            entry.tags = tags

        # Determine variables from current active if not provided
        if variables is None:
            active = self._get_version(entry, entry.active_version)
            variables = active.variables if active else []

        # Create new version
        new_version_num = max(v.version for v in entry.versions) + 1
        now = datetime.now(timezone.utc).isoformat()

        new_version = PromptVersion(
            version=new_version_num,
            template=template,
            variables=variables,
            created_at=now,
            created_by=created_by,
            changelog=changelog or f"Updated to version {new_version_num}",
        )

        entry.versions.append(new_version)
        entry.active_version = new_version_num

        self._save_to_disk()
        logger.info("Prompt updated", prompt_id=prompt_id, version=new_version_num)
        return entry

    def delete_prompt(self, prompt_id: str) -> bool:
        """Delete a prompt. Returns False if not found."""
        if prompt_id not in self.prompts:
            return False
        del self.prompts[prompt_id]
        self._save_to_disk()
        logger.info("Prompt deleted", prompt_id=prompt_id)
        return True

    def get_versions(self, prompt_id: str) -> Optional[List[PromptVersion]]:
        """Get all versions of a prompt."""
        entry = self.prompts.get(prompt_id)
        if not entry:
            return None
        return entry.versions

    def rollback(self, prompt_id: str, version: int) -> PromptEntry:
        """Set active version to a specific version number."""
        entry = self.prompts.get(prompt_id)
        if not entry:
            raise KeyError(f"Prompt '{prompt_id}' not found")

        target = self._get_version(entry, version)
        if not target:
            raise ValueError(f"Version {version} not found for prompt '{prompt_id}'")

        entry.active_version = version
        self._save_to_disk()
        logger.info("Prompt rolled back", prompt_id=prompt_id, version=version)
        return entry

    def render(self, prompt_id: str, variables: Dict[str, str]) -> Optional[str]:
        """Render the active prompt template with given variables."""
        template = self.get_active_template(prompt_id)
        if not template:
            return None

        rendered = template
        for var, value in variables.items():
            rendered = rendered.replace("{" + var + "}", str(value))
        return rendered

    # --- Helpers ---

    def _get_version(self, entry: PromptEntry, version_num: int) -> Optional[PromptVersion]:
        """Find a specific version in a prompt entry."""
        for v in entry.versions:
            if v.version == version_num:
                return v
        return None

    async def cleanup(self):
        """Persist any pending changes."""
        self._save_to_disk()
        logger.info("Prompt store cleanup completed")
