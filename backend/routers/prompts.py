"""REST API endpoints for prompt management."""

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, List, Optional, Any

logger = structlog.get_logger()
router = APIRouter(tags=["prompts"])


# --- Request/Response Models ---

class CreatePromptRequest(BaseModel):
    """Request body for creating a new prompt."""
    prompt_id: str
    name: str
    template: str
    variables: List[str]
    description: str = ""
    tags: List[str] = []
    created_by: str = "user"


class UpdatePromptRequest(BaseModel):
    """Request body for updating a prompt (creates new version)."""
    template: str
    variables: Optional[List[str]] = None
    changelog: str = ""
    created_by: str = "user"
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class RenderPromptRequest(BaseModel):
    """Request body for rendering a prompt with variables."""
    variables: Dict[str, str]


# --- Helper ---

def get_prompt_store(request: Request):
    """Get PromptStore from app state."""
    prompt_store = getattr(request.app.state, "prompt_store", None)
    if not prompt_store:
        try:
            import main
            prompt_store = main.prompt_store
        except (ImportError, AttributeError):
            pass

    if not prompt_store:
        raise HTTPException(status_code=503, detail="Prompt store not available")

    return prompt_store


# --- Endpoints ---

@router.get("/prompts")
async def list_prompts(request: Request):
    """List all prompts with their active version info."""
    store = get_prompt_store(request)
    prompts = store.list_prompts()
    return {"prompts": prompts, "total": len(prompts)}


@router.get("/prompts/{prompt_id}")
async def get_prompt(prompt_id: str, request: Request):
    """Get a prompt with all its version details."""
    store = get_prompt_store(request)
    entry = store.get_prompt(prompt_id)

    if not entry:
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_id}' not found")

    return {
        "prompt_id": prompt_id,
        "name": entry.name,
        "description": entry.description,
        "tags": entry.tags,
        "active_version": entry.active_version,
        "total_versions": len(entry.versions),
        "versions": [v.model_dump() for v in entry.versions],
    }


@router.post("/prompts", status_code=201)
async def create_prompt(body: CreatePromptRequest, request: Request):
    """Create a new prompt template."""
    store = get_prompt_store(request)

    try:
        entry = store.create_prompt(
            prompt_id=body.prompt_id,
            name=body.name,
            template=body.template,
            variables=body.variables,
            description=body.description,
            tags=body.tags,
            created_by=body.created_by,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return {
        "message": f"Prompt '{body.prompt_id}' created successfully",
        "prompt_id": body.prompt_id,
        "version": 1,
    }


@router.put("/prompts/{prompt_id}")
async def update_prompt(prompt_id: str, body: UpdatePromptRequest, request: Request):
    """Update a prompt by creating a new version."""
    store = get_prompt_store(request)

    try:
        entry = store.update_prompt(
            prompt_id=prompt_id,
            template=body.template,
            variables=body.variables,
            changelog=body.changelog,
            created_by=body.created_by,
            name=body.name,
            description=body.description,
            tags=body.tags,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "message": f"Prompt '{prompt_id}' updated to version {entry.active_version}",
        "prompt_id": prompt_id,
        "active_version": entry.active_version,
        "total_versions": len(entry.versions),
    }


@router.delete("/prompts/{prompt_id}")
async def delete_prompt(prompt_id: str, request: Request):
    """Delete a prompt."""
    store = get_prompt_store(request)

    if not store.delete_prompt(prompt_id):
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_id}' not found")

    return {"message": f"Prompt '{prompt_id}' deleted successfully"}


@router.get("/prompts/{prompt_id}/versions")
async def list_versions(prompt_id: str, request: Request):
    """List all versions of a prompt."""
    store = get_prompt_store(request)
    versions = store.get_versions(prompt_id)

    if versions is None:
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_id}' not found")

    entry = store.get_prompt(prompt_id)
    return {
        "prompt_id": prompt_id,
        "active_version": entry.active_version,
        "versions": [v.model_dump() for v in versions],
    }


@router.post("/prompts/{prompt_id}/rollback/{version}")
async def rollback_prompt(prompt_id: str, version: int, request: Request):
    """Roll back a prompt to a specific version."""
    store = get_prompt_store(request)

    try:
        entry = store.rollback(prompt_id, version)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "message": f"Prompt '{prompt_id}' rolled back to version {version}",
        "prompt_id": prompt_id,
        "active_version": entry.active_version,
    }


@router.post("/prompts/{prompt_id}/render")
async def render_prompt(prompt_id: str, body: RenderPromptRequest, request: Request):
    """Render a prompt template with given variables (for testing)."""
    store = get_prompt_store(request)

    rendered = store.render(prompt_id, body.variables)
    if rendered is None:
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_id}' not found")

    entry = store.get_prompt(prompt_id)
    return {
        "prompt_id": prompt_id,
        "version": entry.active_version,
        "rendered_template": rendered,
        "variables_used": body.variables,
    }
