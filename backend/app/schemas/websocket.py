from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ClientInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    frame_width: int | None = Field(default=None, ge=1, le=4096)
    frame_height: int | None = Field(default=None, ge=1, le=4096)


class StartMessage(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=())
    type: Literal["start"]
    mode: Literal["isolated_sign"] = "isolated_sign"
    model_id: str | None = Field(default=None, max_length=64)
    # Omitted: use the active pack from settings. null: no pack.
    domain_pack: str | None = Field(default=None, max_length=64)
    # Attach to a session created via POST /api/v1/sessions.
    session_id: str | None = Field(default=None, max_length=36)
    client: ClientInfo | None = None


class ControlMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["stop", "ping", "correct", "phrase_undo", "phrase_clear", "reset"]
    event_id: int | None = None
    label: str | None = Field(default=None, max_length=120)
