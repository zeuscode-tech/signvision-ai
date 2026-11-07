from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ApiModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class ErrorInfo(ApiModel):
    code: str
    message: str
    details: Any = None
    request_id: str


class ErrorResponse(ApiModel):
    error: ErrorInfo


class DeletedResponse(ApiModel):
    deleted: bool = True
