"""Small permission stack inspired by layered tool gates — rewritten, 3 layers not 7."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from hx.models import Harness
from hx.tools import CATALOG

Decision = Literal["allow", "deny", "needs_approval"]


class PermissionResult(BaseModel):
    decision: Decision
    reason: str
    tool: str


def authorize(harness: Harness, tool_name: str) -> PermissionResult:
    if tool_name not in CATALOG:
        return PermissionResult(decision="deny", reason="unknown tool", tool=tool_name)
    access = harness.tool_access
    if tool_name in access.denied:
        return PermissionResult(decision="deny", reason="on deny list", tool=tool_name)
    if tool_name not in access.allowed:
        return PermissionResult(decision="deny", reason="not on allow list", tool=tool_name)
    if tool_name in access.require_approval:
        return PermissionResult(
            decision="needs_approval",
            reason="harness requires approval",
            tool=tool_name,
        )
    return PermissionResult(decision="allow", reason="allowed", tool=tool_name)
