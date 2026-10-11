"""Where memory lives: MongoDB Atlas when MONGODB_URI is set, else plain files under ~/.hx.

Collections (Atlas) / folders (local):
  harness_versions  one doc per version of the general rules (lineage via `parent`)
  cases             one doc per incident (description + markdown body)
  traces            one doc per user turn (what was loaded, what failed, what memory did)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from hx.memory.models import Case, HarnessVersion, Rule, TurnTrace, now


class Store:
    name = "store"

    # --- primitives each backend implements
    def _latest_harness(self) -> Optional[HarnessVersion]: ...
    def _insert_harness(self, hv: HarnessVersion) -> None: ...
    def _replace_harness(self, hv: HarnessVersion) -> None: ...
    def harness_history(self, limit: int = 20) -> list[HarnessVersion]: ...
    def list_cases(self, include_retired: bool = False) -> list[Case]: ...
    def get_case(self, case_id: str) -> Optional[Case]: ...
    def _put_case(self, case: Case) -> None: ...
    def save_trace(self, trace: TurnTrace) -> None: ...
    def get_trace(self, trace_id: str) -> Optional[TurnTrace]: ...
    def recent_traces(self, limit: int = 10) -> list[TurnTrace]: ...

    # --- shared behaviour
    def current_harness(self) -> HarnessVersion:
        hv = self._latest_harness()
        if hv is None:
            hv = HarnessVersion(version=0, parent=None, rules=[], change="init", reason="empty harness")
            self._insert_harness(hv)
        return hv

    def commit_harness(self, rules: list[Rule], change: str, reason: str = "", trace_id: Optional[str] = None) -> HarnessVersion:
        """Structural change (rule added / revised / promoted / retired) -> new version."""
        cur = self.current_harness()
        hv = HarnessVersion(
            version=cur.version + 1,
            parent=cur.version,
            rules=[r.model_copy() for r in rules],
            change=change,
            reason=reason,
            trace_id=trace_id,
        )
        self._insert_harness(hv)
        return hv

    def update_counters(self, hv: HarnessVersion) -> None:
        """Trial/helped/strike counters change every turn; they don't deserve a new version."""
        self._replace_harness(hv)

    def save_case(self, case: Case) -> None:
        case.updated_at = now()
        self._put_case(case)


# =============================================================================== local files
def _dump_front(case: Case) -> str:
    meta = case.model_dump(mode="json", exclude={"body"})
    lines = ["---"] + [f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in meta.items()] + ["---", ""]
    return "\n".join(lines) + case.body.strip() + "\n"


def _load_front(text: str) -> Case:
    if not text.startswith("---"):
        raise ValueError("missing frontmatter")
    _, head, body = text.split("---", 2)
    meta = {}
    for line in head.strip().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        try:
            meta[k.strip()] = json.loads(v.strip())
        except json.JSONDecodeError:
            meta[k.strip()] = v.strip()
    meta["body"] = body.strip()
    return Case.model_validate(meta)


class LocalStore(Store):
    def __init__(self, home: Path):
        self.home = Path(home).expanduser()
        for sub in ("harness", "cases", "traces"):
            (self.home / sub).mkdir(parents=True, exist_ok=True)
        self.name = f"local:{self.home}"

    # harness
    def _hv_path(self, version: int) -> Path:
        return self.home / "harness" / f"v{version:04d}.json"

    def _all_hv(self) -> list[HarnessVersion]:
        out = []
        for p in sorted((self.home / "harness").glob("v*.json")):
            try:
                out.append(HarnessVersion.model_validate_json(p.read_text()))
            except Exception:  # noqa: BLE001 — skip a corrupt file rather than brick the agent
                continue
        return sorted(out, key=lambda h: h.version)

    def _latest_harness(self) -> Optional[HarnessVersion]:
        hvs = self._all_hv()
        return hvs[-1] if hvs else None

    def _insert_harness(self, hv: HarnessVersion) -> None:
        self._hv_path(hv.version).write_text(hv.model_dump_json(indent=2))
        self._render_views()

    def _replace_harness(self, hv: HarnessVersion) -> None:
        self._insert_harness(hv)

    def harness_history(self, limit: int = 20) -> list[HarnessVersion]:
        return list(reversed(self._all_hv()))[:limit]

    # cases
    def _case_path(self, case_id: str) -> Path:
        return self.home / "cases" / f"{case_id}.md"

    def list_cases(self, include_retired: bool = False) -> list[Case]:
        out = []
        for p in (self.home / "cases").glob("*.md"):
            try:
                c = _load_front(p.read_text())
            except Exception:  # noqa: BLE001
                continue
            if include_retired or c.status == "active":
                out.append(c)
        return sorted(out, key=lambda c: c.updated_at, reverse=True)

    def get_case(self, case_id: str) -> Optional[Case]:
        p = self._case_path(case_id)
        if not p.is_file():
            return None
        try:
            return _load_front(p.read_text())
        except Exception:  # noqa: BLE001
            return None

    def _put_case(self, case: Case) -> None:
        self._case_path(case.case_id).write_text(_dump_front(case))
        self._render_views()

    # traces
    def save_trace(self, trace: TurnTrace) -> None:
        (self.home / "traces" / f"{trace.trace_id}.json").write_text(trace.model_dump_json(indent=2))

    def get_trace(self, trace_id: str) -> Optional[TurnTrace]:
        p = self.home / "traces" / f"{trace_id}.json"
        return TurnTrace.model_validate_json(p.read_text()) if p.is_file() else None

    def recent_traces(self, limit: int = 10) -> list[TurnTrace]:
        files = sorted((self.home / "traces").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [TurnTrace.model_validate_json(p.read_text()) for p in files[:limit]]

    # human-readable views (like CLAUDE.md / MEMORY.md)
    def _render_views(self) -> None:
        from hx.memory.cases import render_index
        from hx.memory.harness import render_rules

        hv = self._latest_harness()
        if hv is not None:
            (self.home / "HARNESS.md").write_text(
                f"# Harness v{hv.version}\n\n{render_rules(hv) or '(no rules yet)'}\n"
            )
        (self.home / "CASES.md").write_text("# Case library\n\n" + (render_index(self.list_cases()) or "(empty)") + "\n")


# =============================================================================== MongoDB Atlas
class MongoStore(Store):
    def __init__(self, uri: str, db_name: str = "hx_memory"):
        from pymongo import ASCENDING, DESCENDING, MongoClient

        kwargs: dict = {"serverSelectionTimeoutMS": 6000, "tz_aware": True, "appname": "hx-agent"}
        try:
            import certifi

            kwargs["tlsCAFile"] = certifi.where()
        except ImportError:
            pass
        self.client = MongoClient(uri, **kwargs)
        self.client.admin.command("ping")
        self.db = self.client[db_name]
        self.hv = self.db["harness_versions"]
        self.cases = self.db["cases"]
        self.traces = self.db["traces"]
        self.hv.create_index([("version", DESCENDING)], unique=True)
        self.cases.create_index([("case_id", ASCENDING)], unique=True)
        self.cases.create_index([("status", ASCENDING), ("updated_at", DESCENDING)])
        self.traces.create_index([("trace_id", ASCENDING)], unique=True)
        self.traces.create_index([("created_at", DESCENDING)])
        host = uri.split("@")[-1].split("/")[0].split("?")[0]
        self.name = f"atlas:{host}/{db_name}"

    @staticmethod
    def _clean(doc: Optional[dict]) -> Optional[dict]:
        if doc is None:
            return None
        doc = dict(doc)
        doc.pop("_id", None)
        return doc

    def _latest_harness(self) -> Optional[HarnessVersion]:
        doc = self._clean(self.hv.find_one(sort=[("version", -1)]))
        return HarnessVersion.model_validate(doc) if doc else None

    def _insert_harness(self, hv: HarnessVersion) -> None:
        from pymongo.errors import DuplicateKeyError

        for _ in range(5):
            try:
                self.hv.insert_one(hv.model_dump())
                return
            except DuplicateKeyError:
                # Another session committed first: stack on top of it (last writer wins on rules).
                latest = self._latest_harness()
                hv.parent = latest.version if latest else None
                hv.version = (latest.version + 1) if latest else 0
        raise RuntimeError("could not commit harness version")

    def _replace_harness(self, hv: HarnessVersion) -> None:
        self.hv.replace_one({"version": hv.version}, hv.model_dump(), upsert=True)

    def harness_history(self, limit: int = 20) -> list[HarnessVersion]:
        return [HarnessVersion.model_validate(self._clean(d)) for d in self.hv.find().sort("version", -1).limit(limit)]

    def list_cases(self, include_retired: bool = False) -> list[Case]:
        q = {} if include_retired else {"status": "active"}
        return [Case.model_validate(self._clean(d)) for d in self.cases.find(q).sort("updated_at", -1)]

    def get_case(self, case_id: str) -> Optional[Case]:
        doc = self._clean(self.cases.find_one({"case_id": case_id}))
        return Case.model_validate(doc) if doc else None

    def _put_case(self, case: Case) -> None:
        self.cases.replace_one({"case_id": case.case_id}, case.model_dump(), upsert=True)

    def save_trace(self, trace: TurnTrace) -> None:
        self.traces.replace_one({"trace_id": trace.trace_id}, trace.model_dump(), upsert=True)

    def get_trace(self, trace_id: str) -> Optional[TurnTrace]:
        doc = self._clean(self.traces.find_one({"trace_id": trace_id}))
        return TurnTrace.model_validate(doc) if doc else None

    def recent_traces(self, limit: int = 10) -> list[TurnTrace]:
        return [TurnTrace.model_validate(self._clean(d)) for d in self.traces.find().sort("created_at", -1).limit(limit)]


def open_store(mongodb_uri: str, db_name: str, home: Path, warn: Optional[Callable[[str], None]] = None) -> Store:
    if mongodb_uri:
        try:
            return MongoStore(mongodb_uri, db_name)
        except Exception as err:  # noqa: BLE001
            if warn:
                warn(f"Atlas unavailable ({type(err).__name__}: {str(err)[:120]}); using local memory at {home}")
    return LocalStore(home)
