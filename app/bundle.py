"""Offline archive bundles: canonical export and validated import.

An archive bundle captures everything needed to reconstruct a plan on
another instance without re-solving:

* ``request``  — the exact planning constraints (services, dependency
  edges, per-wave concurrency cap, fault domains, release window);
* ``result``   — the immutable solver outcome (ordered waves and the
  proven-optimal flag), imported *as stored* so the wave assignment is
  never recomputed;
* ``meta``     — provenance metadata (bundle format, producing service
  and version, source plan id, creation/export timestamps).

The canonical representation is ``json.dumps(..., sort_keys=True,
ensure_ascii=False, separators=(",", ":"))`` of the bundle object,
encoded as UTF-8. SHA-256 over those bytes is content addressing: the
same logical plan always produces the same digest regardless of JSON
whitespace, key order or numeric formatting on the wire, and import is
idempotent on that digest.

Import never mutates an existing plan and never re-solves: the stored
waves are copied verbatim. Validation (digest, schema, references,
result consistency) all runs before any write, and the insert happens
inside one transaction, so a failed import performs zero writes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .config import SERVICE_NAME, SERVICE_VERSION
from .errors import PlanningError
from .graph import build_adjacency, find_cycles
from .planner import _validate_request
from .schemas import Dependency, FaultDomain, Issue, PlanCreate

# Archive format. The format is content-addressed by its canonical JSON;
# a different ``format`` id or an incompatible ``format_version`` is
# rejected on import instead of being misinterpreted.
BUNDLE_FORMAT = "release-wave-plan-bundle"
BUNDLE_FORMAT_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Bundle models (strict: no undeclared fields survive a round trip)
# ---------------------------------------------------------------------------


class BundleDependency(Dependency):
    model_config = ConfigDict(extra="forbid")


class BundleFaultDomain(FaultDomain):
    model_config = ConfigDict(extra="forbid")


class BundlePlanRequest(BaseModel):
    """Canonical planning request embedded in a bundle."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, max_length=200)
    services: list[str] = Field(..., min_length=1)
    dependencies: list[BundleDependency] = Field(default_factory=list)
    max_concurrency: int = Field(..., ge=1)
    fault_domains: list[BundleFaultDomain] = Field(default_factory=list)
    max_waves: int | None = Field(default=None, ge=1)


class BundleWave(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(..., ge=1)
    services: list[str] = Field(..., min_length=1)


class BundleResult(BaseModel):
    """Immutable solver outcome archived with the request."""

    model_config = ConfigDict(extra="forbid")

    services: list[str] = Field(..., min_length=1)
    waves: list[BundleWave] = Field(..., min_length=1)
    max_concurrency: int = Field(..., ge=1)
    max_waves: int | None = Field(default=None, ge=1)
    dependencies: list[BundleDependency] = Field(default_factory=list)
    fault_domains: list[BundleFaultDomain] = Field(default_factory=list)
    optimal: bool


class BundleServiceMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str


class BundleMeta(BaseModel):
    """Format/version metadata; part of the hashed payload."""

    model_config = ConfigDict(extra="forbid")

    format: str
    format_version: str
    service: BundleServiceMeta


class PlanBundleProvenance(BaseModel):
    """Instance-specific export provenance; not covered by the digest."""

    model_config = ConfigDict(extra="forbid")

    source_plan_id: int | None = Field(
        default=None,
        ge=1,
        description="Id of the plan on the exporting instance; informational "
        "only (the importing instance assigns a fresh id)",
    )
    created_at: datetime = Field(
        ..., description="Creation timestamp of the original plan"
    )
    exported_at: datetime = Field(..., description="Bundle export timestamp")


class PlanBundle(BaseModel):
    """The hashed archive payload: request + solver result + version metadata.

    Only these instance-independent fields participate in the canonical
    digest, so the same logical plan exported twice (or by two different
    instances) hashes identically.
    """

    model_config = ConfigDict(extra="forbid")

    request: BundlePlanRequest
    result: BundleResult
    meta: BundleMeta


class PlanBundleEnvelope(BaseModel):
    """Wire envelope of GET /plans/{id}/bundle.

    ``sha256`` is the digest of the canonical JSON of ``bundle`` alone;
    ``provenance`` documents where and when the export happened without
    changing the content address.
    """

    model_config = ConfigDict(extra="forbid")

    sha256: str
    bundle: PlanBundle
    provenance: PlanBundleProvenance


class PlanImportRequest(BaseModel):
    """Body of POST /plans/import: the exported envelope is replayed verbatim.

    ``provenance`` is optional and ignored (the importing instance never
    adopts another instance's plan id or timestamps).
    """

    model_config = ConfigDict(extra="forbid")

    sha256: str = Field(
        ...,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 hex digest of the canonical JSON bundle payload",
    )
    bundle: PlanBundle
    provenance: PlanBundleProvenance | None = None


# ---------------------------------------------------------------------------
# Canonical serialization & content digest
# ---------------------------------------------------------------------------


def canonical_bytes(bundle: PlanBundle) -> bytes:
    """Deterministic UTF-8 JSON bytes the digest is computed over.

    Keys are sorted recursively, no insignificant whitespace, non-ASCII
    service ids preserved; this is the single normalization used by both
    export and import, so whitespace/key-order differences on the wire
    never invalidate a genuine archive.
    """
    payload: Any = bundle.model_dump(mode="json")
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def digest_bundle(bundle: PlanBundle) -> str:
    return hashlib.sha256(canonical_bytes(bundle)).hexdigest()


def _utcnow_iso() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _canonical_request_kwargs(request_payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a stored request to its canonical, order-independent form.

    Constraint semantics ignore declaration order, so services and fault
    members are sorted and edges are ordered; this guarantees identical
    content always hashes identically regardless of input order or
    Python's hash-seeded set iteration.
    """
    return {
        "name": request_payload.get("name"),
        "services": sorted(request_payload["services"]),
        "dependencies": sorted(
            (
                {"service": d["service"], "depends_on": d["depends_on"]}
                for d in request_payload.get("dependencies", [])
            ),
            key=lambda d: (d["service"], d["depends_on"]),
        ),
        "max_concurrency": request_payload["max_concurrency"],
        "fault_domains": [
            {"domain": fd["domain"], "services": sorted(fd["services"])}
            for fd in sorted(
                request_payload.get("fault_domains", []),
                key=lambda fd: fd["domain"],
            )
        ],
        "max_waves": request_payload.get("max_waves"),
    }


def _canonical_result_kwargs(result_payload: dict[str, Any]) -> dict[str, Any]:
    """Same order-independent normalization for the echoed constraints."""
    kwargs = _canonical_request_kwargs(
        {
            "name": None,
            "services": result_payload["services"],
            "dependencies": result_payload.get("dependencies", []),
            "max_concurrency": result_payload["max_concurrency"],
            "fault_domains": result_payload.get("fault_domains", []),
            "max_waves": result_payload.get("max_waves"),
        }
    )
    kwargs.pop("name")
    return kwargs


def build_bundle(
    *,
    request_payload: dict[str, Any],
    result_payload: dict[str, Any],
    optimal: bool,
    source_plan_id: int,
    created_at: str,
    exported_at: datetime | None = None,
) -> tuple[PlanBundle, PlanBundleProvenance]:
    """Assemble a strict (bundle, provenance) pair from stored JSON.

    The bundle carries only instance-independent content and feeds the
    digest; provenance carries the source plan id and timestamps.
    """
    request = BundlePlanRequest.model_validate(
        _canonical_request_kwargs(request_payload)
    )
    result_kwargs = _canonical_result_kwargs(result_payload)
    result_kwargs["optimal"] = bool(optimal)
    # Wave members are sorted by the solver; keep their stored order.
    result_kwargs["waves"] = [
        {"index": w["index"], "services": list(w["services"])}
        for w in result_payload["waves"]
    ]
    result = BundleResult.model_validate(result_kwargs)
    exported_at = exported_at or _utcnow_iso()
    bundle = PlanBundle(
        request=request,
        result=result,
        meta=BundleMeta(
            format=BUNDLE_FORMAT,
            format_version=BUNDLE_FORMAT_VERSION,
            service=BundleServiceMeta(name=SERVICE_NAME, version=SERVICE_VERSION),
        ),
    )
    provenance = PlanBundleProvenance(
        source_plan_id=source_plan_id,
        created_at=datetime.fromisoformat(created_at),
        exported_at=exported_at,
    )
    return bundle, provenance


# ---------------------------------------------------------------------------
# Import validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedImport:
    """Everything the DB layer needs, derived from a verified envelope."""

    digest: str
    name: str | None
    request_payload: dict[str, Any]
    result_payload: dict[str, Any]
    wave_count: int
    optimal: bool


def _issue(
    code: str,
    message: str,
    *,
    constraint: str,
    service: str | None = None,
    related_services: list[str] | None = None,
    context: dict[str, Any] | None = None,
) -> Issue:
    return Issue(
        code=code,
        message=message,
        constraint=constraint,
        service=service,
        related_services=related_services or [],
        context=context or {},
    )


def _bundle_error(issues: list[Issue], message: str) -> PlanningError:
    return PlanningError(message, issues)


def prepare_import(payload: PlanImportRequest) -> PreparedImport:
    """Validate an import envelope and prepare the row to persist.

    Order: digest first (cheap integrity gate), then format/version,
    schema (already enforced by pydantic), dependency/fault-domain
    references and the dependency DAG, and finally result consistency.
    Raises :class:`PlanningError` (422) with localizable issues; the
    caller must not have written anything at that point.
    """
    actual = digest_bundle(payload.bundle)
    if actual != payload.sha256:
        raise _bundle_error(
            [
                _issue(
                    "bundle_checksum_mismatch",
                    message=(
                        "The bundle's canonical SHA-256 does not match the "
                        "declared digest; the archive is corrupted or was "
                        "modified after export."
                    ),
                    constraint="sha256",
                    context={
                        "declared": payload.sha256,
                        "actual": actual,
                    },
                )
            ],
            "The bundle digest does not match its content.",
        )

    meta = payload.bundle.meta
    if (
        meta.format != BUNDLE_FORMAT
        or meta.format_version != BUNDLE_FORMAT_VERSION
    ):
        raise _bundle_error(
            [
                _issue(
                    "bundle_format_unsupported",
                    message=(
                        f"Bundle format '{meta.format}' version "
                        f"'{meta.format_version}' is not supported; expected "
                        f"'{BUNDLE_FORMAT}' version '{BUNDLE_FORMAT_VERSION}'."
                    ),
                    constraint="meta.format_version",
                    context={
                        "expected_format": BUNDLE_FORMAT,
                        "expected_version": BUNDLE_FORMAT_VERSION,
                        "actual_format": meta.format,
                        "actual_version": meta.format_version,
                    },
                )
            ],
            "The bundle format is not supported.",
        )

    req = payload.bundle.request
    res = payload.bundle.result

    # Re-run the exact create-time validation: duplicate declarations,
    # dependency targets and fault-domain membership.
    plan_req = PlanCreate(
        name=req.name,
        services=list(req.services),
        dependencies=[
            Dependency(service=d.service, depends_on=d.depends_on)
            for d in req.dependencies
        ],
        max_concurrency=req.max_concurrency,
        fault_domains=[
            FaultDomain(domain=fd.domain, services=list(fd.services))
            for fd in req.fault_domains
        ],
        max_waves=req.max_waves,
    )
    services, edges, domains, _fault_cliques = _validate_request(plan_req)

    # A plan with a cyclic DAG could never have been exported; reject it
    # with the same localizable issue used at create time.
    adj = build_adjacency(services, edges)
    cycles = find_cycles(services, adj, 20)
    if cycles:
        cycle = cycles[0]
        raise _bundle_error(
            [
                _issue(
                    "dependency_cycle",
                    message=(
                        "Dependency graph of the archived request contains a "
                        f"cycle: {' -> '.join(cycle + [cycle[0]])}"
                    ),
                    constraint="dependencies",
                    service=cycle[0],
                    related_services=cycle,
                    context={"cycles": cycles, "cycle_count": len(cycles)},
                )
            ],
            "The bundle dependency graph contains a cycle.",
        )

    consistency = _result_consistency_issues(
        req, res, services, set(edges), domains
    )
    if consistency:
        raise _bundle_error(
            consistency,
            "The archived result is inconsistent with the archived request.",
        )

    # Canonical, normalized row payloads — identical shapes to what
    # POST /plans stores, so all read endpoints work unchanged.
    request_stored = {
        "name": req.name,
        "services": services,
        "dependencies": [
            {"service": s, "depends_on": d} for s, d in sorted(edges)
        ],
        "max_concurrency": req.max_concurrency,
        "fault_domains": domains,
        "max_waves": req.max_waves,
    }
    result_stored = {
        "services": services,
        "waves": [
            {"index": w.index, "services": list(w.services)} for w in res.waves
        ],
        "max_concurrency": req.max_concurrency,
        "max_waves": req.max_waves,
        "dependencies": request_stored["dependencies"],
        "fault_domains": domains,
        "optimal": res.optimal,
    }
    return PreparedImport(
        digest=payload.sha256,
        name=req.name,
        request_payload=request_stored,
        result_payload=result_stored,
        wave_count=len(res.waves),
        optimal=res.optimal,
    )


def _result_consistency_issues(
    req: BundlePlanRequest,
    res: BundleResult,
    services: list[str],
    edges: set[tuple[str, str]],
    domains: list[dict[str, Any]],
) -> list[Issue]:
    """Verify the archived waves actually solve the archived request.

    The archive must be self-consistent even though its digest is valid:
    a digest only proves the bytes match *some* (possibly re-digested,
    tampered) document, not that the document describes a feasible plan.
    """
    issues: list[Issue] = []

    def add(code: str, message: str, **kw: Any) -> None:
        issues.append(_issue(code, message, **kw))

    known = set(services)
    result_services = list(res.services)

    if len(result_services) != len(set(result_services)):
        dupes = sorted(
            {s for s in result_services if result_services.count(s) > 1}
        )
        add(
            "bundle_result_inconsistent",
            "The archived result lists service(s) more than once: "
            + ", ".join(dupes),
            constraint="result.services",
            service=dupes[0],
            related_services=dupes,
            context={"duplicate_services": dupes},
        )

    if set(result_services) != known:
        missing = sorted(known - set(result_services))
        extra = sorted(set(result_services) - known)
        add(
            "bundle_result_inconsistent",
            "The archived result does not cover the exact request service set.",
            constraint="result.services",
            related_services=sorted(set(missing) | set(extra)),
            context={"missing_services": missing, "unknown_services": extra},
        )

    # Wave structure: contiguous 1..k indexes, known members, no repeats.
    assigned: dict[str, int] = {}
    wave_members: dict[int, list[str]] = {}
    structural_bad = False
    for position, wave in enumerate(res.waves, start=1):
        if wave.index != position:
            structural_bad = True
            add(
                "bundle_result_inconsistent",
                f"Wave at position {position} has index {wave.index}; wave "
                "indexes must be contiguous starting at 1.",
                constraint="result.waves",
                context={"position": position, "index": wave.index},
            )
        for s in wave.services:
            if s not in known:
                structural_bad = True
                add(
                    "bundle_result_inconsistent",
                    f"Wave {wave.index} contains service '{s}', which is not "
                    "part of the archived request.",
                    constraint="result.waves",
                    service=s,
                    context={"wave": wave.index},
                )
            if s in assigned:
                structural_bad = True
                add(
                    "bundle_result_inconsistent",
                    f"Service '{s}' is assigned to both wave {assigned[s]} "
                    f"and wave {wave.index}.",
                    constraint="result.waves",
                    service=s,
                    related_services=[s],
                    context={"waves": [assigned[s], wave.index]},
                )
            else:
                assigned[s] = wave.index
                wave_members.setdefault(wave.index, []).append(s)

    if not structural_bad:
        unassigned = sorted(known - set(assigned))
        if unassigned:
            add(
                "bundle_result_inconsistent",
                "Not every archived service is assigned to a wave: "
                + ", ".join(unassigned),
                constraint="result.waves",
                service=unassigned[0],
                related_services=unassigned,
                context={"unassigned_services": unassigned},
            )

        # Per-wave concurrency cap.
        for wave in res.waves:
            if len(wave.services) > req.max_concurrency:
                add(
                    "bundle_result_inconsistent",
                    f"Wave {wave.index} contains {len(wave.services)} "
                    f"services but the concurrency cap is "
                    f"{req.max_concurrency}.",
                    constraint="result.max_concurrency",
                    context={
                        "wave": wave.index,
                        "service_count": len(wave.services),
                        "max_concurrency": req.max_concurrency,
                    },
                )

        # Dependencies: prerequisite strictly earlier.
        for service, depends_on in sorted(edges):
            if service in assigned and depends_on in assigned:
                if assigned[depends_on] >= assigned[service]:
                    add(
                        "bundle_result_inconsistent",
                        f"Dependency violated: '{service}' is in wave "
                        f"{assigned[service]} but its prerequisite "
                        f"'{depends_on}' is in wave {assigned[depends_on]}; "
                        "the prerequisite must be in an earlier wave.",
                        constraint="result.dependencies",
                        service=service,
                        related_services=[depends_on],
                        context={
                            "service_wave": assigned[service],
                            "dependency_wave": assigned[depends_on],
                        },
                    )

        # Fault-domain anti-affinity: same-domain members in distinct waves.
        for fd in domains:
            by_wave: dict[int, list[str]] = {}
            for m in fd["services"]:
                if m in assigned:
                    by_wave.setdefault(assigned[m], []).append(m)
            for wave_index, members in sorted(by_wave.items()):
                if len(members) > 1:
                    add(
                        "bundle_result_inconsistent",
                        f"Fault domain '{fd['domain']}' places multiple "
                        f"services in wave {wave_index}: "
                        + ", ".join(sorted(members)),
                        constraint="result.fault_domains",
                        service=sorted(members)[0],
                        related_services=sorted(members),
                        context={"domain": fd["domain"], "wave": wave_index},
                    )

        # Release window, when archived, must be respected.
        if req.max_waves is not None and len(res.waves) > req.max_waves:
            add(
                "bundle_result_inconsistent",
                f"The archived result uses {len(res.waves)} waves but the "
                f"release window is {req.max_waves}.",
                constraint="result.max_waves",
                context={
                    "wave_count": len(res.waves),
                    "max_waves": req.max_waves,
                },
            )

    # Scalar constraint echo.
    if res.max_concurrency != req.max_concurrency:
        add(
            "bundle_result_inconsistent",
            "The archived result's max_concurrency does not match the request.",
            constraint="result.max_concurrency",
            context={
                "request": req.max_concurrency,
                "result": res.max_concurrency,
            },
        )
    if res.max_waves != req.max_waves:
        add(
            "bundle_result_inconsistent",
            "The archived result's max_waves does not match the request.",
            constraint="result.max_waves",
            context={"request": req.max_waves, "result": res.max_waves},
        )
    if _edge_set(res) != edges:
        add(
            "bundle_result_inconsistent",
            "The dependency edges archived in the result do not match the "
            "archived request.",
            constraint="result.dependencies",
            context={
                "request_dependencies": [
                    {"service": s, "depends_on": d} for s, d in sorted(edges)
                ],
                "result_dependencies": [
                    {"service": s, "depends_on": d}
                    for s, d in sorted(_edge_set(res))
                ],
            },
        )
    if _domain_set(res) != {(d["domain"], tuple(d["services"])) for d in domains}:
        add(
            "bundle_result_inconsistent",
            "The fault domains archived in the result do not match the "
            "archived request.",
            constraint="result.fault_domains",
        )

    return issues


def _edge_set(res: BundleResult) -> set[tuple[str, str]]:
    return {(d.service, d.depends_on) for d in res.dependencies}


def _domain_set(
    res: BundleResult,
) -> set[tuple[str, tuple[str, ...]]]:
    return {(fd.domain, tuple(fd.services)) for fd in res.fault_domains}
