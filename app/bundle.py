"""Offline plan archives: canonical bundle export and validated import.

A *bundle* is a self-contained JSON document for archiving a release plan
offline and re-importing it later (possibly into another instance). It
carries the effective request constraints, the solved wave result and
version metadata. The payload is serialized as canonical JSON — object
keys sorted, arrays in a fixed order, compact separators, UTF-8 — and the
SHA-256 of that serialization is the bundle digest. The digest
authenticates the content: importing recomputes it and rejects any bundle
whose declared digest does not match.

Export (``GET /plans/{id}/bundle``) is read-only. Import
(``POST /plans/import``) validates, in order, the schema (enforced by the
request model), the digest and the dependency/fault-domain references,
then checks the carried wave result against the constraints. Only then is the
new plan written — in a single transaction, with the carried waves stored
unchanged (no re-solving). A bundle whose digest is already known is
idempotent: the existing plan is returned and nothing is written. Any
validation failure leaves the database untouched.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .errors import PlanningError
from .graph import build_adjacency, find_cycles
from .planner import _cycle_error, _validate_request
from .schemas import BundleResult, Issue, PlanBundle, PlanCreate

BUNDLE_FORMAT = "release-wave-plan-bundle"
BUNDLE_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Canonical serialization and digest
# ---------------------------------------------------------------------------


def canonical_json(payload: dict[str, Any]) -> str:
    """Canonical JSON: sorted object keys, compact separators, UTF-8."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _canonical_request(request: dict[str, Any]) -> dict[str, Any]:
    """Fixed-order form of the request constraints.

    Every array is sorted so the digest does not depend on the order in
    which constraints were listed; duplicate dependency edges collapse
    (creation merges them the same way).
    """
    edges = {
        (d["service"], d["depends_on"]) for d in request.get("dependencies", [])
    }
    domains = [
        {"domain": fd["domain"], "services": sorted(fd["services"])}
        for fd in request.get("fault_domains", [])
    ]
    domains.sort(key=lambda fd: fd["domain"])
    return {
        "name": request.get("name"),
        "services": sorted(request["services"]),
        "dependencies": [
            {"service": service, "depends_on": depends_on}
            for service, depends_on in sorted(edges)
        ],
        "max_concurrency": request["max_concurrency"],
        "fault_domains": domains,
        "max_waves": request.get("max_waves"),
    }


def _canonical_result(result: dict[str, Any]) -> dict[str, Any]:
    """Fixed-order form of the solve result (waves by index, members sorted)."""
    waves = sorted(result["waves"], key=lambda w: w["index"])
    return {
        "waves": [
            {"index": wave["index"], "services": sorted(wave["services"])}
            for wave in waves
        ],
        "wave_count": result["wave_count"],
        "optimal": bool(result["optimal"]),
    }


def canonical_payload(
    request: dict[str, Any], result: dict[str, Any], api_version: str
) -> dict[str, Any]:
    """The signed part of a bundle: constraints, result, version metadata."""
    return {
        "format": BUNDLE_FORMAT,
        "format_version": BUNDLE_FORMAT_VERSION,
        "api_version": api_version,
        "request": _canonical_request(request),
        "result": _canonical_result(result),
    }


def bundle_digest(
    request: dict[str, Any], result: dict[str, Any], api_version: str
) -> str:
    """SHA-256 (hex) of the canonical serialization of the signed payload."""
    canonical = canonical_json(canonical_payload(request, result, api_version))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def plan_digest(
    name: str | None,
    stored_result: dict[str, Any],
    wave_count: int,
    optimal: bool,
    api_version: str,
) -> str:
    """Digest of a stored plan, identical to the digest of its export bundle."""
    request = {
        "name": name,
        "services": stored_result["services"],
        "dependencies": stored_result["dependencies"],
        "max_concurrency": stored_result["max_concurrency"],
        "fault_domains": stored_result["fault_domains"],
        "max_waves": stored_result.get("max_waves"),
    }
    result = {
        "waves": stored_result["waves"],
        "wave_count": wave_count,
        "optimal": optimal,
    }
    return bundle_digest(request, result, api_version)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def build_bundle(row: sqlite3.Row, api_version: str) -> dict[str, Any]:
    """Assemble the export bundle for a stored plan row (read-only)."""
    stored = json.loads(row["result_json"])
    request = {
        "name": row["name"],
        "services": stored["services"],
        "dependencies": stored["dependencies"],
        "max_concurrency": stored["max_concurrency"],
        "fault_domains": stored["fault_domains"],
        "max_waves": stored.get("max_waves"),
    }
    result = {
        "waves": stored["waves"],
        "wave_count": row["wave_count"],
        "optimal": bool(row["optimal"]),
    }
    payload = canonical_payload(request, result, api_version)
    payload["sha256"] = bundle_digest(request, result, api_version)
    return payload


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidatedBundle:
    """A bundle that passed every check, normalized for storage."""

    name: str | None
    request_payload: dict[str, Any]
    result_payload: dict[str, Any]
    wave_count: int
    optimal: bool
    digest: str


def _validate_waves(
    plan: PlanCreate,
    result: BundleResult,
    services: list[str],
    edges: list[tuple[str, str]],
    domains: list[dict],
) -> None:
    """Check the carried wave result against the request constraints.

    The waves must be numbered contiguously from 1, cover every service
    exactly once, respect the per-wave concurrency cap, place every
    dependency in a strictly earlier wave and keep fault-domain members
    apart — the same invariants the solver guarantees for plans created
    directly.
    """
    issues: list[Issue] = []
    waves = result.waves

    if result.wave_count != len(waves):
        issues.append(
            Issue(
                code="bundle_wave_count_mismatch",
                message=(
                    f"The declared wave_count {result.wave_count} does not "
                    f"match the {len(waves)} wave(s) in the bundle."
                ),
                constraint="result.wave_count",
                context={
                    "declared_wave_count": result.wave_count,
                    "actual_wave_count": len(waves),
                },
            )
        )

    expected = list(range(1, len(waves) + 1))
    actual = [wave.index for wave in waves]
    if actual != expected:
        issues.append(
            Issue(
                code="bundle_wave_index_invalid",
                message=(
                    "Wave indexes must be the contiguous ascending sequence "
                    f"1..{len(waves)}; got {actual}."
                ),
                constraint="result.waves",
                context={"expected_indexes": expected, "actual_indexes": actual},
            )
        )

    placed = [service for wave in waves for service in wave.services]
    counts = Counter(placed)
    duplicated = sorted(s for s, n in counts.items() if n > 1)
    missing = sorted(set(services) - set(counts))
    unknown = sorted(set(counts) - set(services))
    if missing or unknown or duplicated:
        issues.append(
            Issue(
                code="bundle_wave_service_mismatch",
                message=(
                    "The bundle waves must contain every request service "
                    "exactly once."
                ),
                constraint="result.waves",
                context={
                    "missing_services": missing,
                    "unknown_services": unknown,
                    "duplicated_services": duplicated,
                },
            )
        )

    if issues:
        raise PlanningError(
            "The wave result carried by the bundle is inconsistent with its "
            "request constraints.",
            issues,
        )

    wave_of = {service: wave.index for wave in waves for service in wave.services}

    for wave in waves:
        if len(wave.services) > plan.max_concurrency:
            issues.append(
                Issue(
                    code="bundle_wave_concurrency_exceeded",
                    message=(
                        f"Wave {wave.index} contains {len(wave.services)} "
                        "services, exceeding the per-wave cap of "
                        f"{plan.max_concurrency}."
                    ),
                    constraint="max_concurrency",
                    context={
                        "wave": wave.index,
                        "wave_size": len(wave.services),
                        "max_concurrency": plan.max_concurrency,
                    },
                )
            )

    for service, depends_on in edges:
        if wave_of[depends_on] >= wave_of[service]:
            issues.append(
                Issue(
                    code="bundle_wave_dependency_violation",
                    message=(
                        f"Service '{service}' is in wave {wave_of[service]} "
                        f"but its dependency '{depends_on}' is in wave "
                        f"{wave_of[depends_on]}; dependencies must release "
                        "in a strictly earlier wave."
                    ),
                    constraint="dependencies",
                    service=service,
                    related_services=[depends_on],
                    context={
                        "service_wave": wave_of[service],
                        "dependency_wave": wave_of[depends_on],
                    },
                )
            )

    for fd in domains:
        by_wave: dict[int, str] = {}
        for member in fd["services"]:
            wave_index = wave_of[member]
            if wave_index in by_wave:
                pair = sorted({member, by_wave[wave_index]})
                issues.append(
                    Issue(
                        code="bundle_wave_fault_domain_violation",
                        message=(
                            f"Services '{pair[0]}' and '{pair[1]}' share "
                            f"fault domain '{fd['domain']}' but are both "
                            f"placed in wave {wave_index}."
                        ),
                        constraint="fault_domains",
                        related_services=pair,
                        context={
                            "domain": fd["domain"],
                            "wave": wave_index,
                            "services": pair,
                        },
                    )
                )
            else:
                by_wave[wave_index] = member

    if issues:
        raise PlanningError(
            "The wave result carried by the bundle violates its own "
            "constraints.",
            issues,
        )


def validate_bundle(payload: PlanBundle, settings: Settings) -> ValidatedBundle:
    """Validate an import bundle and normalize it for storage.

    Raises :class:`~app.errors.PlanningError` (HTTP 422) on the first
    failing stage: digest mismatch, invalid dependency/fault-domain
    references or dependency cycles (the same localized issues as plan
    creation), then wave-result consistency violations. When this raises,
    the caller has not written anything.
    """
    request_dict = payload.request.model_dump()
    result_dict = payload.result.model_dump()
    computed = bundle_digest(request_dict, result_dict, payload.api_version)
    if computed != payload.sha256:
        raise PlanningError(
            "The bundle digest does not match its content; the archive is "
            "corrupted or was modified after export.",
            [
                Issue(
                    code="bundle_digest_mismatch",
                    message=(
                        "The declared sha256 does not match the canonical "
                        "serialization of the bundle content."
                    ),
                    constraint="sha256",
                    context={
                        "provided_digest": payload.sha256,
                        "computed_digest": computed,
                    },
                )
            ],
        )

    plan = payload.request
    services, edges, domains, _fault_cliques = _validate_request(plan)
    adj = build_adjacency(services, edges)
    cycles = find_cycles(services, adj, settings.max_reported_cycles)
    if cycles:
        raise _cycle_error(cycles)

    _validate_waves(plan, payload.result, services, edges, domains)

    waves = [
        {"index": wave.index, "services": sorted(wave.services)}
        for wave in payload.result.waves
    ]
    result_payload = {
        "services": sorted(services),
        "waves": waves,
        "max_concurrency": plan.max_concurrency,
        "max_waves": plan.max_waves,
        "dependencies": [
            {"service": service, "depends_on": depends_on}
            for service, depends_on in sorted(edges)
        ],
        "fault_domains": domains,
    }
    return ValidatedBundle(
        name=plan.name,
        request_payload=request_dict,
        result_payload=result_payload,
        wave_count=len(waves),
        optimal=payload.result.optimal,
        digest=payload.sha256,
    )
