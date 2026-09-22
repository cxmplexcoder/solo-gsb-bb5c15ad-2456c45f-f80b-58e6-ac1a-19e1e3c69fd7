"""End-to-end tests for the offline archive: bundle export and import."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    monkeypatch.setenv("PLANNER_DB_PATH", tmp.name)
    monkeypatch.setenv("PLANNER_SOLVER_TIME_LIMIT", "5")
    from importlib import reload

    from app import config, db, main

    reload(config)
    settings = config.load_settings()
    main.app.state.settings = settings
    db.init_db(settings.database_path)

    with TestClient(main.app) as c:
        yield c

    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(tmp.name + suffix)
        except OSError:
            pass


PLAN_PAYLOAD = {
    "name": "payment-platform",
    "services": ["web", "api", "auth", "db", "cache"],
    "dependencies": [
        {"service": "web", "depends_on": "api"},
        {"service": "api", "depends_on": "auth"},
        {"service": "api", "depends_on": "db"},
        {"service": "auth", "depends_on": "db"},
    ],
    "max_concurrency": 2,
    "fault_domains": [{"domain": "az-a", "services": ["db", "cache"]}],
    "max_waves": None,
}


def _create(client, payload=None):
    r = client.post("/plans", json=payload or PLAN_PAYLOAD)
    assert r.status_code == 201, r.text
    return r.json()


def _export(client, plan_id):
    r = client.get(f"/plans/{plan_id}/bundle")
    assert r.status_code == 200, r.text
    return r.json()


def _canonical_digest(bundle_obj) -> str:
    return hashlib.sha256(
        json.dumps(
            bundle_obj,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# GET /plans/{id}/bundle
# ---------------------------------------------------------------------------


def test_bundle_export_contains_request_result_meta_and_digest(client):
    plan = _create(client)
    envelope = _export(client, plan["id"])

    assert set(envelope) == {"sha256", "bundle", "provenance"}
    bundle = envelope["bundle"]
    assert set(bundle) == {"request", "result", "meta"}

    req, res, meta = bundle["request"], bundle["result"], bundle["meta"]
    assert req["name"] == "payment-platform"
    assert req["services"] == sorted(PLAN_PAYLOAD["services"])
    assert req["max_concurrency"] == 2
    assert req["max_waves"] is None
    assert sorted(req["dependencies"], key=lambda d: (d["service"], d["depends_on"]))
    assert req["fault_domains"] == [{"domain": "az-a", "services": ["cache", "db"]}]

    # The archived waves are the exact stored solution, not recomputed.
    assert res["waves"] == plan["waves"]
    assert res["services"] == req["services"]
    assert res["max_concurrency"] == 2
    assert res["dependencies"] == req["dependencies"]
    assert res["fault_domains"] == req["fault_domains"]
    assert res["optimal"] is True

    assert meta == {
        "format": "release-wave-plan-bundle",
        "format_version": "1.0",
        "service": {
            "name": "release-wave-planner",
            "version": meta["service"]["version"],
        },
    }
    assert meta["service"]["version"]

    # Provenance rides outside the hashed payload.
    provenance = envelope["provenance"]
    assert provenance["source_plan_id"] == plan["id"]
    assert provenance["created_at"] == plan["created_at"].replace("+00:00", "Z")
    assert provenance["exported_at"]

    # Digest equals SHA-256 over the canonical JSON of the bundle object.
    assert envelope["sha256"] == _canonical_digest(bundle)
    assert len(envelope["sha256"]) == 64


def test_bundle_export_missing_plan_is_404(client):
    r = client.get("/plans/9999/bundle")
    assert r.status_code == 404
    assert r.json()["error"] == "not_found"
    assert r.json()["issues"][0]["code"] == "plan_not_found"


def test_bundle_export_is_read_only(client):
    plan = _create(client)
    before = client.get("/plans").json()["total"]
    _export(client, plan["id"])
    _export(client, plan["id"])
    assert client.get("/plans").json()["total"] == before
    fetched = client.get(f"/plans/{plan['id']}").json()
    assert fetched["waves"] == plan["waves"]


def test_bundle_digest_is_stable_and_order_independent(client):
    # Dependencies submitted in a different order yield the same plan
    # content and therefore the same canonical digest.
    plan = _create(client)
    digest_1 = _export(client, plan["id"])["sha256"]

    reordered = json.loads(json.dumps(PLAN_PAYLOAD))
    reordered["dependencies"] = list(reversed(reordered["dependencies"]))
    plan2 = _create(client, reordered)
    digest_2 = _export(client, plan2["id"])["sha256"]
    assert digest_1 == digest_2


def test_bundle_digest_changes_with_content(client):
    plan_a = _create(client)
    other = json.loads(json.dumps(PLAN_PAYLOAD))
    other["max_concurrency"] = 1
    plan_b = _create(client, other)
    assert _export(client, plan_a["id"])["sha256"] != _export(
        client, plan_b["id"]
    )["sha256"]


# ---------------------------------------------------------------------------
# POST /plans/import — happy path & idempotency
# ---------------------------------------------------------------------------


def test_import_bundle_creates_new_plan_with_archived_waves(client):
    plan = _create(client)
    envelope = _export(client, plan["id"])

    r = client.post("/plans/import", json=envelope)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["reused"] is False
    assert body["sha256"] == envelope["sha256"]
    assert body["id"] != plan["id"]
    # The wave assignment is copied verbatim — never re-solved.
    assert body["waves"] == plan["waves"]
    assert body["wave_count"] == plan["wave_count"]
    assert body["max_concurrency"] == plan["max_concurrency"]
    assert body["dependencies"] == plan["dependencies"]
    assert body["fault_domains"] == plan["fault_domains"]
    assert body["optimal"] == plan["optimal"]
    assert body["services"] == plan["services"]

    # The restored plan is fully usable by every read endpoint.
    fetched = client.get(f"/plans/{body['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["waves"] == plan["waves"]


def test_import_same_digest_is_idempotent(client):
    plan = _create(client)
    envelope = _export(client, plan["id"])

    first = client.post("/plans/import", json=envelope)
    assert first.status_code == 201
    new_id = first.json()["id"]

    second = client.post("/plans/import", json=envelope)
    assert second.status_code == 200
    body = second.json()
    assert body["reused"] is True
    assert body["id"] == new_id
    assert body["sha256"] == envelope["sha256"]

    # Exactly one imported plan exists (the original + one restored).
    assert client.get("/plans").json()["total"] == 2


def test_import_ignores_wire_formatting_and_key_order(client):
    plan = _create(client)
    envelope = _export(client, plan["id"])

    # Re-serialize with pretty indentation and reorder top-level keys.
    padded = json.loads(json.dumps(envelope, indent=2))
    reordered_body = {
        "bundle": padded["bundle"],
        "sha256": padded["sha256"],
    }
    r = client.post("/plans/import", json=reordered_body)
    assert r.status_code == 201, r.text
    assert r.json()["sha256"] == envelope["sha256"]


def test_imported_bundle_round_trips_with_matching_digest(client):
    plan = _create(client)
    envelope = _export(client, plan["id"])
    imported = client.post("/plans/import", json=envelope).json()

    # Re-exporting the restored plan yields the identical content digest;
    # only provenance (source plan id/timestamps) differs.
    re_exported = _export(client, imported["id"])
    assert re_exported["sha256"] == envelope["sha256"]
    assert re_exported["bundle"]["request"] == envelope["bundle"]["request"]
    assert re_exported["bundle"]["result"] == envelope["bundle"]["result"]
    assert re_exported["provenance"]["source_plan_id"] == imported["id"]


def test_import_preserves_wave_result_without_solving(client):
    # A plan with max_waves set: the archive keeps the exact waves even
    # though import never invokes the solver.
    payload = {
        "services": ["a", "b", "c"],
        "dependencies": [
            {"service": "b", "depends_on": "a"},
            {"service": "c", "depends_on": "b"},
        ],
        "max_concurrency": 3,
        "max_waves": 3,
    }
    plan = _create(client, payload)
    envelope = _export(client, plan["id"])
    body = client.post("/plans/import", json=envelope)
    assert body.status_code == 201, body.text
    assert body.json()["waves"] == plan["waves"]
    assert body.json()["max_waves"] == 3


# ---------------------------------------------------------------------------
# POST /plans/import — validation failures (always zero writes)
# ---------------------------------------------------------------------------


def test_import_checksum_mismatch_is_422_and_writes_nothing(client):
    _create(client)
    envelope = _export(client, 1)
    bad = json.loads(json.dumps(envelope))
    bad["sha256"] = "0" * 64

    r = client.post("/plans/import", json=bad)
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "planning_failed"
    issue = body["issues"][0]
    assert issue["code"] == "bundle_checksum_mismatch"
    assert issue["context"]["declared"] == "0" * 64
    assert issue["context"]["actual"] == envelope["sha256"]

    assert client.get("/plans").json()["total"] == 1


def test_import_tampered_bundle_with_rehashed_digest_is_consistency_422(client):
    _create(client)
    envelope = _export(client, 1)
    tampered = json.loads(json.dumps(envelope))
    # Mutate the archived wave assignment, then re-hash so the digest gate
    # passes: the result-vs-request consistency check must still catch it.
    tampered["bundle"]["result"]["waves"] = [
        {"index": 1, "services": ["web", "db"]}
    ]
    tampered["sha256"] = _canonical_digest(tampered["bundle"])

    r = client.post("/plans/import", json=tampered)
    assert r.status_code == 422
    codes = {i["code"] for i in r.json()["issues"]}
    assert "bundle_result_inconsistent" in codes
    assert client.get("/plans").json()["total"] == 1


def test_import_unknown_dependency_reference_is_422(client):
    _create(client)
    envelope = _export(client, 1)
    bad = json.loads(json.dumps(envelope))
    bad["bundle"]["request"]["dependencies"].append(
        {"service": "web", "depends_on": "ghost"}
    )
    bad["bundle"]["result"]["dependencies"].append(
        {"service": "web", "depends_on": "ghost"}
    )
    bad["sha256"] = _canonical_digest(bad["bundle"])

    r = client.post("/plans/import", json=bad)
    assert r.status_code == 422
    codes = {i["code"] for i in r.json()["issues"]}
    assert "unknown_dependency_target" in codes
    assert client.get("/plans").json()["total"] == 1


def test_import_unknown_fault_domain_member_is_422(client):
    _create(client)
    envelope = _export(client, 1)
    bad = json.loads(json.dumps(envelope))
    bad["bundle"]["request"]["fault_domains"] = [
        {"domain": "az-a", "services": ["db", "ghost"]}
    ]
    bad["bundle"]["result"]["fault_domains"] = bad["bundle"]["request"][
        "fault_domains"
    ]
    bad["sha256"] = _canonical_digest(bad["bundle"])

    r = client.post("/plans/import", json=bad)
    assert r.status_code == 422
    codes = {i["code"] for i in r.json()["issues"]}
    assert "unknown_fault_domain_member" in codes
    assert client.get("/plans").json()["total"] == 1


def test_import_dependency_cycle_is_422(client):
    _create(client)
    envelope = _export(client, 1)
    bad = json.loads(json.dumps(envelope))
    # Add an edge that closes a cycle: db -> web -> api -> auth -> db.
    edge = {"service": "db", "depends_on": "web"}
    bad["bundle"]["request"]["dependencies"].append(edge)
    bad["bundle"]["result"]["dependencies"].append(edge)
    bad["sha256"] = _canonical_digest(bad["bundle"])

    r = client.post("/plans/import", json=bad)
    assert r.status_code == 422
    assert r.json()["issues"][0]["code"] == "dependency_cycle"
    assert client.get("/plans").json()["total"] == 1


def test_import_wave_violates_dependency_order_is_422(client):
    _create(client)
    envelope = _export(client, 1)
    bad = json.loads(json.dumps(envelope))
    # Swap wave order so 'db' no longer precedes 'auth'.
    waves = bad["bundle"]["result"]["waves"]
    waves[0], waves[3] = waves[3], waves[0]
    for i, wave in enumerate(waves, start=1):
        wave["index"] = i
    bad["sha256"] = _canonical_digest(bad["bundle"])

    r = client.post("/plans/import", json=bad)
    assert r.status_code == 422
    issue = next(
        i
        for i in r.json()["issues"]
        if i["code"] == "bundle_result_inconsistent"
        and i["constraint"] == "result.dependencies"
    )
    assert client.get("/plans").json()["total"] == 1


def test_import_wave_violates_concurrency_cap_is_422(client):
    _create(client)
    envelope = _export(client, 1)
    bad = json.loads(json.dumps(envelope))
    bad["bundle"]["result"]["waves"] = [
        {"index": 1, "services": ["api", "auth", "cache", "db", "web"]}
    ]
    bad["sha256"] = _canonical_digest(bad["bundle"])

    r = client.post("/plans/import", json=bad)
    assert r.status_code == 422
    constraints = {
        i["constraint"] for i in r.json()["issues"]
    }
    assert "result.max_concurrency" in constraints
    assert client.get("/plans").json()["total"] == 1


def test_import_wave_violates_fault_domain_is_422(client):
    _create(client)
    envelope = _export(client, 1)
    bad = json.loads(json.dumps(envelope))
    # Put db and cache (same fault domain az-a) into one wave.
    bad["bundle"]["result"]["waves"] = [
        {"index": 1, "services": ["cache", "db"]},
        {"index": 2, "services": ["auth"]},
        {"index": 3, "services": ["api"]},
        {"index": 4, "services": ["web"]},
    ]
    bad["sha256"] = _canonical_digest(bad["bundle"])

    r = client.post("/plans/import", json=bad)
    assert r.status_code == 422
    issue = next(
        i
        for i in r.json()["issues"]
        if i["constraint"] == "result.fault_domains"
    )
    assert issue["code"] == "bundle_result_inconsistent"
    assert set(issue["related_services"]) == {"cache", "db"}
    assert client.get("/plans").json()["total"] == 1


def test_import_unsupported_format_and_version_are_422(client):
    _create(client)
    envelope = _export(client, 1)
    for field, value in (("format", "something-else"), ("format_version", "9.9")):
        bad = json.loads(json.dumps(envelope))
        bad["bundle"]["meta"][field] = value
        bad["sha256"] = _canonical_digest(bad["bundle"])

        r = client.post("/plans/import", json=bad)
        assert r.status_code == 422, (field, r.text)
        issue = r.json()["issues"][0]
        assert issue["code"] == "bundle_format_unsupported"
    assert client.get("/plans").json()["total"] == 1


def test_import_malformed_digest_is_validation_error(client):
    _create(client)
    envelope = _export(client, 1)
    envelope["sha256"] = "not-hex"
    r = client.post("/plans/import", json=envelope)
    assert r.status_code == 422
    assert r.json()["error"] == "validation_error"
    assert client.get("/plans").json()["total"] == 1


def test_import_schema_violation_is_validation_error(client):
    _create(client)
    envelope = _export(client, 1)
    envelope["bundle"]["request"]["max_concurrency"] = 0
    envelope["sha256"] = _canonical_digest(envelope["bundle"])
    r = client.post("/plans/import", json=envelope)
    assert r.status_code == 422
    assert r.json()["error"] == "validation_error"
    assert client.get("/plans").json()["total"] == 1


def test_import_unknown_extra_field_is_validation_error(client):
    _create(client)
    envelope = _export(client, 1)
    envelope["bundle"]["request"]["unexpected"] = True
    envelope["sha256"] = _canonical_digest(envelope["bundle"])
    r = client.post("/plans/import", json=envelope)
    assert r.status_code == 422
    assert r.json()["error"] == "validation_error"


def test_import_failure_leaves_no_partial_plan_after_prior_success(client):
    # One successful import, then a broken bundle: only the good plan exists.
    _create(client)
    envelope = _export(client, 1)
    assert client.post("/plans/import", json=envelope).status_code == 201

    bad = json.loads(json.dumps(envelope))
    bad["sha256"] = "f" * 64
    assert client.post("/plans/import", json=bad).status_code == 422

    listing = client.get("/plans").json()
    assert listing["total"] == 2
    imported = next(
        item for item in listing["items"] if item["id"] != 1
    )
    assert client.get(f"/plans/{imported['id']}").status_code == 200


# ---------------------------------------------------------------------------
# Idempotency across fresh databases (content addressing)
# ---------------------------------------------------------------------------


def test_same_digest_imported_twice_in_isolation_is_one_plan_per_db(client):
    plan = _create(client)
    envelope = _export(client, plan["id"])
    assert client.post("/plans/import", json=envelope).status_code == 201
    assert client.post("/plans/import", json=envelope).status_code == 200


def test_import_explicit_provenance_is_accepted_but_not_adopted(client):
    plan = _create(client)
    envelope = _export(client, plan["id"])
    envelope["provenance"] = {
        "source_plan_id": 98765,
        "created_at": "2000-01-01T00:00:00Z",
        "exported_at": "2000-01-02T00:00:00Z",
    }
    r = client.post("/plans/import", json=envelope)
    assert r.status_code == 201, r.text
    new_id = r.json()["id"]
    assert new_id != 98765
    # The imported plan gets a fresh local creation timestamp.
    assert r.json()["created_at"].startswith("2026-")
    assert r.json()["reused"] is False


def test_import_no_constraints_plan_round_trips(client):
    plan = _create(
        client,
        {"services": ["x", "y"], "dependencies": [], "max_concurrency": 2},
    )
    envelope = _export(client, plan["id"])
    body = client.post("/plans/import", json=envelope)
    assert body.status_code == 201, body.text
    assert body.json()["waves"] == plan["waves"]
    assert body.json()["fault_domains"] == []
    assert body.json()["dependencies"] == []


def test_import_release_window_violation_is_422(client):
    _create(
        client,
        {
            "services": ["a", "b", "c"],
            "dependencies": [
                {"service": "b", "depends_on": "a"},
                {"service": "c", "depends_on": "b"},
            ],
            "max_concurrency": 3,
            "max_waves": 3,
        },
    )
    envelope = _export(client, 1)
    bad = json.loads(json.dumps(envelope))
    bad["bundle"]["request"]["max_waves"] = 2
    bad["bundle"]["result"]["max_waves"] = 2
    bad["sha256"] = _canonical_digest(bad["bundle"])

    r = client.post("/plans/import", json=bad)
    assert r.status_code == 422
    issue = next(
        i
        for i in r.json()["issues"]
        if i["constraint"] == "result.max_waves"
    )
    assert issue["context"] == {"wave_count": 3, "max_waves": 2}
    assert client.get("/plans").json()["total"] == 1


def test_imported_plan_is_usable_by_gate_rollback_and_audit(client):
    plan = _create(client)
    envelope = _export(client, plan["id"])
    restored = client.post("/plans/import", json=envelope).json()
    pid = restored["id"]

    gate = client.post(f"/plans/{pid}/gate", json={})
    assert gate.status_code == 200
    assert gate.json()["waves"][0]["executable"] == ["db"]

    rollback = client.post(
        f"/plans/{pid}/rollback-preview",
        json={"deployed_services": plan["services"]},
    )
    assert rollback.status_code == 200
    assert rollback.json()["wave_count"] == plan["wave_count"]


def test_delete_then_reimport_same_digest_creates_fresh_plan(client):
    plan = _create(client)
    envelope = _export(client, plan["id"])
    first = client.post("/plans/import", json=envelope).json()
    assert client.delete(f"/plans/{first['id']}").status_code == 200

    again = client.post("/plans/import", json=envelope)
    assert again.status_code == 201
    assert again.json()["reused"] is False
    assert again.json()["id"] != first["id"]
