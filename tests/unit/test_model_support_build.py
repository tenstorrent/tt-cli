# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""scripts/build_model_support.py — the model_support.json generator.

The script is a maintainer tool rather than part of the package, so it is loaded
from its path. Its output ships in the wheel and is published with each release,
which is why the committed artifact is checked here too: a stale one would ship.
"""

import importlib.util
import json
import sys
from importlib import resources
from pathlib import Path

import pytest
import tomlkit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "build_model_support.py"

_spec = importlib.util.spec_from_file_location("build_model_support", SCRIPT)
build = importlib.util.module_from_spec(_spec)
# @dataclass resolves annotations through sys.modules[cls.__module__], so the
# module has to be registered before it executes.
sys.modules[_spec.name] = build
_spec.loader.exec_module(build)


# -- a spec small enough to reason about ----------------------------------------------
def _leaf(name, *, parser=None, reasoning=None, tt_config=None, status="COMPLETE"):
    return {
        "model_name": name,
        "status": status,
        "model_type": "LLM",
        "metadata": {"tool_call_parser_name": parser, "reasoning_parser_name": reasoning},
        "device_model_spec": {"max_context": 4096, "override_tt_config": tt_config},
    }


SPEC = {
    "schema_version": "0.1.0",
    "release_version": "0.18.0",
    "model_specs": {
        "org/Demo-8B": {
            "P300X2": {"vLLM": {"tt_transformers": _leaf("Demo-8B", parser="llama3_json")}},
            "T3K": {"vLLM": {"tt_transformers": _leaf("Demo-8B", parser="llama3_json")}},
        },
        "org/Media-1": {
            "P300X2": {"media": {"pipeline": _leaf("Media-1", status="EXPERIMENTAL")}},
            "P150X4": {"media": {"pipeline": _leaf("Media-1", status="EXPERIMENTAL")}},
        },
    },
}


def _override(**kwargs):
    entry = {
        "model": "Demo-8B",
        "reason": "broken",
        "details": "Wedges the board during warmup.",
        "verified_on": "2026-08-14",
    }
    entry.update(kwargs)
    return entry


def _overrides_file(tmp_path, *entries, schema_version=1):
    lines = [f"schema_version = {schema_version}"]
    for entry in entries:
        lines.append("\n[[override]]")
        lines += [f"{key} = {json.dumps(value)}" for key, value in entry.items()]
    path = tmp_path / "overrides.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def _devices(document, name):
    return next(m for m in document["models"] if m["name"] == name)["devices"]


# -- overrides validation -------------------------------------------------------------
def test_a_valid_overrides_file_round_trips(tmp_path):
    overrides = build.load_overrides(_overrides_file(tmp_path, _override(device="p300x2")))
    assert overrides.marks[0]["model"] == "Demo-8B"


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        (_override(reason="mostly-fine"), "unknown reason"),
        (_override(device="P300X2"), "must be lowercase"),
        ({"model": "Demo-8B", "reason": "broken"}, "is missing"),
    ],
)
def test_malformed_overrides_are_rejected(tmp_path, entry, expected):
    with pytest.raises(build.BuildError, match=expected):
        build.load_overrides(_overrides_file(tmp_path, entry))


def test_duplicate_overrides_are_rejected(tmp_path):
    path = _overrides_file(tmp_path, _override(device="p300x2"), _override(device="p300x2"))
    with pytest.raises(build.BuildError, match="duplicate entry"):
        build.load_overrides(path)


def test_an_overrides_file_from_a_future_schema_is_rejected(tmp_path):
    with pytest.raises(build.BuildError, match="schema_version"):
        build.load_overrides(_overrides_file(tmp_path, schema_version=99))


# -- applying overrides ---------------------------------------------------------------
def test_an_override_marks_only_the_device_it_names():
    document, _ = build.build_document(SPEC, build.Overrides(marks=[_override(device="p300x2")]))
    devices = _devices(document, "Demo-8B")
    assert devices["p300x2"]["supported"] is False
    assert devices["p300x2"]["unsupported"]["reason"] == "broken"
    assert devices["t3k"]["supported"] is True
    assert "unsupported" not in devices["t3k"]


def test_an_override_without_a_device_marks_every_device():
    document, _ = build.build_document(SPEC, build.Overrides(marks=[_override()]))
    devices = _devices(document, "Demo-8B")
    assert [d["supported"] for d in devices.values()] == [False, False]


def test_an_override_naming_an_unclaimed_device_is_an_error():
    """A device the model does not claim blocks nothing, and a silent no-op leaves
    a broken model on offer — the failure mode this check exists to stop."""
    with pytest.raises(build.BuildError, match="Did you mean 'p300x2'"):
        build.build_document(SPEC, build.Overrides(marks=[_override(device="p300x22")]))


def test_an_override_for_a_model_outside_the_spec_only_warns():
    """Marks outlive spec versions: one may be waiting for the next server pin."""
    document, warnings = build.build_document(SPEC, build.Overrides(marks=[_override(model="Not-In-Spec")]))
    assert any("Not-In-Spec" in w and "dormant" in w for w in warnings)
    assert all(
        d["supported"] for model in document["models"] for d in model["devices"].values()
    )


def test_override_details_are_reflowed_to_one_line():
    entry = _override(device="p300x2", details="wraps\n  across   lines\n")
    document, _ = build.build_document(SPEC, build.Overrides(marks=[entry]))
    assert _devices(document, "Demo-8B")["p300x2"]["unsupported"]["details"] == (
        "wraps across lines"
    )


# -- device fallbacks ------------------------------------------------------------------
def _fallback(**kwargs):
    rule = {
        "device": "p150x4",
        "serve_as": "p300x2",
        "engines": ["vLLM"],
        "details": "Both are four-chip Blackhole meshes.",
    }
    rule.update(kwargs)
    return rule


def test_a_fallback_fills_a_device_the_spec_omits():
    """Demo-8B has no p150x4 spec; the vLLM plugin maps both labels to the same
    (1, 4) mesh, so it runs there under the p300x2 name."""
    document, _ = build.build_document(SPEC, build.Overrides(fallbacks=[_fallback()]))
    borrowed = _devices(document, "Demo-8B")["p150x4"]
    assert borrowed["supported"] is True
    assert borrowed["serve_as"] == "p300x2"
    assert borrowed["support_source"] == "mesh-equivalent"


def test_device_fallback_does_not_shadow_a_working_native_spec():
    """t3k has its own spec, so it must keep it — serve_as would send the wrong
    device name to a model that never needed the substitution."""
    document, _ = build.build_document(
        SPEC, build.Overrides(fallbacks=[_fallback(device="t3k", serve_as="p300x2")])
    )
    assert "serve_as" not in _devices(document, "Demo-8B")["t3k"]


def test_device_fallback_rescues_a_device_whose_own_spec_is_broken():
    """Runs after the marks: a board with a broken native spec is exactly the
    case worth borrowing for, and the note has to explain the apparent conflict."""
    document, _ = build.build_document(
        SPEC,
        build.Overrides(
            marks=[_override(device="t3k", details="Stale trace region.")],
            fallbacks=[_fallback(device="t3k", serve_as="p300x2")],
        ),
    )
    rescued = _devices(document, "Demo-8B")["t3k"]
    assert rescued["supported"] is True
    assert rescued["serve_as"] == "p300x2"
    assert "Stale trace region." in rescued["note"]


def test_device_fallback_is_limited_to_the_engines_it_names():
    """Media and forge pin to one topology, so only vLLM may substitute."""
    document, _ = build.build_document(SPEC, build.Overrides(fallbacks=[_fallback(device="n150")]))
    assert "n150" not in _devices(document, "Media-1")


def test_device_fallback_does_not_borrow_from_a_broken_source():
    document, _ = build.build_document(
        SPEC,
        build.Overrides(marks=[_override(device="p300x2")], fallbacks=[_fallback()]),
    )
    assert "p150x4" not in _devices(document, "Demo-8B")


def test_a_fallback_records_which_kind_of_substitution_it_is():
    """A mesh swap and a single-chip narrowing are not equally safe, so the
    published list has to say which one a borrowed entry came from."""
    document, _ = build.build_document(
        SPEC,
        build.Overrides(
            fallbacks=[
                _fallback(support_source="single-chip")
            ]
        ),
    )
    assert _devices(document, "Demo-8B")["p150x4"]["support_source"] == "single-chip"


def _fallback_file(tmp_path, **kwargs):
    rule = _fallback(**kwargs)
    lines = ["schema_version = 1", "", "[[device_fallback]]"]
    lines += [f"{key} = {json.dumps(value)}" for key, value in rule.items()]
    path = tmp_path / "o.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"device": "P150X4"}, "must be lowercase"),
        ({"serve_as": "P300X2"}, "must be lowercase"),
        ({"engines": ["vllm"]}, "unknown engine"),
        ({"engines": ["vLLM", "Media"]}, "unknown engine"),
    ],
)
def test_a_fallback_typo_is_rejected_rather_than_silently_doing_nothing(
    tmp_path, kwargs, expected
):
    """Fallbacks are applied by string match and fill nothing when they match
    nothing, so a casing slip would quietly leave models off a board they run on
    — the same failure the marks' lowercase check exists to stop."""
    with pytest.raises(build.BuildError, match=expected):
        build.load_overrides(_fallback_file(tmp_path, **kwargs))


def test_a_fallback_rule_pointing_at_itself_is_rejected(tmp_path):
    path = tmp_path / "o.toml"
    path.write_text(
        'schema_version = 1\n\n[[device_fallback]]\ndevice = "p150x4"\n'
        'serve_as = "p150x4"\nengines = ["vLLM"]\ndetails = "x"\n'
    )
    with pytest.raises(build.BuildError, match="points at itself"):
        build.load_overrides(path)


# -- serve-time overrides --------------------------------------------------------------
def test_a_serve_override_records_only_the_flags_tt_must_pass():
    """The server applies the spec's own override_tt_config, so a flag is recorded
    only where tt is correcting one — serve_overrides is that list."""
    entry = {
        "model": "Media-1",
        "device": "p150x4",
        "docker_image": "ghcr.io/x/media:0.17.0",
        "details": "Spec resolves to an image tt cannot drive.",
    }
    document, _ = build.build_document(SPEC, build.Overrides(serve=[entry]))
    devices = _devices(document, "Media-1")
    assert devices["p150x4"]["serve_overrides"] == {"docker_image": "ghcr.io/x/media:0.17.0"}
    assert "serve_overrides" not in devices["p300x2"]


def test_a_serve_override_without_a_device_applies_to_every_board():
    entry = {
        "model": "Media-1",
        "docker_image": "ghcr.io/x/media:0.17.0",
        "details": "Weights-dir fix, needed on every board.",
    }
    document, _ = build.build_document(SPEC, build.Overrides(serve=[entry]))
    assert all(d["serve_overrides"] for d in _devices(document, "Media-1").values())


def test_a_serve_override_that_sets_nothing_is_rejected(tmp_path):
    path = tmp_path / "o.toml"
    path.write_text(
        'schema_version = 1\n\n[[serve_override]]\nmodel = "Demo-8B"\ndetails = "x"\n'
    )
    with pytest.raises(build.BuildError, match="sets nothing"):
        build.load_overrides(path)


# -- what the spec contributes --------------------------------------------------------
def test_launch_settings_are_carried_from_the_spec():
    """The parsers and tt-config `tt serve` needs come from the spec, never a guess."""
    spec = json.loads(json.dumps(SPEC))
    leaf = spec["model_specs"]["org/Demo-8B"]["P300X2"]["vLLM"]["tt_transformers"]
    leaf["metadata"]["reasoning_parser_name"] = "qwen3"
    leaf["device_model_spec"]["override_tt_config"] = {"trace_region_size": 57}

    device = _devices(build.build_document(spec, build.Overrides())[0], "Demo-8B")["p300x2"]
    assert device["tool_call_parser"] == "llama3_json"
    assert device["reasoning_parser"] == "qwen3"
    assert device["override_tt_config"] == {"trace_region_size": 57}


def test_every_engine_gets_a_tt_model_id():
    """run.py accepts any MODEL_SPECS name, so media and forge entries are
    servable too — the published list must not imply otherwise."""
    document, _ = build.build_document(SPEC, build.Overrides())
    by_name = {m["name"]: m for m in document["models"]}
    assert by_name["Demo-8B"]["tt_model_id"] == "Demo-8B"
    assert by_name["Media-1"]["tt_model_id"] == "Media-1"


def test_the_build_is_deterministic():
    """No timestamps or ordering drift: --check compares rendered bytes, so an
    unstable build would report a diff on every run."""
    first = build.render(build.build_document(SPEC, build.Overrides(marks=[_override(device="p300x2")]))[0])
    second = build.render(build.build_document(SPEC, build.Overrides(marks=[_override(device="p300x2")]))[0])
    assert first == second


# -- the committed artifact -----------------------------------------------------------
def test_the_committed_artifact_matches_its_inputs():
    """What `--check` enforces in CI. model_support.json is generated: a hand edit,
    or an inputs change without a rebuild, ships a file that misstates support."""
    spec = build.load_spec(build.SPEC_PATH)
    overrides = build.load_overrides(build.OVERRIDES_PATH)
    document, _ = build.build_document(spec, overrides)
    assert build.OUTPUT_PATH.read_text() == build.render(document)


def test_the_committed_artifact_matches_the_pinned_inference_server():
    """Bumping the server pin and rebuilding the support list must go together."""
    document = json.loads(
        (resources.files("tenstorrent.modelhub") / "model_support.json").read_text()
    )
    supplement = tomlkit.parse(
        (resources.files("tenstorrent.tools") / "supplement.toml").read_text()
    ).unwrap()
    pinned = supplement["tools"]["tt-inference-server"]["golden_version"]
    assert f"v{document['release_version']}" == pinned


def test_the_committed_artifact_marks_the_boards_studio_verified():
    """The ported tt-studio marks are the reason this file exists; a build that
    quietly dropped them would still be well-formed."""
    document = json.loads(build.OUTPUT_PATH.read_text())
    devices = _devices(document, "Z-Image-Turbo")
    assert devices["p300x2"]["supported"] is False
    assert "board reset" in devices["p300x2"]["unsupported"]["details"]

def test_the_bundled_spec_is_structurally_sound():
    """The build input, checked before it is flattened: a leaf missing model_name
    or status would otherwise surface as a confusing failure downstream."""
    doc = json.loads(build.SPEC_PATH.read_text())
    assert doc["schema_version"] == build.SPEC_SCHEMA_VERSION
    assert len(doc["model_specs"]) >= 50
    for repo, device_map in doc["model_specs"].items():
        for device_type, engine_map in device_map.items():
            for engine, impls in engine_map.items():
                for impl, leaf in impls.items():
                    where = f"{repo} → {device_type} → {engine} → {impl}"
                    assert leaf.get("model_name"), where
                    assert leaf.get("device_type"), where
                    assert leaf.get("status"), where


def test_this_box_has_servable_models_in_the_committed_artifact():
    document = json.loads(build.OUTPUT_PATH.read_text())
    for device in ("p300", "p300x2"):
        assert any(
            m["tt_model_id"]
            and device in m["devices"]
            and m["devices"][device]["supported"]
            for m in document["models"]
        ), f"no servable model for {device}"


def test_the_two_four_chip_blackhole_meshes_agree_where_a_fallback_applies():
    """p300x2 and p150x4 are the same four Blackhole chips in a different
    arrangement, so a model that runs on one through a device_fallback runs on the
    other. A difference means a mark or a rule is scoped to one of the pair by
    mistake — which is how mochi-1-preview ended up marked on only one mesh, and
    how FLUX ended up marked on the other.

    Forge is excluded: it genuinely pins to one topology. Falcon3-7B-Instruct and
    yolox_nano fail on p150x4 despite their p150 specs, and the release spec
    publishes forge models (the Qwen3-Embedding pair from 0.21.0) for p300x2 and
    not p150x4. Those are real asymmetries, not bookkeeping errors, so the
    fallback rules deliberately cover vLLM and media only and this invariant
    follows them.
    """
    document = json.loads(build.OUTPUT_PATH.read_text())
    fallback_engines = {"vLLM", "media"}

    def available(device):
        return {
            m["name"]
            for m in document["models"]
            if (support := m["devices"].get(device))
            and support["supported"]
            and fallback_engines & set(support["engines"])
        }

    assert available("p300x2") == available("p150x4")
