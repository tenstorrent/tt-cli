# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

import json
from pathlib import Path

from tenstorrent.backends.smi import parse_snapshot

DATA = Path(__file__).parent.parent / "fakes" / "data"


def load(name):
    return json.loads((DATA / f"snapshot_{name}.json").read_text())


# normal/multi/empty are verbatim captures of `tt-smi -s` (tt-smi 5.3.0, QuietBox
# with 2x P300, 2026-07-20) — real value shapes: telemetry strings with leading
# spaces, dram_status as a bool, pcie_speed as an int, per-asic entries.


def test_parse_normal_snapshot():
    snap = parse_snapshot(load("normal"))
    assert snap.warnings == []
    assert snap.host["Driver"] == "TT-KMD 2.9.0"
    assert len(snap.devices) == 1
    dev = snap.devices[0]
    assert dev.index == 0
    assert dev.board_type == "p300c"
    assert dev.bus_id == "0000:01:00.0"
    assert dev.temperature_c == 33.8
    assert dev.power_w == 13.0  # parsed from " 13.0" (leading space)
    assert dev.aiclk_mhz == 800
    assert dev.dram_status == "True"  # bool in the snapshot, normalized to str
    assert dev.pcie_speed == "4"  # int in the snapshot, normalized to str
    assert dev.firmware["fw_bundle_version"] == "19.11.0.0"


def test_parse_multi_snapshot():
    snap = parse_snapshot(load("multi"))
    # 2x P300: each dual-asic board shows up as two per-asic entries sharing a board_id
    assert [d.board_type for d in snap.devices] == ["p300c"] * 4
    assert [d.index for d in snap.devices] == [0, 1, 2, 3]
    board_ids = [d.board_id for d in snap.devices]
    assert board_ids[0] == board_ids[1] and board_ids[2] == board_ids[3]
    assert board_ids[0] != board_ids[2]


def test_parse_empty_snapshot():
    snap = parse_snapshot(load("empty"))
    assert snap.devices == []
    assert snap.warnings == []


def test_parse_malformed_never_raises():
    snap = parse_snapshot(load("malformed"))
    assert len(snap.devices) == 2
    dev0 = snap.devices[0]
    assert dev0.board_type == "n150 L"
    assert dev0.voltage_v is None  # "not-a-number" degraded, not crashed
    assert dev0.temperature_c == 38.6
    dev1 = snap.devices[1]
    assert dev1.board_type is None
    assert snap.host == {}
    assert any("voltage" in w for w in snap.warnings)
    assert any("host_info" in w for w in snap.warnings)


def test_parse_garbage_roots():
    assert parse_snapshot({}).devices == []
    assert parse_snapshot({"device_info": "nope"}).devices == []
    assert parse_snapshot(None).warnings  # type: ignore[arg-type]
