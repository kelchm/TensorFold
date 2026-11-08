"""Check actual Zig formatter output using invented CPU counters, never device qualification."""
import copy
import sys
from pathlib import Path

from tools.perf.glm_replay.common import ReplayError, require
from tools.perf.glm_replay.fixture import load_fixture
from tools.perf.glm_replay.results import load_results, validate_results


def check(fixture_path, outputs):
    fixture = load_fixture(fixture_path, allow_synthetic=True)
    require(fixture["kind"] == "synthetic", "interop controls must be explicitly synthetic")
    fixtures = {fixture["fixture_sha256"]: fixture}
    records = load_results(outputs)
    roots = [Path(name).parent for name in outputs]

    def validate(value):
        return validate_results(value, fixtures, allow_synthetic=True, anchors_root=roots)

    validate(records)
    require(len(records) == 4, "expected actual off/on two-rank formatter output")
    negatives = []
    value = copy.deepcopy(records)
    value[0]["completed_ranks"] = "\x00\x01"
    negatives.append(value)
    value = copy.deepcopy(records)
    value[0]["wall_ns"] += 1
    negatives.append(value)
    value = copy.deepcopy(records)
    value[0]["forwards"][0]["phases_ns"]["target"] += 1
    negatives.append(value)
    value = copy.deepcopy(records)
    for record in value[2:]:
        record["session_nonce"] = value[0]["session_nonce"]
    negatives.append(value)
    for value in negatives:
        try:
            validate(value)
        except ReplayError:
            continue
        raise ReplayError("actual formatter negative control was accepted")
    print("PASS synthetic actual Zig writer/common validator: four records, four rejection controls; device UNRUN")


if __name__ == "__main__":
    check(sys.argv[1], sys.argv[2:])
