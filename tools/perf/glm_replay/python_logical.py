"""Typed retained logical snapshots, independent of torch and the frozen engine."""
from .common import fields, hex_id, integer, require

STATE_FIELDS = ("position device_position ring_from index_from conv recurrent latent_codes latent_scales "
                "index_keys index_gates pooled_keys ring_k ring_v")


def initial_positions(fixture):
    positions = {str(s["stream_id"]): len(s["prompt_tokens"]) for s in fixture["streams"]}
    for forward in fixture["preparation"]:
        for part in forward["parts"]:
            positions[str(part["stream_id"])] = part["position"] + part["kept"]
    return positions


def validate_states(states, fixture, positions):
    require(isinstance(states, dict) and set(states) == set(positions), "logical state slot coverage mismatch")
    for sid, state in states.items():
        fields(state, STATE_FIELDS, "logical state")
        pos = positions[sid]
        require(integer(state["position"], "position") == pos
                and integer(state["device_position"], "device position") == pos, "logical state position mismatch")
        require(integer(state["ring_from"], "ring from") == max(0, pos - 2048)
                and integer(state["index_from"], "index from") == pos // 4 * 4, "logical live range mismatch")
        for key in ("conv", "recurrent"):
            hex_id(state[key], key)
        for key in ("latent_codes", "latent_scales", "index_keys", "index_gates", "pooled_keys", "ring_k", "ring_v"):
            require(isinstance(state[key], list) and state[key], "empty logical tensor hashes")
            if key not in ("ring_k", "ring_v"):
                require(len(state[key]) == 11, "DSA layer hash coverage mismatch")
            for sha in state[key]:
                hex_id(sha, key)
        require(len(state["ring_k"]) == len(state["ring_v"]), "drafter K/V hash coverage mismatch")


def validate_capture(capture, fixture, parts, positions, state_slots=None):
    fields(capture, "states kept", "logical capture")
    slots = set(positions) if state_slots is None else state_slots
    validate_states(capture["states"], fixture, {sid: positions[sid] for sid in slots})
    require(isinstance(capture["kept"], dict)
            and set(capture["kept"]) == {str(p["stream_id"]) for p in parts}, "logical kept slot mismatch")
    vocab = fixture["provenance"]["tokenizer"]["vocab_size"]
    for part in parts:
        kept = capture["kept"][str(part["stream_id"])]
        fields(kept, "logits taps selected", "logical kept rows")
        for key in ("logits", "taps", "selected"):
            require(isinstance(kept[key], list) and len(kept[key]) == part["kept"], "logical row count mismatch")
            for value in kept[key]:
                integer(value, "selected token", 0, vocab - 1) if key == "selected" else hex_id(value, key)
