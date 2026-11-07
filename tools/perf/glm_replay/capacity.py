"""Equal logical capacity and explicit per-owner prompt/tape/twin reservations."""
from .common import fields, integer, require


def ceil_div(value, quantum):
    return (value + quantum - 1) // quantum


def reservations(fixture, page_tokens, prompt_twins):
    extents = {s["stream_id"]: len(s["prompt_tokens"]) for s in fixture["streams"]}
    for forward in fixture["preparation"] + fixture["forwards"]:
        for part in forward["parts"]:
            extent = part["position"] + part["offered"]
            extents[part["stream_id"]] = max(extents[part["stream_id"]], extent)
    extents = {sid: extent + 16 for sid, extent in extents.items()}
    owners, twins = [0, 0], [0, 0]
    for stream in fixture["streams"]:
        owners[stream["owner"]] += ceil_div(extents[stream["stream_id"]], page_tokens)
        peer = 1 - stream["owner"]
        twins[peer] = max(twins[peer], prompt_twins * ceil_div(len(stream["prompt_tokens"]), page_tokens))
    return {"owner_pages": [main + twin for main, twin in zip(owners, twins)], "stream_extents": extents}


def minimum_h(fixture, page_tokens, configured_capacity, prompt_twins=1):
    pages = reservations(fixture, page_tokens, prompt_twins)["owner_pages"]
    return max(*pages, ceil_div(max(131072, configured_capacity // 4), page_tokens))


def validate_pool(fixture):
    pool = fixture["config"]["pool"]
    fields(pool, "h page_tokens configured_capacity prompt_twins", "pool")
    for key in ("h", "page_tokens", "configured_capacity"):
        integer(pool[key], f"pool.{key}", 1)
    require(pool["configured_capacity"] >= 64 and pool["configured_capacity"] % 64 == 0,
            "pool: configured capacity must be a positive multiple of 64")
    require(pool["page_tokens"] == 512, "pool: pinned GLM page size must be 512")
    require(integer(pool["prompt_twins"], "pool.prompt_twins", 1) == 1, "pool: exactly one transient twin required")
    needed = minimum_h(fixture, pool["page_tokens"], pool["configured_capacity"], pool["prompt_twins"])
    require(pool["h"] >= needed, "pool: insufficient owner, lookahead, twin or production-minimum capacity")
    extents = reservations(fixture, pool["page_tokens"], pool["prompt_twins"])["stream_extents"]
    require(max(extents.values()) <= pool["configured_capacity"], "pool: stream exceeds configured capacity")
    return {"off_page_argument": 2 * pool["h"], "on_page_argument": 2 * pool["h"] + 4,
            "logical_pages": 2 * pool["h"], "required_h": needed,
            **reservations(fixture, pool["page_tokens"], pool["prompt_twins"])}
