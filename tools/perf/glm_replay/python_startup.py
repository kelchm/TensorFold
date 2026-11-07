"""Safe production startup helpers and explicit fixed-work deviations."""
import importlib
import os

from .common import integer, require

DEVIATIONS = {"admission": "fixed_equal_capacity_pool; production admit and automatic cache budgeting omitted; fit UNRUN",
              "settings": "full launch/environment store agreement and both-rank runtime fingerprints replace server gathers",
              "calibration": "omitted: GlmEngine._calibrate calls Drafter.propose, forbidden by fixed-work contract",
              "scheduler": "no server scheduler or draft-choice cost table is instantiated"}


def launch_settings(production, display_kv, launch, geometry=None):
    geometry = geometry or importlib.import_module("tensorfold.cuda.geometry")
    rows = int(os.environ.get("TF_GLM_PREFILL_ROWS", "") or geometry.PREFILL_ROWS)
    integer(rows, "production prefill rows", 64, 16384)
    require(rows % 64 == 0, "production prefill rows must tile 64")
    grid = production.prompt_grid(rows)
    require(rows == launch["prefill_rows"] and grid == launch["grid"], "metadata differs from production prefill/grid")
    require(display_kv.credit() == 0, "nonzero production display KV credit is unsupported")
    return {"prefill_rows": rows, "grid": grid, "display": 0, "deviations": DEVIATIONS}


def capture_bank(verifier, needed):
    verifier.capture()
    bank = sorted(verifier.graphs)
    require(bank == list(range(1, 33)) and set(needed) <= set(bank), "production graph bank mismatch")
    return bank


def open_transport(nccl, rank, kind):
    require(kind in ("nccl", "roce"), "unsupported production transport")
    nccl.barrier()
    if kind == "roce":
        return importlib.import_module("tensorfold.cuda.roce").RoceComm(nccl, rank, 2)
    return nccl
