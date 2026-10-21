"""Deterministic runtime diagnostics and invariant verification for `bitboard_chess`."""

from __future__ import annotations

import time
from typing import Any

from .board import Board, STARTPOS_FEN, KIWIPETE_FEN, square_name, parse_square
from .search import evaluate

def _domain_probe(seed: int) -> dict:
    board = Board.from_startpos()
    perft1 = board.perft(1)
    perft2 = board.perft(2)
    kiwi = Board.from_fen(KIWIPETE_FEN)
    perft_kiwi_1 = kiwi.perft(1)
    zobrist_ok = (kiwi.zobrist_hash == kiwi.compute_zobrist_hash())
    sq_rt = square_name(parse_square("e4"))
    eval_cp = evaluate(board)
    return {
        "perft1_startpos": perft1,
        "perft2_startpos": perft2,
        "perft1_kiwipete": perft_kiwi_1,
        "zobrist_consistent": zobrist_ok,
        "startpos_eval_cp": eval_cp,
        "square_roundtrip": sq_rt,
        "invariants_ok": (perft1 == 20 and perft2 == 400 and perft_kiwi_1 == 48 and zobrist_ok and eval_cp == 0 and sq_rt == "e4"),
    }


def run_diagnostic_suite(seed: int = 42) -> dict[str, Any]:
    """Execute deterministic domain probe for `bitboard-uci-chess` and return structured metrics."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    t0 = time.perf_counter()
    metrics = _domain_probe(seed)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return {
        "repository": "bitboard-uci-chess",
        "package": "bitboard_chess",
        "seed": seed,
        "elapsed_ms": elapsed_ms,
        "metrics": metrics,
        "status": "ok" if metrics.get("invariants_ok", False) else "failed",
    }


def validate_runtime_invariants(seed: int = 42) -> dict[str, Any]:
    """Validate deterministic reproducibility and domain invariants for `bitboard_chess`."""
    run_a = run_diagnostic_suite(seed=seed)
    run_b = run_diagnostic_suite(seed=seed)
    reproducible = run_a["metrics"] == run_b["metrics"]
    passed = bool(run_a["metrics"].get("invariants_ok", False)) and reproducible
    return {
        "repository": "bitboard-uci-chess",
        "package": "bitboard_chess",
        "seed": seed,
        "reproducible": reproducible,
        "passed": passed,
        "metrics": run_a["metrics"],
    }
