"""PeSTO tapered evaluation, MVV-LVA move ordering, Zobrist Transposition Table, and Negamax Alpha-Beta search."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import time
from typing import Sequence

from .board import (
    BISHOP,
    BLACK,
    KING,
    KNIGHT,
    PAWN,
    QUEEN,
    ROOK,
    WHITE,
    Board,
    Move,
    _iter_bits,
)

MATE_SCORE = 30000
INF_SCORE = 32000

# PeSTO Middlegame and Endgame piece values (centipawns)
MG_VALUE = [82, 337, 365, 477, 1025, 0]
EG_VALUE = [94, 281, 297, 512, 936, 0]
PHASE_INC = [0, 1, 1, 2, 4, 0]
MAX_PHASE = 24

# PeSTO Piece-Square Tables (Rank 8 -> Rank 1 from White's perspective; indexed via sq ^ 56 for White, sq for Black)
MG_PST: list[list[int]] = [
    # Pawn
    [
          0,   0,   0,   0,   0,   0,   0,   0,
         98, 134,  61,  95,  68, 126,  34, -11,
         -6,   7,  26,  31,  65,  56,  25, -20,
        -14,  13,   6,  21,  23,  12,  17, -23,
        -27,  -2,  -5,  12,  17,   6,  10, -25,
        -26,  -4,  -4, -10,   3,   3,  33, -12,
        -35,  -1, -20, -23, -15,  24,  38, -22,
          0,   0,   0,   0,   0,   0,   0,   0,
    ],
    # Knight
    [
        -167, -89, -34, -49,  61, -97, -15, -107,
         -73, -41,  72,  36,  23,  62,   7,  -17,
         -47,  60,  37,  65,  84, 129,  73,   44,
          -9,  17,  19,  53,  37,  69,  18,   22,
         -13,   4,  16,  13,  28,  19,  21,   -8,
         -23,  -9,  12,  10,  19,  17,  25,  -16,
         -29, -53, -12,  -3,  -1,  18, -14,  -19,
        -105, -21, -58, -33, -17, -28, -19,  -23,
    ],
    # Bishop
    [
        -29,   4, -82, -37, -25, -42,   7,  -8,
        -26,  16, -18, -13,  30,  59,  18, -47,
        -16,  37,  43,  40,  35,  50,  37,  -2,
         -4,   5,  19,  50,  37,  37,   7,  -2,
         -6,  13,  13,  26,  34,  12,  10,   4,
          0,  15,  15,  15,  14,  27,  18,  10,
          4,  15,  16,   0,   7,  21,  33,   1,
        -33,  -3, -14, -21, -13, -12, -39, -21,
    ],
    # Rook
    [
         32,  42,  32,  51,  63,   9,  31,  43,
         27,  32,  58,  62,  80,  67,  26,  44,
         -5,  19,  26,  36,  17,  45,  61,  16,
        -24, -11,   7,  26,  24,  35,  -8, -20,
        -36, -26, -12,  -1,   9,  -7,   6, -23,
        -45, -25, -16, -17,   3,   0,  -5, -33,
        -44, -16, -20,  -9,  -1,  11,  -6, -71,
        -19, -13,   1,  17,  16,   7, -37, -26,
    ],
    # Queen
    [
        -28,   0,  29,  12,  59,  44,  43,  45,
        -24, -39,  -5,   1, -16,  57,  28,  54,
        -13, -17,   7,   8,  29,  56,  47,  57,
        -27, -27, -16, -16,  -1,  17,  -2,   1,
         -9, -26,  -9, -10,  -2,  -4,   3,  -3,
        -14,   2, -11,  -2,  -5,   2,  14,   5,
        -35,  -8,  11,   2,   8,  15,  -3,   1,
         -1, -18,  -9,  10, -15, -25, -31, -50,
    ],
    # King
    [
        -65,  23,  16, -15, -56, -34,   2,  13,
         29,  -1, -20,  -7,  -8,  -4, -38, -29,
         -9,  24,   2, -16, -20,   6,  22, -22,
        -17, -20, -12, -27, -30, -25, -14, -36,
        -49,  -1, -27, -39, -46, -44, -33, -51,
        -14, -14, -22, -46, -44, -30, -15, -27,
          1,   7,  -8, -64, -43, -16,   9,   8,
        -15,  36,  12, -54,   8, -28,  24,  14,
    ],
]

EG_PST: list[list[int]] = [
    # Pawn
    [
          0,   0,   0,   0,   0,   0,   0,   0,
        178, 173, 158, 134, 147, 132, 165, 187,
         94, 100,  85,  67,  56,  53,  82,  84,
         32,  24,  13,   5,  -2,   4,  17,  17,
         13,   9,  -3,  -7,  -7,  -8,   3,  -1,
          4,   7,  -6,   1,   0,  -5,  -1,  -8,
         13,   8,   8,  10,  13,   0,   2,  -7,
          0,   0,   0,   0,   0,   0,   0,   0,
    ],
    # Knight
    [
        -58, -38, -13, -28, -31, -27, -63, -99,
        -25,  -8, -25,  -2,  -9, -25, -24, -52,
        -24, -20,  10,   9,  -1,  -9, -19, -41,
        -17,   3,  22,  22,  22,  11,   8, -18,
        -18,  -6,  16,  25,  16,  17,   4, -18,
        -23,  -3,  -1,  15,  10,  -3, -20, -22,
        -42, -20, -10,  -5,  -2, -20, -23, -44,
        -29, -51, -23, -15, -22, -18, -50, -64,
    ],
    # Bishop
    [
        -14, -21, -11,  -8,  -7,  -9, -17, -24,
         -8,  -4,   7, -12,  -3, -13,  -4, -14,
          2,  -8,   0,  -1,  -2,   6,   0,   4,
         -3,   9,  12,   9,  14,  10,   3,   2,
         -6,   3,  13,  19,   7,  10,  -3,  -9,
        -12,  -3,   8,  10,  13,   3,  -7, -15,
        -14, -18,  -7,  -1,   4,  -9, -15, -27,
        -23,  -9, -23,  -5,  -9, -16,  -5, -17,
    ],
    # Rook
    [
         13,  10,  18,  15,  12,  12,   8,   5,
         11,  13,  13,  11,  -3,   3,   8,   3,
          7,   7,   7,   5,   4,  -3,  -5,  -3,
          4,   3,  13,   1,   2,   1,  -1,   2,
          3,   5,   8,   4,  -5,  -6,  -8, -11,
         -4,   0,  -5,  -1,  -7, -12,  -8, -16,
         -6,  -6,   0,   2,  -9,  -9, -11,  -3,
         -9,   2,   3,  -1,  -5, -13,   4, -20,
    ],
    # Queen
    [
         -9,  22,  22,  27,  27,  19,  10,  20,
        -17,  20,  32,  41,  58,  25,  30,   0,
        -20,   6,   9,  49,  47,  35,  19,   9,
          3,  22,  24,  45,  57,  40,  57,  36,
        -18,  28,  19,  47,  31,  34,  39,  23,
        -16, -27,  15,   6,   9,  17,  10,   5,
        -22, -23, -30, -16, -16, -23, -36, -32,
        -33, -28, -22, -43,  -5, -32, -20, -41,
    ],
    # King
    [
        -74, -35, -18, -18, -11,  15,   4, -17,
        -12,  17,  14,  17,  17,  38,  23,  11,
         10,  17,  23,  15,  20,  45,  44,  13,
         -8,  22,  24,  27,  26,  33,  26,   3,
        -18,  -4,  21,  24,  27,  23,   9, -11,
        -19,  -3,  11,  21,  23,  16,   7,  -9,
        -27, -11,   4,  13,  14,   4,  -5, -17,
        -53, -34, -21, -11, -28, -14, -24, -43,
    ],
]


class TTFlag(IntEnum):
    """Transposition Table bound classification."""

    EXACT = 0
    LOWERBOUND = 1
    UPPERBOUND = 2


@dataclass(frozen=True)
class TTEntry:
    """Single Zobrist Transposition Table entry."""

    zobrist_key: int
    depth: int
    score: int
    flag: TTFlag
    best_move: Move | None


class TranspositionTable:
    """Fixed-capacity Zobrist Transposition Table with depth-preferred replacement."""

    def __init__(self, max_entries: int = 65536) -> None:
        self.max_entries = max_entries
        self._table: dict[int, TTEntry] = {}

    def probe(self, zobrist_key: int) -> TTEntry | None:
        return self._table.get(zobrist_key)

    def store(
        self,
        zobrist_key: int,
        depth: int,
        score: int,
        flag: TTFlag,
        best_move: Move | None,
    ) -> None:
        existing = self._table.get(zobrist_key)
        if existing is None or depth >= existing.depth:
            if len(self._table) >= self.max_entries and existing is None:
                # Evict oldest inserted key
                oldest_key = next(iter(self._table))
                del self._table[oldest_key]
            self._table[zobrist_key] = TTEntry(
                zobrist_key=zobrist_key,
                depth=depth,
                score=score,
                flag=flag,
                best_move=best_move,
            )

    def clear(self) -> None:
        self._table.clear()

    def __len__(self) -> int:
        return len(self._table)


def evaluate(board: Board) -> int:
    """Evaluate `board` using PeSTO tapered piece-square tables from side-to-move's perspective."""
    mg = [0, 0]
    eg = [0, 0]
    phase = 0

    for ptype in range(6):
        w_bb = board.pieces[WHITE][ptype]
        for sq in _iter_bits(w_bb):
            pst_idx = sq ^ 56  # Flip rank so Rank 8 is index 0..7
            mg[WHITE] += MG_VALUE[ptype] + MG_PST[ptype][pst_idx]
            eg[WHITE] += EG_VALUE[ptype] + EG_PST[ptype][pst_idx]
            phase += PHASE_INC[ptype]

        b_bb = board.pieces[BLACK][ptype]
        for sq in _iter_bits(b_bb):
            pst_idx = sq  # Black's Rank 1 is index 0..7
            mg[BLACK] += MG_VALUE[ptype] + MG_PST[ptype][pst_idx]
            eg[BLACK] += EG_VALUE[ptype] + EG_PST[ptype][pst_idx]
            phase += PHASE_INC[ptype]

    mg_phase = min(phase, MAX_PHASE)
    eg_phase = MAX_PHASE - mg_phase

    mg_score = mg[board.turn] - mg[1 - board.turn]
    eg_score = eg[board.turn] - eg[1 - board.turn]
    return (mg_score * mg_phase + eg_score * eg_phase) // MAX_PHASE


def mvv_lva_score(board: Board, move: Move) -> int:
    """Score a move using Most Valuable Victim - Least Valuable Attacker (MVV-LVA)."""
    attacker_info = board.piece_at_sq[move.from_sq]
    attacker_ptype = attacker_info[1] if attacker_info is not None else PAWN

    score = 0
    if move.is_en_passant:
        victim_ptype: int | None = PAWN
    else:
        victim_info = board.piece_at_sq[move.to_sq]
        victim_ptype = victim_info[1] if victim_info is not None else None

    if victim_ptype is not None:
        score += 100_000 + 10 * MG_VALUE[victim_ptype] - (MG_VALUE[attacker_ptype] // 10)

    if move.promotion is not None:
        score += 80_000 + MG_VALUE[move.promotion]

    return score


def order_moves(
    board: Board,
    moves: Sequence[Move],
    tt_move: Move | None = None,
) -> list[Move]:
    """Order moves with Transposition Table best move first, followed by MVV-LVA captures."""

    def _priority(m: Move) -> int:
        if tt_move is not None and m == tt_move:
            return 1_000_000
        return mvv_lva_score(board, m)

    return sorted(moves, key=_priority, reverse=True)


@dataclass(frozen=True)
class SearchResult:
    """Result of an iterative-deepening Negamax Alpha-Beta search."""

    best_move: Move | None
    score: int
    depth: int
    nodes: int
    pv: tuple[Move, ...]
    elapsed_ms: float


class Searcher:
    """Negamax Alpha-Beta search engine with Quiescence search and Zobrist Transposition Table."""

    def __init__(self, tt_size: int = 65536) -> None:
        self.tt = TranspositionTable(max_entries=tt_size)
        self.nodes = 0

    def quiescence(self, board: Board, alpha: int, beta: int, ply: int = 0, qply: int = 0) -> int:
        """Capture-only Quiescence search (with full check evasions when in check)."""
        self.nodes += 1
        if board.in_check():
            legal_evasions = board.generate_legal_moves()
            if not legal_evasions:
                return -MATE_SCORE + ply
            if qply >= 6:
                return evaluate(board)
            for move in order_moves(board, legal_evasions):
                board.make_move(move)
                score = -self.quiescence(board, -beta, -alpha, ply + 1, qply + 1)
                board.unmake_move()
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
            return alpha

        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        if qply >= 6:
            return alpha

        captures = order_moves(board, board.generate_legal_captures())
        for move in captures:
            board.make_move(move)
            score = -self.quiescence(board, -beta, -alpha, ply + 1, qply + 1)
            board.unmake_move()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def negamax(self, board: Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        """Negamax search with Alpha-Beta pruning and Transposition Table lookup."""
        alpha_orig = alpha
        zkey = board.zobrist_hash
        tt_entry = self.tt.probe(zkey)
        tt_move: Move | None = None

        if tt_entry is not None:
            tt_move = tt_entry.best_move
            if tt_entry.depth >= depth and ply > 0 and abs(tt_entry.score) < MATE_SCORE - 100:
                if tt_entry.flag == TTFlag.EXACT:
                    return tt_entry.score
                if tt_entry.flag == TTFlag.LOWERBOUND:
                    alpha = max(alpha, tt_entry.score)
                elif tt_entry.flag == TTFlag.UPPERBOUND:
                    beta = min(beta, tt_entry.score)
                if alpha >= beta:
                    return tt_entry.score

        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply=ply)

        self.nodes += 1
        legal_moves = board.generate_legal_moves()
        if not legal_moves:
            if board.in_check():
                return -MATE_SCORE + ply
            return 0

        if board.halfmove_clock >= 100:
            return 0

        ordered = order_moves(board, legal_moves, tt_move=tt_move)
        best_score = -INF_SCORE
        best_move: Move | None = ordered[0]

        for move in ordered:
            board.make_move(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.unmake_move()

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        if best_score <= alpha_orig:
            flag = TTFlag.UPPERBOUND
        elif best_score >= beta:
            flag = TTFlag.LOWERBOUND
        else:
            flag = TTFlag.EXACT

        self.tt.store(zkey, depth, best_score, flag, best_move)
        return best_score

    def _extract_pv(self, board: Board, max_len: int) -> list[Move]:
        pv: list[Move] = []
        seen: set[int] = set()
        for _ in range(max_len):
            zkey = board.zobrist_hash
            if zkey in seen:
                break
            seen.add(zkey)
            entry = self.tt.probe(zkey)
            if entry is None or entry.best_move is None:
                break
            if entry.best_move not in board.generate_legal_moves():
                break
            pv.append(entry.best_move)
            board.make_move(entry.best_move)
        for _ in range(len(pv)):
            board.unmake_move()
        return pv

    def search(self, board: Board, max_depth: int = 4) -> SearchResult:
        """Run iterative deepening search from depth 1 up to `max_depth`."""
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        t0 = time.perf_counter()
        self.nodes = 0
        best_move: Move | None = None
        best_score = 0
        pv: list[Move] = []

        legal_root = board.generate_legal_moves()
        if not legal_root:
            score = -MATE_SCORE if board.in_check() else 0
            return SearchResult(None, score, 0, 0, (), 0.0)

        for d in range(1, max_depth + 1):
            best_score = self.negamax(board, d, -INF_SCORE, INF_SCORE, ply=0)
            pv = self._extract_pv(board, d)
            if pv:
                best_move = pv[0]
            elif best_move is None:
                best_move = legal_root[0]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return SearchResult(
            best_move=best_move,
            score=best_score,
            depth=max_depth,
            nodes=self.nodes,
            pv=tuple(pv),
            elapsed_ms=elapsed_ms,
        )
