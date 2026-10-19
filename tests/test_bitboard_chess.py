"""Comprehensive pytest test suite for bitboard-uci-chess."""

from __future__ import annotations

import pytest

from bitboard_chess import (
    BISHOP,
    BLACK,
    KIWIPETE_FEN,
    KNIGHT,
    MATE_SCORE,
    PAWN,
    QUEEN,
    ROOK,
    STARTPOS_FEN,
    TTFlag,
    UCIEngine,
    WHITE,
    Board,
    Move,
    Searcher,
    bishop_attacks,
    evaluate,
    king_attacks,
    knight_attacks,
    mvv_lva_score,
    order_moves,
    parse_square,
    queen_attacks,
    rook_attacks,
    square_name,
)


def test_square_coordinate_roundtrip() -> None:
    assert parse_square("a1") == 0
    assert parse_square("h1") == 7
    assert parse_square("e4") == 28
    assert parse_square("h8") == 63
    for sq in range(64):
        assert parse_square(square_name(sq)) == sq


def test_leaper_attack_masks_knight_and_king() -> None:
    # Knight on d4 (sq 27) attacks 8 squares; on a1 (sq 0) attacks b3 (17) and c2 (10)
    assert knight_attacks(27).bit_count() == 8
    assert knight_attacks(0) == (1 << parse_square("b3")) | (1 << parse_square("c2"))
    # King on e4 (28) attacks 8 squares; on h8 (63) attacks 3 squares
    assert king_attacks(28).bit_count() == 8
    assert king_attacks(63).bit_count() == 3


def test_sliding_ray_attacks_with_blockers() -> None:
    d4 = parse_square("d4")
    d7 = parse_square("d7")
    d8 = parse_square("d8")
    occ = 1 << d7
    r_att = rook_attacks(d4, occ)
    # d7 (blocker) is included, d8 (behind blocker) is masked out
    assert bool(r_att & (1 << d7)) is True
    assert bool(r_att & (1 << d8)) is False

    g7 = parse_square("g7")
    h8 = parse_square("h8")
    b_att = bishop_attacks(d4, 1 << g7)
    assert bool(b_att & (1 << g7)) is True
    assert bool(b_att & (1 << h8)) is False
    assert queen_attacks(d4, occ) == (rook_attacks(d4, occ) | bishop_attacks(d4, occ))


def test_fen_serialization_roundtrip() -> None:
    for fen in (STARTPOS_FEN, KIWIPETE_FEN):
        board = Board.from_fen(fen)
        assert board.to_fen() == fen


def test_perft_startpos_depths_1_to_3() -> None:
    board = Board.from_startpos()
    assert board.perft(1) == 20
    assert board.perft(2) == 400
    assert board.perft(3) == 8902


def test_perft_kiwipete_depths_1_and_2() -> None:
    board = Board.from_fen(KIWIPETE_FEN)
    assert board.perft(1) == 48
    assert board.perft(2) == 2039


def test_en_passant_and_promotion_legality() -> None:
    # White pawn on e5, Black just double-pushed d7->d5 (ep square d6)
    board = Board.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    ep_move = board.parse_uci_move("e5d6")
    assert ep_move.is_en_passant is True
    board.make_move(ep_move)
    assert board.piece_at_sq[parse_square("d5")] is None
    assert board.piece_at_sq[parse_square("d6")] == (WHITE, PAWN)
    board.unmake_move()
    assert board.to_fen() == "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"


def test_castling_blocked_when_passing_through_attacked_square() -> None:
    # Black rook on f8 attacks f1, so White cannot castle kingside (e1g1), only queenside (e1c1)
    board = Board.from_fen("4kr2/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    uci_moves = {m.to_uci() for m in board.generate_legal_moves()}
    assert "e1c1" in uci_moves
    assert "e1g1" not in uci_moves


def test_zobrist_incremental_hash_matches_full_recomputation() -> None:
    board = Board.from_fen(KIWIPETE_FEN)
    for move in board.generate_legal_moves()[:15]:
        board.make_move(move)
        assert board.zobrist_hash == board.compute_zobrist_hash()
        for reply in board.generate_legal_moves()[:8]:
            board.make_move(reply)
            assert board.zobrist_hash == board.compute_zobrist_hash()
            board.unmake_move()
        board.unmake_move()
        assert board.zobrist_hash == board.compute_zobrist_hash()


def test_zobrist_transposition_equivalence_across_move_orders() -> None:
    b1 = Board.from_startpos()
    for m in ("g1f3", "g8f6", "b1c3", "b8c6"):
        b1.push_uci(m)

    b2 = Board.from_startpos()
    for m in ("b1c3", "b8c6", "g1f3", "g8f6"):
        b2.push_uci(m)

    assert b1.to_fen() == b2.to_fen()
    assert b1.zobrist_hash == b2.zobrist_hash


def test_pesto_evaluation_symmetry_and_material_advantage() -> None:
    start_board = Board.from_startpos()
    assert evaluate(start_board) == 0

    # Remove Black's queen -> White has huge positive evaluation
    up_queen = Board.from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert evaluate(up_queen) > 900


def test_mvv_lva_orders_pawn_captures_queen_above_queen_captures_pawn() -> None:
    # White pawn on e4 can capture Black queen on d5; White queen on d1 can capture Black pawn on d4
    board = Board.from_fen("4k3/8/8/3q4/3pP3/8/8/3QK3 w - - 0 1")
    pxq = board.parse_uci_move("e4d5")
    qxp = board.parse_uci_move("d1d4")
    quiet = board.parse_uci_move("e1f2")
    assert mvv_lva_score(board, pxq) > mvv_lva_score(board, qxp) > mvv_lva_score(board, quiet)

    ordered = order_moves(board, [quiet, qxp, pxq])
    assert ordered[0] == pxq
    assert ordered[1] == qxp


def test_search_finds_mate_in_one_back_rank() -> None:
    # White rook on a1 delivers back-rank mate Ra1-a8#
    board = Board.from_fen("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    searcher = Searcher()
    res = searcher.search(board, max_depth=2)
    assert res.best_move is not None
    assert res.best_move.to_uci() == "a1a8"
    assert res.score >= MATE_SCORE - 5


def test_search_finds_mate_in_two_tactical_puzzle() -> None:
    # Forced Mate in 2: 1. Rb8+ Nxb8 2. Rxb8# (Na6 only defends b8 and cannot interpose on c8..f8)
    board = Board.from_fen("6k1/5ppp/n7/8/8/8/1R6/1R4K1 w - - 0 1")
    searcher = Searcher()
    res = searcher.search(board, max_depth=3)
    assert res.best_move is not None
    assert res.best_move.to_uci() == "b2b8"
    assert res.score >= MATE_SCORE - 5


def test_transposition_table_stores_and_probes_entries() -> None:
    board = Board.from_startpos()
    searcher = Searcher()
    res = searcher.search(board, max_depth=3)
    entry = searcher.tt.probe(board.zobrist_hash)
    assert entry is not None
    assert entry.depth == 3
    assert entry.flag == TTFlag.EXACT
    assert entry.best_move == res.best_move


def test_uci_protocol_command_lifecycle() -> None:
    engine = UCIEngine()
    uci_out = engine.handle_command("uci")
    assert any("Abdullah Azmi" in line for line in uci_out)
    assert uci_out[-1] == "uciok"

    assert engine.handle_command("isready") == ["readyok"]
    engine.handle_command("position startpos moves e2e4 e7e5")
    go_out = engine.handle_command("go depth 2")
    assert len(go_out) == 2
    assert go_out[0].startswith("info depth 2")
    assert go_out[1].startswith("bestmove ")

    perft_out = engine.handle_command("perft 2")
    assert perft_out == ["Nodes searched: 835"]

    engine.handle_command("quit")
    assert engine.should_quit is True
