# bitboard-uci-chess

A 64-bit bitboard chess engine and Universal Chess Interface (**UCI**) driver written in pure Python, born out of rapid and blitz games at the Marshall Chess Club and Washington Square Park in New York City.

## Architecture & Engine Features

1. **64-Bit Bitboard Representation (`board.py`)**
   - Twelve piece bitboards (`2 colors × 6 piece types`) plus color occupancy and aggregate occupancy masks over `a1=0 .. h8=63` (Little-Endian Rank-File Mapping, LERF).
   - **Precomputed Attack Masks:** O(1) lookup tables for Knights (`KNIGHT_ATTACKS[64]`), Kings (`KING_ATTACKS[64]`), and Pawns (`PAWN_ATTACKS[2][64]`).
   - **Classical Ray Sliding Attacks:** Precomputed directional ray masks (`RAY_ATTACKS[8][64]`) using LSB/MSB bit-scanning (`int.bit_length()`) to resolve blocked rook, bishop, and queen rays in $O(1)$ per ray without square-by-square loops:
     $$\text{Attacks}(s, \text{occ}) = \text{Ray}_d(s) \oplus \text{Ray}_d(\text{first\_blocker}(s, \text{occ}))$$
   - **Incremental 64-Bit Zobrist Hashing:** Deterministic PRNG-seeded Zobrist keys across piece-squares, side-to-move, 16 castling-right states, and en-passant files, updated incrementally during `make_move` / `unmake_move`.

2. **Search & Tactical Engine (`search.py`)**
   - **PeSTO Tapered Evaluation:** Middlegame and endgame piece-square tables interpolated continuously by non-pawn material phase ($P \in [0, 24]$):
     $$E_{\text{tapered}} = \frac{P \cdot E_{\text{MG}} + (24 - P) \cdot E_{\text{EG}}}{24}$$
   - **Negamax Alpha-Beta with Iterative Deepening:** Principal Variation (PV) tracking and ply-adjusted checkmate distance scoring (`MATE_SCORE - ply`) to guarantee shortest-mate selection.
   - **Transposition Table (TT):** Zobrist-indexed cache storing `EXACT`, `LOWERBOUND`, and `UPPERBOUND` bounds alongside the best tactical move for instant hash-move ordering.
   - **MVV-LVA & Quiescence Search:** Most Valuable Victim / Least Valuable Attacker capture prioritization plus capture-only quiescence search at horizon nodes to eliminate the horizon effect.

3. **UCI Protocol Driver (`uci.py`)**
   - Full support for `uci`, `isready`, `ucinewgame`, `position startpos | fen ... [moves ...]`, `go depth | movetime | wtime/btime`, `perft`, `d`, and `quit`.

## Quickstart

```python
from bitboard_chess import Board, Searcher, UCIEngine

# 1. Verify exact Perft move-tree enumeration from initial position
board = Board.from_startpos()
print("Perft(1):", board.perft(1))  # 20
print("Perft(2):", board.perft(2))  # 400
print("Perft(3):", board.perft(3))  # 8902

# 2. Solve a tactical Mate-in-2 puzzle
puzzle = Board.from_fen("r2qk2r/pb4pp/1n2Pb2/2B2Q2/p1p5/2P5/2B2PPP/RN2R1K1 w - - 1 0")
searcher = Searcher()
result = searcher.search(puzzle, max_depth=4)
print(f"Best move: {result.best_move.to_uci()} (score: {result.score})")
```

## Running Tests

```bash
pytest
```

## License

MIT © [Abdullah Azmi](https://abdullahazmi.com)


## CLI Diagnostics & Invariant Verification

Run the built-in deterministic diagnostic probe and invariant self-check directly via the module CLI:

```bash
PYTHONPATH=src python -m bitboard_chess --self-test --json
```

Run the complete regression test suite:

```bash
PYTHONPATH=src pytest -q
```
