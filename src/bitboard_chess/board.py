"""64-bit bitboard representation, precomputed ray/leaper attack tables, Zobrist hashing, and legal move generator."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterator

U64_MASK = 0xFFFFFFFFFFFFFFFF

WHITE = 0
BLACK = 1

PAWN = 0
KNIGHT = 1
BISHOP = 2
ROOK = 3
QUEEN = 4
KING = 5

CASTLE_WK = 1
CASTLE_WQ = 2
CASTLE_BK = 4
CASTLE_BQ = 8

STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE_FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

PIECE_CHARS = {
    (WHITE, PAWN): "P",
    (WHITE, KNIGHT): "N",
    (WHITE, BISHOP): "B",
    (WHITE, ROOK): "R",
    (WHITE, QUEEN): "Q",
    (WHITE, KING): "K",
    (BLACK, PAWN): "p",
    (BLACK, KNIGHT): "n",
    (BLACK, BISHOP): "b",
    (BLACK, ROOK): "r",
    (BLACK, QUEEN): "q",
    (BLACK, KING): "k",
}
CHAR_TO_PIECE = {v: k for k, v in PIECE_CHARS.items()}
PROMO_TO_CHAR = {KNIGHT: "n", BISHOP: "b", ROOK: "r", QUEEN: "q"}
CHAR_TO_PROMO = {v: k for k, v in PROMO_TO_CHAR.items()}

# Ray directions: 0..3 positive (N, E, NE, NW), 4..7 negative (S, W, SW, SE)
DIR_N = 0
DIR_E = 1
DIR_NE = 2
DIR_NW = 3
DIR_S = 4
DIR_W = 5
DIR_SW = 6
DIR_SE = 7

KNIGHT_ATTACKS: list[int] = [0] * 64
KING_ATTACKS: list[int] = [0] * 64
PAWN_ATTACKS: list[list[int]] = [[0] * 64, [0] * 64]
RAY_ATTACKS: list[list[int]] = [[0] * 64 for _ in range(8)]


def _init_attack_tables() -> None:
    knight_offsets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    king_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    ray_deltas = [
        (DIR_N, 1, 0),
        (DIR_E, 0, 1),
        (DIR_NE, 1, 1),
        (DIR_NW, 1, -1),
        (DIR_S, -1, 0),
        (DIR_W, 0, -1),
        (DIR_SW, -1, -1),
        (DIR_SE, -1, 1),
    ]

    for sq in range(64):
        r, f = divmod(sq, 8)
        kmask = 0
        for dr, df in knight_offsets:
            nr, nf = r + dr, f + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                kmask |= 1 << (nr * 8 + nf)
        KNIGHT_ATTACKS[sq] = kmask

        gmask = 0
        for dr, df in king_offsets:
            nr, nf = r + dr, f + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                gmask |= 1 << (nr * 8 + nf)
        KING_ATTACKS[sq] = gmask

        # White pawn attacks (+1 rank)
        wpmask = 0
        if r + 1 < 8:
            if f - 1 >= 0:
                wpmask |= 1 << ((r + 1) * 8 + (f - 1))
            if f + 1 < 8:
                wpmask |= 1 << ((r + 1) * 8 + (f + 1))
        PAWN_ATTACKS[WHITE][sq] = wpmask

        # Black pawn attacks (-1 rank)
        bpmask = 0
        if r - 1 >= 0:
            if f - 1 >= 0:
                bpmask |= 1 << ((r - 1) * 8 + (f - 1))
            if f + 1 < 8:
                bpmask |= 1 << ((r - 1) * 8 + (f + 1))
        PAWN_ATTACKS[BLACK][sq] = bpmask

        for d_idx, dr, df in ray_deltas:
            rmask = 0
            nr, nf = r + dr, f + df
            while 0 <= nr < 8 and 0 <= nf < 8:
                rmask |= 1 << (nr * 8 + nf)
                nr += dr
                nf += df
            RAY_ATTACKS[d_idx][sq] = rmask


_init_attack_tables()


# Deterministic 64-bit Zobrist tables
_rng = random.Random(0x4D41525348414C4C)  # "MARSHALL" seed
ZOBRIST_PIECE: list[list[list[int]]] = [
    [[_rng.getrandbits(64) for _ in range(64)] for _ in range(6)] for _ in range(2)
]
ZOBRIST_SIDE: int = _rng.getrandbits(64)
ZOBRIST_CASTLE: list[int] = [_rng.getrandbits(64) for _ in range(16)]
ZOBRIST_EP_FILE: list[int] = [_rng.getrandbits(64) for _ in range(8)]

# Castling rights mask applied on every move for `from_sq` and `to_sq`
CASTLE_RIGHTS_MASK: list[int] = [15] * 64
CASTLE_RIGHTS_MASK[0] &= ~CASTLE_WQ
CASTLE_RIGHTS_MASK[4] &= ~(CASTLE_WK | CASTLE_WQ)
CASTLE_RIGHTS_MASK[7] &= ~CASTLE_WK
CASTLE_RIGHTS_MASK[56] &= ~CASTLE_BQ
CASTLE_RIGHTS_MASK[60] &= ~(CASTLE_BK | CASTLE_BQ)
CASTLE_RIGHTS_MASK[63] &= ~CASTLE_BK


def square_name(sq: int) -> str:
    """Convert 0..63 square index to algebraic coordinate ('a1'..'h8')."""
    r, f = divmod(sq, 8)
    return f"{chr(ord('a') + f)}{r + 1}"


def parse_square(coord: str) -> int:
    """Convert algebraic coordinate ('a1'..'h8') to 0..63 square index."""
    if len(coord) != 2:
        raise ValueError(f"Invalid square coordinate: {coord!r}")
    f = ord(coord[0].lower()) - ord("a")
    r = int(coord[1]) - 1
    if not (0 <= r < 8 and 0 <= f < 8):
        raise ValueError(f"Square out of range: {coord!r}")
    return r * 8 + f


def knight_attacks(sq: int) -> int:
    """Return 64-bit attack mask for a knight on `sq`."""
    return KNIGHT_ATTACKS[sq]


def king_attacks(sq: int) -> int:
    """Return 64-bit attack mask for a king on `sq`."""
    return KING_ATTACKS[sq]


def _positive_ray_attacks(d_idx: int, sq: int, occupied: int) -> int:
    attacks = RAY_ATTACKS[d_idx][sq]
    blockers = attacks & occupied
    if blockers:
        first_blocker = (blockers & -blockers).bit_length() - 1
        attacks ^= RAY_ATTACKS[d_idx][first_blocker]
    return attacks


def _negative_ray_attacks(d_idx: int, sq: int, occupied: int) -> int:
    attacks = RAY_ATTACKS[d_idx][sq]
    blockers = attacks & occupied
    if blockers:
        first_blocker = blockers.bit_length() - 1
        attacks ^= RAY_ATTACKS[d_idx][first_blocker]
    return attacks


def rook_attacks(sq: int, occupied: int) -> int:
    """Compute sliding rook attacks from `sq` given board occupancy bitboard."""
    return (
        _positive_ray_attacks(DIR_N, sq, occupied)
        | _positive_ray_attacks(DIR_E, sq, occupied)
        | _negative_ray_attacks(DIR_S, sq, occupied)
        | _negative_ray_attacks(DIR_W, sq, occupied)
    )


def bishop_attacks(sq: int, occupied: int) -> int:
    """Compute sliding bishop attacks from `sq` given board occupancy bitboard."""
    return (
        _positive_ray_attacks(DIR_NE, sq, occupied)
        | _positive_ray_attacks(DIR_NW, sq, occupied)
        | _negative_ray_attacks(DIR_SW, sq, occupied)
        | _negative_ray_attacks(DIR_SE, sq, occupied)
    )


def queen_attacks(sq: int, occupied: int) -> int:
    """Compute sliding queen attacks from `sq` given board occupancy bitboard."""
    return rook_attacks(sq, occupied) | bishop_attacks(sq, occupied)


def _iter_bits(bb: int) -> Iterator[int]:
    """Yield set bit indices from least significant to most significant."""
    while bb:
        lsb = bb & -bb
        yield lsb.bit_length() - 1
        bb ^= lsb


@dataclass(frozen=True)
class Move:
    """Chess move representation supporting promotions, en passant, and castling."""

    from_sq: int
    to_sq: int
    promotion: int | None = None
    is_en_passant: bool = False
    is_castling: bool = False

    def to_uci(self) -> str:
        promo_str = PROMO_TO_CHAR[self.promotion] if self.promotion is not None else ""
        return f"{square_name(self.from_sq)}{square_name(self.to_sq)}{promo_str}"


@dataclass(frozen=True)
class _UndoState:
    move: Move
    moved_piece: int
    captured_piece: int | None
    captured_sq: int
    castling_rights: int
    ep_square: int | None
    halfmove_clock: int
    zobrist_hash: int


class Board:
    """64-bit bitboard chess position with incremental Zobrist hashing and full move legality."""

    def __init__(self) -> None:
        self.pieces: list[list[int]] = [[0] * 6 for _ in range(2)]
        self.occupied_co: list[int] = [0, 0]
        self.occupied: int = 0
        self.piece_at_sq: list[tuple[int, int] | None] = [None] * 64
        self.turn: int = WHITE
        self.castling_rights: int = 0
        self.ep_square: int | None = None
        self.halfmove_clock: int = 0
        self.fullmove_number: int = 1
        self.zobrist_hash: int = 0
        self._history: list[_UndoState] = []

    @classmethod
    def from_startpos(cls) -> Board:
        """Create a board initialized to the standard chess starting position."""
        return cls.from_fen(STARTPOS_FEN)

    @classmethod
    def from_fen(cls, fen: str) -> Board:
        """Parse a Forsyth-Edwards Notation (FEN) string into a `Board`."""
        board = cls()
        parts = fen.strip().split()
        if len(parts) < 4:
            raise ValueError(f"Invalid FEN string: {fen!r}")

        ranks = parts[0].split("/")
        if len(ranks) != 8:
            raise ValueError(f"FEN must have 8 ranks: {fen!r}")

        for r_idx, rank_str in enumerate(reversed(ranks)):
            f_idx = 0
            for ch in rank_str:
                if ch.isdigit():
                    f_idx += int(ch)
                elif ch in CHAR_TO_PIECE:
                    color, ptype = CHAR_TO_PIECE[ch]
                    sq = r_idx * 8 + f_idx
                    board._put_piece(color, ptype, sq)
                    f_idx += 1
                else:
                    raise ValueError(f"Unexpected FEN character {ch!r} in {fen!r}")

        board.turn = WHITE if parts[1] == "w" else BLACK

        c_str = parts[2]
        c_rights = 0
        if "K" in c_str:
            c_rights |= CASTLE_WK
        if "Q" in c_str:
            c_rights |= CASTLE_WQ
        if "k" in c_str:
            c_rights |= CASTLE_BK
        if "q" in c_str:
            c_rights |= CASTLE_BQ
        board.castling_rights = c_rights

        ep_str = parts[3]
        board.ep_square = None if ep_str == "-" else parse_square(ep_str)
        board.halfmove_clock = int(parts[4]) if len(parts) > 4 else 0
        board.fullmove_number = int(parts[5]) if len(parts) > 5 else 1
        board.zobrist_hash = board.compute_zobrist_hash()
        return board

    def to_fen(self) -> str:
        """Serialize the current board position to standard FEN."""
        rank_strs: list[str] = []
        for r in range(7, -1, -1):
            empty = 0
            chars: list[str] = []
            for f in range(8):
                sq = r * 8 + f
                info = self.piece_at_sq[sq]
                if info is None:
                    empty += 1
                else:
                    if empty:
                        chars.append(str(empty))
                        empty = 0
                    chars.append(PIECE_CHARS[info])
            if empty:
                chars.append(str(empty))
            rank_strs.append("".join(chars))

        turn_str = "w" if self.turn == WHITE else "b"
        c_chars: list[str] = []
        if self.castling_rights & CASTLE_WK:
            c_chars.append("K")
        if self.castling_rights & CASTLE_WQ:
            c_chars.append("Q")
        if self.castling_rights & CASTLE_BK:
            c_chars.append("k")
        if self.castling_rights & CASTLE_BQ:
            c_chars.append("q")
        castle_str = "".join(c_chars) if c_chars else "-"
        ep_str = square_name(self.ep_square) if self.ep_square is not None else "-"
        return f"{'/'.join(rank_strs)} {turn_str} {castle_str} {ep_str} {self.halfmove_clock} {self.fullmove_number}"

    def _put_piece(self, color: int, ptype: int, sq: int) -> None:
        bit = 1 << sq
        self.pieces[color][ptype] |= bit
        self.occupied_co[color] |= bit
        self.occupied |= bit
        self.piece_at_sq[sq] = (color, ptype)

    def _remove_piece(self, color: int, ptype: int, sq: int) -> None:
        bit = ~(1 << sq) & U64_MASK
        self.pieces[color][ptype] &= bit
        self.occupied_co[color] &= bit
        self.occupied &= bit
        self.piece_at_sq[sq] = None

    def compute_zobrist_hash(self) -> int:
        """Compute full 64-bit Zobrist hash from scratch."""
        h = 0
        for sq, info in enumerate(self.piece_at_sq):
            if info is not None:
                c, p = info
                h ^= ZOBRIST_PIECE[c][p][sq]
        if self.turn == BLACK:
            h ^= ZOBRIST_SIDE
        h ^= ZOBRIST_CASTLE[self.castling_rights]
        if self.ep_square is not None:
            h ^= ZOBRIST_EP_FILE[self.ep_square % 8]
        return h

    def king_square(self, color: int) -> int:
        """Return the square index (0..63) of `color`'s king."""
        kbb = self.pieces[color][KING]
        if not kbb:
            raise RuntimeError(f"King missing for color {color}")
        return (kbb & -kbb).bit_length() - 1

    def is_square_attacked(self, sq: int, by_color: int) -> bool:
        """Return True if square `sq` is attacked by any piece of `by_color`."""
        p_pieces = self.pieces[by_color]
        # Attacked by pawns of `by_color` means a pawn of `1 - by_color` on `sq` would attack them
        if PAWN_ATTACKS[1 - by_color][sq] & p_pieces[PAWN]:
            return True
        if KNIGHT_ATTACKS[sq] & p_pieces[KNIGHT]:
            return True
        if KING_ATTACKS[sq] & p_pieces[KING]:
            return True
        if rook_attacks(sq, self.occupied) & (p_pieces[ROOK] | p_pieces[QUEEN]):
            return True
        if bishop_attacks(sq, self.occupied) & (p_pieces[BISHOP] | p_pieces[QUEEN]):
            return True
        return False

    def in_check(self, color: int | None = None) -> bool:
        """Return True if `color` (defaults to side to move) has its king in check."""
        c = self.turn if color is None else color
        return self.is_square_attacked(self.king_square(c), 1 - c)

    def make_move(self, move: Move) -> None:
        """Execute `move` on the board and update incremental Zobrist hash."""
        us = self.turn
        them = 1 - us
        info = self.piece_at_sq[move.from_sq]
        if info is None:
            raise ValueError(f"No piece on source square {square_name(move.from_sq)}")
        _, moved_ptype = info

        captured_ptype: int | None = None
        captured_sq = move.to_sq

        if move.is_en_passant:
            captured_sq = move.to_sq - 8 if us == WHITE else move.to_sq + 8
            captured_ptype = PAWN
        else:
            dest_info = self.piece_at_sq[move.to_sq]
            if dest_info is not None:
                captured_ptype = dest_info[1]

        self._history.append(
            _UndoState(
                move=move,
                moved_piece=moved_ptype,
                captured_piece=captured_ptype,
                captured_sq=captured_sq,
                castling_rights=self.castling_rights,
                ep_square=self.ep_square,
                halfmove_clock=self.halfmove_clock,
                zobrist_hash=self.zobrist_hash,
            )
        )

        h = self.zobrist_hash
        if self.ep_square is not None:
            h ^= ZOBRIST_EP_FILE[self.ep_square % 8]
        h ^= ZOBRIST_CASTLE[self.castling_rights]

        # Remove moving piece from source square
        self._remove_piece(us, moved_ptype, move.from_sq)
        h ^= ZOBRIST_PIECE[us][moved_ptype][move.from_sq]

        # Remove captured piece if any
        if captured_ptype is not None:
            self._remove_piece(them, captured_ptype, captured_sq)
            h ^= ZOBRIST_PIECE[them][captured_ptype][captured_sq]

        # Place moving piece (or promoted piece) on destination square
        placed_ptype = move.promotion if move.promotion is not None else moved_ptype
        self._put_piece(us, placed_ptype, move.to_sq)
        h ^= ZOBRIST_PIECE[us][placed_ptype][move.to_sq]

        # Handle rook movement during castling
        if move.is_castling:
            if move.to_sq == 6:  # White O-O
                self._remove_piece(WHITE, ROOK, 7)
                self._put_piece(WHITE, ROOK, 5)
                h ^= ZOBRIST_PIECE[WHITE][ROOK][7] ^ ZOBRIST_PIECE[WHITE][ROOK][5]
            elif move.to_sq == 2:  # White O-O-O
                self._remove_piece(WHITE, ROOK, 0)
                self._put_piece(WHITE, ROOK, 3)
                h ^= ZOBRIST_PIECE[WHITE][ROOK][0] ^ ZOBRIST_PIECE[WHITE][ROOK][3]
            elif move.to_sq == 62:  # Black O-O
                self._remove_piece(BLACK, ROOK, 63)
                self._put_piece(BLACK, ROOK, 61)
                h ^= ZOBRIST_PIECE[BLACK][ROOK][63] ^ ZOBRIST_PIECE[BLACK][ROOK][61]
            elif move.to_sq == 58:  # Black O-O-O
                self._remove_piece(BLACK, ROOK, 56)
                self._put_piece(BLACK, ROOK, 59)
                h ^= ZOBRIST_PIECE[BLACK][ROOK][56] ^ ZOBRIST_PIECE[BLACK][ROOK][59]

        # Update castling rights
        self.castling_rights &= CASTLE_RIGHTS_MASK[move.from_sq] & CASTLE_RIGHTS_MASK[move.to_sq]
        h ^= ZOBRIST_CASTLE[self.castling_rights]

        # Update en passant square
        self.ep_square = None
        if moved_ptype == PAWN and abs(move.to_sq - move.from_sq) == 16:
            self.ep_square = (move.from_sq + move.to_sq) // 2
            h ^= ZOBRIST_EP_FILE[self.ep_square % 8]

        # Update clocks and side to move
        if moved_ptype == PAWN or captured_ptype is not None:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        if us == BLACK:
            self.fullmove_number += 1

        self.turn = them
        h ^= ZOBRIST_SIDE
        self.zobrist_hash = h

    def unmake_move(self) -> Move:
        """Revert the most recent move and restore board state."""
        state = self._history.pop()
        move = state.move
        them = self.turn
        us = 1 - them

        placed_ptype = move.promotion if move.promotion is not None else state.moved_piece
        self._remove_piece(us, placed_ptype, move.to_sq)
        self._put_piece(us, state.moved_piece, move.from_sq)

        if state.captured_piece is not None:
            self._put_piece(them, state.captured_piece, state.captured_sq)

        if move.is_castling:
            if move.to_sq == 6:
                self._remove_piece(WHITE, ROOK, 5)
                self._put_piece(WHITE, ROOK, 7)
            elif move.to_sq == 2:
                self._remove_piece(WHITE, ROOK, 3)
                self._put_piece(WHITE, ROOK, 0)
            elif move.to_sq == 62:
                self._remove_piece(BLACK, ROOK, 61)
                self._put_piece(BLACK, ROOK, 63)
            elif move.to_sq == 58:
                self._remove_piece(BLACK, ROOK, 59)
                self._put_piece(BLACK, ROOK, 56)

        self.turn = us
        if us == BLACK:
            self.fullmove_number -= 1
        self.castling_rights = state.castling_rights
        self.ep_square = state.ep_square
        self.halfmove_clock = state.halfmove_clock
        self.zobrist_hash = state.zobrist_hash
        return move

    def generate_pseudo_legal_moves(self) -> list[Move]:
        """Generate all pseudo-legal moves for the active side."""
        moves: list[Move] = []
        us = self.turn
        them = 1 - us
        our_occ = self.occupied_co[us]
        their_occ = self.occupied_co[them]
        all_occ = self.occupied
        not_ours = ~our_occ & U64_MASK

        # 1. Pawn moves
        pawns = self.pieces[us][PAWN]
        step = 8 if us == WHITE else -8
        start_rank = 1 if us == WHITE else 6
        promo_rank = 7 if us == WHITE else 0

        for from_sq in _iter_bits(pawns):
            r = from_sq // 8
            to_sq = from_sq + step
            if 0 <= to_sq < 64 and not (all_occ & (1 << to_sq)):
                if to_sq // 8 == promo_rank:
                    for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                        moves.append(Move(from_sq, to_sq, promotion=promo))
                else:
                    moves.append(Move(from_sq, to_sq))
                    if r == start_rank:
                        double_sq = from_sq + 2 * step
                        if not (all_occ & (1 << double_sq)):
                            moves.append(Move(from_sq, double_sq))

            # Pawn captures
            cap_targets = PAWN_ATTACKS[us][from_sq] & their_occ
            for cap_sq in _iter_bits(cap_targets):
                if cap_sq // 8 == promo_rank:
                    for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                        moves.append(Move(from_sq, cap_sq, promotion=promo))
                else:
                    moves.append(Move(from_sq, cap_sq))

            # En passant capture
            if self.ep_square is not None and (PAWN_ATTACKS[us][from_sq] & (1 << self.ep_square)):
                moves.append(Move(from_sq, self.ep_square, is_en_passant=True))

        # 2. Knight moves
        for from_sq in _iter_bits(self.pieces[us][KNIGHT]):
            targets = KNIGHT_ATTACKS[from_sq] & not_ours
            for to_sq in _iter_bits(targets):
                moves.append(Move(from_sq, to_sq))

        # 3. Bishop moves
        for from_sq in _iter_bits(self.pieces[us][BISHOP]):
            targets = bishop_attacks(from_sq, all_occ) & not_ours
            for to_sq in _iter_bits(targets):
                moves.append(Move(from_sq, to_sq))

        # 4. Rook moves
        for from_sq in _iter_bits(self.pieces[us][ROOK]):
            targets = rook_attacks(from_sq, all_occ) & not_ours
            for to_sq in _iter_bits(targets):
                moves.append(Move(from_sq, to_sq))

        # 5. Queen moves
        for from_sq in _iter_bits(self.pieces[us][QUEEN]):
            targets = queen_attacks(from_sq, all_occ) & not_ours
            for to_sq in _iter_bits(targets):
                moves.append(Move(from_sq, to_sq))

        # 6. King moves & Castling
        for from_sq in _iter_bits(self.pieces[us][KING]):
            targets = KING_ATTACKS[from_sq] & not_ours
            for to_sq in _iter_bits(targets):
                moves.append(Move(from_sq, to_sq))

            if us == WHITE and from_sq == 4:
                if (
                    (self.castling_rights & CASTLE_WK)
                    and not (all_occ & ((1 << 5) | (1 << 6)))
                    and not self.is_square_attacked(4, BLACK)
                    and not self.is_square_attacked(5, BLACK)
                    and not self.is_square_attacked(6, BLACK)
                ):
                    moves.append(Move(4, 6, is_castling=True))
                if (
                    (self.castling_rights & CASTLE_WQ)
                    and not (all_occ & ((1 << 1) | (1 << 2) | (1 << 3)))
                    and not self.is_square_attacked(4, BLACK)
                    and not self.is_square_attacked(3, BLACK)
                    and not self.is_square_attacked(2, BLACK)
                ):
                    moves.append(Move(4, 2, is_castling=True))
            elif us == BLACK and from_sq == 60:
                if (
                    (self.castling_rights & CASTLE_BK)
                    and not (all_occ & ((1 << 61) | (1 << 62)))
                    and not self.is_square_attacked(60, WHITE)
                    and not self.is_square_attacked(61, WHITE)
                    and not self.is_square_attacked(62, WHITE)
                ):
                    moves.append(Move(60, 62, is_castling=True))
                if (
                    (self.castling_rights & CASTLE_BQ)
                    and not (all_occ & ((1 << 57) | (1 << 58) | (1 << 59)))
                    and not self.is_square_attacked(60, WHITE)
                    and not self.is_square_attacked(59, WHITE)
                    and not self.is_square_attacked(58, WHITE)
                ):
                    moves.append(Move(60, 58, is_castling=True))

        return moves

    def generate_legal_moves(self) -> list[Move]:
        """Generate all strictly legal moves for the active side."""
        us = self.turn
        legal: list[Move] = []
        for move in self.generate_pseudo_legal_moves():
            self.make_move(move)
            if not self.is_square_attacked(self.king_square(us), 1 - us):
                legal.append(move)
            self.unmake_move()
        return legal

    def generate_legal_captures(self) -> list[Move]:
        """Generate all strictly legal capture and promotion moves for Quiescence search."""
        us = self.turn
        captures: list[Move] = []
        for move in self.generate_pseudo_legal_moves():
            is_cap = move.is_en_passant or (self.piece_at_sq[move.to_sq] is not None) or (move.promotion is not None)
            if not is_cap:
                continue
            self.make_move(move)
            if not self.is_square_attacked(self.king_square(us), 1 - us):
                captures.append(move)
            self.unmake_move()
        return captures

    def parse_uci_move(self, uci_str: str) -> Move:
        """Match a UCI coordinate move string (e.g. 'e2e4', 'e7e8q') against legal moves."""
        norm = uci_str.strip().lower()
        for move in self.generate_legal_moves():
            if move.to_uci() == norm:
                return move
        raise ValueError(f"Illegal or unrecognized UCI move {uci_str!r} in position {self.to_fen()}")

    def push_uci(self, uci_str: str) -> Move:
        """Parse and execute a UCI move string on the board."""
        move = self.parse_uci_move(uci_str)
        self.make_move(move)
        return move

    def perft(self, depth: int) -> int:
        """Count total leaf nodes at `depth` plies from current position."""
        if depth <= 0:
            return 1
        legal_moves = self.generate_legal_moves()
        if depth == 1:
            return len(legal_moves)
        nodes = 0
        for move in legal_moves:
            self.make_move(move)
            nodes += self.perft(depth - 1)
            self.unmake_move()
        return nodes
