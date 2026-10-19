"""Universal Chess Interface (UCI) protocol parser and command driver."""

from __future__ import annotations

import sys

from .board import STARTPOS_FEN, Board
from .search import MATE_SCORE, Searcher


class UCIEngine:
    """Stateful UCI protocol handler wrapping `Board` and `Searcher`."""

    ENGINE_NAME = "BitboardChess 0.2.0"
    ENGINE_AUTHOR = "Abdullah Azmi"

    def __init__(self) -> None:
        self.board = Board.from_startpos()
        self.searcher = Searcher()
        self.should_quit = False

    def handle_command(self, raw_line: str) -> list[str]:
        """Process a single UCI command string and return protocol output lines."""
        line = raw_line.strip()
        if not line:
            return []

        tokens = line.split()
        cmd = tokens[0]

        if cmd == "uci":
            return [
                f"id name {self.ENGINE_NAME}",
                f"id author {self.ENGINE_AUTHOR}",
                "option name Hash type spin default 64 min 1 max 1024",
                "uciok",
            ]

        if cmd == "isready":
            return ["readyok"]

        if cmd == "ucinewgame":
            self.board = Board.from_startpos()
            self.searcher.tt.clear()
            return []

        if cmd == "position":
            self._handle_position(tokens[1:])
            return []

        if cmd == "go":
            return self._handle_go(tokens[1:])

        if cmd == "perft":
            depth = int(tokens[1]) if len(tokens) > 1 else 1
            nodes = self.board.perft(depth)
            return [f"Nodes searched: {nodes}"]

        if cmd == "d":
            return [
                f"Fen: {self.board.to_fen()}",
                f"Key: {self.board.zobrist_hash:016x}",
            ]

        if cmd == "quit":
            self.should_quit = True
            return []

        return []

    def _handle_position(self, args: list[str]) -> None:
        if not args:
            return

        moves_idx: int | None = None
        if "moves" in args:
            moves_idx = args.index("moves")

        pos_spec = args[:moves_idx] if moves_idx is not None else args
        move_tokens = args[moves_idx + 1 :] if moves_idx is not None else []

        if pos_spec and pos_spec[0] == "startpos":
            self.board = Board.from_startpos()
        elif pos_spec and pos_spec[0] == "fen":
            fen_str = " ".join(pos_spec[1:])
            self.board = Board.from_fen(fen_str if fen_str else STARTPOS_FEN)

        for mv_str in move_tokens:
            self.board.push_uci(mv_str)

    def _handle_go(self, args: list[str]) -> list[str]:
        depth = 3
        if "depth" in args:
            idx = args.index("depth")
            if idx + 1 < len(args):
                depth = max(1, int(args[idx + 1]))
        elif "movetime" in args:
            depth = 3

        result = self.searcher.search(self.board, max_depth=depth)
        pv_str = " ".join(m.to_uci() for m in result.pv)
        if abs(result.score) >= MATE_SCORE - 100:
            plies_to_mate = MATE_SCORE - abs(result.score)
            mate_moves = max(1, (plies_to_mate + 1) // 2)
            if result.score < 0:
                mate_moves = -mate_moves
            score_str = f"mate {mate_moves}"
        else:
            score_str = f"cp {result.score}"

        best_str = result.best_move.to_uci() if result.best_move is not None else "0000"
        return [
            f"info depth {result.depth} score {score_str} nodes {result.nodes} pv {pv_str}".strip(),
            f"bestmove {best_str}",
        ]


def main() -> None:
    """Run interactive UCI stdio loop."""
    engine = UCIEngine()
    for line in sys.stdin:
        for out in engine.handle_command(line):
            print(out, flush=True)
        if engine.should_quit:
            break


if __name__ == "__main__":
    main()
