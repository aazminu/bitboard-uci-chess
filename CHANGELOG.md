# Changelog — `bitboard-uci-chess`

All notable engineering changes to `bitboard-uci-chess` (`bitboard_chess`) are documented in this file.

## [0.3.0] — 2026-10-24
### Added
- Documented CLI diagnostic usage (`python -m bitboard_chess --self-test --json`) and deterministic invariant verification workflows in `README.md`.
- Added release history and semantic versioning audit trail in `CHANGELOG.md`.

## [0.2.1] — 2026-10-23
### Added
- Added automated CLI and runtime invariant regression suite in `tests/test_diagnostics_cli.py` covering seed determinism, negative-seed rejection, JSON output schema, and `--self-test` exit codes.

## [0.2.0] — 2026-10-21
### Added
- Added `src/bitboard_chess/diagnostics.py` (`run_diagnostic_suite`, `validate_runtime_invariants`) and module CLI runner `src/bitboard_chess/__main__.py`.
- Exported diagnostic helpers in `bitboard_chess.__all__`.

## [0.1.0] — 2026-10-19
### Added
- Initial architecture and core test suite: 64-bit bitboard chess move generator and UCI engine with Zobrist transposition tables, Negamax alpha-beta search, MVV-LVA ordering, and PeSTO tapered evaluation.
