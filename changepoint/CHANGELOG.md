# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
This project follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- …
- …

### Changed
- …
- …

### Deprecated
- …

### Removed
- …

### Fixed
- …

---

## [0.15.0] - 2025-10-03
### Changed

- Updated dependencies including rv, rand, and nalgebra.
- Updated to Rust 2024 Edition.

## [0.14.2] - 2024-07-14
### Added
- Add `serde` support (optional) for internal `RunLengthDist` struct.  
- Introduce `Bocpd::with_capacity(cap: usize)` for pre‑allocation.

### Changed
- Update version in `Cargo.toml` to `0.14.2`.  
- Adjust internal use of `rv` crate to latest version compatibility.  
- Improve error messages when passed empty input slices.

### Fixed
- Resolve overflow in computation of hazard function at extreme `t`.  
- Fix lifetime issue in `Argpcp` with kernel parameter references.

---

## [0.14.1] - 2023-08-08
### Changed
- Bump minor version from 0.14.0 → 0.14.1.  
- Update documentation comments in `lib.rs` and example code.  
- Minor refactoring: move `map_changepoints` from `utils` to root lib.

### Fixed
- Correct incorrect branching in `utils::map_changepoints` for empty history.  
- Fix warning in `no_std` build guard in `Bocpd`.

---

## [0.14.0] - 2023-06-15
### Added
- First release on `0.14.x` track.  
- Support for new algorithm parameter in `Argpcp`: `noise_level`.  
- Add example integration test in `tests/` illustrating full pipeline through `BocpdTruncated`.

### Changed
- Breaking change: `Bocpd::new()` now requires explicit prior instead of default.  
- Reworked internal module structure: move `utils`, `distribution`, and `hazard` under `internal/`.  
- Simplify trait bounds on generic input types for `step`.

### Fixed
- Fix panic when calling `.step()` on uninitialized detector.  
- Correct bug in normalization of run‑length probabilities under underflow.

---

## [0.13.1] - 2023-01-28
### Added
- Introduce `BocpdTruncated` struct, for truncated run‑length support.  
- Add `Argpcp` (Gaussian Process based changepoint detection).  
- Add functions in `utils` for mapping run‑length posterior to changepoint indices.

### Changed
- Refactor `RunLengthDist` representation for memory efficiency.  
- Change default hazard rate function signature.

### Fixed
- Boundary case fix when no change point is detected (empty output).  
- Bugfix: correct conditional branch in `distribution::update`.


[0.13.1]: https://github.com/promised-ai/changepoint/compare/2b40f54571de8724c3fe57158849102c19e2a2a6...rust-v0.13.1
[0.14.0]: https://github.com/promised-ai/changepoint/compare/rust-v0.13.1...rust-v0.14.0
[0.14.1]: https://github.com/promised-ai/changepoint/compare/rust-v0.14.0...rust-v0.14.1
[0.14.2]: https://github.com/promised-ai/changepoint/compare/rust-v0.14.1...rust-v0.14.2
[0.15.0]: https://github.com/promised-ai/changepoint/compare/rust-v0.14.2...rust-v0.15.0
[Unreleased]: https://github.com/promised-ai/changepoint/compare/rust-v0.15.0...
