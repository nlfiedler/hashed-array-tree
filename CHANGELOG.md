# Change Log

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/).
This file follows the convention described at
[Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [1.2.0] - 2025-11-05
### Fixed
- Panic when dropping an empty `IntoIterator`.
### Added
- Add `hat!` macro, add `split_off()` and `truncate()` functions.

## [1.1.0] - 2025-10-28
### Added
- Add `Clone` implementation for elements that implement `Clone`.
- Add `Send` implementation for elements that implement `Send`.
- Add `append()`, `swap()`, `dedup_by()`, and `sort_unstable_by()` functions.

## [1.0.0] - 2025-10-24
### Changed
- Initial release
