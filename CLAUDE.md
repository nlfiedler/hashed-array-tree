# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`hashed-array-tree` is a single-crate Rust library implementing the Hashed Array Tree (HAT) data structure described by Edward Sitarski (Dr. Dobb's Journal, Sept. 1996). It behaves like a growable array (`push`/`pop`/index access) but keeps unused space to O(√N) instead of the up-to-50% overhead of `Vec`. It does **not** support inserts or removes at arbitrary positions — the only mid-array mutation is `swap_remove`.

## Commands

```shell
cargo build                  # build the library
cargo test                   # run the unit test suite (tests live in `mod tests` in src/lib.rs)
cargo test <name>            # run a single test by (substring) name
cargo doc --open             # build and view the rustdoc
cargo run --example benchmarks   # timing comparisons vs. Vec
cargo run --example leak_test    # exercise allocation/deallocation paths
```

Minimum supported Rust version is 1.85.0 (edition 2024).

### Memory-leak / Address Sanitizer testing

Sanitizer runs require Rust nightly and work best on Linux, so they run inside the Docker container in `containers/`:

```shell
cd containers
docker compose build --pull
docker compose run leaktest
sh leak.sh        # runs leak_test example + test suite under -Zsanitizer=address
```

`leak.sh` uses `cargo run/test -Zbuild-std ... -Zsanitizer=address`. Clean output means no leaks detected.

## Architecture

The entire implementation is in `src/lib.rs` (~1700 lines). There is one public type, `HashedArrayTree<T>`.

### Core layout

A HAT is a two-level structure:
- `index: Vec<*mut T>` — the "dope vector" / top array holding raw pointers to separately-allocated data blocks ("leaves").
- Each leaf is a manually-allocated block of `T`. Both the index and every leaf have length `l = 2^k` (powers of two: 4, 8, 16, 32, …).

Element `i` is addressed by bit math, not division: the high bits (`i >> k`) select the leaf in `index`, the low bits (`i & k_mask`) select the slot within that leaf. This is what makes get/set O(1).

Key fields on the struct: `count` (live elements), `k`/`k_mask`/`l` (current power-of-two sizing), and `upper_limit`/`lower_limit` (thresholds that trigger reshaping).

### Growth and shrink ("expand"/"compress")

When `count` hits `upper_limit`, the structure **expands**: `k` increases, so `l` doubles, and elements are re-laid-out into larger leaves. When `count` drops to `lower_limit`, it **compresses** into smaller leaves. This reshaping is the O(N) worst case for `push`/`pop`; steady-state pushes only allocate one new leaf at a time, and pops deallocate empty leaves. When touching `push`/`pop`/`expand`/`compress`, keep the `k`/`k_mask`/`l`/limit invariants consistent — they are assumed everywhere else.

### Unsafe code

This crate is fundamentally `unsafe`: it calls `alloc`/`dealloc` directly, copies elements with raw pointers, and manually manages `Drop`. Any change to allocation, reshaping, or iterators must preserve the invariant that exactly `count` slots are initialized and every allocated leaf is freed exactly once. Validate such changes with the sanitizer container above, not just `cargo test`.

### Trait surface and iterators

`HashedArrayTree<T>` implements `Index`/`IndexMut`, `Default`, `Display`, `Drop`, `Clone` (when `T: Clone`), `Send` (when `T: Send`), and `FromIterator`. There are three iterator types — `ArrayIter` (`iter`), `ArrayIterMut` (`iter_mut`), and `ArrayIntoIter` (`into_iter`) — each implementing `ExactSizeIterator` with a real `size_hint`. `ArrayIntoIter` has its own `Drop` to free not-yet-yielded elements (note the historical bug fixed in 1.2.0: dropping an empty `IntoIterator`).

The `hat!` macro (à la `vec!`) constructs a populated HAT.

## Conventions

- Keep `CHANGELOG.md` updated (Keep a Changelog format, SemVer) and bump the version in `Cargo.toml` for any user-visible change.
- New public methods that mirror `Vec` (e.g. `append`, `swap`, `dedup_by`, `sort_unstable_by`, `split_off`, `truncate`) should match the standard library's signature and semantics where practical.
