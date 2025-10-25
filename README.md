# Hashed Array Trees

## Overview

This Rust crate provides an implementation of hashed array trees as described by Edward Sitarski in the Algorithm Alley section of the September 1996 edition of Dr. Dobb's Journal.

This data structure supports `push` and `pop` operations and does _not_ support inserts or removes at other locations within the array. One exception is the `swap/remove` operation which will retrieve a value from a specified index, overwrite that slot with the value at the end of the array, decrement the count, and return the retrieved value.

### Memory Usage

Compared to the `Vec` type in the Rust standard library, this data structure will have substantially less unused space, on the order of O(√N). The dope vector contributes to the overhead of this data structure, and that is on the order of O(√N). Based on the current implementation of `Vec`, as much as 50% of the space may be unused since it has a growth factor of 2.

## Examples

A simple example copied from the unit tests.

```rust
let mut sut = HashedArrayTree::<usize>::new();
for value in 0..13 {
    sut.push(value);
}
assert_eq!(sut.len(), 13);
assert_eq!(sut.capacity(), 16);
for value in 0..13 {
    assert_eq!(sut[value], value);
}
```

## Supported Rust Versions

The Rust edition is set to `2024` and hence version `1.85.0` is the minimum supported version.

## Troubleshooting

### Memory Leaks

Finding memory leaks with [Address Sanitizer](https://clang.llvm.org/docs/AddressSanitizer.html) is fairly [easy](https://doc.rust-lang.org/beta/unstable-book/compiler-flags/sanitizer.html) and seems to work best on Linux. The shell script below gives a quick demonstration of running one of the examples with ASAN analysis enabled.

```shell
#!/bin/sh
env RUSTDOCFLAGS=-Zsanitizer=address RUSTFLAGS=-Zsanitizer=address \
    cargo run -Zbuild-std --target x86_64-unknown-linux-gnu --release --example leak_test
```

## References

* [HATs: Hashed Array Trees (1996)](https://jacobfilipp.com/DrDobbs/articles/DDJ/1996/9609/9609n/9609n.htm)
    - An archived version of the original article from Dr. Dobb's Journal.

## Other Implementations

* [Aca-S/hashed-array-tree](https://github.com/Aca-S/hashed-array-tree) (C++)
