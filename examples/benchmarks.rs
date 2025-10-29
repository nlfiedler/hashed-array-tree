//
// Copyright (c) 2025 Nathan Fiedler
//
use hashed_array_tree::HashedArrayTree;
use std::time::Instant;

fn benchmark_hat(size: usize) {
    let mut coll: HashedArrayTree<usize> = HashedArrayTree::new();
    let start = Instant::now();
    for value in 0..size {
        coll.push(value);
    }
    let duration = start.elapsed();
    println!("HAT create: {:?}", duration);

    // test sequenced access for entire collection
    let start = Instant::now();
    for (index, value) in coll.iter().enumerate() {
        assert_eq!(*value, index);
    }
    let duration = start.elapsed();
    println!("HAT ordered: {:?}", duration);

    let unused = coll.capacity() - coll.len();
    println!("unused capacity: {unused}");

    // test popping all elements from the array
    let start = Instant::now();
    while !coll.is_empty() {
        coll.pop();
    }
    let duration = start.elapsed();
    println!("HAT pop-all: {:?}", duration);
    println!("final capacity: {}", coll.capacity());
}

fn benchmark_vector(size: usize) {
    let start = Instant::now();
    let mut coll: Vec<usize> = Vec::new();
    for value in 0..size {
        coll.push(value);
    }
    let duration = start.elapsed();
    println!("vector create: {:?}", duration);

    // test sequenced access for entire collection
    let start = Instant::now();
    for (index, value) in coll.iter().enumerate() {
        assert_eq!(*value, index);
    }
    let duration = start.elapsed();
    println!("vector ordered: {:?}", duration);

    let unused = coll.capacity() - coll.len();
    println!("unused capacity: {unused}");

    // test popping all elements from the vector
    let start = Instant::now();
    while !coll.is_empty() {
        coll.pop();
    }
    let duration = start.elapsed();
    println!("vector pop-all: {:?}", duration);
    println!("final capacity: {}", coll.capacity());
}

fn main() {
    let size = 100_000_000;
    println!("\ncreating HashedArrayTree...");
    benchmark_hat(size);
    println!("\ncreating Vec...");
    benchmark_vector(size);
}
