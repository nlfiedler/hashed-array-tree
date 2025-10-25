//
// Copyright (c) 2025 Nathan Fiedler
//
use hashed_array_tree::HashedArrayTree;

//
// Create and drop collections and iterators in order to test for memory leaks.
// Must allocate Strings in order to fully test the drop implementation.
//
fn main() {
    // add only enough values to allocate one segment
    let mut array: HashedArrayTree<String> = HashedArrayTree::new();
    for _ in 0..2 {
        let value = ulid::Ulid::new().to_string();
        array.push(value);
    }
    drop(array);

    // add enough values to allocate a bunch of blocks but leave the last one
    // partially filled
    let mut array: HashedArrayTree<String> = HashedArrayTree::new();
    for _ in 0..72 {
        let value = ulid::Ulid::new().to_string();
        array.push(value);
    }
    drop(array);

    // test pushing many elements then popping all of them
    let mut array: HashedArrayTree<String> = HashedArrayTree::new();
    for _ in 0..512 {
        let value = ulid::Ulid::new().to_string();
        array.push(value);
    }
    while !array.is_empty() {
        array.pop();
    }

    // IntoIterator: add exactly one element to test special case
    let mut array: HashedArrayTree<String> = HashedArrayTree::new();
    let value = ulid::Ulid::new().to_string();
    array.push(value);
    let itty = array.into_iter();
    drop(itty);

    // IntoIterator: add enough values to allocate a bunch of blocks
    let mut array: HashedArrayTree<String> = HashedArrayTree::new();
    for _ in 0..256 {
        let value = ulid::Ulid::new().to_string();
        array.push(value);
    }
    // skip enough elements to pass over a block then drop
    for (index, value) in array.into_iter().enumerate() {
        if index == 24 {
            println!("24: {value}");
            // exit the iterator early intentionally
            break;
        }
    }
}
