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

    // test append() handling of dealloc and preventing drop
    let odds = ["one", "three", "five", "seven", "nine"];
    let mut array = HashedArrayTree::<String>::new();
    for item in odds {
        array.push(item.to_owned());
    }
    let evens = ["two", "four", "six", "eight"];
    let mut other = HashedArrayTree::<String>::new();
    for item in evens {
        other.push(item.to_owned());
    }
    array.append(&mut other);

    // test dedup_by() handling of dealloc and drop
    let inputs = ["foo", "bar", "Bar", "baz", "bar"];
    let mut array = HashedArrayTree::<String>::new();
    for item in inputs {
        array.push(item.to_owned());
    }
    array.dedup_by(|a, b| a.eq_ignore_ascii_case(b));

    // test split_off()
    let inputs = ["foo", "bar", "baz", "quux", "one", "two", "tree"];
    let mut array = HashedArrayTree::<String>::new();
    for item in inputs {
        array.push(item.to_owned());
    }
    let _ = array.split_off(4);

    // test truncate()
    let inputs = ["foo", "bar", "baz", "quux", "one", "two", "tree"];
    let mut array = HashedArrayTree::<String>::new();
    for item in inputs {
        array.push(item.to_owned());
    }
    array.truncate(4);

    println!("hashed array tree tests complete");
}
