//
// Copyright (c) 2025 Nathan Fiedler
//

//! An implementation of Hashed Array Trees by Edward Sitarski.
//! 
//! From the original article:
//! 
//! > To overcome the limitations of variable-length arrays, I created a data
//! > structure that has fast constant access time like an array, but mostly
//! > avoids copying elements when it grows. I call this new structure a
//! > "Hashed-Array Tree" (HAT) because it combines some of the features of hash
//! > tables, arrays, and trees.
//! 
//! To achieve this, the data structure uses a standard growable vector to
//! reference separate data blocks which hold the array elements. The index and
//! the blocks are at most O(√N) in size. As more elements are added, the size
//! of the index and blocks will grow by powers of two (4, 8, 16, 32, etc).
//!
//! # Memory Usage
//!
//! An empty hased array tree is approximately 72 bytes in size, and while
//! holding elements it will have a space overhead on the order of O(√N). As
//! elements are added the array will grow by allocating additional data blocks.
//! Likewise, as elements are removed from the end of the array, data blocks
//! will be deallocated as they become empty.
//! 
//! # Performance
//! 
//! The get and set operations are O(1) while the push and pop may take O(N) in
//! the worst case, if the array needs to be grown or shrunk.
//!
//! # Safety
//!
//! Because this data structure is allocating memory, copying bytes using
//! pointers, and de-allocating memory as needed, there are many `unsafe` blocks
//! throughout the code.

use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::fmt;
use std::ops::{Index, IndexMut};

/// Hashed Array Tree (HAT) described by Edward Sitarski.
pub struct HashedArrayTree<T> {
    /// top array that holds pointers to data blocks ("leaves")
    index: Vec<*mut T>,
    /// number of elements in the array
    count: usize,
    /// index and leaves are 2^k in length
    k: usize,
    /// bit-mask to get the index into a leaf array
    k_mask: usize,
    /// the number of slots in the top array and leaves
    l: usize,
    /// when size increases to upper_limit, an expand is required
    upper_limit: usize,
    /// when size decreases to lower_limit, a compress is required
    lower_limit: usize,
}

impl<T> HashedArrayTree<T> {
    /// Returns a hashed array tree with zero capacity.
    pub fn new() -> Self {
        let index: Vec<*mut T> = vec![];
        Self {
            index,
            count: 0,
            k: 2,
            k_mask: 3,
            l: 4,
            upper_limit: 16,
            lower_limit: 0,
        }
    }

    /// Double the capacity of this array by combining its leaves into new
    /// leaves of double the capacity.
    fn expand(&mut self) {
        let l_prime = 1 << (self.k + 1);
        let old_index: Vec<*mut T> = std::mem::take(&mut self.index);
        let mut iter = old_index.into_iter();
        while let Some(a) = iter.next() {
            let layout = Layout::array::<T>(l_prime).expect("unexpected overflow");
            let buffer = unsafe {
                let ptr = alloc(layout).cast::<T>();
                if ptr.is_null() {
                    handle_alloc_error(layout);
                }
                ptr
            };
            if let Some(b) = iter.next() {
                let b_dst = unsafe { buffer.add(self.l) };
                let old_layout = Layout::array::<T>(self.l).expect("unexpected overflow");
                unsafe {
                    std::ptr::copy(a, buffer, self.l);
                    std::ptr::copy(b, b_dst, self.l);
                    dealloc(a as *mut u8, old_layout);
                    dealloc(b as *mut u8, old_layout);
                }
            } else {
                let old_layout = Layout::array::<T>(self.l).expect("unexpected overflow");
                unsafe {
                    std::ptr::copy(a, buffer, self.l);
                    dealloc(a as *mut u8, old_layout);
                }
            }
            self.index.push(buffer);
        }
        self.k += 1;
        self.k_mask = (1 << self.k) - 1;
        self.l = 1 << self.k;
        self.upper_limit = self.l * self.l;
        self.lower_limit = self.upper_limit / 8;
    }

    /// Appends an element to the back of a collection.
    ///
    /// # Panics
    ///
    /// Panics if a new block is allocated that would exceed `isize::MAX` _bytes_.
    ///
    /// # Time complexity
    ///
    /// O(N) in the worst case (expand).
    pub fn push(&mut self, value: T) {
        let len = self.count;
        if len >= self.upper_limit {
            self.expand();
        }
        if len >= self.capacity() {
            let layout = Layout::array::<T>(self.l).expect("unexpected overflow");
            let buffer = unsafe {
                let ptr = alloc(layout).cast::<T>();
                if ptr.is_null() {
                    handle_alloc_error(layout);
                }
                ptr
            };
            self.index.push(buffer);
        }
        let block = len >> self.k;
        let slot = len & self.k_mask;
        unsafe { std::ptr::write(self.index[block].add(slot), value) }
        self.count += 1;
    }

    /// Appends an element if there is sufficient spare capacity, otherwise an
    /// error is returned with the element.
    ///
    /// # Time complexity
    ///
    /// O(N) in the worst case (expand).
    pub fn push_within_capacity(&mut self, value: T) -> Result<(), T> {
        if self.capacity() <= self.count {
            Err(value)
        } else {
            self.push(value);
            Ok(())
        }
    }

    /// Retrieve a reference to the element at the given offset.
    ///
    /// # Time complexity
    ///
    /// Constant time.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.count {
            None
        } else {
            let block = index >> self.k;
            let slot = index & self.k_mask;
            unsafe { Some(&*self.index[block].add(slot)) }
        }
    }

    /// Returns a mutable reference to an element.
    ///
    /// # Time complexity
    ///
    /// Constant time.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.count {
            None
        } else {
            let block = index >> self.k;
            let slot = index & self.k_mask;
            unsafe { (self.index[block].add(slot)).as_mut() }
        }
    }

    /// Shrink the capacity of this array by splitting its leaves into new
    /// leaves of half the capacity.
    fn compress(&mut self) {
        let old_index: Vec<*mut T> = std::mem::take(&mut self.index);
        for old_buffer in old_index.into_iter() {
            let half = self.l / 2;
            let layout = Layout::array::<T>(half).expect("unexpected overflow");
            let a = unsafe {
                let ptr = alloc(layout).cast::<T>();
                if ptr.is_null() {
                    handle_alloc_error(layout);
                }
                ptr
            };
            let b = unsafe {
                let ptr = alloc(layout).cast::<T>();
                if ptr.is_null() {
                    handle_alloc_error(layout);
                }
                ptr
            };
            unsafe {
                std::ptr::copy(old_buffer, a, half);
                std::ptr::copy(old_buffer.add(half), b, half);
            };
            let layout = Layout::array::<T>(self.l).expect("unexpected overflow");
            unsafe {
                dealloc(old_buffer as *mut u8, layout);
            }
            self.index.push(a);
            self.index.push(b);
        }
        self.k -= 1;
        self.k_mask = (1 << self.k) - 1;
        self.l = 1 << self.k;
        self.upper_limit = self.l * self.l;
        self.lower_limit = self.upper_limit / 8;
    }

    /// Removes the last element from the array and returns it, or `None` if the
    /// array is empty.
    ///
    /// # Time complexity
    ///
    /// O(N) in the worst case (shrink).
    pub fn pop(&mut self) -> Option<T> {
        if self.count > 0 {
            let index = self.count - 1;
            // avoid compressing the leaves smaller than 4
            if index < self.lower_limit && self.k > 2 {
                self.compress();
            }
            let block = index >> self.k;
            let slot = index & self.k_mask;
            let ret = unsafe { Some(std::ptr::read(self.index[block].add(slot))) };
            if slot == 0 {
                // prune leaves as they become empty
                let ptr = self.index.pop().unwrap();
                let layout = Layout::array::<T>(self.l).expect("unexpected overflow");
                unsafe {
                    dealloc(ptr as *mut u8, layout);
                }
            }
            self.count -= 1;
            ret
        } else {
            None
        }
    }

    /// Removes and returns the last element from an array if the predicate
    /// returns true, or `None`` if the predicate returns `false`` or the array
    /// is empty (the predicate will not be called in that case).
    ///
    /// # Time complexity
    ///
    /// O(N) in the worst case (shrink).
    pub fn pop_if(&mut self, predicate: impl FnOnce(&mut T) -> bool) -> Option<T> {
        if self.count == 0 {
            None
        } else if let Some(last) = self.get_mut(self.count - 1) {
            if predicate(last) { self.pop() } else { None }
        } else {
            None
        }
    }

    /// Removes an element from the array and returns it.
    ///
    /// The removed element is replaced by the last element of the array.
    ///
    /// This does not preserve ordering of the remaining elements.
    ///
    /// # Time complexity
    ///
    /// O(N) in the worst case (shrink).
    pub fn swap_remove(&mut self, index: usize) -> T {
        if index >= self.count {
            panic!(
                "swap_remove index (is {index}) should be < len (is {})",
                self.count
            );
        }
        // retreive the value at index before overwriting
        let block = index >> self.k;
        let slot = index & self.k_mask;
        unsafe {
            let index_ptr = self.index[block].add(slot);
            let value = index_ptr.read();
            // find the pointer of the last element and copy to index pointer
            let block = (self.count - 1) >> self.k;
            let slot = (self.count - 1) & self.k_mask;
            let last_ptr = self.index[block].add(slot);
            std::ptr::copy(last_ptr, index_ptr, 1);
            if slot == 0 {
                // prune leaves as they become empty
                let ptr = self.index.pop().unwrap();
                let layout = Layout::array::<T>(self.l).expect("unexpected overflow");
                dealloc(ptr as *mut u8, layout);
            }
            self.count -= 1;
            value
        }
    }

    // Returns an iterator over the array.
    //
    // The iterator yields all items from start to end.
    pub fn iter(&self) -> ArrayIter<'_, T> {
        ArrayIter {
            array: self,
            index: 0,
        }
    }

    /// Return the number of elements in the array.
    ///
    /// # Time complexity
    ///
    /// Constant time.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns the total number of elements the array can hold without
    /// reallocating.
    ///
    /// # Time complexity
    ///
    /// Constant time.
    pub fn capacity(&self) -> usize {
        (1 << self.k) * self.index.len()
    }

    /// Returns true if the array has a length of 0.
    ///
    /// # Time complexity
    ///
    /// Constant time.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Clears the array, removing all values and deallocating all leaves.
    ///
    /// # Time complexity
    ///
    /// O(N) if elements are droppable, otherwise O(√N)
    pub fn clear(&mut self) {
        use std::ptr::{drop_in_place, slice_from_raw_parts_mut};

        if self.count > 0 && std::mem::needs_drop::<T>() {
            // find the last leaf that contains values and drop them
            let last_index = self.count - 1;
            let last_block = last_index >> self.k;
            let last_slot = last_index & self.k_mask;
            unsafe {
                // last_slot is pointing at the last element, need to add
                // one to include it in the slice
                drop_in_place(slice_from_raw_parts_mut(
                    self.index[last_block],
                    last_slot + 1,
                ));
            }

            // drop the values in all of the preceding leaves
            for block in 0..last_block {
                unsafe {
                    drop_in_place(slice_from_raw_parts_mut(self.index[block], self.l));
                }
            }
        }

        // deallocate all leaves using the index as the source of truth
        let layout = Layout::array::<T>(self.l).expect("unexpected overflow");
        for block in 0..self.index.len() {
            unsafe {
                dealloc(self.index[block] as *mut u8, layout);
            }
        }
        self.index.clear();

        self.count = 0;
        self.k = 2;
        self.k_mask = 3;
        self.l = 1 << self.k;
        self.upper_limit = self.l * self.l;
        self.lower_limit = 0;
    }
}

impl<T> Default for HashedArrayTree<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> fmt::Display for HashedArrayTree<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HashedArrayTree(k: {}, count: {}, dope: {})",
            self.k,
            self.count,
            self.index.len(),
        )
    }
}

impl<T> Index<usize> for HashedArrayTree<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        let Some(item) = self.get(index) else {
            panic!("index out of bounds: {}", index);
        };
        item
    }
}

impl<T> IndexMut<usize> for HashedArrayTree<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let Some(item) = self.get_mut(index) else {
            panic!("index out of bounds: {}", index);
        };
        item
    }
}

impl<T> Drop for HashedArrayTree<T> {
    fn drop(&mut self) {
        self.clear();
    }
}

impl<A> FromIterator<A> for HashedArrayTree<A> {
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        let mut arr: HashedArrayTree<A> = HashedArrayTree::new();
        for value in iter {
            arr.push(value)
        }
        arr
    }
}

/// Immutable hashed array tree iterator.
pub struct ArrayIter<'a, T> {
    array: &'a HashedArrayTree<T>,
    index: usize,
}

impl<'a, T> Iterator for ArrayIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.array.get(self.index);
        self.index += 1;
        value
    }
}

/// An iterator that moves out of a hashed array tree.
pub struct ArrayIntoIter<T> {
    index: usize,
    k: usize,
    k_mask: usize,
    count: usize,
    dope: Vec<*mut T>,
}

impl<T> Iterator for ArrayIntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.count {
            let block = self.index >> self.k;
            let slot = self.index & self.k_mask;
            self.index += 1;
            unsafe { Some((self.dope[block].add(slot)).read()) }
        } else {
            None
        }
    }
}

impl<T> IntoIterator for HashedArrayTree<T> {
    type Item = T;
    type IntoIter = ArrayIntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let mut me = std::mem::ManuallyDrop::new(self);
        let dope = std::mem::take(&mut me.index);
        ArrayIntoIter {
            index: 0,
            count: me.count,
            k: me.k,
            k_mask: me.k_mask,
            dope,
        }
    }
}

impl<T> Drop for ArrayIntoIter<T> {
    fn drop(&mut self) {
        use std::ptr::{drop_in_place, slice_from_raw_parts_mut};
        let block_len = 1 << self.k;

        if std::mem::needs_drop::<T>() {
            let first_block = self.index >> self.k;
            let first_slot = self.index & self.k_mask;
            let last_block = (self.count - 1) >> self.k;
            let last_slot = (self.count - 1) & self.k_mask;
            if first_block == last_block {
                // special-case, remaining values are in only one leaf
                if first_slot <= last_slot {
                    unsafe {
                        // last_slot is pointing at the last element, need to
                        // add one to include it in the slice
                        drop_in_place(slice_from_raw_parts_mut(
                            self.dope[first_block].add(first_slot),
                            last_slot - first_slot + 1,
                        ));
                    }
                }
            } else {
                // drop the remaining values in the first leaf
                if block_len < self.count {
                    unsafe {
                        drop_in_place(slice_from_raw_parts_mut(
                            self.dope[first_block].add(first_slot),
                            block_len - first_slot,
                        ));
                    }
                }

                // drop the values in the last leaf
                unsafe {
                    drop_in_place(slice_from_raw_parts_mut(
                        self.dope[last_block],
                        last_slot + 1,
                    ));
                }

                // drop the values in all of the other leaves
                for block in first_block + 1..last_block {
                    unsafe {
                        drop_in_place(slice_from_raw_parts_mut(self.dope[block], block_len));
                    }
                }
            }
        }

        // deallocate all of the leaves
        for block in 0..self.dope.len() {
            let layout = Layout::array::<T>(block_len).expect("unexpected overflow");
            unsafe {
                dealloc(self.dope[block] as *mut u8, layout);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let sut = HashedArrayTree::<usize>::new();
        assert!(sut.get(0).is_none());
    }

    #[test]
    #[should_panic(expected = "index out of bounds:")]
    fn test_index_out_of_bounds() {
        let mut sut: HashedArrayTree<i32> = HashedArrayTree::new();
        sut.push(10);
        sut.push(20);
        let _ = sut[2];
    }

    #[test]
    #[should_panic(expected = "index out of bounds:")]
    fn test_index_mut_out_of_bounds() {
        let mut sut: HashedArrayTree<i32> = HashedArrayTree::new();
        sut.push(10);
        sut.push(20);
        sut[2] = 30;
    }

    #[test]
    fn test_push_no_expand() {
        // push few enough elements to avoid an expansion
        let mut sut = HashedArrayTree::<usize>::new();
        for value in 0..13 {
            sut.push(value);
        }
        assert_eq!(sut.len(), 13);
        assert_eq!(sut.capacity(), 16);
        for value in 0..13 {
            assert_eq!(sut[value], value);
        }
        // pop enough to cause the last leaf to be freed
        sut.pop();
        assert_eq!(sut.len(), 12);
        assert_eq!(sut.capacity(), 12);
    }

    #[test]
    fn test_push_with_expand() {
        // push few enough elements to cause an expansion
        let mut sut = HashedArrayTree::<usize>::new();
        for value in 0..64 {
            sut.push(value);
        }
        assert_eq!(sut.len(), 64);
        assert_eq!(sut.capacity(), 64);
        for value in 0..64 {
            assert_eq!(sut[value], value);
        }
    }

    #[test]
    fn test_expand_and_compress() {
        // add enough to cause multiple expansions
        let mut sut = HashedArrayTree::<usize>::new();
        for value in 0..1024 {
            sut.push(value);
        }
        assert_eq!(sut.len(), 1024);
        assert_eq!(sut.capacity(), 1024);
        // remove enough to cause multiple compressions
        for _ in 0..960 {
            sut.pop();
        }
        // ensure the correct elements remain
        assert_eq!(sut.len(), 64);
        assert_eq!(sut.capacity(), 64);
        for value in 0..64 {
            assert_eq!(sut[value], value);
        }
    }

    #[test]
    fn test_push_get_get_mut() {
        let mut sut = HashedArrayTree::<usize>::new();
        for value in 0..12 {
            sut.push(value);
        }
        assert_eq!(sut.get(2), Some(&2));
        if let Some(value) = sut.get_mut(1) {
            *value = 11;
        } else {
            panic!("get_mut() returned None")
        }
        assert_eq!(sut[0], 0);
        assert_eq!(sut[1], 11);
        assert_eq!(sut[2], 2);
    }

    #[test]
    fn test_push_within_capacity() {
        // empty array has no allocated space
        let mut sut = HashedArrayTree::<u32>::new();
        assert_eq!(sut.push_within_capacity(101), Err(101));
        // will have 4 slots after a single allocation
        sut.push(1);
        sut.push(2);
        assert_eq!(sut.push_within_capacity(3), Ok(()));
        assert_eq!(sut.push_within_capacity(4), Ok(()));
        assert_eq!(sut.push_within_capacity(5), Err(5));
    }

    #[test]
    fn test_pop_small() {
        let mut sut = HashedArrayTree::<usize>::new();
        assert!(sut.is_empty());
        assert_eq!(sut.len(), 0);
        for value in 0..15 {
            sut.push(value);
        }
        assert!(!sut.is_empty());
        assert_eq!(sut.len(), 15);
        for value in (0..15).rev() {
            assert_eq!(sut.pop(), Some(value));
        }
        assert!(sut.is_empty());
        assert_eq!(sut.len(), 0);
        assert_eq!(sut.capacity(), 0);
    }

    #[test]
    fn test_pop_if() {
        let mut sut = HashedArrayTree::<u32>::new();
        assert!(sut.pop_if(|_| panic!("should not be called")).is_none());
        for value in 0..10 {
            sut.push(value);
        }
        assert!(sut.pop_if(|_| false).is_none());
        let maybe = sut.pop_if(|v| *v == 9);
        assert_eq!(maybe.unwrap(), 9);
        assert!(sut.pop_if(|v| *v == 9).is_none());
    }

    #[test]
    fn test_clear_and_reuse_ints() {
        let mut sut: HashedArrayTree<i32> = HashedArrayTree::new();
        for value in 0..512 {
            sut.push(value);
        }
        assert_eq!(sut.len(), 512);
        sut.clear();
        assert_eq!(sut.len(), 0);
        for value in 0..512 {
            sut.push(value);
        }
        for idx in 0..512 {
            let maybe = sut.get(idx);
            assert!(maybe.is_some(), "{idx} is none");
            let actual = maybe.unwrap();
            assert_eq!(idx, *actual as usize);
        }
    }

    #[test]
    fn test_clear_and_reuse_strings() {
        let mut sut: HashedArrayTree<String> = HashedArrayTree::new();
        for _ in 0..512 {
            let value = ulid::Ulid::new().to_string();
            sut.push(value);
        }
        assert_eq!(sut.len(), 512);
        sut.clear();
        assert_eq!(sut.len(), 0);
        for _ in 0..512 {
            let value = ulid::Ulid::new().to_string();
            sut.push(value);
        }
        assert_eq!(sut.len(), 512);
        // implicitly drop()
    }

    #[test]
    fn test_from_iterator() {
        let mut inputs: Vec<i32> = Vec::new();
        for value in 0..10_000 {
            inputs.push(value);
        }
        let sut: HashedArrayTree<i32> = inputs.into_iter().collect();
        assert_eq!(sut.len(), 10_000);
        for idx in 0..10_000i32 {
            let maybe = sut.get(idx as usize);
            assert!(maybe.is_some(), "{idx} is none");
            let actual = maybe.unwrap();
            assert_eq!(idx, *actual);
        }
    }

    #[test]
    fn test_iterator() {
        let mut sut = HashedArrayTree::<usize>::new();
        for value in 0..512 {
            sut.push(value);
        }
        assert_eq!(sut.len(), 512);
        for (idx, elem) in sut.iter().enumerate() {
            assert_eq!(sut[idx], *elem);
        }
    }

    #[test]
    fn test_into_iterator() {
        // an array that only requires a single segment
        let inputs = [
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        ];
        let mut sut: HashedArrayTree<String> = HashedArrayTree::new();
        for item in inputs {
            sut.push(item.to_owned());
        }
        for (idx, elem) in sut.into_iter().enumerate() {
            assert_eq!(inputs[idx], elem);
        }
        // sut.len(); // error: ownership of sut was moved
    }

    #[test]
    fn test_into_iterator_drop_tiny() {
        // an array that only requires a single segment and only some need to be
        // dropped after partially iterating the values
        let inputs = [
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        ];
        let mut sut: HashedArrayTree<String> = HashedArrayTree::new();
        for item in inputs {
            sut.push(item.to_owned());
        }
        for (idx, _) in sut.into_iter().enumerate() {
            if idx > 2 {
                break;
            }
        }
        // implicitly drop()
    }

    #[test]
    fn test_into_iterator_drop_large() {
        // by adding 512 values and iterating less than 64 times, there will be
        // values in the first segment and some in the last segment, and two
        // segments inbetween that all need to be dropped
        let mut sut: HashedArrayTree<String> = HashedArrayTree::new();
        for _ in 0..512 {
            let value = ulid::Ulid::new().to_string();
            sut.push(value);
        }
        for (idx, _) in sut.into_iter().enumerate() {
            if idx >= 30 {
                break;
            }
        }
        // implicitly drop()
    }

    #[test]
    fn test_swap_remove_single_segment() {
        let mut sut: HashedArrayTree<u32> = HashedArrayTree::new();
        sut.push(1);
        sut.push(2);
        assert_eq!(sut.len(), 2);
        let one = sut.swap_remove(0);
        assert_eq!(one, 1);
        assert_eq!(sut[0], 2);
    }

    #[test]
    fn test_swap_remove_prune_empty() {
        let mut sut: HashedArrayTree<u32> = HashedArrayTree::new();
        for value in 0..13 {
            sut.push(value);
        }
        assert_eq!(sut.len(), 13);
        assert_eq!(sut.capacity(), 16);
        let value = sut.swap_remove(8);
        assert_eq!(value, 8);
        assert_eq!(sut[8], 12);
        assert_eq!(sut.len(), 12);
        assert_eq!(sut.capacity(), 12);
    }

    #[test]
    fn test_swap_remove_multiple_segments() {
        let mut sut: HashedArrayTree<u32> = HashedArrayTree::new();
        for value in 0..512 {
            sut.push(value);
        }
        assert_eq!(sut.len(), 512);
        let eighty = sut.swap_remove(80);
        assert_eq!(eighty, 80);
        assert_eq!(sut.pop(), Some(510));
        assert_eq!(sut[80], 511);
    }

    #[test]
    #[should_panic(expected = "swap_remove index (is 0) should be < len (is 0)")]
    fn test_swap_remove_panic_empty() {
        let mut sut: HashedArrayTree<u32> = HashedArrayTree::new();
        sut.swap_remove(0);
    }

    #[test]
    #[should_panic(expected = "swap_remove index (is 1) should be < len (is 1)")]
    fn test_swap_remove_panic_range_edge() {
        let mut sut: HashedArrayTree<u32> = HashedArrayTree::new();
        sut.push(1);
        sut.swap_remove(1);
    }

    #[test]
    #[should_panic(expected = "swap_remove index (is 2) should be < len (is 1)")]
    fn test_swap_remove_panic_range_exceed() {
        let mut sut: HashedArrayTree<u32> = HashedArrayTree::new();
        sut.push(1);
        sut.swap_remove(2);
    }

    #[test]
    fn test_push_get_many_instances_ints() {
        // test allocating, filling, and then dropping many instances
        for _ in 0..1_000 {
            let mut sut: HashedArrayTree<usize> = HashedArrayTree::new();
            for value in 0..10_000 {
                sut.push(value);
            }
            assert_eq!(sut.len(), 10_000);
        }
    }

    #[test]
    fn test_push_get_many_instances_strings() {
        // test allocating, filling, and then dropping many instances
        for _ in 0..1_000 {
            let mut sut: HashedArrayTree<String> = HashedArrayTree::new();
            for _ in 0..1_000 {
                let value = ulid::Ulid::new().to_string();
                sut.push(value);
            }
            assert_eq!(sut.len(), 1_000);
        }
    }
}
