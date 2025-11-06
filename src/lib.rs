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
use std::cmp::Ordering;
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
        unsafe { self.index[block].add(slot).write(value) }
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

    /// Like [`Self::get()`] but without the `Option` wrapper.
    ///
    /// Will panic if the index is out of bounds.
    fn raw_get(&self, index: usize) -> &T {
        let block = index >> self.k;
        let slot = index & self.k_mask;
        unsafe { &*self.index[block].add(slot) }
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
            Some(self.raw_get(index))
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

    /// Like [`Self::pop()`] but without the `Option` wrapper.
    ///
    /// Will panic if the array is empty.
    pub fn raw_pop(&mut self) -> T {
        let index = self.count - 1;
        // avoid compressing the leaves smaller than 4
        if index < self.lower_limit && self.k > 2 {
            self.compress();
        }
        let block = index >> self.k;
        let slot = index & self.k_mask;
        let ret = unsafe { self.index[block].add(slot).read() };
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
    }

    /// Removes the last element from the array and returns it, or `None` if the
    /// array is empty.
    ///
    /// # Time complexity
    ///
    /// O(N) in the worst case (shrink).
    pub fn pop(&mut self) -> Option<T> {
        if self.count > 0 {
            Some(self.raw_pop())
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
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
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

    /// Swap two elements in the array.
    ///
    /// # Panics
    ///
    /// Panics if either index is out of bounds.
    pub fn swap(&mut self, a: usize, b: usize) {
        if a >= self.count {
            panic!("swap a (is {a}) should be < len (is {})", self.count);
        }
        if b >= self.count {
            panic!("swap b (is {b}) should be < len (is {})", self.count);
        }
        // save the value in slot a before overwriting with value from slot b,
        // then write the saved value to slot b
        let a_block = a >> self.k;
        let a_slot = a & self.k_mask;
        let b_block = b >> self.k;
        let b_slot = b & self.k_mask;
        unsafe {
            let a_ptr = self.index[a_block].add(a_slot);
            let value = a_ptr.read();
            let b_ptr = self.index[b_block].add(b_slot);
            std::ptr::copy(b_ptr, a_ptr, 1);
            b_ptr.write(value);
        }
    }

    /// Sorts the slice in ascending order with a comparison function,
    /// **without** preserving the initial order of equal elements.
    ///
    /// This sort is unstable (i.e., may reorder equal elements), in-place
    /// (i.e., does not allocate), and *O*(*n* \* log(*n*)) worst-case.
    ///
    /// Implements the standard heapsort algorithm.
    pub fn sort_unstable_by<F>(&mut self, mut compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        if self.count < 2 {
            return;
        }
        let mut start = self.count / 2;
        let mut end = self.count;
        while end > 1 {
            if start > 0 {
                start -= 1;
            } else {
                end -= 1;
                self.swap(end, 0);
            }
            let mut root = start;
            let mut child = 2 * root + 1;
            while child < end {
                if child + 1 < end && compare(&self[child], &self[child + 1]) == Ordering::Less {
                    child += 1;
                }
                if compare(&self[root], &self[child]) == Ordering::Less {
                    self.swap(root, child);
                    root = child;
                    child = 2 * root + 1;
                } else {
                    break;
                }
            }
        }
    }

    /// Removes all but the first of consecutive elements in the array
    /// satisfying a given equality relation.
    ///
    /// The `same_bucket` function is passed references to two elements from the
    /// array and must determine if the elements compare equal. The elements are
    /// passed in reverse order from their order in the array, so if
    /// `same_bucket(a, b)` returns `true`, `a` is removed.
    ///
    /// If the array is sorted, this removes all duplicates.
    pub fn dedup_by<F>(&mut self, mut same_bucket: F)
    where
        F: FnMut(&T, &T) -> bool,
    {
        let len = self.len();
        if len <= 1 {
            return;
        }

        // Check if any duplicates exist to avoid allocating, copying, and
        // deallocating memory if nothing needs to be removed.
        let mut first_duplicate_idx: usize = 1;
        while first_duplicate_idx != len {
            let found_duplicate = {
                let prev = self.raw_get(first_duplicate_idx - 1);
                let current = self.raw_get(first_duplicate_idx);
                same_bucket(current, prev)
            };
            if found_duplicate {
                break;
            }
            first_duplicate_idx += 1;
        }
        if first_duplicate_idx == len {
            return;
        }

        // duplicates exist, build a new array of only unique values; steal the
        // old index and sizes, then clear the rest of the properties in order
        // to start over
        let index = std::mem::take(&mut self.index);
        let old_l = self.l;
        let mut remaining = self.count - 1;
        self.count = 0;
        self.clear();
        let layout = Layout::array::<T>(old_l).expect("unexpected overflow");

        // read the first value (we know there are at least two) to get the
        // process started
        let mut prev = unsafe { index[0].read() };
        let mut slot = 1;
        for buffer in index.into_iter() {
            while slot < old_l {
                let current = unsafe { buffer.add(slot).read() };
                if same_bucket(&current, &prev) {
                    drop(current);
                } else {
                    self.push(prev);
                    prev = current;
                }
                remaining -= 1;
                if remaining == 0 {
                    break;
                }
                slot += 1;
            }
            unsafe {
                dealloc(buffer as *mut u8, layout);
            }
            slot = 0;
        }
        // the last element always gets saved
        self.push(prev);
    }

    /// Moves all the elements of `other` into `self`, leaving `other` empty.
    pub fn append(&mut self, other: &mut Self) {
        let index = std::mem::take(&mut other.index);
        let mut remaining = other.count;
        // prevent other from performing any dropping
        other.count = 0;
        let layout = Layout::array::<T>(other.l).expect("unexpected overflow");
        for buffer in index.into_iter() {
            for slot in 0..other.l {
                let value = unsafe { buffer.add(slot).read() };
                self.push(value);
                remaining -= 1;
                if remaining == 0 {
                    break;
                }
            }
            unsafe {
                dealloc(buffer as *mut u8, layout);
            }
        }
    }

    /// Splits the collection into two at the given index.
    ///
    /// Returns a newly allocated array containing the elements in the range
    /// `[at, len)`. After the call, the original array will be left containing
    /// the elements `[0, at)`.
    ///
    /// # Panics
    ///
    /// Panics if `at > len`.
    ///
    /// # Time complexity
    ///
    /// O(N)
    pub fn split_off(&mut self, at: usize) -> Self {
        if at >= self.count {
            panic!("index out of bounds: {at}");
        }
        // Unlike Vec, cannot simply cut the array into two parts, the structure
        // is built around the number of elements in the array, and that changes
        // no matter what the value of `at` might be.
        //
        // As such, pop elements from self and push onto other, maintaining the
        // correct structure of both arrays, then reverse the elements in the
        // other array to correct the order.
        let mut other = Self::new();
        while self.count > at {
            other.push(self.raw_pop());
        }
        let mut low = 0;
        let mut high = other.count - 1;
        while low < high {
            unsafe {
                let lp = other.index[low >> other.k].add(low & other.k_mask);
                let value = lp.read();
                let hp = other.index[high >> other.k].add(high & other.k_mask);
                std::ptr::copy(hp, lp, 1);
                hp.write(value);
            }
            low += 1;
            high -= 1;
        }
        other
    }

    /// Shortens the array, keeping the first `len` elements and dropping the
    /// rest.
    ///
    /// If `len` is greater or equal to the array's current length, this has no
    /// effect.
    ///
    /// # Time complexity
    ///
    /// O(N)
    pub fn truncate(&mut self, len: usize) {
        while self.count > len {
            self.raw_pop();
            // intentionally dropping the value
        }
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

impl<T: Clone> Clone for HashedArrayTree<T> {
    fn clone(&self) -> Self {
        let mut result = HashedArrayTree::<T>::new();
        for value in self.iter() {
            result.push(value.clone());
        }
        result
    }
}

unsafe impl<T: Send> Send for HashedArrayTree<T> {}

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

        if self.count > 0 && std::mem::needs_drop::<T>() {
            // if completely exhausted, the first block/slot will be past the
            // end of the array and thus skip all drops below
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
            } else if first_block < last_block {
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

/// Creates a [`HashedArrayTree`] containing the arguments.
///
/// `hat!` allows `HashedArrayTree`s to be defined with the same syntax as array
/// expressions, much like the `vec!` mocro from the standard library.
#[macro_export]
macro_rules! hat {
    () => {
        HashedArrayTree::new()
    };
    // Takes a comma-separated list of expressions
    ( $($item:expr),* $(,)? ) => {
        {
            let mut result = HashedArrayTree::new();
            $(
                result.push($item);
            )*
            result
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hat_macro() {
        let sut: HashedArrayTree<usize> = hat![];
        assert_eq!(sut.len(), 0);
        assert_eq!(sut.capacity(), 0);

        let sut: HashedArrayTree<usize> = hat![1];
        assert_eq!(sut.len(), 1);
        assert_eq!(sut.capacity(), 4);

        let sut = hat![1, 2, 3, 4, 5, 6, 7, 8, 9,];
        assert_eq!(sut.len(), 9);
        assert_eq!(sut.capacity(), 12);
        assert_eq!(sut[0], 1);
        assert_eq!(sut[1], 2);
        assert_eq!(sut[2], 3);
        assert_eq!(sut[3], 4);
        assert_eq!(sut[4], 5);
        assert_eq!(sut[5], 6);
        assert_eq!(sut[6], 7);
        assert_eq!(sut[7], 8);
        assert_eq!(sut[8], 9);
    }

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
    fn test_clone_ints() {
        let mut sut = HashedArrayTree::<usize>::new();
        for value in 0..512 {
            sut.push(value);
        }
        let cloned = sut.clone();
        let ai = sut.iter();
        let bi = cloned.iter();
        for (a, b) in ai.zip(bi) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_clone_strings() {
        let mut sut = HashedArrayTree::<String>::new();
        for _ in 0..64 {
            let value = ulid::Ulid::new().to_string();
            sut.push(value);
        }
        let cloned = sut.clone();
        let ai = sut.iter();
        let bi = cloned.iter();
        for (a, b) in ai.zip(bi) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_swap() {
        let mut sut = HashedArrayTree::<usize>::new();
        sut.push(1);
        sut.push(2);
        sut.push(3);
        sut.push(4);
        sut.swap(1, 3);
        assert_eq!(sut[0], 1);
        assert_eq!(sut[1], 4);
        assert_eq!(sut[2], 3);
        assert_eq!(sut[3], 2);
    }

    #[test]
    #[should_panic(expected = "swap a (is 1) should be < len (is 1)")]
    fn test_swap_panic_a() {
        let mut sut = HashedArrayTree::<usize>::new();
        sut.push(1);
        sut.swap(1, 2);
    }

    #[test]
    #[should_panic(expected = "swap b (is 1) should be < len (is 1)")]
    fn test_swap_panic_b() {
        let mut sut = HashedArrayTree::<usize>::new();
        sut.push(1);
        sut.swap(0, 1);
    }

    #[test]
    fn test_sort_unstable_by_ints() {
        let mut sut = HashedArrayTree::<usize>::new();
        sut.push(10);
        sut.push(1);
        sut.push(100);
        sut.push(20);
        sut.push(2);
        sut.push(99);
        sut.push(88);
        sut.push(77);
        sut.push(66);
        sut.sort_unstable_by(|a, b| a.cmp(b));
        assert_eq!(sut[0], 1);
        assert_eq!(sut[1], 2);
        assert_eq!(sut[2], 10);
        assert_eq!(sut[3], 20);
        assert_eq!(sut[4], 66);
        assert_eq!(sut[5], 77);
        assert_eq!(sut[6], 88);
        assert_eq!(sut[7], 99);
        assert_eq!(sut[8], 100);
    }

    #[test]
    fn test_sort_unstable_by_strings() {
        let mut sut = HashedArrayTree::<String>::new();
        sut.push("cat".into());
        sut.push("ape".into());
        sut.push("zebra".into());
        sut.push("dog".into());
        sut.push("bird".into());
        sut.push("tapir".into());
        sut.push("monkey".into());
        sut.push("giraffe".into());
        sut.push("frog".into());
        sut.sort_unstable_by(|a, b| a.cmp(b));
        assert_eq!(sut[0], "ape");
        assert_eq!(sut[1], "bird");
        assert_eq!(sut[2], "cat");
        assert_eq!(sut[3], "dog");
        assert_eq!(sut[4], "frog");
        assert_eq!(sut[5], "giraffe");
        assert_eq!(sut[6], "monkey");
        assert_eq!(sut[7], "tapir");
        assert_eq!(sut[8], "zebra");
    }

    #[test]
    fn test_append() {
        let odds = ["one", "three", "five", "seven", "nine"];
        let mut sut = HashedArrayTree::<String>::new();
        for item in odds {
            sut.push(item.to_owned());
        }
        let evens = ["two", "four", "six", "eight", "ten"];
        let mut other = HashedArrayTree::<String>::new();
        for item in evens {
            other.push(item.to_owned());
        }
        sut.append(&mut other);
        assert_eq!(sut.len(), 10);
        assert_eq!(sut.capacity(), 12);
        assert_eq!(other.len(), 0);
        assert_eq!(other.capacity(), 0);
        sut.sort_unstable_by(|a, b| a.cmp(b));
        assert_eq!(sut[0], "eight");
        assert_eq!(sut[1], "five");
        assert_eq!(sut[2], "four");
        assert_eq!(sut[3], "nine");
        assert_eq!(sut[4], "one");
        assert_eq!(sut[5], "seven");
        assert_eq!(sut[6], "six");
        assert_eq!(sut[7], "ten");
        assert_eq!(sut[8], "three");
        assert_eq!(sut[9], "two");
    }

    #[test]
    fn test_dedup_by_tiny() {
        let mut sut = HashedArrayTree::<String>::new();
        sut.push("one".into());
        sut.dedup_by(|a, b| a == b);
        assert_eq!(sut.len(), 1);
        assert_eq!(sut[0], "one");
    }

    #[test]
    fn test_dedup_by_2_dupes() {
        let mut sut = HashedArrayTree::<String>::new();
        sut.push("one".into());
        sut.push("one".into());
        sut.dedup_by(|a, b| a == b);
        assert_eq!(sut.len(), 1);
        assert_eq!(sut[0], "one");
    }

    #[test]
    fn test_dedup_by_2_unique() {
        let mut sut = HashedArrayTree::<String>::new();
        sut.push("one".into());
        sut.push("two".into());
        sut.dedup_by(|a, b| a == b);
        assert_eq!(sut.len(), 2);
        assert_eq!(sut[0], "one");
        assert_eq!(sut[1], "two");
    }

    #[test]
    fn test_dedup_by_all_unique() {
        let inputs = [
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        ];
        let mut sut = HashedArrayTree::<String>::new();
        for item in inputs {
            sut.push(item.to_owned());
        }
        sut.dedup_by(|a, b| a == b);
        assert_eq!(sut.len(), 9);
        for (idx, elem) in sut.into_iter().enumerate() {
            assert_eq!(inputs[idx], elem);
        }
    }

    #[test]
    fn test_dedup_by_all_dupes() {
        let inputs = [
            "one", "one", "one", "one", "one", "one", "one", "one", "one", "one",
        ];
        let mut sut = HashedArrayTree::<String>::new();
        for item in inputs {
            sut.push(item.to_owned());
        }
        assert_eq!(sut.len(), 10);
        sut.dedup_by(|a, b| a == b);
        assert_eq!(sut.len(), 1);
        assert_eq!(inputs[0], "one");
    }

    #[test]
    fn test_dedup_by_some_dupes_ints() {
        let inputs = [1, 2, 2, 3, 2];
        let mut sut = HashedArrayTree::<usize>::new();
        for item in inputs {
            sut.push(item.to_owned());
        }
        assert_eq!(sut.len(), 5);
        sut.dedup_by(|a, b| a == b);
        assert_eq!(sut.len(), 4);
        assert_eq!(sut[0], 1);
        assert_eq!(sut[1], 2);
        assert_eq!(sut[2], 3);
        assert_eq!(sut[3], 2);
    }

    #[test]
    fn test_dedup_by_some_dupes_strings() {
        let inputs = ["foo", "bar", "Bar", "baz", "bar"];
        let mut sut = HashedArrayTree::<String>::new();
        for item in inputs {
            sut.push(item.to_owned());
        }
        assert_eq!(sut.len(), 5);
        sut.dedup_by(|a, b| a.eq_ignore_ascii_case(b));
        assert_eq!(sut.len(), 4);
        assert_eq!(sut[0], "foo");
        assert_eq!(sut[1], "bar");
        assert_eq!(sut[2], "baz");
        assert_eq!(sut[3], "bar");
    }

    #[test]
    #[should_panic(expected = "index out of bounds: 10")]
    fn test_split_off_bounds_panic() {
        let mut sut = HashedArrayTree::<usize>::new();
        sut.split_off(10);
    }

    #[test]
    fn test_split_off_middle() {
        let mut sut = HashedArrayTree::<usize>::new();
        for value in 0..16 {
            sut.push(value);
        }
        assert_eq!(sut.len(), 16);
        assert_eq!(sut.capacity(), 16);
        let other = sut.split_off(8);
        assert_eq!(sut.len(), 8);
        assert_eq!(sut.capacity(), 8);
        for value in 0..8 {
            assert_eq!(sut[value], value);
        }
        assert_eq!(other.len(), 8);
        assert_eq!(other.capacity(), 8);
        for (index, value) in (8..16).enumerate() {
            assert_eq!(other[index], value);
        }
    }

    #[test]
    fn test_split_off_almost_start() {
        let mut sut = HashedArrayTree::<usize>::new();
        for value in 0..16 {
            sut.push(value);
        }
        assert_eq!(sut.len(), 16);
        assert_eq!(sut.capacity(), 16);
        let other = sut.split_off(1);
        assert_eq!(sut.len(), 1);
        assert_eq!(sut.capacity(), 4);
        for value in 0..1 {
            assert_eq!(sut[value], value);
        }
        assert_eq!(other.len(), 15);
        assert_eq!(other.capacity(), 16);
        for (index, value) in (1..16).enumerate() {
            assert_eq!(other[index], value);
        }
    }

    #[test]
    fn test_split_off_almost_end() {
        let mut sut = HashedArrayTree::<usize>::new();
        for value in 0..16 {
            sut.push(value);
        }
        assert_eq!(sut.len(), 16);
        assert_eq!(sut.capacity(), 16);
        let other = sut.split_off(15);
        assert_eq!(sut.len(), 15);
        assert_eq!(sut.capacity(), 16);
        for value in 0..15 {
            assert_eq!(sut[value], value);
        }
        assert_eq!(other.len(), 1);
        assert_eq!(other.capacity(), 4);
        assert_eq!(other[0], 15);
    }

    #[test]
    fn test_split_off_odd_other() {
        let mut sut = HashedArrayTree::<usize>::new();
        for value in 0..16 {
            sut.push(value);
        }
        assert_eq!(sut.len(), 16);
        assert_eq!(sut.capacity(), 16);
        let other = sut.split_off(11);
        assert_eq!(sut.len(), 11);
        assert_eq!(sut.capacity(), 12);
        for value in 0..11 {
            assert_eq!(sut[value], value);
        }
        assert_eq!(other.len(), 5);
        assert_eq!(other.capacity(), 8);
        for (index, value) in (11..16).enumerate() {
            assert_eq!(other[index], value);
        }
    }

    #[test]
    fn test_truncate_typical() {
        let mut sut = hat![1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(sut.len(), 8);
        assert_eq!(sut.capacity(), 8);
        sut.truncate(5);
        assert_eq!(sut.len(), 5);
        assert_eq!(sut.capacity(), 8);
        for (index, value) in (1..6).enumerate() {
            assert_eq!(sut[index], value);
        }
    }

    #[test]
    fn test_truncate_out_of_bounds() {
        let mut sut = hat![1, 2, 3, 4, 5,];
        assert_eq!(sut.len(), 5);
        assert_eq!(sut.capacity(), 8);
        sut.truncate(8);
        assert_eq!(sut.len(), 5);
        assert_eq!(sut.capacity(), 8);
        for (index, value) in (1..6).enumerate() {
            assert_eq!(sut[index], value);
        }
    }

    #[test]
    fn test_truncate_to_empty() {
        let mut sut = hat![1, 2, 3, 4, 5,];
        assert_eq!(sut.len(), 5);
        assert_eq!(sut.capacity(), 8);
        sut.truncate(0);
        assert_eq!(sut.len(), 0);
        assert_eq!(sut.capacity(), 0);
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
    fn test_into_iterator_edge_case() {
        // iterate to the end (of the last data block)
        let inputs = [
            "one", "two", "three", "four", "five", "six", "seven", "eight",
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
    fn test_into_iterator_multiple_leaves() {
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
    fn test_into_iterator_drop_empty() {
        let sut: HashedArrayTree<String> = HashedArrayTree::new();
        assert_eq!(sut.into_iter().count(), 0);
    }

    #[test]
    fn test_into_iterator_drop_single_leaf() {
        // an array that only requires a single segment and only some need to be
        // dropped after partially iterating the values
        let inputs = ["one", "two", "three", "four"];
        let mut sut: HashedArrayTree<String> = HashedArrayTree::new();
        for item in inputs {
            sut.push(item.to_owned());
        }
        for (idx, _) in sut.into_iter().enumerate() {
            if idx > 1 {
                break;
            }
        }
        // implicitly drop()
    }

    #[test]
    fn test_into_iterator_drop_large() {
        // by adding 512 values and iterating less than 32 times, there will be
        // values in the first segment and some in the last segment, and two
        // segments inbetween that all need to be dropped
        let mut sut: HashedArrayTree<String> = HashedArrayTree::new();
        for _ in 0..512 {
            let value = ulid::Ulid::new().to_string();
            sut.push(value);
        }
        for (idx, _) in sut.into_iter().enumerate() {
            if idx >= 28 {
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
