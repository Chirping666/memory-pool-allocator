use super::*;

#[test]
fn test_basic_allocation() {
    type Alloc = MemoryPoolAllocator<1024, 64>;
    let mut mem = [0u8; 1024];
    let allocator = unsafe { Alloc::new(mem.as_mut_ptr()) };

    let layout = Layout::from_size_align(16, 8).unwrap();
    let ptr = allocator.try_allocate(layout).unwrap();

    // Check pointer alignment
    assert_eq!(ptr as usize % 8, 0);

    // Deallocate
    assert!(allocator.try_deallocate(ptr).is_ok());

    #[cfg(feature = "statistics")]
    {
        let stats = allocator.stats.lock();
        assert_eq!(stats.allocated_chunks, 0);
    }
}

#[test]
fn test_multiple_allocations() {
    type Alloc = MemoryPoolAllocator<1024, 64>;
    let mut mem = [0u8; 1024];
    let allocator = unsafe { Alloc::new(mem.as_mut_ptr()) };

    let layout = Layout::from_size_align(16, 8).unwrap();
    let mut ptrs = [core::ptr::null_mut(); 10];
    let mut count = 0;

    // Allocate multiple blocks
    for i in 0..10 {
        match allocator.try_allocate(layout) {
            Ok(ptr) => {
                ptrs[i] = ptr;
                count += 1;
            }
            Err(_) => break,
        }
    }

    assert!(count > 0);

    // Verify all pointers are different and aligned
    for i in 0..count {
        assert_eq!(ptrs[i] as usize % 8, 0);
        for j in (i + 1)..count {
            assert_ne!(ptrs[i], ptrs[j]);
        }
    }

    // Deallocate all
    for i in 0..count {
        assert!(allocator.try_deallocate(ptrs[i]).is_ok());
    }

    #[cfg(feature = "statistics")]
    {
        let stats = allocator.stats.lock();
        assert_eq!(stats.allocated_chunks, 0);
    }
}

#[test]
fn test_complex_alignment() {
    type Alloc = MemoryPoolAllocator<1024, 64>;
    let mut mem = [0u8; 1024];
    let allocator = unsafe { Alloc::new(mem.as_mut_ptr()) };

    // Test various alignments
    let layout1 = Layout::from_size_align(10, 1).unwrap();
    let layout2 = Layout::from_size_align(20, 4).unwrap();
    let layout3 = Layout::from_size_align(30, 16).unwrap();

    let ptr1 = allocator.try_allocate(layout1).unwrap();
    let ptr2 = allocator.try_allocate(layout2).unwrap();
    let ptr3 = allocator.try_allocate(layout3).unwrap();

    assert_eq!(ptr1 as usize % 1, 0);
    assert_eq!(ptr2 as usize % 4, 0);
    assert_eq!(ptr3 as usize % 16, 0);

    allocator.try_deallocate(ptr1).unwrap();
    allocator.try_deallocate(ptr2).unwrap();
    allocator.try_deallocate(ptr3).unwrap();

    // Pool should be completely free
    assert_eq!(allocator.count_free_chunks(), 64);
    assert_eq!(allocator.count_allocated_chunks(), 0);
}

#[test]
fn test_alignment_handling() {
    type Alloc = MemoryPoolAllocator<2048, 32>;
    #[repr(align(32))]
    struct Aligned {
        mem: [u8; 1024],
    }
    let mut aligned = Aligned { mem: [0; 1024] };
    let allocator = unsafe { Alloc::new(aligned.mem.as_mut_ptr()) };

    // Allocate with different alignments
    let layout1 = Layout::from_size_align(32, 16).unwrap();
    let ptr1 = allocator.try_allocate(layout1).unwrap();
    assert_eq!(ptr1 as usize % 16, 0);

    let layout2 = Layout::from_size_align(64, 32).unwrap();
    let ptr2 = allocator.try_allocate(layout2).unwrap();
    assert_eq!(ptr2 as usize % 32, 0);

    // Deallocate
    allocator.try_deallocate(ptr1).unwrap();
    allocator.try_deallocate(ptr2).unwrap();
}

#[test]
#[cfg(feature = "statistics")]
fn test_error_handling() {
    type Alloc = MemoryPoolAllocator<512, 32>;
    let mut mem = [0u8; 1024];
    let allocator = unsafe { Alloc::new(mem.as_mut_ptr()) };

    let layout = Layout::from_size_align(16, 8).unwrap();

    // Try allocating more than available
    for _ in 0..32 {
        assert!(allocator.try_allocate(layout).is_ok());
    }

    // Next allocation should fail
    assert!(allocator.try_allocate(layout).is_err());

    // Check stats
    let stats = allocator.stats.lock();
    assert!(stats.allocation_errors > 0);
}

#[test]
fn test_full_pool() {
    type Alloc = MemoryPoolAllocator<64, 4>;
    let mut mem = [0u8; 1024];
    let allocator = unsafe { Alloc::new(mem.as_mut_ptr()) };

    let layout = Layout::from_size_align(16, 8).unwrap();

    // Fill the pool
    let ptr1 = allocator.try_allocate(layout).unwrap();
    let ptr2 = allocator.try_allocate(layout).unwrap();
    let ptr3 = allocator.try_allocate(layout).unwrap();
    let ptr4 = allocator.try_allocate(layout).unwrap();

    // Next allocation should fail
    assert!(allocator.try_allocate(layout).is_err());

    // Free one and try again
    allocator.try_deallocate(ptr2).unwrap();
    let ptr5 = allocator.try_allocate(layout).unwrap();

    // Clean up
    allocator.try_deallocate(ptr1).unwrap();
    allocator.try_deallocate(ptr3).unwrap();
    allocator.try_deallocate(ptr4).unwrap();
    allocator.try_deallocate(ptr5).unwrap();
}

#[test]
fn test_allocation_with_alignment_padding() {
    type Alloc = MemoryPoolAllocator<256, 16>;
    #[repr(align(32))]
    struct AlignedPool {
        mem: [u8; 512],
    }

    let mut pool = AlignedPool { mem: [0; 512] };
    let allocator = unsafe { Alloc::new(pool.mem.as_mut_ptr()) };

    let basic_layout = Layout::from_size_align(16, 1).unwrap();
    let align_layout = Layout::from_size_align(16, 32).unwrap();

    let first = allocator.try_allocate(basic_layout).unwrap();
    let second = allocator.try_allocate(basic_layout).unwrap();
    let third = allocator.try_allocate(basic_layout).unwrap();

    allocator.try_deallocate(second).unwrap();
    allocator.try_deallocate(third).unwrap();

    let aligned_ptr = allocator.try_allocate(align_layout).unwrap();
    assert_eq!(aligned_ptr as usize % 32, 0);

    let fourth = allocator.try_allocate(basic_layout).unwrap();

    allocator.try_deallocate(first).unwrap();
    allocator.try_deallocate(aligned_ptr).unwrap();
    allocator.try_deallocate(fourth).unwrap();

    assert_eq!(allocator.count_allocated_chunks(), 0);
    assert_eq!(allocator.count_free_chunks(), 16);
}

#[test]
#[cfg(feature = "zero-on-free")]
fn test_memory_zeroing() {
    type Alloc = MemoryPoolAllocator<64, 4>;
    let mut mem = [0u8; 1024];
    let allocator = unsafe { Alloc::new(mem.as_mut_ptr()) };

    let layout = Layout::from_size_align(16, 8).unwrap();
    let ptr = allocator.try_allocate(layout).unwrap();

    // Write pattern after the metadata (first 8 bytes)
    unsafe {
        let data_ptr = ptr.add(8);
        core::ptr::write_bytes(data_ptr, 0xAB, 8);
    }

    // Deallocate
    allocator.try_deallocate(ptr).unwrap();

    // Reallocate
    let ptr2 = allocator.try_allocate(layout).unwrap();

    // Check if the data portion is zeroed (skip metadata)
    let mut buffer = [0u8; 8];
    unsafe {
        let data_ptr = ptr2.add(8);
        core::ptr::copy_nonoverlapping(data_ptr, buffer.as_mut_ptr(), 8);
    }
    assert!(buffer.iter().all(|&b| b == 0));

    allocator.try_deallocate(ptr2).unwrap();
}

#[test]
fn test_coalescing() {
    type Alloc = MemoryPoolAllocator<64, 8>; // 8 chunks of 8 bytes each
    let mut mem = [0u8; 64];
    let allocator = unsafe { Alloc::new(mem.as_mut_ptr()) };

    // Initially, should have 8 free chunks
    assert_eq!(allocator.count_free_chunks(), 8);

    // Allocate 3 blocks
    let layout = Layout::from_size_align(8, 1).unwrap(); // 1 chunk each
    let ptr1 = allocator.try_allocate(layout).unwrap();
    let ptr2 = allocator.try_allocate(layout).unwrap();
    let ptr3 = allocator.try_allocate(layout).unwrap();

    // Should have 5 free chunks, 3 allocated
    assert_eq!(allocator.count_free_chunks(), 5);
    assert_eq!(allocator.count_allocated_chunks(), 3);

    // Free the middle one
    allocator.try_deallocate(ptr2).unwrap();

    // Should still have 6 free chunks total (5 + 1), but fragmented
    assert_eq!(allocator.count_free_chunks(), 6);
    assert_eq!(allocator.count_allocated_chunks(), 2);

    // Free the first one (should coalesce with the freed middle one)
    allocator.try_deallocate(ptr1).unwrap();

    // Should have 7 free chunks, 1 allocated
    assert_eq!(allocator.count_free_chunks(), 7);
    assert_eq!(allocator.count_allocated_chunks(), 1);

    // Free the last one (should coalesce everything)
    allocator.try_deallocate(ptr3).unwrap();

    // Should be back to all 8 chunks free
    assert_eq!(allocator.count_free_chunks(), 8);
    assert_eq!(allocator.count_allocated_chunks(), 0);
}

// Just for fun:
struct BadVec<'a, T> {
    size: usize,
    slice: &'a mut [T],
}

impl<'a, T> BadVec<'a, T> {
    fn new(size: usize, slice: &'a mut [T]) -> BadVec<'a, T> {
        BadVec { size, slice }
    }

    fn set(&mut self, idx: usize, data: T) -> Result<(), ()> {
        if idx < self.size {
            self.slice[idx] = data;
            Ok(())
        } else {
            Err(())
        }
    }

    fn get(&mut self, idx: usize) -> Option<&mut T> {
        if idx < self.size {
            Some(&mut self.slice[idx])
        } else {
            None
        }
    }
}

#[test]
fn try_out_bad_vec() {
    type Alloc = MemoryPoolAllocator<64, 8>;
    let mut mem = [0u8; 64];
    let allocator = unsafe { Alloc::new(mem.as_mut_ptr()) };

    let layout = Layout::from_size_align(10, 8).unwrap();
    let ptr = allocator.try_allocate(layout).unwrap();
    let slice = unsafe { core::slice::from_raw_parts_mut(ptr, 10) };

    let mut my_bad_vec = BadVec::new(10, slice);
    for i in 0..10 {
        my_bad_vec.set(i, i as u8).unwrap();
    }

    match my_bad_vec.get(5) {
        Some(data) => {
            assert_eq!(*data, 5);
        }
        None => {
            panic!("No data found!");
        }
    }
}

#[test]
fn test_zero_size_allocation() {
    type Alloc = MemoryPoolAllocator<64, 8>;
    let mut mem = [0u8; 64];
    let allocator = unsafe { Alloc::new(mem.as_mut_ptr()) };

    let layout = Layout::from_size_align(0, 1).unwrap();
    let ptr = allocator.try_allocate(layout).unwrap();

    // Zero-size allocations should return a non-null dangling pointer
    assert!(!ptr.is_null());

    // Should be safe to "deallocate" (should be a no-op or return error)
    // Note: Actual behavior depends on implementation
}

#[test]
fn test_maximum_size_allocation() {
    type Alloc = MemoryPoolAllocator<64, 4>; // 4 chunks of 16 bytes each
    let mut mem = [0u8; 64];
    let allocator = unsafe { Alloc::new(mem.as_mut_ptr()) };

    // Try to allocate the entire pool
    let layout = Layout::from_size_align(64, 1).unwrap();
    let ptr = allocator.try_allocate(layout).unwrap();

    // Should succeed
    assert!(!ptr.is_null());
    assert_eq!(allocator.count_free_chunks(), 0);
    assert_eq!(allocator.count_allocated_chunks(), 4);

    // Any further allocation should fail
    let small_layout = Layout::from_size_align(1, 1).unwrap();
    assert!(allocator.try_allocate(small_layout).is_err());

    // Clean up
    allocator.try_deallocate(ptr).unwrap();
    assert_eq!(allocator.count_free_chunks(), 4);
}

#[test]
fn test_oversized_allocation() {
    type Alloc = MemoryPoolAllocator<64, 4>;
    let mut mem = [0u8; 64];
    let allocator = unsafe { Alloc::new(mem.as_mut_ptr()) };

    // Try to allocate more than the entire pool
    let layout = Layout::from_size_align(128, 1).unwrap();
    assert!(allocator.try_allocate(layout).is_err());

    // Pool should remain unchanged
    assert_eq!(allocator.count_free_chunks(), 4);
    assert_eq!(allocator.count_allocated_chunks(), 0);
}

#[test]
fn test_power_of_two_alignments() {
    type Alloc = MemoryPoolAllocator<1024, 64>; // 64 chunks of 16 bytes each
    let mut mem = [0u8; 1024];
    let allocator = unsafe { Alloc::new(mem.as_mut_ptr()) };

    let alignments = [1, 2, 4, 8, 16, 32, 64];
    let mut ptrs = [0u8 as *mut u8; 7];

    for i in 0..7 {
        let layout = Layout::from_size_align(8, alignments[i]).unwrap();
        if let Ok(ptr) = allocator.try_allocate(layout) {
            assert_eq!(
                ptr as usize % alignments[i],
                0,
                "Failed alignment for {}",
                alignments[i]
            );
            ptrs[i] = ptr;
        }
    }

    // Clean up
    for ptr in ptrs {
        allocator.try_deallocate(ptr).unwrap();
    }
}

#[test]
fn test_large_alignment_requirements() {
    type Alloc = MemoryPoolAllocator<512, 32>; // 32 chunks of 16 bytes each
    let mut mem = [0u8; 512];
    let allocator = unsafe { Alloc::new(mem.as_mut_ptr()) };

    // Test alignment larger than chunk size
    let layout = Layout::from_size_align(8, 64).unwrap();
    if let Ok(ptr) = allocator.try_allocate(layout) {
        assert_eq!(ptr as usize % 64, 0);
        allocator.try_deallocate(ptr).unwrap();
    }

    // Test very large alignment (should fail)
    let layout = Layout::from_size_align(8, 1024).unwrap();
    assert!(allocator.try_allocate(layout).is_err());
}

#[test]
fn test_fragmentation_pattern() {
    type Alloc = MemoryPoolAllocator<128, 16>; // 16 chunks of 8 bytes each
    let mut mem = [0u8; 128];
    let allocator = unsafe { Alloc::new(mem.as_mut_ptr()) };

    let layout = Layout::from_size_align(8, 1).unwrap();

    // Allocate every other chunk to create fragmentation
    let mut ptrs = [(0usize, 0u8 as *mut u8); 8];
    for i in 0..8 {
        if let Ok(ptr) = allocator.try_allocate(layout) {
            ptrs[i] = (i, ptr);
        }
    }

    // Free every other allocation to create holes
    for (i, ptr) in ptrs.iter() {
        if i % 2 == 0 {
            allocator.try_deallocate(*ptr).unwrap();
        }
    }

    // Try to allocate larger blocks that require coalescing
    let large_layout = Layout::from_size_align(16, 1).unwrap(); // 2 chunks
    let large_ptr = allocator.try_allocate(large_layout);

    // Clean up remaining allocations
    for (i, ptr) in ptrs.iter() {
        if i % 2 == 1 {
            allocator.try_deallocate(*ptr).unwrap();
        }
    }

    if let Ok(ptr) = large_ptr {
        allocator.try_deallocate(ptr).unwrap();
    }

    // Should be back to fully free
    assert_eq!(allocator.count_free_chunks(), 16);
}

#[test]
fn test_alternating_alloc_free() {
    type Alloc = MemoryPoolAllocator<64, 8>;
    let mut mem = [0u8; 64];
    let allocator = unsafe { Alloc::new(mem.as_mut_ptr()) };

    let layout = Layout::from_size_align(8, 1).unwrap();

    // Repeatedly allocate and immediately free
    for _ in 0..100 {
        let ptr = allocator.try_allocate(layout).unwrap();
        allocator.try_deallocate(ptr).unwrap();
    }

    // Pool should be completely free and unfragmented
    assert_eq!(allocator.count_free_chunks(), 8);
    assert_eq!(allocator.count_allocated_chunks(), 0);

    // Should be able to allocate the entire pool as one block
    let full_layout = Layout::from_size_align(64, 1).unwrap();
    let full_ptr = allocator.try_allocate(full_layout).unwrap();
    allocator.try_deallocate(full_ptr).unwrap();
}

#[test]
fn test_double_free_detection() {
    type Alloc = MemoryPoolAllocator<64, 4>;
    let mut mem = [0u8; 64];
    let allocator = unsafe { Alloc::new(mem.as_mut_ptr()) };

    let layout = Layout::from_size_align(16, 1).unwrap();
    let ptr = allocator.try_allocate(layout).unwrap();

    // First free should succeed
    assert!(allocator.try_deallocate(ptr).is_ok());

    // Second free should fail
    assert!(allocator.try_deallocate(ptr).is_err());
}

#[test]
fn test_invalid_pointer_deallocation() {
    type Alloc = MemoryPoolAllocator<64, 4>;
    let mut mem = [0u8; 64];
    let allocator = unsafe { Alloc::new(mem.as_mut_ptr()) };

    // Try to free a pointer not from this allocator
    let fake_ptr = 0x1000 as *mut u8;
    assert!(allocator.try_deallocate(fake_ptr).is_err());

    // Try to free null pointer
    assert!(allocator.try_deallocate(core::ptr::null_mut()).is_err());
}

#[test]
fn test_invalid_layout() {
    type Alloc = MemoryPoolAllocator<64, 4>;
    let mut mem = [0u8; 64];
    let allocator = unsafe { Alloc::new(mem.as_mut_ptr()) };

    // Non-power-of-two alignment (this should be caught by Layout::from_size_align)
    // but if somehow created, our allocator should reject it

    // Alignment larger than pool size
    if let Ok(layout) = Layout::from_size_align(8, 128) {
        assert!(allocator.try_allocate(layout).is_err());
    }
}

#[test]
fn test_single_byte_allocations() {
    type Alloc = MemoryPoolAllocator<32, 32>; // 32 chunks of 1 byte each
    let mut mem = [0u8; 32];
    let allocator = unsafe { Alloc::new(mem.as_mut_ptr()) };

    let layout = Layout::from_size_align(1, 1).unwrap();
    let mut ptrs = [0u8 as *mut u8; 32];

    // Allocate 32 single bytes
    for i in 0..32 {
        if let Ok(ptr) = allocator.try_allocate(layout) {
            ptrs[i] = ptr;
        } else {
            break;
        }
    }

    // Should have allocated all 32 chunks
    assert_eq!(allocator.count_allocated_chunks(), 32);
    assert_eq!(allocator.count_free_chunks(), 0);

    // Next allocation should fail
    assert!(allocator.try_allocate(layout).is_err());

    // Free all
    for ptr in ptrs {
        allocator.try_deallocate(ptr).unwrap();
    }

    assert_eq!(allocator.count_free_chunks(), 32);
}

#[test]
fn test_unaligned_pool_base() {
    // Test with a pool that's not optimally aligned
    type Alloc = MemoryPoolAllocator<64, 8>;
    let mut mem = [0u8; 68]; // Extra bytes for alignment adjustment

    // Force misalignment by offsetting into the array
    let misaligned_ptr = unsafe { mem.as_mut_ptr().add(3) }; // +3 bytes offset
    let allocator = unsafe { Alloc::new(misaligned_ptr) };

    // Small allocations should still work
    let layout = Layout::from_size_align(4, 1).unwrap();
    let ptr = allocator.try_allocate(layout).unwrap();

    // Large alignment requirements might fail due to misaligned pool
    let aligned_layout = Layout::from_size_align(4, 16).unwrap();
    let result = allocator.try_allocate(aligned_layout);
    // This might succeed or fail depending on the specific misalignment

    allocator.try_deallocate(ptr).unwrap();
    if let Ok(aligned_ptr) = result {
        allocator.try_deallocate(aligned_ptr).unwrap();
    }
}
