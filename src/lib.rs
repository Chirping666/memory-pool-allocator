// Copyright 2024 Chirping666
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0>
// or the MIT license <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your option.

#![no_std]

use core::{alloc::Layout, ptr::NonNull};
use parking_lot::Mutex;

/// Fixed-size memory pool allocator with compile-time size configuration
/// 
/// # Type Parameters
/// * `N` - Total pool size in bytes
/// * `M` - Number of chunks to divide the pool into
/// 
/// # Invariants
/// * `N` must be exactly divisible by `M`
/// * Each chunk is `N/M` bytes
pub struct MemoryPoolAllocator<const N: usize, const M: usize>
where
    [u8; N]: Sized,
    [BlockInfo; M]: Sized,
{
    inner: Mutex<PoolInner<N, M>>,
    stats: Mutex<PoolStats>,
}

struct PoolInner<const N: usize, const M: usize> {
    pool: [u8; N],
    blocks: [BlockInfo; M],
}

/// Tracks the state of each chunk in the pool
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlockInfo {
    /// Start of a free block with total chunk count
    FreeStart(usize),
    /// Continuation of a free block
    FreeContinuation,
    /// Start of an allocated block with total chunk count
    AllocatedStart(usize),
    /// Continuation of an allocated block
    AllocatedContinuation,
}

/// Allocation errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocError {
    /// Not enough contiguous space
    OutOfMemory,
    /// Invalid layout parameters
    InvalidLayout,
    /// Pointer not from this allocator
    InvalidPointer,
    /// Pointer not currently allocated
    NotAllocated,
}

impl core::fmt::Display for AllocError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfMemory => write!(f, "Out of memory"),
            Self::InvalidLayout => write!(f, "Invalid layout parameters"),
            Self::InvalidPointer => write!(f, "Pointer not from this allocator"),
            Self::NotAllocated => write!(f, "Pointer not currently allocated"),
        }
    }
}

/// Pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_size: usize,
    pub chunk_size: usize,
    pub total_chunks: usize,
    pub allocated_bytes: usize,
    pub free_bytes: usize,
    pub allocated_blocks: usize,
    pub free_blocks: usize,
    pub largest_free_block: usize,
    pub fragmentation: usize, // Percentage * 100 (e.g., 2500 = 25%)
    pub allocation_errors: usize,
    pub deallocation_errors: usize,
}

impl PoolStats {
    const fn new(total_size: usize, chunk_size: usize, total_chunks: usize) -> Self {
        Self {
            total_size,
            chunk_size,
            total_chunks,
            allocated_bytes: 0,
            free_bytes: total_size,
            allocated_blocks: 0,
            free_blocks: 1, // Initially one large free block
            largest_free_block: total_size,
            fragmentation: 0,
            allocation_errors: 0,
            deallocation_errors: 0,
        }
    }
}

// Safety: Pool data is protected by mutex
unsafe impl<const N: usize, const M: usize> Sync for MemoryPoolAllocator<N, M> {}
unsafe impl<const N: usize, const M: usize> Send for MemoryPoolAllocator<N, M> {}

impl<const N: usize, const M: usize> MemoryPoolAllocator<N, M>
where
    [u8; N]: Sized,
    [BlockInfo; M]: Sized,
{
    /// Size of each chunk in bytes
    const CHUNK_SIZE: usize = N / M;

    /// Creates a new memory pool allocator
    pub const fn new() -> Self {
        assert!(N % M == 0, "N must be exactly divisible by M");
        assert!(N > 0 && M > 0, "Pool and chunk count must be positive");
        
        let mut blocks = [BlockInfo::FreeContinuation; M];
        blocks[0] = BlockInfo::FreeStart(M);
        
        Self {
            inner: Mutex::new(PoolInner { 
                pool: [0u8; N], 
                blocks 
            }),
            stats: Mutex::new(PoolStats::new(N, N / M, M)),
        }
    }

    /// Attempts to allocate memory with the specified layout
    pub fn try_allocate(&self, layout: Layout) -> Result<*mut u8, AllocError> {
        // Handle zero-size allocations
        if layout.size() == 0 {
            return Ok(NonNull::dangling().as_ptr());
        }

        // Validate layout
        if !layout.align().is_power_of_two() || layout.align() > N {
            self.stats.lock().allocation_errors += 1;
            return Err(AllocError::InvalidLayout);
        }

        let chunks_needed = Self::bytes_to_chunks(layout.size());
        let mut inner = self.inner.lock();
        
        // Find a suitable free block
        let result = self.find_and_allocate_block(&mut inner, layout, chunks_needed);
        
        match result {
            Ok(ptr) => {
                self.update_stats_after_allocation(chunks_needed);
                Ok(ptr)
            }
            Err(e) => {
                self.stats.lock().allocation_errors += 1;
                Err(e)
            }
        }
    }

    /// Attempts to deallocate previously allocated memory
    pub fn try_deallocate(&self, ptr: *mut u8, layout: Layout) -> Result<(), AllocError> {
        // Handle null and zero-size
        if ptr.is_null() || layout.size() == 0 {
            self.stats.lock().deallocation_errors += 1;
            return Err(AllocError::InvalidPointer);
        }

        let mut inner = self.inner.lock();
        
        // Validate pointer is within pool
        let pool_start = inner.pool.as_ptr() as usize;
        let pool_end = pool_start + N;
        let ptr_addr = ptr as usize;
        
        if ptr_addr < pool_start || ptr_addr >= pool_end {
            self.stats.lock().deallocation_errors += 1;
            return Err(AllocError::InvalidPointer);
        }

        // Find the chunk and validate it's allocated
        let byte_offset = ptr_addr - pool_start;
        let chunk_idx = byte_offset / Self::CHUNK_SIZE;
        
        if chunk_idx >= M {
            self.stats.lock().deallocation_errors += 1;
            return Err(AllocError::InvalidPointer);
        }

        match inner.blocks[chunk_idx] {
            BlockInfo::AllocatedStart(chunk_count) => {
                // Clear memory if feature enabled
                #[cfg(feature = "zero-on-free")]
                self.clear_memory(&mut inner.pool, chunk_idx, chunk_count);
                
                // Mark as free and coalesce
                self.mark_chunks_free(&mut inner.blocks, chunk_idx, chunk_count);
                self.coalesce_free_blocks(&mut inner.blocks, chunk_idx);
                
                self.update_stats_after_deallocation(chunk_count);
                Ok(())
            }
            _ => {
                self.stats.lock().deallocation_errors += 1;
                Err(AllocError::NotAllocated)
            }
        }
    }

    /// Returns current pool statistics
    pub fn get_stats(&self) -> PoolStats {
        let stats = self.stats.lock();
        stats.clone()
    }

    /// Returns the size of each chunk in bytes
    pub const fn chunk_size() -> usize {
        Self::CHUNK_SIZE
    }

    /// Returns the total number of chunks
    pub const fn chunk_count() -> usize {
        M
    }

    // === Private helper methods ===

    /// Converts bytes to number of chunks needed (rounds up)
    #[inline]
    fn bytes_to_chunks(bytes: usize) -> usize {
        (bytes + Self::CHUNK_SIZE - 1) / Self::CHUNK_SIZE
    }

    /// Aligns an address up to the specified alignment
    #[inline]
    fn align_up(addr: usize, align: usize) -> Option<usize> {
        let mask = align - 1;
        addr.checked_add(mask).map(|v| v & !mask)
    }

    /// Finds a suitable free block and allocates it
    fn find_and_allocate_block(
        &self,
        inner: &mut PoolInner<N, M>,
        layout: Layout,
        chunks_needed: usize,
    ) -> Result<*mut u8, AllocError> {
        let pool_base = inner.pool.as_ptr() as usize;
        let mut chunk_idx = 0;

        while chunk_idx < M {
            if let BlockInfo::FreeStart(block_chunks) = inner.blocks[chunk_idx] {
                // Check if block can accommodate aligned allocation
                let block_addr = pool_base + (chunk_idx * Self::CHUNK_SIZE);
                
                let aligned_addr = match Self::align_up(block_addr, layout.align()) {
                    Some(addr) => addr,
                    None => {
                        chunk_idx += block_chunks;
                        continue;
                    }
                };

                // Validate aligned address is within pool
                if aligned_addr >= pool_base + N {
                    chunk_idx += block_chunks;
                    continue;
                }

                let alignment_waste = aligned_addr - block_addr;
                let alignment_chunks = Self::bytes_to_chunks(alignment_waste);
                
                if alignment_chunks + chunks_needed <= block_chunks {
                    return Ok(self.split_and_allocate_block(
                        inner,
                        chunk_idx,
                        block_chunks,
                        alignment_chunks,
                        chunks_needed,
                        aligned_addr,
                    ));
                }
                
                chunk_idx += block_chunks;
            } else {
                chunk_idx += 1;
            }
        }

        Err(AllocError::OutOfMemory)
    }

    /// Splits a free block and allocates part of it
    fn split_and_allocate_block(
        &self,
        inner: &mut PoolInner<N, M>,
        block_start: usize,
        block_size: usize,
        alignment_offset: usize,
        alloc_size: usize,
        aligned_addr: usize,
    ) -> *mut u8 {
        let alloc_start = block_start + alignment_offset;

        // Mark alignment padding as free if present
        if alignment_offset > 0 {
            self.mark_chunks(&mut inner.blocks, block_start, alignment_offset, true);
        }

        // Mark allocated chunks
        self.mark_chunks(&mut inner.blocks, alloc_start, alloc_size, false);

        // Mark remaining chunks as free if present
        let remaining = block_size - alignment_offset - alloc_size;
        if remaining > 0 {
            let remaining_start = alloc_start + alloc_size;
            self.mark_chunks(&mut inner.blocks, remaining_start, remaining, true);
        }

        aligned_addr as *mut u8
    }

    /// Marks a range of chunks as free or allocated
    fn mark_chunks(
        &self,
        blocks: &mut [BlockInfo],
        start: usize,
        count: usize,
        is_free: bool,
    ) {
        debug_assert!(start + count <= M, "Chunk range exceeds pool size");
        
        blocks[start] = if is_free {
            BlockInfo::FreeStart(count)
        } else {
            BlockInfo::AllocatedStart(count)
        };

        let continuation = if is_free {
            BlockInfo::FreeContinuation
        } else {
            BlockInfo::AllocatedContinuation
        };

        for i in (start + 1)..(start + count) {
            blocks[i] = continuation;
        }
    }

    /// Marks chunks as free (convenience wrapper)
    #[inline]
    fn mark_chunks_free(&self, blocks: &mut [BlockInfo], start: usize, count: usize) {
        self.mark_chunks(blocks, start, count, true);
    }

    /// Coalesces adjacent free blocks into larger blocks
    fn coalesce_free_blocks(&self, blocks: &mut [BlockInfo], hint_chunk: usize) {
        // Find the start of the free region containing hint_chunk
        let mut region_start = hint_chunk;
        
        // Walk backward to find the beginning of our free region
        while region_start > 0 {
            match blocks[region_start - 1] {
                BlockInfo::FreeContinuation => region_start -= 1,
                BlockInfo::FreeStart(_) => {
                    region_start -= 1;
                    break;
                }
                _ => break,
            }
        }

        // Walk further backward to include any adjacent free blocks
        while region_start > 0 {
            // Find the start of the potential previous block
            let mut prev_start = region_start - 1;
            while prev_start > 0 && matches!(blocks[prev_start], BlockInfo::FreeContinuation) {
                prev_start -= 1;
            }

            // Check if it's a free block that ends where our region starts
            if let BlockInfo::FreeStart(prev_count) = blocks[prev_start] {
                if prev_start + prev_count == region_start {
                    region_start = prev_start;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        // Find the end of the free region
        let mut region_end = hint_chunk;
        while region_end < M {
            match blocks[region_end] {
                BlockInfo::FreeStart(count) => region_end += count,
                BlockInfo::FreeContinuation => region_end += 1,
                _ => break,
            }
        }

        // Mark the entire region as one free block
        let total_chunks = region_end - region_start;
        if total_chunks > 0 {
            self.mark_chunks_free(blocks, region_start, total_chunks);
        }
    }

    /// Clears memory in the specified chunk range
    #[cfg(feature = "zero-on-free")]
    fn clear_memory(&self, pool: &mut [u8], chunk_start: usize, chunk_count: usize) {
        let byte_start = chunk_start * Self::CHUNK_SIZE;
        let byte_count = chunk_count * Self::CHUNK_SIZE;
        let byte_end = (byte_start + byte_count).min(N);
        
        pool[byte_start..byte_end].fill(0);
    }

    /// Updates statistics after successful allocation
    fn update_stats_after_allocation(&self, chunks: usize) {
        let mut stats = self.stats.lock();
        let bytes = chunks * Self::CHUNK_SIZE;
        
        stats.allocated_blocks += 1;
        stats.allocated_bytes += bytes;
        stats.free_bytes = stats.free_bytes.saturating_sub(bytes);
        
        // Recalculate fragmentation and free blocks
        self.recalculate_fragmentation_stats(&mut stats);
    }

    /// Updates statistics after successful deallocation
    fn update_stats_after_deallocation(&self, chunks: usize) {
        let mut stats = self.stats.lock();
        let bytes = chunks * Self::CHUNK_SIZE;
        
        stats.allocated_blocks = stats.allocated_blocks.saturating_sub(1);
        stats.allocated_bytes = stats.allocated_bytes.saturating_sub(bytes);
        stats.free_bytes += bytes;
        
        // Recalculate fragmentation and free blocks
        self.recalculate_fragmentation_stats(&mut stats);
    }

    /// Recalculates fragmentation-related statistics
    fn recalculate_fragmentation_stats(&self, stats: &mut PoolStats) {
        let inner = self.inner.lock();
        let mut free_blocks = 0;
        let mut largest_free = 0;
        let mut i = 0;

        while i < M {
            if let BlockInfo::FreeStart(count) = inner.blocks[i] {
                free_blocks += 1;
                largest_free = largest_free.max(count * Self::CHUNK_SIZE);
                i += count;
            } else {
                i += 1;
            }
        }

        stats.free_blocks = free_blocks;
        stats.largest_free_block = largest_free;

        // Calculate fragmentation: (1 - largest_free/total_free) * 100
        if stats.free_bytes > 0 && free_blocks > 1 {
            let frag = ((stats.free_bytes - largest_free) * 10000) / stats.free_bytes;
            stats.fragmentation = frag;
        } else {
            stats.fragmentation = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_allocation() {
        type Alloc = MemoryPoolAllocator<1024, 64>;
        let allocator = Alloc::new();
        
        let layout = Layout::from_size_align(32, 8).unwrap();
        let ptr = allocator.try_allocate(layout).unwrap();
        assert!(!ptr.is_null());
        assert_eq!(ptr as usize % 8, 0); // Check alignment
        
        assert!(allocator.try_deallocate(ptr, layout).is_ok());
    }

    #[test]
    fn test_multiple_allocations() {
        type Alloc = MemoryPoolAllocator<1024, 64>;
        let allocator = Alloc::new();
        
        let layout = Layout::from_size_align(16, 8).unwrap();
        let mut ptrs = [0u8 as *mut u8; 10];
        
        // Allocate multiple blocks
        for i in 0..10 {
            match allocator.try_allocate(layout) {
                Ok(ptr) => ptrs[i] = ptr,
                Err(_) => break,
            }
        }
        
        assert!(!ptrs.is_empty());
        
        // Deallocate all
        for ptr in ptrs {
            assert!(allocator.try_deallocate(ptr, layout).is_ok());
        }
        
        // Check stats
        let stats = allocator.get_stats();
        assert_eq!(stats.allocated_blocks, 0);
        assert_eq!(stats.free_bytes, 1024);
    }

    #[test]
    fn test_coalescing() {
        type Alloc = MemoryPoolAllocator<256, 16>;
        let allocator = Alloc::new();
        
        let layout = Layout::from_size_align(16, 8).unwrap();
        
        // Allocate three adjacent blocks
        let ptr1 = allocator.try_allocate(layout).unwrap();
        let ptr2 = allocator.try_allocate(layout).unwrap();
        let ptr3 = allocator.try_allocate(layout).unwrap();
        
        // Deallocate in order that requires coalescing
        assert!(allocator.try_deallocate(ptr2, layout).is_ok());
        assert!(allocator.try_deallocate(ptr1, layout).is_ok());
        assert!(allocator.try_deallocate(ptr3, layout).is_ok());
        
        // Should have one large free block
        let stats = allocator.get_stats();
        assert_eq!(stats.free_blocks, 1);
        assert_eq!(stats.largest_free_block, 256);
    }

    #[test]
    fn test_alignment_handling() {
        type Alloc = MemoryPoolAllocator<512, 32>;
        let allocator = Alloc::new();
        
        // Test various alignments
        for align in [1, 2, 4, 8, 16, 32, 64].iter() {
            let layout = Layout::from_size_align(24, *align).unwrap();
            match allocator.try_allocate(layout) {
                Ok(ptr) => {
                    assert_eq!(ptr as usize % align, 0);
                    assert!(allocator.try_deallocate(ptr, layout).is_ok());
                }
                Err(_) => {} // Large alignments might fail
            }
        }
    }

    #[test]
    fn test_error_handling() {
        type Alloc = MemoryPoolAllocator<128, 8>;
        let allocator = Alloc::new();
        
        // Invalid layout - alignment not power of 2
        let stats_before = allocator.get_stats();
        assert!(allocator.try_allocate(Layout::from_size_align(16, 3).unwrap()).is_err());
        let stats_after = allocator.get_stats();
        assert_eq!(stats_after.allocation_errors, stats_before.allocation_errors + 1);
        
        // Allocation too large
        assert!(allocator.try_allocate(Layout::from_size_align(256, 8).unwrap()).is_err());
        
        // Invalid deallocation
        let fake_ptr = 0x1234 as *mut u8;
        assert!(allocator.try_deallocate(fake_ptr, Layout::from_size_align(16, 8).unwrap()).is_err());
    }

    #[test]
    fn test_fragmentation_stats() {
        type Alloc = MemoryPoolAllocator<256, 16>;
        let allocator = Alloc::new();
        
        let layout = Layout::from_size_align(16, 8).unwrap();
        
        // Create fragmentation by allocating every other block
        let mut ptrs = [0u8 as *mut u8; 8];
        for i in 0..8 {
            if let Ok(ptr) = allocator.try_allocate(layout) {
                if i % 2 == 0 {
                    ptrs[i] = ptr;
                } else {
                    // Immediately free to create gaps
                    allocator.try_deallocate(ptr, layout).unwrap();
                }
            }
        }
        
        let stats = allocator.get_stats();
        assert!(stats.fragmentation > 0);
        assert_eq!(stats.free_blocks, 4); // Should have 4 separate free blocks
        
        // Clean up
        for ptr in ptrs {
            allocator.try_deallocate(ptr, layout).unwrap();
        }
    }

    #[test]
    #[cfg(feature = "zero-on-free")]
    fn test_memory_zeroing() {
        type Alloc = MemoryPoolAllocator<64, 4>;
        let allocator = Alloc::new();
        
        let layout = Layout::from_size_align(16, 8).unwrap();
        let ptr = allocator.try_allocate(layout).unwrap();
        
        // Write some data
        unsafe {
            core::ptr::write_bytes(ptr, 0xFF, 16);
        }
        
        // Deallocate
        allocator.try_deallocate(ptr, layout).unwrap();
        
        // Reallocate same memory
        let ptr2 = allocator.try_allocate(layout).unwrap();
        assert_eq!(ptr, ptr2); // Should get same address
        
        // Check if zeroed
        let mut buffer = [0u8; 16];
        unsafe {
            core::ptr::copy_nonoverlapping(ptr2, buffer.as_mut_ptr(), 16);
        }
        assert!(buffer.iter().all(|&b| b == 0));
        
        allocator.try_deallocate(ptr2, layout).unwrap();
    }
}