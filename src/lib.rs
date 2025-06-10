// Copyright 2024 <your name or organization>
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your option.
// This file may not be copied, modified, or distributed except according to those terms.

#![no_std]

use core::{
    alloc::{GlobalAlloc, Layout}, 
    ptr::{null_mut, NonNull},
};
use parking_lot::Mutex;

/// A fixed-size memory allocator that manages memory in chunks/blocks
///
/// # Type Parameters
/// * `N` - Total size of the memory pool in bytes
/// * `M` - Number of blocks/chunks to divide the pool into
///
/// The allocator divides the memory pool into M equal-sized chunks, where each chunk
/// is N/M bytes. This means N must be exactly divisible by M (N % M == 0).
///
/// # Panics
/// The `new()` constructor will panic if N is not exactly divisible by M.
///
/// # Example
/// ```
/// use memory_pool_allocator::MemoryPoolAllocator;
///
/// // Create a 1024-byte pool divided into 64 chunks (16 bytes each)
/// let allocator = MemoryPoolAllocator::<1024, 64>::new();
/// ```
pub struct MemoryPoolAllocator<const N: usize, const M: usize>
where
    [u8; N]: Sized, // Ensure N is a valid size for an array
    [BlockInfo; M]: Sized, // Ensure M is a valid size for an array
{
    /// Inner mutex to protect the memory pool state
    /// This allows safe concurrent access to the pool
    /// across multiple threads.
    inner: Mutex<PoolInner<N, M>>,
}

struct PoolInner<const N: usize, const M: usize> {
    pool: [u8; N],
    blocks: [BlockInfo; M],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlockInfo {
    FreeStart(usize),
    FreeContinuation,
    AllocatedStart(usize),
    AllocatedContinuation,
}

unsafe impl<const N: usize, const M: usize> Sync for MemoryPoolAllocator<N, M> {}
unsafe impl<const N: usize, const M: usize> Send for MemoryPoolAllocator<N, M> {}

impl<const N: usize, const M: usize> MemoryPoolAllocator<N, M>
where 
    [u8; N]: Sized, // Ensure N is a valid size for an array
    [BlockInfo; M]: Sized, // Ensure M is a valid size for an array
{
    /// The size of each chunk in bytes
    const CHUNK_SIZE: usize = N / M;


    /// Creates a new memory pool allocator
    /// # Returns
    /// * `MemoryPoolAllocator<N, M>` - A new instance of the memory pool allocator
    /// 
    /// # Panics
    /// This function will panic if `N` is not exactly divisible by `M`.
    /// This ensures that the memory pool can be evenly divided into `M` chunks.
    pub const fn new() -> Self {
        assert!(N % M == 0);
        let pool = [0u8; N];
        let mut blocks = [BlockInfo::FreeContinuation; M];
        blocks[0] = BlockInfo::FreeStart(M);
        MemoryPoolAllocator { 
            inner: Mutex::new(PoolInner { pool, blocks }),
        }
    }

    #[inline]
    /// Converts a byte offset to a chunk index
    fn byte_to_chunk(byte_offset: usize) -> usize {
        byte_offset / Self::CHUNK_SIZE
    }

    #[inline]
    /// Converts a chunk index to a byte offset
    fn chunk_to_byte(chunk_index: usize) -> usize {
        chunk_index * Self::CHUNK_SIZE
    }

    #[inline]
    /// Converts a size in bytes to the number of chunks needed
    fn size_to_chunks(size: usize) -> usize {
        (size + Self::CHUNK_SIZE - 1) / Self::CHUNK_SIZE
    }

    /// Allocates a block of memory with the specified layout
    /// # Arguments
    /// * `layout` - The layout describing the size and alignment of the memory block
    /// 
    /// # Returns
    /// * `*mut u8` - Pointer to the allocated memory block, or null pointer if allocation fails
    /// 
    /// # Safety
    /// The caller must ensure that the pointer is valid and properly aligned for the requested layout.
    /// The caller is also responsible for deallocating the memory using `deallocate`.
    pub fn allocate(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        let alignment = layout.align();
        if size == 0 {
            return NonNull::dangling().as_ptr();
        }
        if !alignment.is_power_of_two() || alignment > N {
            return null_mut();
        }
        let chunks_needed = Self::size_to_chunks(size);
        let mut inner = self.inner.lock();
        let pool_addr = inner.pool.as_mut_ptr() as usize;
        let mut chunk_idx = 0;
        while chunk_idx < inner.blocks.len() {
            if let BlockInfo::FreeStart(block_chunks) = inner.blocks[chunk_idx] {
                let block_byte_start = Self::chunk_to_byte(chunk_idx);
                let block_addr = pool_addr + block_byte_start;
                let aligned_addr = align_up(block_addr, alignment);
                let alignment_offset_bytes = aligned_addr - block_addr;
                let alignment_offset_chunks = Self::size_to_chunks(alignment_offset_bytes);
                if alignment_offset_chunks + chunks_needed <= block_chunks {
                    return self.allocate_in_block(
                        &mut inner, 
                        chunk_idx, 
                        block_chunks, 
                        alignment_offset_chunks, 
                        chunks_needed,
                        aligned_addr
                    );
                }
                chunk_idx += block_chunks;
            } else {
                chunk_idx += 1;
            }
        }
        null_mut()
    }

    /// Allocates memory in a specific block, handling alignment and marking the blocks
    /// # Arguments
    /// * `inner` - Mutable reference to the inner state of the memory pool
    /// * `block_start_chunk` - The starting chunk index of the block to allocate from
    /// * `block_size_chunks` - The total number of chunks in the block
    /// * `alignment_offset_chunks` - The number of chunks to skip for alignment
    /// * `alloc_size_chunks` - The number of chunks to allocate
    /// * `aligned_addr` - The aligned address to start the allocation from
    /// 
    /// # Returns
    /// * `*mut u8` - Pointer to the allocated memory block
    /// 
    /// This function updates the block information to reflect the allocation,
    /// marking the allocated range and any remaining free space.
    /// It also handles the case where the allocation requires skipping some initial chunks
    /// for alignment purposes.
    /// 
    /// # Safety
    /// The caller must ensure that the `inner` state is properly locked and that the
    /// `block_start_chunk` and `block_size_chunks` are valid indices within the `blocks` array.
    /// The caller must also ensure that the `aligned_addr` is a valid address within the pool.
    fn allocate_in_block(
        &self,
        inner: &mut PoolInner<N, M>,
        block_start_chunk: usize,
        block_size_chunks: usize,
        alignment_offset_chunks: usize,
        alloc_size_chunks: usize,
        aligned_addr: usize,
    ) -> *mut u8 {
        let alloc_start_chunk = block_start_chunk + alignment_offset_chunks;
        if alignment_offset_chunks > 0 {
            self.mark_chunk_range(
                &mut inner.blocks, 
                block_start_chunk, 
                alignment_offset_chunks, 
                BlockInfo::FreeStart(alignment_offset_chunks)
            );
        }
        self.mark_chunk_range(
            &mut inner.blocks, 
            alloc_start_chunk, 
            alloc_size_chunks, 
            BlockInfo::AllocatedStart(alloc_size_chunks)
        );
        let remaining_chunks = block_size_chunks - alignment_offset_chunks - alloc_size_chunks;
        if remaining_chunks > 0 {
            let remaining_start_chunk = alloc_start_chunk + alloc_size_chunks;
            self.mark_chunk_range(
                &mut inner.blocks, 
                remaining_start_chunk, 
                remaining_chunks, 
                BlockInfo::FreeStart(remaining_chunks)
            );
        }
        aligned_addr as *mut u8
    }

    /// Marks a range of chunks with a specific block type
    /// # Arguments
    /// * `blocks` - Mutable slice of block information
    /// * `start_chunk` - The starting chunk index to mark
    /// * `chunk_count` - The number of chunks to mark
    /// * `block_type` - The type of block to mark the range with
    /// 
    /// This function updates the `blocks` slice to reflect the new state of the memory pool.
    /// It handles both the start of a new block and continuation blocks.
    /// 
    /// # Safety
    /// The caller must ensure that `start_chunk` is a valid index within the `blocks` slice
    /// and that `chunk_count` does not exceed the bounds of the slice.
    /// If `start_chunk` is out of bounds or `chunk_count` is zero, the function does nothing.
    #[inline]
    #[allow(clippy::cast_possible_wrap)]
    #[allow(clippy::cast_sign_loss)]
    fn mark_chunk_range(
        &self, 
        blocks: &mut [BlockInfo], 
        start_chunk: usize, 
        chunk_count: usize, 
        block_type: BlockInfo
    ) {
        if start_chunk >= blocks.len() || chunk_count == 0 {
            return;
        }
        let end_chunk = core::cmp::min(start_chunk + chunk_count, blocks.len());
        blocks[start_chunk] = block_type;
        let continuation = match block_type {
            BlockInfo::FreeStart(_) => BlockInfo::FreeContinuation,
            BlockInfo::AllocatedStart(_) => BlockInfo::AllocatedContinuation,
            _ => return,
        };
        for i in (start_chunk + 1)..end_chunk {
            blocks[i] = continuation;
        }
    }

    /// Deallocates a previously allocated memory block
    /// # Arguments
    /// * `ptr` - Pointer to the memory block to deallocate
    /// * `layout` - The layout describing the size and alignment of the memory block
    /// 
    /// This function checks if the pointer is valid and within the bounds of the memory pool.
    /// If the pointer is valid, it marks the corresponding chunks as free and attempts to coalesce
    /// adjacent free blocks to reduce fragmentation.
    /// 
    /// # Safety
    /// The caller must ensure that the pointer was allocated by this allocator and that the layout
    /// matches the original allocation. The pointer must not be null or point outside the pool bounds.
    /// If the pointer is null or the layout size is zero, the function does nothing.
    /// If the pointer is outside the bounds of the pool, it also does nothing.
    /// If the pointer does not correspond to an allocated block, it returns without doing anything.
    /// 
    /// # Note
    /// This function does not check if the pointer was previously allocated by this allocator.
    /// It assumes that the caller is responsible for ensuring that the pointer is valid.
    /// It also does not check if the layout size is larger than the allocated size.
    /// If the layout size is larger than the allocated size, it will not clear the entire block.
    /// It only clears the bytes that were actually allocated.
    /// If the pointer is not aligned to the chunk size, it will not deallocate anything.
    pub fn deallocate(&self, ptr: *mut u8, layout: Layout) {
        if ptr.is_null() || layout.size() == 0 {
            return;
        }
        let mut inner = self.inner.lock();
        let pool_addr = inner.pool.as_mut_ptr() as usize;
        let ptr_addr = ptr as usize;
        if ptr_addr < pool_addr || ptr_addr >= pool_addr + N {
            return;
        }
        let byte_offset = ptr_addr - pool_addr;
        let chunk_idx = Self::byte_to_chunk(byte_offset);
        if chunk_idx >= inner.blocks.len() {
            return;
        }
        match inner.blocks[chunk_idx] {
            BlockInfo::AllocatedStart(chunk_count) => {
                let allocated_bytes = chunk_count * Self::CHUNK_SIZE;
                if layout.size() > allocated_bytes {
                    return;
                }
                let actual_byte_start = Self::chunk_to_byte(chunk_idx);
                let clear_size = core::cmp::min(allocated_bytes, N - actual_byte_start);
                for i in 0..clear_size {
                    if actual_byte_start + i < N {
                        inner.pool[actual_byte_start + i] = 0;
                    }
                }
                self.mark_chunk_range(
                    &mut inner.blocks, 
                    chunk_idx, 
                    chunk_count, 
                    BlockInfo::FreeStart(chunk_count)
                );
                self.coalesce_free_blocks(&mut inner.blocks, chunk_idx);
            },
            _ => return,
        }
    }

    
    /// Coalesces adjacent free blocks into a single larger free block
    /// # Arguments
    /// * `blocks` - Mutable slice of block information
    /// * `start_chunk` - The starting chunk index to begin coalescing from
    /// 
    /// This function checks for adjacent free blocks before and after the specified start chunk.
    /// It merges them into a single free block if possible, updating the block information accordingly.
    /// 
    /// # Safety
    /// The caller must ensure that `start_chunk` is a valid index within the `blocks` slice.
    /// If `start_chunk` is out of bounds or does not point to a free block, the function does nothing.
    fn coalesce_free_blocks(&self, blocks: &mut [BlockInfo], start_chunk: usize) {
        // Move region_start backward to the start of the free region
        let mut region_start = start_chunk;
        while region_start > 0 {
            match blocks[region_start - 1] {
                BlockInfo::FreeStart(_) | BlockInfo::FreeContinuation => region_start -= 1,
                _ => break,
            }
        }
        // Move region_end forward to the end of the free region
        let mut region_end = start_chunk;
        while region_end < blocks.len() {
            match blocks[region_end] {
                BlockInfo::FreeStart(count) => region_end += count,
                BlockInfo::FreeContinuation => region_end += 1,
                _ => break,
            }
        }
        let total_chunks = region_end - region_start;
        if total_chunks > 0 {
            self.mark_chunk_range(blocks, region_start, total_chunks, BlockInfo::FreeStart(total_chunks));
        }
    }

    /// Returns the total amount of free space in bytes
    /// 
    /// This method iterates through all blocks and sums up the size of free blocks.
    /// The result is guaranteed to not exceed the total pool size N.
    /// 
    /// # Returns
    /// * `usize` - Total number of free bytes
    pub fn get_free_space(&self) -> usize {
        let inner = self.inner.lock();
        let mut total_free_chunks = 0;
        let mut chunk_idx = 0;
        
        while chunk_idx < inner.blocks.len() {
            match inner.blocks[chunk_idx] {
                BlockInfo::FreeStart(chunk_count) => {
                    total_free_chunks += chunk_count;
                    chunk_idx += chunk_count;
                },
                BlockInfo::AllocatedStart(chunk_count) => {
                    chunk_idx += chunk_count;
                },
                _ => chunk_idx += 1,
            }
        }
        
        core::cmp::min(total_free_chunks * Self::CHUNK_SIZE, N)
    }

    /// Returns the total amount of allocated space in bytes
    /// 
    /// Calculated as the difference between total pool size and free space.
    /// 
    /// # Returns
    /// * `usize` - Total number of allocated bytes
    #[inline]
    pub fn get_used_space(&self) -> usize {
        N - self.get_free_space()
    }

    /// Returns detailed statistics about the memory pool's current state
    /// 
    /// Gathers information about:
    /// - Total size and chunk size
    /// - Number of allocated and free blocks
    /// - Amount of allocated and free bytes
    /// - Fragmentation metrics
    /// 
    /// # Returns
    /// * `PoolStats` - Structure containing various statistics
    pub fn get_stats(&self) -> PoolStats {
        let inner = self.inner.lock();
        let mut stats = PoolStats::default();
        let mut chunk_idx = 0;

        while chunk_idx < inner.blocks.len() {
            match inner.blocks[chunk_idx] {
                BlockInfo::FreeStart(chunk_count) => {
                    let byte_count = core::cmp::min(
                        chunk_count * Self::CHUNK_SIZE,
                        N - Self::chunk_to_byte(chunk_idx)
                    );
                    stats.free_bytes += byte_count;
                    stats.free_blocks += 1;
                    if byte_count > stats.largest_free_block {
                        stats.largest_free_block = byte_count;
                    }
                    chunk_idx += chunk_count;
                },
                BlockInfo::AllocatedStart(chunk_count) => {
                    let byte_count = core::cmp::min(
                        chunk_count * Self::CHUNK_SIZE,
                        N - Self::chunk_to_byte(chunk_idx)
                    );
                    stats.allocated_bytes += byte_count;
                    stats.allocated_blocks += 1;
                    chunk_idx += chunk_count;
                },
                _ => chunk_idx += 1,
            }
        }

        stats.total_size = N;
        stats.chunk_size = Self::CHUNK_SIZE;
        stats.total_chunks = M;
        stats.fragmentation = if stats.free_bytes > 0 && stats.free_blocks > 0 {
            ((stats.free_bytes - stats.largest_free_block) * 10000) / stats.free_bytes
        } else {
            0
        };

        stats
    }

    #[inline]
    /// Returns the size of each chunk in bytes
    pub const fn chunk_size() -> usize {
        Self::CHUNK_SIZE
    }

    #[inline]
    /// Returns the total number of chunks in the memory pool
    pub const fn chunk_count() -> usize {
        M
    }
}

#[derive(Debug, Default)]
/// Statistics structure for the memory pool allocator
/// Contains various metrics about the current state of the memory pool
pub struct PoolStats {
    pub total_size: usize,
    pub allocated_bytes: usize,
    pub free_bytes: usize,
    pub allocated_blocks: usize,
    pub free_blocks: usize,
    pub largest_free_block: usize,
    // fragmentation represented as percentage * 100
    // e.g., 2500 means 25% fragmented
    pub fragmentation: usize,
    pub chunk_size: usize,
    pub total_chunks: usize,
}

#[inline]
/// Aligns an address up to the nearest multiple of the specified alignment
/// # Arguments
/// * `addr` - The address to align
/// * `align` - The alignment value (must be a power of two)
////// # Returns
/// * `usize` - The aligned address
fn align_up(addr: usize, align: usize) -> usize {
    (addr + align - 1) & !(align - 1)
}

/// Global allocator implementation for the MemoryPoolAllocator
/// This allows the allocator to be used as the default global allocator in Rust applications.
unsafe impl<const N: usize, const M: usize> GlobalAlloc for MemoryPoolAllocator<N, M> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.allocate(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.deallocate(ptr, layout);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_calculations() {
        type Alloc1 = MemoryPoolAllocator<1024, 64>;
        type Alloc2 = MemoryPoolAllocator<1008, 63>;
        assert_eq!(Alloc1::chunk_size(), 16);
        assert_eq!(Alloc1::chunk_count(), 64);
        assert_eq!(Alloc2::chunk_count(), 63);
    }

    #[test]
    fn test_basic_allocation() {
        type Alloc = MemoryPoolAllocator<1024, 64>;
        let allocator = Alloc::new();
        let layout = Layout::from_size_align(64, 8).unwrap();
        let ptr = allocator.allocate(layout);
        assert!(!ptr.is_null());
        assert_eq!(ptr as usize % 8, 0);
        allocator.deallocate(ptr, layout);
    }

    #[test]
    fn test_small_allocations() {
        type Alloc = MemoryPoolAllocator<1024, 64>;
        let allocator = Alloc::new();
        for size in [1, 4, 8, 12, 15] {
            let layout = Layout::from_size_align(size, 1).unwrap();
            let ptr = allocator.allocate(layout);
            assert!(!ptr.is_null());
            allocator.deallocate(ptr, layout);
        }
    }

    #[test]
    fn test_alignment_with_chunks() {
        type Alloc = MemoryPoolAllocator<1024, 64>;
        let allocator = Alloc::new();
        for align in [1, 2, 4, 8, 16, 32] {
            let layout = Layout::from_size_align(32, align).unwrap();
            let ptr = allocator.allocate(layout);
            assert!(!ptr.is_null());
            assert_eq!(ptr as usize % align, 0);
            allocator.deallocate(ptr, layout);
        }
    }

    #[test]
    fn test_memory_overhead_reduction() {
        type Alloc = MemoryPoolAllocator<4096, 256>;
        let allocator = Alloc::new();
        let stats = allocator.get_stats();
        assert_eq!(stats.total_chunks, 256);
        assert_eq!(stats.chunk_size, 16);
        assert_eq!(stats.total_size, 4096);
    }

    #[test]
    fn test_coalescing_with_chunks() {
        type Alloc = MemoryPoolAllocator<1024, 64>;
        let allocator = Alloc::new();
        let layout = Layout::from_size_align(64, 8).unwrap();
        let ptr1 = allocator.allocate(layout);
        let ptr2 = allocator.allocate(layout);
        let ptr3 = allocator.allocate(layout);
        let initial_free = allocator.get_free_space();
        allocator.deallocate(ptr2, layout);
        allocator.deallocate(ptr1, layout);
        allocator.deallocate(ptr3, layout);
        assert!(allocator.get_free_space() >= initial_free + 192);
    }

    #[test]
    fn test_stats_with_chunks() {
        type Alloc = MemoryPoolAllocator<1024, 64>;
        let allocator = Alloc::new();
        let stats = allocator.get_stats();
        assert_eq!(stats.total_size, 1024);
        assert_eq!(stats.chunk_size, 16);
        assert_eq!(stats.total_chunks, 64);
        assert_eq!(stats.free_bytes, 1024);
        assert_eq!(stats.allocated_bytes, 0);
        let layout = Layout::from_size_align(100, 8).unwrap();
        let _ptr = allocator.allocate(layout);
        let stats = allocator.get_stats();
        assert_eq!(stats.allocated_bytes, 112);
        assert_eq!(stats.free_bytes, 1024 - 112);
    }

    #[test]
    fn test_partial_chunk_deallocation() {
        type Alloc = MemoryPoolAllocator<1024, 64>;
        let allocator = Alloc::new();
        let layout = Layout::from_size_align(10, 1).unwrap();
        let ptr = allocator.allocate(layout);
        assert!(!ptr.is_null());
        allocator.deallocate(ptr, layout);
        assert_eq!(allocator.get_free_space(), 1024);
    }

    #[test]
    fn test_zero_size_allocation() {
        type Alloc = MemoryPoolAllocator<1024, 64>;
        let allocator = Alloc::new();
        let layout = Layout::from_size_align(0, 1).unwrap();
        let ptr = allocator.allocate(layout);
        assert!(!ptr.is_null());
        allocator.deallocate(ptr, layout);
    }

    #[test]
    fn test_large_allocation_failure() {
        type Alloc = MemoryPoolAllocator<96, 6>;
        let allocator = Alloc::new();
        let layout = Layout::from_size_align(200, 8).unwrap();
        let ptr = allocator.allocate(layout);
        assert!(ptr.is_null());
    }

    #[test]
    fn test_memory_efficiency() {
        type Alloc = MemoryPoolAllocator<4096, 256>;
        let allocator = Alloc::new();
        let layout = Layout::from_size_align(48, 16).unwrap();
        let ptr1 = allocator.allocate(layout);
        let ptr2 = allocator.allocate(layout);
        let stats = allocator.get_stats();
        assert_eq!(stats.allocated_blocks, 2);
        assert_eq!(stats.allocated_bytes, 96);
        allocator.deallocate(ptr1, layout);
        allocator.deallocate(ptr2, layout);
    }
}