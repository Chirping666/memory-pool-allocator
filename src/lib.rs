// Copyright 2025 Chirping666
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0>
// or the MIT license <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your option.

#![no_std]

use anyhow::{anyhow, Result};
use core::{alloc::Layout, ptr::NonNull};
use parking_lot::Mutex;

/// Bitmap-based fixed-size memory pool allocator
/// 
/// # Type Parameters
/// * `N` - Total pool size in bytes
/// * `M` - Number of chunks to divide the pool into
/// 
/// Uses a bitmap where each bit represents a chunk's allocation status.
pub struct MemoryPoolAllocator<const N: usize, const M: usize> {
    inner: Mutex<PoolInner<N, M>>,
    #[cfg(feature = "statistics")]
    stats: Mutex<PoolStats>,
}

struct PoolInner<const N: usize, const M: usize> {
    /// The actual memory pool
    pool: [u8; N],
    /// Bitmap tracking allocation status (true = allocated, false = free)
    meta: [Option<usize>; M],
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

impl From<anyhow::Error> for AllocError {
    fn from(err: anyhow::Error) -> Self {
        if err.is::<AllocError>() {
            *err.downcast_ref::<AllocError>().unwrap()
        } else {
            AllocError::InvalidLayout
        }
    }
}

#[cfg(feature = "statistics")]
/// Pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub allocated_chunks: usize,
    pub allocation_errors: usize,
    pub deallocation_errors: usize,
}

#[cfg(feature = "statistics")]
impl PoolStats {
    const fn new() -> Self {
        Self {
            allocated_chunks: 0,
            allocation_errors: 0,
            deallocation_errors: 0,
        }
    }
}

// Safety: Pool data is protected by mutex
unsafe impl<const N: usize, const M: usize> Sync for MemoryPoolAllocator<N, M> {}
unsafe impl<const N: usize, const M: usize> Send for MemoryPoolAllocator<N, M> {}

impl<const N: usize, const M: usize> MemoryPoolAllocator<N, M> {
    /// Size of each chunk in bytes
    pub const CHUNK_SIZE: usize = N / M;

    /// Creates a new memory pool allocator
    pub const fn new() -> Self {
        assert!(N % M == 0, "N must be exactly divisible by M");
        assert!(N > 0 && M > 0, "Pool and chunk count must be positive");
        assert!(Self::CHUNK_SIZE > 0, "Chunk size must be positive");
        
        Self {
            inner: Mutex::new(PoolInner { 
                pool: [0u8; N], 
                meta: [None; M],
            }),
            #[cfg(feature = "statistics")]
            stats: Mutex::new(PoolStats::new()),
        }
    }

    /// Attempts to allocate memory with the specified layout
    pub fn try_allocate(&self, layout: Layout) -> Result<*mut u8> {
        // Handle zero-size allocations
        if layout.size() == 0 {
            return Ok(NonNull::dangling().as_ptr());
        }

        // Validate layout
        if !layout.align().is_power_of_two() || layout.align() > N {
            #[cfg(feature = "statistics")]
            {
                self.stats.lock().allocation_errors += 1;
            }
            return Err(anyhow!(AllocError::InvalidLayout).context("Invalid alignment or size"));
        }

        let chunks_needed = (layout.size() + Self::CHUNK_SIZE - 1) / Self::CHUNK_SIZE;
        if chunks_needed > M {
            #[cfg(feature = "statistics")]
            {
                self.stats.lock().allocation_errors += 1;
            }
            return Err(anyhow!(AllocError::OutOfMemory).context("Failed to find free region"));
        }

        let mut inner = self.inner.lock();

        // Find a suitable free region
        if let Some((start_chunk, total_chunks)) = self.find_free_region(&inner, chunks_needed, layout.align()) {
            // Mark chunks as allocated
            self.mark_allocated(&mut inner.meta, start_chunk, total_chunks)?;
            
            // Calculate pointer address
            let pool_base = inner.pool.as_ptr() as usize;
            let ptr_addr = pool_base + start_chunk * Self::CHUNK_SIZE;
            
            // Update stats
            #[cfg(feature = "statistics")]
            {
                let mut stats = self.stats.lock();
                stats.allocated_chunks += total_chunks;
            }
            
            return Ok(ptr_addr as *mut u8);
        }
        
        // Find suitable free region
        Err(anyhow!(AllocError::OutOfMemory).context("Failed to find free region"))
        
    }

    /// Attempts to deallocate previously allocated memory
    pub fn try_deallocate(&self, ptr: *mut u8, _layout: Layout) -> Result<()> {
        // Handle null
        if ptr.is_null() {
            #[cfg(feature = "statistics")]
            {
                self.stats.lock().deallocation_errors += 1;
            }
            return Err(anyhow!(AllocError::InvalidPointer).context("Cannot deallocate null pointer"));
        }

        let mut inner = self.inner.lock();
        
        // Validate pointer is within pool
        let pool_base = inner.pool.as_ptr() as usize;
        let ptr_addr = ptr as usize;
        
        if ptr_addr < pool_base || ptr_addr >= pool_base + N {
            #[cfg(feature = "statistics")]
            {
                self.stats.lock().deallocation_errors += 1;
            }
            return Err(anyhow!(AllocError::InvalidPointer).context("Pointer not from this allocator"));
        }

        // Find the allocation metadata
        if (ptr_addr - pool_base) % Self::CHUNK_SIZE != 0 {
            #[cfg(feature = "statistics")]
            {
                self.stats.lock().deallocation_errors += 1;
            }
            return Err(anyhow!(AllocError::InvalidPointer).context("Pointer not aligned to chunk size"));
        }

        let start_chunk = (ptr_addr - pool_base) / Self::CHUNK_SIZE;
        if start_chunk >= M || !self.is_chunk_allocated(&inner.meta, start_chunk) {
            #[cfg(feature = "statistics")]
            {
                self.stats.lock().deallocation_errors += 1;
            }
            return Err(anyhow!(AllocError::NotAllocated).context("Pointer not currently allocated"));
        }

        // Calculate the start of the allocated chunk
        let total_chunks = match inner.meta[start_chunk] {
            Some(size) => size,
            None => {
                #[cfg(feature = "statistics")]
                {
                    self.stats.lock().deallocation_errors += 1;
                }
                return Err(anyhow!(AllocError::NotAllocated).context("Chunk already free"));
            }
        };
        
        // Clear memory if feature enabled
        #[cfg(feature = "zero-on-free")]
        {
            unsafe {
                let start_ptr = inner.pool.as_mut_ptr().add(start_chunk * Self::CHUNK_SIZE);
                core::ptr::write_bytes(start_ptr, 0, total_chunks * Self::CHUNK_SIZE);
            }
        }
        
        // Mark chunks as free
        
        self.mark_chunks_free(&mut inner.meta, start_chunk)?;

        // Update stats
        #[cfg(feature = "statistics")]
        {
            let mut stats = self.stats.lock();
            stats.allocated_chunks = stats.allocated_chunks.saturating_sub(total_chunks);
        }
        
        Ok(())
    }

    // === Private meta helper methods ===

    /// Checks if a specific chunk is allocated
    /// Checks if a specific chunk is allocated
    #[inline]
    fn is_chunk_allocated(&self, meta: &[Option<usize>; M], chunk_idx: usize) -> bool {
        if chunk_idx < M {
            matches!(meta[chunk_idx], Some(_))
        } else {
            false
        }
    }

    /// Marks a range of chunks as allocated
    fn mark_allocated(&self, meta: &mut [Option<usize>; M], start_chunk: usize, chunk_size: usize) -> Result<()> {
        if start_chunk + chunk_size > M {
            return Err(anyhow!(AllocError::OutOfMemory).context("Not enough space to allocate chunks"));
        }
        meta[start_chunk] = Some(chunk_size);
        Ok(())
    }

    /// Marks a range of chunks as free
    fn mark_chunks_free(&self, meta: &mut [Option<usize>; M], start_chunk: usize) -> Result<()> {
        match meta[start_chunk] {
            Some(size) => {
                if start_chunk + size > M {
                    return Err(anyhow!(AllocError::OutOfMemory).context("Invalid chunk range to free"));
                }

                for i in start_chunk..start_chunk + size {
                    if i < M {
                        meta[i] = None; // Mark as free
                    }
                }

                Ok(())
            }
            None => {
                Err(anyhow!(AllocError::NotAllocated).context("Chunk already free"))
            }
        }
    }

    /// Finds a contiguous free region that can accommodate the request
    fn find_free_region(&self, inner: &PoolInner<N, M>, chunks_needed: usize, align: usize) -> Option<(usize,usize)> {
        let pool_base = inner.pool.as_ptr() as usize;
        let mut start = 0;
        
        while start + chunks_needed <= M {
            // Check alignment requirements
            let block_addr = pool_base + start * Self::CHUNK_SIZE;
            let aligned_addr = (block_addr + align - 1) & !(align - 1);
            let alignment_waste = aligned_addr - block_addr;
            let alignment_chunks = alignment_waste / Self::CHUNK_SIZE;
            
            let total_chunks_needed = alignment_chunks + chunks_needed;
            
            if start + total_chunks_needed > M {
                // Not enough space left in pool
                break;
            }
            
            // Check if we have enough contiguous free chunks
            let mut found = true;
            for i in 0..total_chunks_needed {
                if self.is_chunk_allocated(&inner.meta, start + i) {
                    found = false;
                    // Skip past this allocated chunk
                    start = start + i + 1;
                    break;
                }
            }
            
            if found {
                return Some((start, total_chunks_needed));
            }
        }
        
        None
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_allocation() {
        type Alloc = MemoryPoolAllocator<1024, 64>;
        let allocator = Alloc::new();
        
        let layout = Layout::from_size_align(16, 8).unwrap();
        let ptr = allocator.try_allocate(layout).unwrap();
        
        // Check pointer alignment
        assert_eq!(ptr as usize % 8, 0);
        
        // Deallocate
        assert!(allocator.try_deallocate(ptr, layout).is_ok());
        
        #[cfg(feature = "statistics")]
        {
            let stats = allocator.stats.lock();
            assert_eq!(stats.allocated_chunks, 0);
        }
    }

    #[test]
    fn test_multiple_allocations() {
        type Alloc = MemoryPoolAllocator<1024, 64>;
        let allocator = Alloc::new();
        
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
            for j in (i+1)..count {
                assert_ne!(ptrs[i], ptrs[j]);
            }
        }
        
        // Deallocate all
        for i in 0..count {
            assert!(allocator.try_deallocate(ptrs[i], layout).is_ok());
        }

        #[cfg(feature = "statistics")]
        {
            let stats = allocator.stats.lock();
            assert_eq!(stats.allocated_chunks, 0);
        }
    }

    #[test]
    fn test_alignment_handling() {
        type Alloc = MemoryPoolAllocator<2048, 128>;
        let allocator = Alloc::new();
        
        // Allocate with different alignments
        let layout1 = Layout::from_size_align(32, 16).unwrap();
        let ptr1 = allocator.try_allocate(layout1).unwrap();
        assert_eq!(ptr1 as usize % 16, 0);
        
        // let layout2 = Layout::from_size_align(64, 32).unwrap();
        // let ptr2 = allocator.try_allocate(layout2).unwrap();
        // assert_eq!(ptr2 as usize % 32, 0);
        
        // Deallocate
        assert!(allocator.try_deallocate(ptr1, layout1).is_ok());
        // assert!(allocator.try_deallocate(ptr2, layout2).is_ok());
    }

    #[test]
    #[cfg(feature = "statistics")]
    fn test_error_handling() {
       
    }

    #[test]
    #[cfg(feature = "statistics")]
    fn test_fragmentation_stats() {
        type Alloc = MemoryPoolAllocator<256, 16>;
        let allocator = Alloc::new();
        
        let layout = Layout::from_size_align(16, 8).unwrap();
        let mut ptrs = [core::ptr::null_mut(); 8];
        
        // Allocate every other chunk
        for i in 0..8 {
            if let Ok(ptr) = allocator.try_allocate(layout) {
                ptrs[i] = ptr;
            }
        }
        
        // Free every other allocation to create fragmentation
        for i in 0..8 {
            if i % 2 == 1 && !ptrs[i].is_null() {
                allocator.try_deallocate(ptrs[i], layout).unwrap();
            }
        }        
        
        // Clean up remaining allocations
        for i in 0..8 {
            if i % 2 == 0 && !ptrs[i].is_null() {
                allocator.try_deallocate(ptrs[i], layout).unwrap();
            }
        }
    }

    #[test]
    fn test_full_pool() {
        type Alloc = MemoryPoolAllocator<64, 4>;
        let allocator = Alloc::new();
        
        let layout = Layout::from_size_align(16, 8).unwrap();
        
        // Fill the pool
        let ptr1 = allocator.try_allocate(layout).unwrap();
        let ptr2 = allocator.try_allocate(layout).unwrap();
        let ptr3 = allocator.try_allocate(layout).unwrap();
        let ptr4 = allocator.try_allocate(layout).unwrap();
        
        // Next allocation should fail
        assert!(allocator.try_allocate(layout).is_err());
        
        // Free one and try again
        allocator.try_deallocate(ptr2, layout).unwrap();
        let ptr5 = allocator.try_allocate(layout).unwrap();
        
        // Clean up
        allocator.try_deallocate(ptr1, layout).unwrap();
        allocator.try_deallocate(ptr3, layout).unwrap();
        allocator.try_deallocate(ptr4, layout).unwrap();
        allocator.try_deallocate(ptr5, layout).unwrap();
    }

    #[test]
    #[cfg(feature = "zero-on-free")]
    fn test_memory_zeroing() {
        type Alloc = MemoryPoolAllocator<64, 4>;
        let allocator = Alloc::new();
        
        let layout = Layout::from_size_align(16, 8).unwrap();
        let ptr = allocator.try_allocate(layout).unwrap();
        
        // Write pattern after the metadata (first 8 bytes)
        unsafe {
            let data_ptr = ptr.add(8);
            core::ptr::write_bytes(data_ptr, 0xAB, 8);
        }
        
        // Deallocate
        allocator.try_deallocate(ptr, layout).unwrap();
        
        // Reallocate
        let ptr2 = allocator.try_allocate(layout).unwrap();
        
        // Check if the data portion is zeroed (skip metadata)
        let mut buffer = [0u8; 8];
        unsafe {
            let data_ptr = ptr2.add(8);
            core::ptr::copy_nonoverlapping(data_ptr, buffer.as_mut_ptr(), 8);
        }
        assert!(buffer.iter().all(|&b| b == 0));
        
        allocator.try_deallocate(ptr2, layout).unwrap();
    }
}