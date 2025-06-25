//! # memory-pool-allocator
//!
//! A no_std, thread-safe, array chunk tracking-based fixed-size memory pool allocator.
//!
//! ## Features
//! - User-provided memory pool (via `*mut u8`)
//! - Chunked allocation with array-based chunk tracking
//! - Optional statistics and zero-on-free features
//! - Thread-safe via `parking_lot::Mutex`
//!
//! ### Default Features
//! - **zero-on-free**: Zeroes memory of each allocation when it is deallocated.
//! - **zero-on-drop**: Zeroes the entire memory pool when the allocator is dropped.
//! - **statistics**: Tracks allocation and deallocation statistics (number of allocated chunks, allocation/deallocation errors).
//!
//! ## Type Parameters
//! - `N`: Total pool size in bytes
//! - `M`: Number of chunks to divide the pool into
//!
//! ## Safety
//! - The user must provide a pointer to a memory region of at least `N` bytes, aligned to the maximum alignment required by allocations.
//! - The allocator does not manage the lifetime of the memory pool; the user is responsible for ensuring it is valid for the allocator's lifetime.
//! - If the pool is not sufficiently aligned, allocations with higher alignment requirements may fail or result in undefined behavior.
//!
//! ## Example
//! ```rust
//! # use memory_pool_allocator::MemoryPoolAllocator;
//! # use core::alloc::Layout;
//! #[repr(align(64))]
//! struct Aligned {
//!     mem: [u8; 1024]
//! }
//! let mut aligned = Aligned { mem: [0; 1024] };
//! let allocator = unsafe { MemoryPoolAllocator::<1024, 64>::new(aligned.mem.as_mut_ptr()) };
//! let layout = Layout::from_size_align(128, 64).unwrap();
//! let ptr = allocator.try_allocate(layout).unwrap();
//! assert_eq!(ptr as usize % 64, 0);
//! allocator.try_deallocate(ptr).unwrap();
//! ```
//!
//! ## License
//! Licensed under either of Apache License, Version 2.0 or MIT license at your option.

// Copyright 2025 Chirping666
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0>
// or the MIT license <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your option.

#![no_std]

use anyhow::{anyhow, Result};
use core::{alloc::Layout, ptr::NonNull};
use parking_lot::Mutex;

/// Array chunk tracking-based fixed-size memory pool allocator
/// 
/// # Type Parameters
/// * `N` - Total pool size in bytes
/// * `M` - Number of chunks to divide the pool into
/// 
pub struct MemoryPoolAllocator<const N: usize, const M: usize> {
    inner: Mutex<PoolInner<N, M>>,
    #[cfg(feature = "statistics")]
    stats: Mutex<PoolStats>,
}

struct PoolInner<const N: usize, const M: usize> {
    /// Pointer to the actual memory pool (user-provided)
    pool: *mut u8,
    /// Allocation tracking
    meta: [MetaInfo; M],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MetaInfo {
    /// Start of a number of free chunks
    Free(usize),
    
    FreeContinuation,
    
    AllocStart {
        size: usize, // Size of the allocation
        ptr_offset: usize, // Offset from the chunk base
    },
    /// Chunk is allocated
    AllocContinuation,
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
    // Compile-time assertion
    const _DIVISIBILITY: () = assert!(N % M == 0, "Pool size N must be exactly divisible by chunk count M");
    const _NON_ZERO_CHUNK_NUM: () = assert!(M > 0, "Must have at least one chunk");
    const _NON_ZERO_POOL_SIZE: () = assert!(N > 0, "Pool size must be greater than zero");
    const _N_GR_THAN_OR_EQ_TO_M: () = assert!(N >= M, "Pool size N must be greater than or equal to chunk count M");

    /// Size of each chunk in bytes
    pub const CHUNK_SIZE: usize = N / M;


    /// # Safety
    /// The caller must ensure the pointer is valid for reads/writes of N bytes and properly aligned for all pool operations.
    pub const unsafe fn new(pool: *mut u8) -> Self {
        let mut meta = [MetaInfo::Free; M];
        meta[0] = MetaInfo::FreeStart(M); // Initialize first chunk as FreeStart
        Self {
            inner: Mutex::new(PoolInner {
                pool,
                meta,
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
        let pool_base = inner.pool as usize;

        // Find a suitable free region
        if let Some((start_chunk, total_chunks)) = self.find_free_region(&inner, chunks_needed, layout.align()) {
            self.mark_allocated(&mut inner.meta, start_chunk, total_chunks)?;

            // Calculate pointer address
            let ptr_addr = pool_base + start_chunk * Self::CHUNK_SIZE;

            #[cfg(feature = "statistics")]
            {
                let mut stats = self.stats.lock();
                stats.allocated_chunks += total_chunks;
            }
            return Ok(ptr_addr as *mut u8);
        }

        #[cfg(feature = "statistics")]
        {
            self.stats.lock().allocation_errors += 1;
        }
        Err(anyhow!(AllocError::OutOfMemory).context("Failed to find free region"))
    }

    /// Attempts to deallocate previously allocated memory
    pub fn try_deallocate(&self, ptr: *mut u8) -> Result<()> {
        // Handle null
        if ptr.is_null() {
            #[cfg(feature = "statistics")]
            {
                self.stats.lock().deallocation_errors += 1;
            }
            return Err(anyhow!(AllocError::InvalidPointer).context("Cannot deallocate null pointer"));
        }

        let mut inner = self.inner.lock();
        let pool_base = inner.pool as usize;
        let ptr_addr = ptr as usize;

        // Validate pointer is within pool
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
            MetaInfo::Ptr(size) => size,
            _ => {
                #[cfg(feature = "statistics")]
                {
                    self.stats.lock().deallocation_errors += 1;
                }
                return Err(anyhow!(AllocError::NotAllocated).context("Chunk not allocated with Ptr"));
            }
        };
        
        // Clear memory if feature enabled
        #[cfg(feature = "zero-on-free")]
        {
            unsafe {
                let start_ptr = (pool_base + start_chunk * Self::CHUNK_SIZE) as *mut u8;
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
    #[inline]
    fn is_allocated(&self, meta: &[MetaInfo; M], chunk_idx: usize) -> bool {
        todo!()
    }

    /// Marks a range of chunks as allocated (Ptr/Allocated)
    fn mark_allocated(&self, meta: &mut [MetaInfo; M], start_chunk: usize, chunk_size: usize) -> Result<()> {
        todo!()
    }

    /// Marks a range of chunks as free and coalesces with adjacent free regions
    fn mark_chunks_free(&self, meta: &mut [MetaInfo; M], start_chunk: usize) -> Result<()> {
        todo!()
    }

    /// Finds a contiguous free region that can accommodate the request, considering alignment
    fn find_free_region(&self, inner: &PoolInner<N, M>, chunks_needed: usize, align: usize) -> Option<(usize,usize)> {
        todo!()
    }

}

#[cfg(feature = "zero-on-drop")]
impl<const N: usize, const M: usize> Drop for MemoryPoolAllocator<N, M> {
    fn drop(&mut self) {
        // Ensure all chunks are freed before dropping
        let inner = self.inner.lock();
        unsafe {
            core::ptr::write_bytes(inner.pool, 0, N);
        }
    }
}

#[cfg(test)]
mod tests {
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
            for j in (i+1)..count {
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
    fn test_alignment_handling() {
        type Alloc = MemoryPoolAllocator<2048, 32>;
        #[repr(align(32))]
        struct Aligned {
            mem: [u8;1024]
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
}