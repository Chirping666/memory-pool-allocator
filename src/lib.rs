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
//! ## Default Features
//! - **zero-on-free**: Zeroes memory of each allocation when it is deallocated.
//! - **zero-on-drop**: Zeroes the entire memory pool when the allocator is dropped.
//! - **statistics**: Tracks allocation and deallocation statistics (number of allocated chunks, allocation/deallocation errors).
//!
//! ## Optional Feature
//! - **debug**: Adds assertions of pool consistency for debug builds.
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

#[cfg(test)]
mod tests;

use anyhow::{Result, anyhow};
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
        size: usize,       // Size of the allocation
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

/// Information about an allocation for deallocation
struct AllocationInfo {
    start_chunk: usize,
    total_chunks: usize,
}

// Safety: Pool data is protected by mutex
unsafe impl<const N: usize, const M: usize> Sync for MemoryPoolAllocator<N, M> {}
unsafe impl<const N: usize, const M: usize> Send for MemoryPoolAllocator<N, M> {}

impl<const N: usize, const M: usize> MemoryPoolAllocator<N, M> {
    // Compile-time assertion
    const _DIVISIBILITY: () = assert!(
        N % M == 0,
        "Pool size N must be exactly divisible by chunk count M"
    );
    const _NON_ZERO_CHUNK_NUM: () = assert!(M > 0, "Must have at least one chunk");
    const _NON_ZERO_POOL_SIZE: () = assert!(N > 0, "Pool size must be greater than zero");
    const _N_GR_THAN_OR_EQ_TO_M: () = assert!(
        N >= M,
        "Pool size N must be greater than or equal to chunk count M"
    );

    /// Size of each chunk in bytes
    pub const CHUNK_SIZE: usize = N / M;

    /// # Safety
    /// The caller must ensure the pointer is valid for reads/writes of N bytes and properly aligned for all pool operations.
    pub const unsafe fn new(pool: *mut u8) -> Self {
        let mut meta = [MetaInfo::FreeContinuation; M];
        meta[0] = MetaInfo::Free(M); // Initialize first chunk as Free with size M
        Self {
            inner: Mutex::new(PoolInner { pool, meta }),
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
        if chunks_needed > M || chunks_needed == 0 {
            #[cfg(feature = "statistics")]
            {
                self.stats.lock().allocation_errors += 1;
            }
            return Err(anyhow!(AllocError::OutOfMemory).context("Allocation too large"));
        }

        let mut inner = self.inner.lock();
        let pool_base = inner.pool as usize;

        // Find a suitable free region
        if let Some((start_chunk, total_chunks, aligned_ptr)) =
            self.find_free_region(&inner, chunks_needed, layout.align(), pool_base)
        {
            self.mark_allocated(
                &mut inner.meta,
                start_chunk,
                total_chunks,
                aligned_ptr,
                pool_base,
            )?;

            #[cfg(feature = "debug")]
            #[cfg(debug_assertions)]
            {
                debug_assert!(
                    self.validate_pool_consistency(&inner.meta),
                    "Pool consistency check failed after allocation"
                );
            }

            #[cfg(feature = "statistics")]
            {
                let mut stats = self.stats.lock();
                stats.allocated_chunks += total_chunks;
            }

            return Ok(aligned_ptr as *mut u8);
        }

        #[cfg(feature = "statistics")]
        {
            self.stats.lock().allocation_errors += 1;
        }
        Err(anyhow!(AllocError::OutOfMemory).context("No suitable free region found"))
    }

    /// Attempts to deallocate previously allocated memory  
    pub fn try_deallocate(&self, ptr: *mut u8) -> Result<()> {
        // Handle null pointer
        if ptr.is_null() {
            #[cfg(feature = "statistics")]
            {
                self.stats.lock().deallocation_errors += 1;
            }
            return Err(
                anyhow!(AllocError::InvalidPointer).context("Cannot deallocate null pointer")
            );
        }

        let mut inner = self.inner.lock();
        let pool_base = inner.pool as usize;
        let ptr_addr = ptr as usize;

        // Validate pointer is within pool bounds
        if ptr_addr < pool_base || ptr_addr >= pool_base + N {
            #[cfg(feature = "statistics")]
            {
                self.stats.lock().deallocation_errors += 1;
            }
            return Err(
                anyhow!(AllocError::InvalidPointer).context("Pointer not from this allocator")
            );
        }

        // Find the allocation that contains this pointer
        let allocation_info =
            match self.find_allocation_containing_ptr(&inner.meta, ptr_addr, pool_base) {
                Ok(info) => info,
                Err(_) => {
                    #[cfg(feature = "statistics")]
                    {
                        self.stats.lock().deallocation_errors += 1;
                    }
                    return Err(anyhow!(AllocError::NotAllocated)
                        .context("Pointer not currently allocated"));
                }
            };

        // Clear memory if feature enabled
        #[cfg(feature = "zero-on-free")]
        {
            unsafe {
                let start_ptr =
                    (pool_base + allocation_info.start_chunk * Self::CHUNK_SIZE) as *mut u8;
                core::ptr::write_bytes(
                    start_ptr,
                    0,
                    allocation_info.total_chunks * Self::CHUNK_SIZE,
                );
            }
        }

        // Mark chunks as free and coalesce
        self.mark_chunks_free(
            &mut inner.meta,
            allocation_info.start_chunk,
            allocation_info.total_chunks,
        )?;

        #[cfg(feature = "debug")]
        #[cfg(debug_assertions)]
        {
            debug_assert!(
                self.validate_pool_consistency(&inner.meta),
                "Pool consistency check failed after deallocation"
            );
        }

        // Update stats
        #[cfg(feature = "statistics")]
        {
            let mut stats = self.stats.lock();
            stats.allocated_chunks = stats
                .allocated_chunks
                .saturating_sub(allocation_info.total_chunks);
        }

        Ok(())
    }

    // === Private helper methods ===

    /// Find the allocation that contains the given pointer
    fn find_allocation_containing_ptr(
        &self,
        meta: &[MetaInfo; M],
        ptr_addr: usize,
        pool_base: usize,
    ) -> Result<AllocationInfo> {
        // Check which chunk contains this pointer
        let containing_chunk = (ptr_addr - pool_base) / Self::CHUNK_SIZE;
        if containing_chunk >= M {
            return Err(anyhow!(AllocError::InvalidPointer).context("Pointer beyond pool bounds"));
        }

        // Find the start of the allocation by scanning backwards
        let mut scan_chunk = containing_chunk;
        loop {
            match meta[scan_chunk] {
                MetaInfo::AllocStart { size, ptr_offset } => {
                    // Found the allocation start
                    let end_chunk = scan_chunk + size;
                    if containing_chunk < end_chunk {
                        // Check if this is the right allocation by validating the pointer
                        let expected_ptr = pool_base + scan_chunk * Self::CHUNK_SIZE + ptr_offset;
                        if ptr_addr == expected_ptr {
                            return Ok(AllocationInfo {
                                start_chunk: scan_chunk,
                                total_chunks: size,
                            });
                        }
                    }
                    return Err(anyhow!(AllocError::NotAllocated)
                        .context("Pointer not matching expected allocation"));
                }
                MetaInfo::AllocContinuation => {
                    // Continue scanning backwards
                    if scan_chunk == 0 {
                        return Err(
                            anyhow!(AllocError::NotAllocated).context("No allocation start found")
                        );
                    }
                    scan_chunk -= 1;
                }
                _ => {
                    return Err(anyhow!(AllocError::NotAllocated).context("Pointer in free region"));
                }
            }
        }
    }

    /// Finds a contiguous free region that can accommodate the request, considering alignment
    /// Returns (start_chunk, total_chunks_to_reserve, aligned_user_pointer)
    fn find_free_region(
        &self,
        inner: &PoolInner<N, M>,
        chunks_needed: usize,
        align: usize,
        pool_base: usize,
    ) -> Option<(usize, usize, usize)> {
        let mut i = 0;

        while i < M {
            match inner.meta[i] {
                MetaInfo::Free(free_size) => {
                    // Found a free region, try to allocate within it
                    let free_start = i;
                    let free_end = i + free_size;

                    // Try different starting positions within this free region
                    for try_start in free_start..free_end {
                        // Calculate address of this chunk
                        let chunk_addr = pool_base + try_start * Self::CHUNK_SIZE;

                        // Calculate aligned address
                        let aligned_addr = (chunk_addr + align - 1) & !(align - 1);

                        // How many bytes of alignment padding do we need?
                        let alignment_offset = aligned_addr - chunk_addr;

                        // Convert alignment offset to chunks (round up)
                        let alignment_chunks =
                            (alignment_offset + Self::CHUNK_SIZE - 1) / Self::CHUNK_SIZE;

                        // Total chunks needed: alignment padding + actual allocation
                        let total_chunks = alignment_chunks + chunks_needed;

                        // Check if we have enough space in this free region
                        if try_start + total_chunks <= free_end {
                            // Recalculate the actual aligned pointer within our reserved space
                            let reserved_start_addr = pool_base + try_start * Self::CHUNK_SIZE;
                            let final_aligned_addr =
                                (reserved_start_addr + align - 1) & !(align - 1);

                            // Ensure the aligned address is within our reserved region
                            let reserved_end_addr =
                                pool_base + (try_start + total_chunks) * Self::CHUNK_SIZE;
                            if final_aligned_addr + (chunks_needed * Self::CHUNK_SIZE)
                                <= reserved_end_addr
                            {
                                return Some((try_start, total_chunks, final_aligned_addr));
                            }
                        }
                    }

                    // Move to the next region
                    i = free_end;
                }
                _ => {
                    // Not a free region, move to next chunk
                    i += 1;
                }
            }
        }

        None
    }

    /// Marks a range of chunks as allocated
    fn mark_allocated(
        &self,
        meta: &mut [MetaInfo; M],
        start_chunk: usize,
        total_chunks: usize,
        user_ptr: usize,
        pool_base: usize,
    ) -> Result<()> {
        if start_chunk + total_chunks > M {
            return Err(anyhow!(AllocError::OutOfMemory).context("Allocation exceeds pool bounds"));
        }

        // Calculate the offset of the user pointer from the chunk base
        let chunk_base_addr = pool_base + start_chunk * Self::CHUNK_SIZE;
        let ptr_offset = user_ptr - chunk_base_addr;

        // Determine the full free region that contains this allocation
        let mut region_start = start_chunk;
        while region_start > 0 && matches!(meta[region_start - 1], MetaInfo::FreeContinuation) {
            region_start -= 1;
        }

        let free_region_size = match meta.get(region_start) {
            Some(MetaInfo::Free(size)) => *size,
            Some(
                MetaInfo::FreeContinuation
                | MetaInfo::AllocStart { .. }
                | MetaInfo::AllocContinuation,
            ) => {
                return Err(anyhow!(AllocError::OutOfMemory)
                    .context("Attempted to allocate from a non-free region"));
            }
            None => {
                return Err(anyhow!(AllocError::OutOfMemory)
                    .context("Allocation region start out of bounds"));
            }
        };

        let region_end = region_start + free_region_size;
        if start_chunk + total_chunks > region_end {
            return Err(anyhow!(AllocError::OutOfMemory)
                .context("Allocation exceeds available free region"));
        }

        // Temporarily mark the entire region as continuations so we can rebuild
        for idx in region_start..region_end {
            meta[idx] = MetaInfo::FreeContinuation;
        }

        // Restore any leading free region before the allocation
        let leading_free = start_chunk.saturating_sub(region_start);
        if leading_free > 0 {
            Self::set_free_region(meta, region_start, leading_free);
        }

        // Mark the first chunk with the allocation info
        meta[start_chunk] = MetaInfo::AllocStart {
            size: total_chunks,
            ptr_offset,
        };

        // Mark subsequent chunks as allocated continuations
        for i in 1..total_chunks {
            meta[start_chunk + i] = MetaInfo::AllocContinuation;
        }

        // Restore any trailing free region after the allocation
        let allocation_end = start_chunk + total_chunks;
        if allocation_end < region_end {
            Self::set_free_region(meta, allocation_end, region_end - allocation_end);
        }

        #[cfg(feature = "debug")]
        #[cfg(debug_assertions)]
        {
            debug_assert!(
                self.validate_pool_consistency(meta),
                "Pool consistency check failed after marking allocation"
            );
        }

        Ok(())
    }

    /// Marks chunks as free and coalesces with adjacent free regions
    fn mark_chunks_free(
        &self,
        meta: &mut [MetaInfo; M],
        start_chunk: usize,
        chunk_count: usize,
    ) -> Result<()> {
        if start_chunk + chunk_count > M {
            return Err(anyhow!(AllocError::OutOfMemory).context("Invalid chunk range"));
        }

        let left_region = if start_chunk > 0 {
            Self::free_region_info(meta, start_chunk - 1)
        } else {
            None
        };

        let right_index = start_chunk + chunk_count;
        let right_region = if right_index < M {
            Self::free_region_info(meta, right_index)
        } else {
            None
        };

        let mut region_start = start_chunk;
        if let Some((left_start, _)) = left_region {
            region_start = left_start;
        }

        let mut region_end = start_chunk + chunk_count;
        if let Some((right_start, right_size)) = right_region {
            region_end = core::cmp::max(region_end, right_start + right_size);
        }

        for idx in region_start..region_end {
            meta[idx] = MetaInfo::FreeContinuation;
        }

        Self::set_free_region(meta, region_start, region_end - region_start);

        #[cfg(feature = "debug")]
        #[cfg(debug_assertions)]
        {
            debug_assert!(
                self.validate_pool_consistency(meta),
                "Pool consistency check failed after marking chunks free"
            );
        }

        Ok(())
    }

    fn free_region_info(meta: &[MetaInfo; M], idx: usize) -> Option<(usize, usize)> {
        if idx >= M {
            return None;
        }

        match meta[idx] {
            MetaInfo::Free(size) => Some((idx, size)),
            MetaInfo::FreeContinuation => {
                let mut start = idx;
                while start > 0 && matches!(meta[start - 1], MetaInfo::FreeContinuation) {
                    start -= 1;
                }

                if let MetaInfo::Free(size) = meta[start] {
                    Some((start, size))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn set_free_region(meta: &mut [MetaInfo; M], start: usize, len: usize) {
        if len == 0 {
            return;
        }

        meta[start] = MetaInfo::Free(len);
        for offset in 1..len {
            meta[start + offset] = MetaInfo::FreeContinuation;
        }
    }

    /// Get current pool statistics and debug info
    #[cfg(feature = "statistics")]
    pub fn get_stats(&self) -> PoolStats {
        self.stats.lock().clone()
    }

    /// Get a snapshot of the current pool state for debugging
    // #[cfg(debug_assertions)]
    // pub fn debug_pool_state(&self) -> Vec<(usize, MetaInfo)> {
    //     let inner = self.inner.lock();
    //     inner.meta.iter().enumerate().map(|(i, &meta)| (i, meta)).collect()
    // }

    /// Validate pool consistency (debug builds only)
    #[cfg(feature = "debug")]
    #[cfg(debug_assertions)]
    fn validate_pool_consistency(&self, meta: &[MetaInfo; M]) -> bool {
        let mut i = 0;
        while i < M {
            match meta[i] {
                MetaInfo::Free(size) => {
                    if i + size > M {
                        return false; // Size exceeds bounds
                    }
                    // Check that following chunks are FreeContinuation
                    for j in 1..size {
                        if !matches!(meta[i + j], MetaInfo::FreeContinuation) {
                            return false;
                        }
                    }
                    i += size;
                }
                MetaInfo::AllocStart { size, .. } => {
                    if i + size > M {
                        return false; // Size exceeds bounds
                    }
                    // Check that following chunks are AllocContinuation
                    for j in 1..size {
                        if !matches!(meta[i + j], MetaInfo::AllocContinuation) {
                            return false;
                        }
                    }
                    i += size;
                }
                MetaInfo::FreeContinuation | MetaInfo::AllocContinuation => {
                    // These should only appear as continuations, not at scan positions
                    return false;
                }
            }
        }
        true
    }

    // This is probably not needed at all.
    // /// Checks if a specific chunk is allocated (used by tests/debug)
    // #[allow(dead_code)]
    // #[cfg(test)]
    // fn is_allocated(&self, meta: &[MetaInfo; M], chunk_idx: usize) -> bool {
    //     if chunk_idx >= M {
    //         return false;
    //     }
    //     matches!(
    //         meta[chunk_idx],
    //         MetaInfo::AllocStart { .. } | MetaInfo::AllocContinuation
    //     )
    // }

    // /// Test helper function to check if a pointer is properly aligned
    // #[allow(dead_code)]
    // #[cfg(test)]
    // fn is_properly_aligned(ptr: *mut u8, align: usize) -> bool {
    //     (ptr as usize) % align == 0
    // }

    /// Test helper to get total free space in chunks
    #[cfg(test)]
    fn count_free_chunks(&self) -> usize {
        let inner = self.inner.lock();
        let mut count = 0;
        let mut i = 0;
        while i < M {
            match inner.meta[i] {
                MetaInfo::Free(size) => {
                    count += size;
                    i += size;
                }
                MetaInfo::AllocStart { size, .. } => {
                    i += size;
                }
                _ => i += 1,
            }
        }
        count
    }

    /// Test helper to get total allocated space in chunks
    #[cfg(test)]
    fn count_allocated_chunks(&self) -> usize {
        let inner = self.inner.lock();
        let mut count = 0;
        let mut i = 0;
        while i < M {
            match inner.meta[i] {
                MetaInfo::Free(size) => {
                    i += size;
                }
                MetaInfo::AllocStart { size, .. } => {
                    count += size;
                    i += size;
                }
                _ => i += 1,
            }
        }
        count
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
