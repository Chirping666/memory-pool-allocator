use super::*;

/// Find the allocation that contains the given pointer
fn find_allocation_containing_ptr(chunk_size: usize, meta: &[MetaInfo; M], ptr_addr: usize, pool_base: usize) -> Result<AllocationInfo> {
    // Check which chunk contains this pointer
    let containing_chunk = (ptr_addr - pool_base) / chunk_size;
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
                    let expected_ptr = pool_base + scan_chunk * chunk_size + ptr_offset;
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
                    return Err(anyhow!(AllocError::NotAllocated).context("No allocation start found"));
                }
                scan_chunk -= 1;
            }
            _ => {
                return Err(anyhow!(AllocError::NotAllocated)
                    .context("Pointer in free region"));
            }
        }
    }
}

/// Checks if a specific chunk is allocated (used by tests/debug)
#[allow(dead_code)]
fn is_allocated(&self, meta: &[MetaInfo; M], chunk_idx: usize) -> bool {
    if chunk_idx >= M {
        return false;
    }
    matches!(meta[chunk_idx], 
        MetaInfo::AllocStart { .. } | MetaInfo::AllocContinuation)
}

/// Finds a contiguous free region that can accommodate the request, considering alignment
/// Returns (start_chunk, total_chunks_to_reserve, aligned_user_pointer)
fn find_free_region(&self, inner: &PoolInner<N, M>, chunks_needed: usize, align: usize, pool_base: usize) -> Option<(usize, usize, usize)> {
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
                    let chunk_addr = pool_base + try_start * chunk_size;
                    
                    // Calculate aligned address
                    let aligned_addr = (chunk_addr + align - 1) & !(align - 1);
                    
                    // How many bytes of alignment padding do we need?
                    let alignment_offset = aligned_addr - chunk_addr;
                    
                    // Convert alignment offset to chunks (round up)
                    let alignment_chunks = (alignment_offset + chunk_size - 1) / chunk_size;
                    
                    // Total chunks needed: alignment padding + actual allocation
                    let total_chunks = alignment_chunks + chunks_needed;
                    
                    // Check if we have enough space in this free region
                    if try_start + total_chunks <= free_end {
                        // Verify all chunks in the range are actually free
                        let all_free = (try_start..(try_start + total_chunks)).all(|check_idx| {
                            matches!(inner.meta[check_idx], MetaInfo::Free(_) | MetaInfo::FreeContinuation)
                        });
                        
                        if all_free {
                            // Recalculate the actual aligned pointer within our reserved space
                            let reserved_start_addr = pool_base + try_start * chunk_size;
                            let final_aligned_addr = (reserved_start_addr + align - 1) & !(align - 1);
                            
                            // Ensure the aligned address is within our reserved region
                            let reserved_end_addr = pool_base + (try_start + total_chunks) * chunk_size;
                            if final_aligned_addr + (chunks_needed * chunk_size) <= reserved_end_addr {
                                return Some((try_start, total_chunks, final_aligned_addr));
                            }
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
fn mark_allocated(&self, meta: &mut [MetaInfo; M], start_chunk: usize, total_chunks: usize, user_ptr: usize, pool_base: usize) -> Result<()> {
    if start_chunk + total_chunks > M {
        return Err(anyhow!(AllocError::OutOfMemory).context("Allocation exceeds pool bounds"));
    }

    // Calculate the offset of the user pointer from the chunk base
    let chunk_base_addr = pool_base + start_chunk * chunk_size;
    let ptr_offset = user_ptr - chunk_base_addr;

    // Mark the first chunk with the allocation info
    meta[start_chunk] = MetaInfo::AllocStart {
        size: total_chunks,
        ptr_offset,
    };
    
    // Mark subsequent chunks as allocated continuations
    for i in 1..total_chunks {
        meta[start_chunk + i] = MetaInfo::AllocContinuation;
    }

    // Handle any remaining free space after our allocation
    let after_allocation = start_chunk + total_chunks;
    if after_allocation < M {
        // Check if there are contiguous free chunks after our allocation
        let mut remaining_free = 0;
        let mut scan_idx = after_allocation;
        
        while scan_idx < M && matches!(meta[scan_idx], MetaInfo::FreeContinuation) {
            remaining_free += 1;
            scan_idx += 1;
        }
        
        // If there are free chunks, mark them properly
        if remaining_free > 0 {
            meta[after_allocation] = MetaInfo::Free(remaining_free);
            // The rest remain as FreeContinuation (they're already FreeContinuation)
        }
    }

    #[cfg(debug_assertions)]
    {
        debug_assert!(self.validate_pool_consistency(meta), "Pool consistency check failed after marking allocation");
    }

    Ok(())
}

/// Marks chunks as free and coalesces with adjacent free regions
fn mark_chunks_free(&self, meta: &mut [MetaInfo; M], start_chunk: usize, chunk_count: usize) -> Result<()> {
    if start_chunk + chunk_count > M {
        return Err(anyhow!(AllocError::OutOfMemory).context("Invalid chunk range"));
    }

    // First, mark all chunks in the allocation as free continuations
    for i in start_chunk..(start_chunk + chunk_count) {
        meta[i] = MetaInfo::FreeContinuation;
    }

    // Now rebuild all free region markers by scanning the entire metadata array
    // This is simpler and safer than trying to do complex coalescing
    self.rebuild_free_markers(meta);

    #[cfg(debug_assertions)]
    {
        debug_assert!(self.validate_pool_consistency(meta), "Pool consistency check failed after marking chunks free");
    }

    Ok(())
}

/// Rebuild free region markers after freeing chunks
/// This scans the entire metadata array and properly marks free regions
fn rebuild_free_markers(&self, meta: &mut [MetaInfo; M]) {
    let mut i = 0;
    while i < M {
        match meta[i] {
            MetaInfo::FreeContinuation | MetaInfo::Free(_) => {
                // Found the start of a free region (or need to rebuild it), count its size
                let start = i;
                let mut size = 0;
                
                // Count consecutive free chunks (both FreeContinuation and Free(_))
                while i < M && matches!(meta[i], MetaInfo::FreeContinuation | MetaInfo::Free(_)) {
                    size += 1;
                    i += 1;
                }
                
                // Mark this free region properly
                if size > 0 {
                    meta[start] = MetaInfo::Free(size);
                    // Mark the rest as FreeContinuation
                    for j in (start + 1)..(start + size) {
                        if j < M {
                            meta[j] = MetaInfo::FreeContinuation;
                        }
                    }
                }
            }
            MetaInfo::AllocStart { size, .. } => {
                // Skip this entire allocation
                i += size;
            }
            MetaInfo::AllocContinuation => {
                // This shouldn't happen at a scan position, but handle it gracefully
                i += 1;
            }
        }
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

/// Test helper function to check if a pointer is properly aligned
#[cfg(test)]
fn is_properly_aligned(ptr: *mut u8, align: usize) -> bool {
    (ptr as usize) % align == 0
}

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