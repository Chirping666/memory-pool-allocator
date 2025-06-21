#![cfg(test)]
use core::alloc::Layout;
use heapless::Vec;
use memory_pool_allocator::MemoryPoolAllocator;

#[test]
fn heapless_vec_allocation() {
    type Alloc = MemoryPoolAllocator<1024, 64>;
    let allocator = Alloc::new();
    let layout = Layout::from_size_align(16, 8).unwrap();
    let mut ptrs: Vec<*mut u8, 16> = Vec::new();
    for _ in 0..16 {
        let ptr = allocator.allocate(layout);
        assert!(!ptr.is_null());
        ptrs.push(ptr).unwrap();
    }
    for ptr in ptrs {
        allocator.deallocate(ptr, layout);
    }
}
