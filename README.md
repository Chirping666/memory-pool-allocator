# memory-pool-allocator

A fixed-size, thread-safe memory pool allocator for Rust, supporting custom chunk sizes and efficient allocation/deallocation. Made with the help of AI such as Claude and Copilot.

## Features
- No_std compatible
- Thread-safe via `parking_lot::Mutex`
- Dual-licensed (Apache-2.0 OR MIT)
- Customizable pool size and chunk count via const generics
- Fast allocation and deallocation for fixed-size blocks
- Usage statistics
- User-provided memory pool (via ```rust *mut u8 ```)
- Default features: `zero-on-free`, `zero-on-drop` and `statistics`.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
memory-pool-allocator = "1.0"
```

Or, if you do not want statistics and zeroing, add:
```toml
[dependencies]
memory-pool-allocator = { version = "1.0", default-features = false }
```

Example:

```rust
use memory_pool_allocator::MemoryPoolAllocator;
use core::alloc::Layout;

#[repr(align(64))]
struct Aligned {
    mem: [u8; 1024]
}
let mut aligned = Aligned { mem: [0; 1024] };
let allocator = unsafe { MemoryPoolAllocator::<1024, 64>::new(aligned.mem.as_mut_ptr()) };
let layout = Layout::from_size_align(128, 64).unwrap();
let ptr = allocator.try_allocate(layout).unwrap();
assert_eq!(ptr as usize % 64, 0);
allocator.try_deallocate(ptr).unwrap();
```

## Safety
- The user must provide a pointer to a memory region of at least `N` bytes, aligned to the maximum alignment required by allocations.
- The allocator does not manage the lifetime of the memory pool; the user is responsible for ensuring it is valid for the allocator's lifetime.
- If the pool is not sufficiently aligned, allocations with higher alignment requirements may fail or result in undefined behavior.
- `N` (total bytes) must be exactly divisible by `M` (number of chunks).

## Default Features
- **zero-on-free**: Zeroes memory of each allocation when it is deallocated.
- **zero-on-drop**: Zeroes the entire memory pool when the allocator is dropped.
- **statistics**: Tracks allocation and deallocation statistics (number of allocated chunks, allocation/deallocation errors).

## License
Licensed under either of
 - Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 - MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
at your option.
