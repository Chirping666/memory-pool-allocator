# memory-pool-allocator

A fixed-size, thread-safe memory pool allocator for Rust, supporting custom chunk sizes and efficient allocation/deallocation.

## Features
- No_std compatible
- Thread-safe via `parking_lot::Mutex`
- Dual-licensed (Apache-2.0 OR MIT)
- Customizable pool size and chunk count via const generics
- Fast allocation and deallocation for fixed-size blocks
- Fragmentation and usage statistics

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
memory-pool-allocator = "0.1"
```

Example:

```rust
use memory_pool_allocator::MemoryPoolAllocator;

// Create a 1024-byte pool divided into 64 chunks (16 bytes each)
let allocator = MemoryPoolAllocator::<1024, 64>::new();

let layout = core::alloc::Layout::from_size_align(32, 8).unwrap();
let ptr = allocator.allocate(layout);
assert!(!ptr.is_null());
allocator.deallocate(ptr, layout);
```

## Safety
- `N` (total bytes) must be exactly divisible by `M` (number of chunks).
- The allocator does not track ownership of pointers; the user must ensure only valid pointers are deallocated.

## License
Licensed under either of
 - Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 - MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
at your option.
