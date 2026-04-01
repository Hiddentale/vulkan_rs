#![allow(unused_imports)]
#![allow(clippy::too_many_arguments)]
use crate::error::{VkResult, check, enumerate_two_call, fill_two_call};
use crate::vk::bitmasks::*;
use crate::vk::constants::*;
use crate::vk::enums::*;
use crate::vk::handles::*;
use crate::vk::structs::*;
impl crate::Device {
    ///Wraps [`vkGetDeviceProcAddr`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDeviceProcAddr.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns a function pointer for a device-level command. This is the
    ///device-specific equivalent of `get_instance_proc_addr` and returns
    ///pointers dispatched directly through the device's driver, bypassing
    ///the loader trampoline.
    ///
    ///In normal usage you do not need to call this yourself, `Device`
    ///loads all function pointers automatically at creation time. Use this
    ///only if you need a command that is not yet exposed as a wrapper
    ///method, or for raw interop scenarios.
    ///
    ///The returned pointer is only valid for the device it was queried
    ///from. Passing a command name that the device does not support
    ///returns a null pointer.
    pub unsafe fn get_device_proc_addr(&self, p_name: *const core::ffi::c_char) {
        let fp = self
            .commands()
            .get_device_proc_addr
            .expect("vkGetDeviceProcAddr not loaded");
        unsafe { fp(self.handle(), p_name) };
    }
    ///Wraps [`vkDestroyDevice`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyDevice.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `device` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a logical device and frees its resources. All objects
    ///created from this device (buffers, images, pipelines, command pools,
    ///etc.) **must** be destroyed before calling this.
    ///
    ///A safe teardown order:
    ///
    ///1. `device_wait_idle`, ensure no GPU work is in flight.
    ///2. Destroy all device-child objects (pipelines, buffers, images,
    ///   views, descriptor pools, command pools, fences, semaphores, etc.).
    ///3. `free_memory` for all device memory allocations.
    ///4. `destroy_device`.
    ///
    ///After this call the `Device` handle is invalid. Do not use it or any
    ///object created from it.
    pub unsafe fn destroy_device(&self, allocator: Option<&AllocationCallbacks>) {
        let fp = self
            .commands()
            .destroy_device
            .expect("vkDestroyDevice not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), alloc_ptr) };
    }
    ///Wraps [`vkGetDeviceQueue`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDeviceQueue.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves a queue handle for a queue that was requested at device
    ///creation time. The `queue_family_index` and `queue_index` must match
    ///a family and index that was included in the `DeviceCreateInfo`'s
    ///`queue_create_infos`.
    ///
    ///Queue handles are implicitly owned by the device, they do not need
    ///to be destroyed and become invalid when the device is destroyed.
    ///
    ///Queues retrieved this way have no special flags. If you created
    ///queues with `DeviceQueueCreateFlags` (e.g. protected queues), use
    ///`get_device_queue2` instead.
    ///
    ///It is common to retrieve queues once after device creation and store
    ///them for the lifetime of the device.
    pub unsafe fn get_device_queue(&self, queue_family_index: u32, queue_index: u32) -> Queue {
        let fp = self
            .commands()
            .get_device_queue
            .expect("vkGetDeviceQueue not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(self.handle(), queue_family_index, queue_index, &mut out) };
        out
    }
    ///Wraps [`vkQueueSubmit`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkQueueSubmit.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_DEVICE_LOST`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `queue` (self) must be valid and not destroyed.
    ///- `queue` must be externally synchronized.
    ///- `fence` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///`queue_submit` is the primary way to send recorded command buffers
    ///to the GPU. Each `SubmitInfo` specifies:
    ///
    ///- **Wait semaphores + stage masks**: the submission waits on these
    ///  semaphores at the given pipeline stages before executing.
    ///- **Command buffers**: executed in array order within the submission.
    ///- **Signal semaphores**: signalled when all command buffers complete.
    ///
    ///**Fence**: pass a fence to know when the *entire batch* of submissions
    ///completes on the CPU side. Passing `Fence::null()` means there is no
    ///CPU-visible signal, you must use semaphores or `queue_wait_idle`
    ///instead.
    ///
    ///Minimize `queue_submit` calls. Each call has driver overhead; batching
    ///multiple `SubmitInfo` entries into one call is cheaper than separate
    ///calls.
    ///
    ///**Thread safety**: a `Queue` must be externally synchronized. If
    ///multiple threads submit to the same queue, you need a mutex.
    pub unsafe fn queue_submit(
        &self,
        queue: Queue,
        p_submits: &[SubmitInfo],
        fence: Fence,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .queue_submit
            .expect("vkQueueSubmit not loaded");
        check(unsafe { fp(queue, p_submits.len() as u32, p_submits.as_ptr(), fence) })
    }
    ///Wraps [`vkQueueWaitIdle`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkQueueWaitIdle.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_DEVICE_LOST`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `queue` (self) must be valid and not destroyed.
    ///- `queue` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Blocks the calling thread until all commands submitted to this queue
    ///have completed. Equivalent to submitting a fence and immediately
    ///waiting on it, but simpler.
    ///
    ///Use this for quick synchronization in non-performance-critical paths
    ///(e.g. during teardown or after a one-shot transfer). In a render
    ///loop, prefer fences or timeline semaphores for finer-grained
    ///control, `queue_wait_idle` stalls the CPU and prevents overlap
    ///between CPU and GPU work.
    ///
    ///The queue must be externally synchronized: do not call this while
    ///another thread is submitting to the same queue.
    pub unsafe fn queue_wait_idle(&self, queue: Queue) -> VkResult<()> {
        let fp = self
            .commands()
            .queue_wait_idle
            .expect("vkQueueWaitIdle not loaded");
        check(unsafe { fp(queue) })
    }
    ///Wraps [`vkDeviceWaitIdle`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDeviceWaitIdle.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_DEVICE_LOST`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Blocks the calling thread until **all** queues on this device are
    ///idle. This is the nuclear option for synchronization, it drains
    ///every queue completely.
    ///
    ///Typical uses:
    ///
    ///- **Before destroying the device**: ensures no GPU work is in flight
    ///  before you start tearing down resources.
    ///- **Before a swapchain resize**: guarantees all frames are done so
    ///  image views and framebuffers can be safely recreated.
    ///
    ///Avoid calling this in a render loop. It forces a full CPU–GPU
    ///round-trip and prevents any overlap. Use per-frame fences or
    ///timeline semaphores instead.
    pub unsafe fn device_wait_idle(&self) -> VkResult<()> {
        let fp = self
            .commands()
            .device_wait_idle
            .expect("vkDeviceWaitIdle not loaded");
        check(unsafe { fp(self.handle()) })
    }
    ///Wraps [`vkAllocateMemory`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkAllocateMemory.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_INVALID_EXTERNAL_HANDLE`
    ///- `VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Memory allocation in Vulkan is expensive. Prefer sub-allocating from
    ///large blocks using a memory allocator (e.g., `gpu-allocator`) rather
    ///than calling this for every buffer or image.
    ///
    ///The returned `DeviceMemory` must be freed with `free_memory` when no
    ///longer needed. Vulkan does not garbage-collect device memory.
    pub unsafe fn allocate_memory(
        &self,
        p_allocate_info: &MemoryAllocateInfo,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<DeviceMemory> {
        let fp = self
            .commands()
            .allocate_memory
            .expect("vkAllocateMemory not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_allocate_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkFreeMemory`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkFreeMemory.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `memory` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Frees a device memory allocation. All buffers and images bound to
    ///this memory must already be destroyed, freeing memory while objects
    ///are still bound is undefined behaviour.
    ///
    ///If the memory is currently mapped, it is implicitly unmapped before
    ///being freed. You do not need to call `unmap_memory` first, although
    ///doing so explicitly is a common defensive practice.
    ///
    ///Vulkan has a per-device allocation limit
    ///(`max_memory_allocation_count`, often 4096). Sub-allocating from
    ///large blocks and freeing them as a group keeps you well within this
    ///limit.
    pub unsafe fn free_memory(
        &self,
        memory: DeviceMemory,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .free_memory
            .expect("vkFreeMemory not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), memory, alloc_ptr) };
    }
    ///Wraps [`vkMapMemory`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkMapMemory.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_MEMORY_MAP_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `memory` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Maps device memory into the host address space so the CPU can read or
    ///write it. The memory must have been allocated from a memory type with
    ///`MEMORY_PROPERTY_HOST_VISIBLE`.
    ///
    ///**Persistently mapped memory**: it is valid (and recommended) to keep
    ///memory mapped for the lifetime of the allocation. Map once after
    ///`allocate_memory` and unmap only before `free_memory`. This avoids
    ///repeated map/unmap overhead.
    ///
    ///**Coherency**: if the memory type does not include
    ///`MEMORY_PROPERTY_HOST_COHERENT`, you must call:
    ///
    ///- `flush_mapped_memory_ranges` after CPU writes, before GPU reads.
    ///- `invalidate_mapped_memory_ranges` after GPU writes, before CPU reads.
    ///
    ///Host-coherent memory skips both calls at the cost of slightly lower
    ///throughput on some architectures.
    ///
    ///**Alignment**: when sub-allocating from a large allocation, ensure
    ///offsets respect `non_coherent_atom_size` for flush/invalidate ranges.
    pub unsafe fn map_memory(
        &self,
        memory: DeviceMemory,
        offset: u64,
        size: u64,
        flags: MemoryMapFlags,
        pp_data: *mut *mut core::ffi::c_void,
    ) -> VkResult<()> {
        let fp = self.commands().map_memory.expect("vkMapMemory not loaded");
        check(unsafe { fp(self.handle(), memory, offset, size, flags, pp_data) })
    }
    ///Wraps [`vkUnmapMemory`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkUnmapMemory.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `memory` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Unmaps a previously mapped device memory region. After this call the
    ///host pointer returned by `map_memory` is invalid.
    ///
    ///Unmapping is only strictly necessary before `free_memory` if you
    ///want to be explicit. Freeing memory implicitly unmaps it.
    ///
    ///For persistently mapped memory (the recommended pattern), you
    ///typically map once after allocation and unmap only during teardown.
    ///There is no performance penalty for keeping memory mapped.
    ///
    ///If the memory type is not `HOST_COHERENT`, make sure to call
    ///`flush_mapped_memory_ranges` after your final writes before
    ///unmapping, to ensure the GPU sees the latest data.
    pub unsafe fn unmap_memory(&self, memory: DeviceMemory) {
        let fp = self
            .commands()
            .unmap_memory
            .expect("vkUnmapMemory not loaded");
        unsafe { fp(self.handle(), memory) };
    }
    ///Wraps [`vkFlushMappedMemoryRanges`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkFlushMappedMemoryRanges.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Flushes CPU writes to mapped non-coherent memory so the GPU can see
    ///them. Only needed when the memory type does **not** include
    ///`MEMORY_PROPERTY_HOST_COHERENT`.
    ///
    ///Call this **after** writing data through the mapped pointer and
    ///**before** the GPU reads it (i.e. before the relevant
    ///`queue_submit`).
    ///
    ///**Alignment**: the `offset` and `size` of each range must be
    ///multiples of `non_coherent_atom_size` (from physical device limits),
    ///or `offset` must be zero and `size` must be `VK_WHOLE_SIZE`. Failing
    ///to align causes undefined behaviour on some implementations.
    ///
    ///Multiple ranges can be flushed in a single call. Batch them when
    ///updating several sub-allocations within the same memory object.
    ///
    ///If you are using host-coherent memory, this call is unnecessary and
    ///can be skipped entirely.
    pub unsafe fn flush_mapped_memory_ranges(
        &self,
        p_memory_ranges: &[MappedMemoryRange],
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .flush_mapped_memory_ranges
            .expect("vkFlushMappedMemoryRanges not loaded");
        check(unsafe {
            fp(
                self.handle(),
                p_memory_ranges.len() as u32,
                p_memory_ranges.as_ptr(),
            )
        })
    }
    ///Wraps [`vkInvalidateMappedMemoryRanges`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkInvalidateMappedMemoryRanges.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Invalidates CPU caches for mapped non-coherent memory so the CPU
    ///can see data written by the GPU. The counterpart to
    ///`flush_mapped_memory_ranges`.
    ///
    ///Call this **after** the GPU has finished writing (e.g. after
    ///`wait_for_fences` on the relevant submission) and **before** reading
    ///the data through the mapped pointer.
    ///
    ///The same alignment rules apply: `offset` and `size` must be
    ///multiples of `non_coherent_atom_size`, or use offset zero with
    ///`VK_WHOLE_SIZE`.
    ///
    ///If you are using host-coherent memory, this call is unnecessary,
    ///GPU writes are automatically visible to the CPU. Most desktop GPUs
    ///offer host-coherent memory types for host-visible heaps.
    pub unsafe fn invalidate_mapped_memory_ranges(
        &self,
        p_memory_ranges: &[MappedMemoryRange],
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .invalidate_mapped_memory_ranges
            .expect("vkInvalidateMappedMemoryRanges not loaded");
        check(unsafe {
            fp(
                self.handle(),
                p_memory_ranges.len() as u32,
                p_memory_ranges.as_ptr(),
            )
        })
    }
    ///Wraps [`vkGetDeviceMemoryCommitment`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDeviceMemoryCommitment.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries how many bytes of a lazily-allocated memory object are
    ///currently backed by physical storage. Only meaningful for memory
    ///allocated with `MEMORY_PROPERTY_LAZILY_ALLOCATED`.
    ///
    ///Lazily-allocated memory is primarily used for transient framebuffer
    ///attachments on tile-based GPUs (mobile). The driver may not back the
    ///full allocation with physical memory until tiles actually need it.
    ///
    ///On desktop GPUs this typically returns the full allocation size since
    ///lazy allocation is rarely supported. Check
    ///`memory_properties.memory_types` for the `LAZILY_ALLOCATED` flag
    ///before relying on this.
    pub unsafe fn get_device_memory_commitment(&self, memory: DeviceMemory) -> u64 {
        let fp = self
            .commands()
            .get_device_memory_commitment
            .expect("vkGetDeviceMemoryCommitment not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(self.handle(), memory, &mut out) };
        out
    }
    ///Wraps [`vkGetBufferMemoryRequirements`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetBufferMemoryRequirements.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns the memory requirements (size, alignment, compatible memory
    ///type bits) for a buffer. Must be called before `bind_buffer_memory`.
    ///
    ///The returned `memory_type_bits` is a bitmask where bit *i* is set if
    ///memory type *i* (from `get_physical_device_memory_properties`) is
    ///compatible. Pick a type that satisfies both this mask and your
    ///desired properties (`HOST_VISIBLE`, `DEVICE_LOCAL`, etc.).
    ///
    ///The `alignment` value must be respected when sub-allocating: the
    ///offset passed to `bind_buffer_memory` must be a multiple of it.
    ///
    ///For Vulkan 1.1+, prefer `get_buffer_memory_requirements2` which
    ///supports dedicated allocation queries via
    ///`MemoryDedicatedRequirements`.
    pub unsafe fn get_buffer_memory_requirements(&self, buffer: Buffer) -> MemoryRequirements {
        let fp = self
            .commands()
            .get_buffer_memory_requirements
            .expect("vkGetBufferMemoryRequirements not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(self.handle(), buffer, &mut out) };
        out
    }
    ///Wraps [`vkBindBufferMemory`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkBindBufferMemory.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `buffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Binds a `DeviceMemory` allocation (or a region of one) to a buffer.
    ///Must be called before the buffer is used in any command.
    ///
    ///**Requirements**:
    ///
    ///1. Call `get_buffer_memory_requirements` first.
    ///2. The `memory_type_index` used for allocation must have a bit set
    ///   in the returned `memory_type_bits` mask.
    ///3. `memory_offset` must be a multiple of the returned `alignment`.
    ///4. `memory_offset + size` must not exceed the allocation size.
    ///
    ///**Sub-allocation**: multiple buffers can share one `DeviceMemory`
    ///allocation at different offsets. This is strongly recommended,
    ///drivers have a per-allocation limit (`max_memory_allocation_count`,
    ///often 4096) and each allocation has overhead.
    ///
    ///Once bound, the memory binding cannot be changed for the lifetime of
    ///the buffer. Destroy the buffer before freeing its backing memory.
    pub unsafe fn bind_buffer_memory(
        &self,
        buffer: Buffer,
        memory: DeviceMemory,
        memory_offset: u64,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .bind_buffer_memory
            .expect("vkBindBufferMemory not loaded");
        check(unsafe { fp(self.handle(), buffer, memory, memory_offset) })
    }
    ///Wraps [`vkGetImageMemoryRequirements`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetImageMemoryRequirements.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns the memory requirements (size, alignment, compatible memory
    ///type bits) for an image. Must be called before `bind_image_memory`.
    ///
    ///Image memory requirements can differ significantly based on tiling,
    ///format, and usage flags. An `IMAGE_TILING_OPTIMAL` image typically
    ///requires `DEVICE_LOCAL` memory and has stricter alignment than a
    ///linear image.
    ///
    ///When sub-allocating linear and optimal images from the same memory
    ///object, the `buffer_image_granularity` device limit applies. You may
    ///need extra padding between the two to satisfy this constraint.
    ///
    ///For Vulkan 1.1+, prefer `get_image_memory_requirements2` which
    ///supports dedicated allocation queries.
    pub unsafe fn get_image_memory_requirements(&self, image: Image) -> MemoryRequirements {
        let fp = self
            .commands()
            .get_image_memory_requirements
            .expect("vkGetImageMemoryRequirements not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(self.handle(), image, &mut out) };
        out
    }
    ///Wraps [`vkBindImageMemory`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkBindImageMemory.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `image` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Binds a `DeviceMemory` allocation (or a region of one) to an image.
    ///Must be called after `create_image` and before the image is used.
    ///
    ///**Requirements**:
    ///
    ///1. Call `get_image_memory_requirements` first.
    ///2. The `memory_type_index` used for allocation must have a bit set
    ///   in the returned `memory_type_bits` mask.
    ///3. `memory_offset` must be a multiple of the returned `alignment`.
    ///
    ///**Dedicated allocations**: some drivers perform better when certain
    ///images (especially swapchain-sized color or depth targets) have their
    ///own allocation. Query `get_image_memory_requirements2` with
    ///`MemoryDedicatedRequirements` to check whether the driver prefers or
    ///requires a dedicated allocation.
    ///
    ///**Sub-allocation**: like buffers, multiple images can share one
    ///allocation at different offsets. Respect alignment from the memory
    ///requirements, and note that linear and optimal-tiling images may
    ///need `buffer_image_granularity` spacing between them.
    ///
    ///Once bound, the memory binding is permanent. Destroy the image
    ///before freeing its backing memory.
    pub unsafe fn bind_image_memory(
        &self,
        image: Image,
        memory: DeviceMemory,
        memory_offset: u64,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .bind_image_memory
            .expect("vkBindImageMemory not loaded");
        check(unsafe { fp(self.handle(), image, memory, memory_offset) })
    }
    ///Wraps [`vkGetImageSparseMemoryRequirements`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetImageSparseMemoryRequirements.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the sparse memory requirements for an image created with
    ///one of the `IMAGE_CREATE_SPARSE_*` flags. Returns a list of sparse
    ///image format properties describing the memory layout for each
    ///image aspect (color, depth, stencil, metadata).
    ///
    ///Sparse resources allow partially-resident textures where only some
    ///mip levels or regions are backed by physical memory. This is an
    ///advanced feature primarily used for virtual texturing and terrain
    ///streaming.
    ///
    ///If the image was not created with sparse flags, this returns an
    ///empty list. Check `physical_device_features.sparse_binding` before
    ///using sparse resources.
    pub unsafe fn get_image_sparse_memory_requirements(
        &self,
        image: Image,
    ) -> Vec<SparseImageMemoryRequirements> {
        let fp = self
            .commands()
            .get_image_sparse_memory_requirements
            .expect("vkGetImageSparseMemoryRequirements not loaded");
        fill_two_call(|count, data| unsafe { fp(self.handle(), image, count, data) })
    }
    ///Wraps [`vkQueueBindSparse`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkQueueBindSparse.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_DEVICE_LOST`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `queue` (self) must be valid and not destroyed.
    ///- `queue` must be externally synchronized.
    ///- `fence` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Binds sparse memory regions to sparse resources (buffers or images).
    ///This is the only way to change the memory backing of a sparse
    ///resource after creation.
    ///
    ///Sparse binding supports:
    ///
    ///- **Partial residency**: bind memory to individual mip tail regions
    ///  or image tiles, leaving others unbound.
    ///- **Aliasing**: multiple sparse resources can alias the same memory
    ///  region (with `IMAGE_CREATE_SPARSE_ALIASED`).
    ///- **Dynamic re-binding**: swap memory pages at runtime for virtual
    ///  texturing or streaming.
    ///
    ///The bind operation is asynchronous and can synchronize with
    ///semaphores, similar to `queue_submit`. The queue must support sparse
    ///binding (check `QUEUE_SPARSE_BINDING`).
    ///
    ///This is an advanced feature. Most applications use fully-bound
    ///resources with `bind_buffer_memory` / `bind_image_memory` instead.
    pub unsafe fn queue_bind_sparse(
        &self,
        queue: Queue,
        p_bind_info: &[BindSparseInfo],
        fence: Fence,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .queue_bind_sparse
            .expect("vkQueueBindSparse not loaded");
        check(unsafe { fp(queue, p_bind_info.len() as u32, p_bind_info.as_ptr(), fence) })
    }
    ///Wraps [`vkCreateFence`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateFence.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Fences are the primary CPU–GPU synchronization primitive. The CPU
    ///blocks on `wait_for_fences` until the GPU signals the fence.
    ///
    ///**Initial state**: create with `FENCE_CREATE_SIGNALED` when the
    ///fence is used in a frame loop that waits before the first submit.
    ///Without this flag the first `wait_for_fences` would block forever.
    ///
    ///**Typical frame loop pattern**:
    ///
    ///1. `wait_for_fences`, block until the previous frame's GPU work
    ///   completes.
    ///2. `reset_fences`, reset back to unsignaled.
    ///3. Record and submit commands, passing the fence to `queue_submit`.
    ///
    ///A fence can only be associated with one submission at a time.
    ///Submitting with a fence that is already pending is an error.
    ///
    ///For GPU–GPU synchronization (between queue submissions) use
    ///semaphores instead. Fences are strictly for CPU-visible signalling.
    pub unsafe fn create_fence(
        &self,
        p_create_info: &FenceCreateInfo,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<Fence> {
        let fp = self
            .commands()
            .create_fence
            .expect("vkCreateFence not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyFence`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyFence.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `fence` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a fence object. The fence must not be in use by any
    ///pending `queue_submit` call, wait on it or call `device_wait_idle`
    ///before destroying.
    ///
    ///Fences are lightweight objects but are still tracked by the driver.
    ///Destroy them during teardown or when they are no longer part of your
    ///synchronization scheme.
    pub unsafe fn destroy_fence(&self, fence: Fence, allocator: Option<&AllocationCallbacks>) {
        let fp = self
            .commands()
            .destroy_fence
            .expect("vkDestroyFence not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), fence, alloc_ptr) };
    }
    ///Wraps [`vkResetFences`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkResetFences.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `pFences` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Resets one or more fences to the unsignaled state. Must be called
    ///before reusing a fence in a new `queue_submit` call.
    ///
    ///The fence must not be currently waited on by `wait_for_fences` from
    ///another thread. A common safe pattern:
    ///
    ///1. `wait_for_fences`, blocks until signaled.
    ///2. `reset_fences`, immediately reset after the wait returns.
    ///3. Submit new work with the fence.
    ///
    ///Resetting a fence that is already unsignaled is valid but wasteful.
    ///Resetting a fence that is pending (submitted but not yet signaled)
    ///is an error.
    pub unsafe fn reset_fences(&self, p_fences: &[Fence]) -> VkResult<()> {
        let fp = self
            .commands()
            .reset_fences
            .expect("vkResetFences not loaded");
        check(unsafe { fp(self.handle(), p_fences.len() as u32, p_fences.as_ptr()) })
    }
    ///Wraps [`vkGetFenceStatus`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetFenceStatus.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_DEVICE_LOST`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Non-blocking check of whether a fence is signaled. Returns
    ///`VK_SUCCESS` if signaled, `VK_NOT_READY` if still pending.
    ///
    ///Use this for polling patterns where you want to do other work while
    ///waiting:
    ///
    ///```text
    ///loop {
    ///    if get_fence_status(fence) == VK_SUCCESS { break; }
    ///    // do other work...
    ///}
    ///```
    ///
    ///For blocking waits, prefer `wait_for_fences` which is more efficient
    ///than a spin loop, it lets the CPU sleep until the driver signals.
    ///
    ///This call can also return device-lost errors, so check the result
    ///even in non-error paths.
    pub unsafe fn get_fence_status(&self, fence: Fence) -> VkResult<()> {
        let fp = self
            .commands()
            .get_fence_status
            .expect("vkGetFenceStatus not loaded");
        check(unsafe { fp(self.handle(), fence) })
    }
    ///Wraps [`vkWaitForFences`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkWaitForFences.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_DEVICE_LOST`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Blocks the calling thread until one or all of the given fences are
    ///signaled, or until the timeout expires.
    ///
    ///**`wait_all`**: when `true`, the call returns only after *every*
    ///fence in the list is signaled. When `false`, it returns as soon as
    ///*any* one fence is signaled.
    ///
    ///**Timeout**: specified in nanoseconds. `u64::MAX` means wait
    ///indefinitely. A timeout of zero performs a non-blocking check
    ///(equivalent to polling `get_fence_status` on each fence).
    ///
    ///Returns `VK_TIMEOUT` if the timeout expires before the condition is
    ///met. This is not an error, check the return value and handle it
    ///(e.g. log a warning or retry).
    ///
    ///**Typical frame loop**:
    ///
    ///```text
    ///wait_for_fences(&[frame_fence], true, u64::MAX)
    ///reset_fences(&[frame_fence])
    #[doc = "// record and submit..."]
    ///```
    ///
    ///After `wait_for_fences` returns successfully, all GPU work
    ///associated with those fences is complete and the resources are safe
    ///to reuse.
    pub unsafe fn wait_for_fences(
        &self,
        p_fences: &[Fence],
        wait_all: u32,
        timeout: u64,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .wait_for_fences
            .expect("vkWaitForFences not loaded");
        check(unsafe {
            fp(
                self.handle(),
                p_fences.len() as u32,
                p_fences.as_ptr(),
                wait_all,
                timeout,
            )
        })
    }
    ///Wraps [`vkCreateSemaphore`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateSemaphore.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a semaphore for GPU–GPU synchronization between queue
    ///submissions. Unlike fences (CPU–GPU), semaphores are invisible to
    ///the CPU, they are signaled and waited on entirely within
    ///`queue_submit` or `queue_present_khr`.
    ///
    ///**Binary semaphores** (the default) have two states: signaled and
    ///unsignaled. A submission signals the semaphore, and a later
    ///submission waits on it, which also resets it to unsignaled.
    ///
    ///**Timeline semaphores** (Vulkan 1.2+) have a monotonically
    ///increasing 64-bit counter. Create one by chaining
    ///`SemaphoreTypeCreateInfo` with `SEMAPHORE_TYPE_TIMELINE`. Timeline
    ///semaphores can be waited on and signaled from the CPU as well via
    ///`wait_semaphores` and `signal_semaphore`.
    ///
    ///Common uses:
    ///
    ///- Synchronize between a graphics queue submit and a present.
    ///- Order a transfer upload before a render pass that consumes it.
    ///- Coordinate work across different queue families.
    pub unsafe fn create_semaphore(
        &self,
        p_create_info: &SemaphoreCreateInfo,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<Semaphore> {
        let fp = self
            .commands()
            .create_semaphore
            .expect("vkCreateSemaphore not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroySemaphore`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroySemaphore.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `semaphore` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a semaphore. The semaphore must not be referenced by any
    ///pending queue submission, either as a wait or signal semaphore.
    ///
    ///Wait for all submissions that use this semaphore to complete (via
    ///fences or `device_wait_idle`) before destroying it.
    pub unsafe fn destroy_semaphore(
        &self,
        semaphore: Semaphore,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_semaphore
            .expect("vkDestroySemaphore not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), semaphore, alloc_ptr) };
    }
    ///Wraps [`vkCreateEvent`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateEvent.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates an event, a fine-grained synchronisation primitive that
    ///can be signaled and waited on from both the host (CPU) and the
    ///device (GPU).
    ///
    ///Events are most useful for split barriers: signal an event at one
    ///point in a command buffer, do other work, then wait on it later.
    ///This gives the GPU more flexibility to overlap execution compared to
    ///a single `cmd_pipeline_barrier`.
    ///
    ///**Host-side usage**: `set_event` and `reset_event` signal and reset
    ///from the CPU. `get_event_status` polls the current state. However,
    ///host-signaled events cannot be reliably waited on by the GPU on all
    ///implementations, use them primarily for GPU–GPU sync within a
    ///queue.
    ///
    ///Events are lightweight and cheap to create.
    pub unsafe fn create_event(
        &self,
        p_create_info: &EventCreateInfo,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<Event> {
        let fp = self
            .commands()
            .create_event
            .expect("vkCreateEvent not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyEvent`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyEvent.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `event` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys an event. The event must not be referenced by any pending
    ///command buffer. Wait for all relevant submissions to complete before
    ///destroying.
    pub unsafe fn destroy_event(&self, event: Event, allocator: Option<&AllocationCallbacks>) {
        let fp = self
            .commands()
            .destroy_event
            .expect("vkDestroyEvent not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), event, alloc_ptr) };
    }
    ///Wraps [`vkGetEventStatus`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetEventStatus.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_DEVICE_LOST`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns whether an event is currently signaled or unsignaled.
    ///Returns `VK_EVENT_SET` if signaled, `VK_EVENT_RESET` if not.
    ///
    ///This is a non-blocking host-side query. Use it to poll for
    ///GPU-signaled events when you need to know the result without
    ///blocking. For blocking synchronisation, use fences instead.
    pub unsafe fn get_event_status(&self, event: Event) -> VkResult<()> {
        let fp = self
            .commands()
            .get_event_status
            .expect("vkGetEventStatus not loaded");
        check(unsafe { fp(self.handle(), event) })
    }
    ///Wraps [`vkSetEvent`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkSetEvent.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `event` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Signals an event from the host (CPU). After this call,
    ///`get_event_status` returns `VK_EVENT_SET`.
    ///
    ///Host-signaled events are primarily useful for host–host
    ///synchronisation or as a manual control mechanism. For GPU–GPU
    ///synchronisation, prefer `cmd_set_event` recorded in a command
    ///buffer.
    pub unsafe fn set_event(&self, event: Event) -> VkResult<()> {
        let fp = self.commands().set_event.expect("vkSetEvent not loaded");
        check(unsafe { fp(self.handle(), event) })
    }
    ///Wraps [`vkResetEvent`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkResetEvent.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `event` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Resets an event to the unsignaled state from the host (CPU). The
    ///event must not be waited on by any pending `cmd_wait_events` call.
    ///
    ///After resetting, the event can be signaled again by `set_event` or
    ///`cmd_set_event`.
    pub unsafe fn reset_event(&self, event: Event) -> VkResult<()> {
        let fp = self
            .commands()
            .reset_event
            .expect("vkResetEvent not loaded");
        check(unsafe { fp(self.handle(), event) })
    }
    ///Wraps [`vkCreateQueryPool`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateQueryPool.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a pool of query slots. Queries let you measure GPU
    ///performance and gather statistics without stalling the pipeline.
    ///
    ///**Query types**:
    ///
    ///- `OCCLUSION`: counts how many samples pass the depth test. Useful
    ///  for visibility culling, render a bounding box, check the count.
    ///- `PIPELINE_STATISTICS`: counts shader invocations, primitives,
    ///  clipping, etc. Must be enabled via
    ///  `pipeline_statistics_query` device feature.
    ///- `TIMESTAMP`: records a GPU timestamp. Use two timestamps and the
    ///  `timestamp_period` device property to measure elapsed time.
    ///
    ///Queries must be reset before use with `cmd_reset_query_pool` (or
    ///`reset_query_pool` on Vulkan 1.2+). Results are retrieved with
    ///`get_query_pool_results` or copied into a buffer with
    ///`cmd_copy_query_pool_results`.
    pub unsafe fn create_query_pool(
        &self,
        p_create_info: &QueryPoolCreateInfo,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<QueryPool> {
        let fp = self
            .commands()
            .create_query_pool
            .expect("vkCreateQueryPool not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyQueryPool`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyQueryPool.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `queryPool` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a query pool and frees its resources. All command buffers
    ///that reference this pool must have completed execution before
    ///destroying it.
    pub unsafe fn destroy_query_pool(
        &self,
        query_pool: QueryPool,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_query_pool
            .expect("vkDestroyQueryPool not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), query_pool, alloc_ptr) };
    }
    ///Wraps [`vkGetQueryPoolResults`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetQueryPoolResults.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_DEVICE_LOST`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Reads query results from a query pool into a host buffer. This is
    ///the CPU-side retrieval path, for GPU-side copies into a device
    ///buffer, use `cmd_copy_query_pool_results` instead.
    ///
    ///**Key flags**:
    ///
    ///- `QUERY_RESULT_64`: return 64-bit results. Always use this for
    ///  timestamp and pipeline statistics queries to avoid overflow.
    ///- `QUERY_RESULT_WAIT`: block until all requested queries are
    ///  available. Without this flag, unavailable queries return
    ///  `VK_NOT_READY` and their slots are left untouched.
    ///- `QUERY_RESULT_WITH_AVAILABILITY`: append an availability value
    ///  after each result (non-zero if available). Useful for polling
    ///  without blocking.
    ///- `QUERY_RESULT_PARTIAL`: return whatever data is available even
    ///  for incomplete queries. Only meaningful for occlusion queries.
    ///
    ///**Stride**: the `stride` parameter is the byte distance between
    ///successive query results in your output buffer. It must be at least
    ///large enough to hold the result plus the optional availability value.
    ///
    ///Queries that have not been started or not yet completed return
    ///`VK_NOT_READY` unless `QUERY_RESULT_WAIT` is set.
    pub unsafe fn get_query_pool_results(
        &self,
        query_pool: QueryPool,
        first_query: u32,
        query_count: u32,
        data_size: usize,
        p_data: *mut core::ffi::c_void,
        stride: u64,
        flags: QueryResultFlags,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_query_pool_results
            .expect("vkGetQueryPoolResults not loaded");
        check(unsafe {
            fp(
                self.handle(),
                query_pool,
                first_query,
                query_count,
                data_size,
                p_data,
                stride,
                flags,
            )
        })
    }
    ///Wraps [`vkResetQueryPool`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkResetQueryPool.html).
    /**
    Provided by **VK_BASE_VERSION_1_2**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Resets a range of queries in a pool from the host (CPU). This is the
    ///Vulkan 1.2 host-side alternative to `cmd_reset_query_pool`, which
    ///resets queries from a command buffer.
    ///
    ///Host-side reset is simpler, call it directly without recording a
    ///command buffer. Requires the `host_query_reset` feature (core in
    ///Vulkan 1.2).
    ///
    ///Queries must be reset before use. Resetting a query that is in use
    ///by a pending command buffer is an error.
    pub unsafe fn reset_query_pool(
        &self,
        query_pool: QueryPool,
        first_query: u32,
        query_count: u32,
    ) {
        let fp = self
            .commands()
            .reset_query_pool
            .expect("vkResetQueryPool not loaded");
        unsafe { fp(self.handle(), query_pool, first_query, query_count) };
    }
    ///Wraps [`vkCreateBuffer`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateBuffer.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Examples
    ///
    ///```no_run
    ///# let (_, instance) = vulkan_rs::test_helpers::create_test_instance().expect("test setup");
    ///# let phys = unsafe { instance.enumerate_physical_devices() }.expect("no devices");
    ///# let p = [1.0f32];
    ///# let qi = vulkan_rs::vk::structs::DeviceQueueCreateInfo::builder().queue_priorities(&p);
    ///# let qis = [*qi];
    ///# let di = vulkan_rs::vk::structs::DeviceCreateInfo::builder().queue_create_infos(&qis);
    ///# let device = unsafe { instance.create_device(phys[0], &di, None) }.expect("device creation");
    ///use vulkan_rs::vk::structs::*;
    ///use vulkan_rs::vk::bitmasks::*;
    ///
    ///let info = BufferCreateInfo::builder()
    ///    .size(1024)
    ///    .usage(BufferUsageFlagBits::VERTEX_BUFFER)
    ///    .sharing_mode(vulkan_rs::vk::enums::SharingMode::EXCLUSIVE);
    ///let buffer = unsafe { device.create_buffer(&info, None) }
    ///    .expect("buffer creation failed");
    #[doc = "// Use buffer..."]
    ///unsafe { device.destroy_buffer(buffer, None) };
    ///# unsafe { device.destroy_device(None) };
    ///# unsafe { instance.destroy_instance(None) };
    ///```
    pub unsafe fn create_buffer(
        &self,
        p_create_info: &BufferCreateInfo,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<Buffer> {
        let fp = self
            .commands()
            .create_buffer
            .expect("vkCreateBuffer not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyBuffer`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyBuffer.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `buffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a buffer object. The buffer must not be in use by any
    ///pending GPU work, wait on the relevant fences or call
    ///`device_wait_idle` before destroying.
    ///
    ///Destroying a buffer does **not** free its backing memory. Call
    ///`free_memory` separately (or let your sub-allocator reclaim the
    ///region).
    ///
    ///Destroy order: destroy the buffer first, then free the memory. Not
    ///the reverse, freeing memory while a buffer is still bound to it is
    ///undefined behaviour.
    pub unsafe fn destroy_buffer(&self, buffer: Buffer, allocator: Option<&AllocationCallbacks>) {
        let fp = self
            .commands()
            .destroy_buffer
            .expect("vkDestroyBuffer not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), buffer, alloc_ptr) };
    }
    ///Wraps [`vkCreateBufferView`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateBufferView.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a view into a buffer that interprets its contents as a
    ///typed array of texels. Buffer views are used with:
    ///
    ///- **Uniform texel buffers** (`BUFFER_USAGE_UNIFORM_TEXEL_BUFFER`):
    ///  read-only typed access from shaders via `samplerBuffer` /
    ///  `textureBuffer` in GLSL.
    ///- **Storage texel buffers** (`BUFFER_USAGE_STORAGE_TEXEL_BUFFER`):
    ///  read-write typed access from shaders via `imageBuffer` in GLSL.
    ///
    ///The format, offset, and range define the view window into the
    ///buffer. The format must be supported for the buffer view usage,
    ///check `format_properties.buffer_features`.
    ///
    ///Buffer views are less common than image views. They are mainly used
    ///for large, flat data arrays (e.g. particle attributes, lookup
    ///tables) that benefit from format conversion on read.
    pub unsafe fn create_buffer_view(
        &self,
        p_create_info: &BufferViewCreateInfo,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<BufferView> {
        let fp = self
            .commands()
            .create_buffer_view
            .expect("vkCreateBufferView not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyBufferView`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyBufferView.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `bufferView` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a buffer view. The view must not be referenced by any
    ///descriptor set that is bound in a pending command buffer.
    ///
    ///Destroy buffer views before the underlying buffer.
    pub unsafe fn destroy_buffer_view(
        &self,
        buffer_view: BufferView,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_buffer_view
            .expect("vkDestroyBufferView not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), buffer_view, alloc_ptr) };
    }
    ///Wraps [`vkCreateImage`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateImage.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_COMPRESSION_EXHAUSTED_EXT`
    ///- `VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///After creating an image you must bind memory to it before use:
    ///
    ///1. Query requirements with `get_image_memory_requirements`.
    ///2. Allocate from a compatible memory type with `allocate_memory`.
    ///3. Bind with `bind_image_memory`.
    ///
    ///Choose `IMAGE_TILING_OPTIMAL` for GPU-side textures and render targets.
    ///Use `IMAGE_TILING_LINEAR` only when you need direct CPU access to the
    ///texel layout (e.g. CPU readback), and check format support first with
    ///`get_physical_device_image_format_properties`.
    ///
    ///The `initial_layout` must be `UNDEFINED` or `PREINITIALIZED`.
    ///Most applications use `UNDEFINED` and transition via a pipeline barrier.
    ///
    ///Destroy with `destroy_image` when no longer needed. Do not destroy an
    ///image that is still referenced by a framebuffer or image view.
    pub unsafe fn create_image(
        &self,
        p_create_info: &ImageCreateInfo,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<Image> {
        let fp = self
            .commands()
            .create_image
            .expect("vkCreateImage not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyImage`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyImage.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `image` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys an image object. The image must not be in use by any
    ///pending GPU work, and all image views referencing this image must
    ///already be destroyed.
    ///
    ///Destroying an image does **not** free its backing memory. Call
    ///`free_memory` separately after destroying the image.
    ///
    ///Safe teardown order for an image:
    ///
    ///1. Wait for all GPU work using the image to complete.
    ///2. Destroy all `ImageView` objects referencing the image.
    ///3. Destroy any `Framebuffer` objects that included those views.
    ///4. `destroy_image`.
    ///5. Free or reclaim the backing memory.
    pub unsafe fn destroy_image(&self, image: Image, allocator: Option<&AllocationCallbacks>) {
        let fp = self
            .commands()
            .destroy_image
            .expect("vkDestroyImage not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), image, alloc_ptr) };
    }
    ///Wraps [`vkGetImageSubresourceLayout`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetImageSubresourceLayout.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the memory layout (offset, size, row pitch, array pitch,
    ///depth pitch) of a specific subresource within a linear-tiling
    ///image. Only valid for images created with `IMAGE_TILING_LINEAR`.
    ///For optimal-tiling images, use `get_image_subresource_layout2`.
    pub unsafe fn get_image_subresource_layout(
        &self,
        image: Image,
        p_subresource: &ImageSubresource,
    ) -> SubresourceLayout {
        let fp = self
            .commands()
            .get_image_subresource_layout
            .expect("vkGetImageSubresourceLayout not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(self.handle(), image, p_subresource, &mut out) };
        out
    }
    ///Wraps [`vkCreateImageView`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateImageView.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///An image view selects a subset of an image's subresources and
    ///reinterprets them for a specific use (sampling, color attachment, etc.).
    ///
    ///Common pitfalls:
    ///
    ///- **Aspect mask**: use `COLOR` for color formats, `DEPTH` and/or
    ///  `STENCIL` for depth/stencil formats. Getting this wrong causes
    ///  validation errors that are not always obvious.
    ///- **Format compatibility**: the view format must be compatible with the
    ///  image's format. Using `IMAGE_CREATE_MUTABLE_FORMAT` on the image
    ///  relaxes this to any format in the same size-compatibility class.
    ///- **View type vs image type**: a 2D image can back a `VIEW_TYPE_2D` or
    ///  `VIEW_TYPE_2D_ARRAY`. A 3D image cannot be viewed as 2D without
    ///  `VK_EXT_image_2d_view_of_3d`.
    ///
    ///Destroy with `destroy_image_view` when no longer needed. Destroy image
    ///views *before* destroying the underlying image.
    pub unsafe fn create_image_view(
        &self,
        p_create_info: &ImageViewCreateInfo,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<ImageView> {
        let fp = self
            .commands()
            .create_image_view
            .expect("vkCreateImageView not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyImageView`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyImageView.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `imageView` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys an image view. The view must not be referenced by any
    ///pending GPU work or by any framebuffer that is still in use.
    ///
    ///Destroy image views **before** the underlying image. Destroy any
    ///framebuffers that reference the view before destroying the view
    ///itself.
    pub unsafe fn destroy_image_view(
        &self,
        image_view: ImageView,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_image_view
            .expect("vkDestroyImageView not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), image_view, alloc_ptr) };
    }
    ///Wraps [`vkCreateShaderModule`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateShaderModule.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_INVALID_SHADER_NV`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///The input must be valid SPIR-V bytecode. The `code` slice is `&[u32]`
    ///and must be aligned to 4 bytes, which Rust's `&[u32]` guarantees.
    ///
    ///Use the `bytecode::read_spv` helper to load a `.spv` file from disk
    ///with correct alignment.
    ///
    ///Shader modules can be destroyed immediately after pipeline creation;
    ///the driver copies what it needs during `create_graphics_pipelines` or
    ///`create_compute_pipelines`. Destroying early keeps the handle count
    ///low.
    pub unsafe fn create_shader_module(
        &self,
        p_create_info: &ShaderModuleCreateInfo,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<ShaderModule> {
        let fp = self
            .commands()
            .create_shader_module
            .expect("vkCreateShaderModule not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyShaderModule`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyShaderModule.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `shaderModule` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a shader module. The module must not be in use by any
    ///pipeline. Shader modules can be safely destroyed after pipeline
    ///creation since the driver copies the SPIR-V at creation time.
    pub unsafe fn destroy_shader_module(
        &self,
        shader_module: ShaderModule,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_shader_module
            .expect("vkDestroyShaderModule not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), shader_module, alloc_ptr) };
    }
    ///Wraps [`vkCreatePipelineCache`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreatePipelineCache.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a pipeline cache that stores compiled pipeline state to
    ///speed up future pipeline creation.
    ///
    ///**Initial data**: pass previously serialized cache data (from
    ///`get_pipeline_cache_data`) to warm the cache on startup. The driver
    ///validates the header and silently ignores data from incompatible
    ///driver versions or hardware, it is always safe to pass stale data.
    ///
    ///A single cache can be shared across all pipeline creation calls in
    ///the application. Multiple threads can use the same cache
    ///concurrently, the driver handles internal synchronization.
    ///
    ///**Recommended workflow**:
    ///
    ///1. On startup, load cache data from disk and create the cache.
    ///2. Pass the cache to every `create_graphics_pipelines` and
    ///   `create_compute_pipelines` call.
    ///3. On shutdown, serialize with `get_pipeline_cache_data` and write
    ///   to disk.
    pub unsafe fn create_pipeline_cache(
        &self,
        p_create_info: &PipelineCacheCreateInfo,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<PipelineCache> {
        let fp = self
            .commands()
            .create_pipeline_cache
            .expect("vkCreatePipelineCache not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyPipelineCache`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyPipelineCache.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `pipelineCache` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a pipeline cache. Pipelines that were created using this
    ///cache remain valid, the cache is only needed during creation, not
    ///at runtime.
    ///
    ///Serialize the cache with `get_pipeline_cache_data` before destroying
    ///if you want to persist it to disk for the next session.
    pub unsafe fn destroy_pipeline_cache(
        &self,
        pipeline_cache: PipelineCache,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_pipeline_cache
            .expect("vkDestroyPipelineCache not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), pipeline_cache, alloc_ptr) };
    }
    ///Wraps [`vkGetPipelineCacheData`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPipelineCacheData.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Serializes the contents of a pipeline cache into a byte buffer for
    ///storage on disk. The data includes a vendor-specific header that the
    ///driver uses to validate compatibility on reload.
    ///
    ///Call this with a null data pointer first to query the required buffer
    ///size, then allocate and call again. The wrapper handles this
    ///two-call pattern for you.
    ///
    ///The cache data is **not portable** across different GPU vendors,
    ///driver versions, or pipeline cache UUIDs. Always check the header
    ///or let the driver reject incompatible data silently on reload via
    ///`create_pipeline_cache`.
    ///
    ///Write the data to a file (e.g. `pipeline_cache.bin`) and load it on
    ///the next application start to avoid redundant shader compilation.
    pub unsafe fn get_pipeline_cache_data(
        &self,
        pipeline_cache: PipelineCache,
        p_data: *mut core::ffi::c_void,
    ) -> VkResult<usize> {
        let fp = self
            .commands()
            .get_pipeline_cache_data
            .expect("vkGetPipelineCacheData not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), pipeline_cache, &mut out, p_data) })?;
        Ok(out)
    }
    ///Wraps [`vkMergePipelineCaches`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkMergePipelineCaches.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `dstCache` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Merges one or more source caches into a destination cache. Useful
    ///when multiple threads each use their own cache during parallel
    ///pipeline creation, merge them into a single cache before
    ///serializing to disk.
    ///
    ///The source caches are not modified or destroyed by this call. The
    ///destination cache receives all entries from the sources that it does
    ///not already contain. Duplicate entries are ignored.
    ///
    ///After merging, you can destroy the source caches if they are no
    ///longer needed.
    pub unsafe fn merge_pipeline_caches(
        &self,
        dst_cache: PipelineCache,
        p_src_caches: &[PipelineCache],
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .merge_pipeline_caches
            .expect("vkMergePipelineCaches not loaded");
        check(unsafe {
            fp(
                self.handle(),
                dst_cache,
                p_src_caches.len() as u32,
                p_src_caches.as_ptr(),
            )
        })
    }
    ///Wraps [`vkCreatePipelineBinariesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreatePipelineBinariesKHR.html).
    /**
    Provided by **VK_KHR_pipeline_binary**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates pipeline binary objects from either a pipeline create info
    ///or previously serialized binary data. Pipeline binaries capture
    ///compiled shader code in a device-specific format, enabling fast
    ///pipeline recreation without recompilation.
    ///
    ///Two creation paths via `PipelineBinaryCreateInfoKHR`:
    ///
    ///- **From pipeline create info + key**: compiles shaders and
    ///  produces binaries. Use `get_pipeline_key_khr` to obtain the
    ///  key first.
    ///- **From serialized data**: restores binaries saved with
    ///  `get_pipeline_binary_data_khr` from a prior run. This skips
    ///  compilation entirely.
    ///
    ///The output is written to `PipelineBinaryHandlesInfoKHR`. Call
    ///once with a null `pipelines` pointer to query the count, then
    ///again with an allocated array.
    ///
    ///Pipeline binaries are more portable than pipeline caches, they
    ///can be validated, versioned, and stored in application-managed
    ///files rather than opaque blobs.
    pub unsafe fn create_pipeline_binaries_khr(
        &self,
        p_create_info: &PipelineBinaryCreateInfoKHR,
        allocator: Option<&AllocationCallbacks>,
        p_binaries: *mut PipelineBinaryHandlesInfoKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .create_pipeline_binaries_khr
            .expect("vkCreatePipelineBinariesKHR not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, p_binaries) })
    }
    ///Wraps [`vkDestroyPipelineBinaryKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyPipelineBinaryKHR.html).
    /**
    Provided by **VK_KHR_pipeline_binary**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `pipelineBinary` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a pipeline binary handle. After destruction, the binary
    ///cannot be used to create pipelines or retrieve data.
    ///
    ///Pipeline binaries are independent of the pipelines created from
    ///them, destroying a binary does not affect any pipeline that was
    ///already created using it.
    pub unsafe fn destroy_pipeline_binary_khr(
        &self,
        pipeline_binary: PipelineBinaryKHR,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_pipeline_binary_khr
            .expect("vkDestroyPipelineBinaryKHR not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), pipeline_binary, alloc_ptr) };
    }
    ///Wraps [`vkGetPipelineKeyKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPipelineKeyKHR.html).
    /**
    Provided by **VK_KHR_pipeline_binary**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Computes a pipeline key that identifies the pipeline configuration
    ///for use with pipeline binaries. The key is a hash of the pipeline
    ///create info that lets you look up previously cached binaries.
    ///
    ///Pass a `PipelineCreateInfoKHR` referencing the pipeline create
    ///info (graphics, compute, or ray tracing) to get its key. Pass
    ///`None` to get an empty key structure for use as an output
    ///parameter.
    ///
    ///Store the key alongside serialized binary data from
    ///`get_pipeline_binary_data_khr`. On subsequent runs, compute the
    ///key for the current pipeline configuration and check if a matching
    ///binary exists before falling back to full compilation.
    pub unsafe fn get_pipeline_key_khr(
        &self,
        p_pipeline_create_info: Option<&PipelineCreateInfoKHR>,
        p_pipeline_key: *mut PipelineBinaryKeyKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_pipeline_key_khr
            .expect("vkGetPipelineKeyKHR not loaded");
        let p_pipeline_create_info_ptr =
            p_pipeline_create_info.map_or(core::ptr::null(), core::ptr::from_ref);
        check(unsafe { fp(self.handle(), p_pipeline_create_info_ptr, p_pipeline_key) })
    }
    ///Wraps [`vkGetPipelineBinaryDataKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPipelineBinaryDataKHR.html).
    /**
    Provided by **VK_KHR_pipeline_binary**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_NOT_ENOUGH_SPACE_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Serializes a pipeline binary into a byte buffer for offline
    ///storage. The data can be saved to disk and later passed to
    ///`create_pipeline_binaries_khr` to skip shader compilation on
    ///subsequent application launches.
    ///
    ///Uses the two-call pattern: call with a null `p_pipeline_binary_data`
    ///to query the required `data_size`, allocate a buffer, then call
    ///again to fill it.
    ///
    ///The output also includes a `PipelineBinaryKeyKHR` that identifies
    ///the binary. Store the key alongside the data, it is required
    ///when recreating the binary.
    ///
    ///Serialized data is device-specific and may become invalid after
    ///driver updates. Applications should handle creation failure
    ///gracefully by falling back to full recompilation.
    pub unsafe fn get_pipeline_binary_data_khr(
        &self,
        p_info: &PipelineBinaryDataInfoKHR,
        p_pipeline_binary_key: *mut PipelineBinaryKeyKHR,
        p_pipeline_binary_data: *mut core::ffi::c_void,
    ) -> VkResult<usize> {
        let fp = self
            .commands()
            .get_pipeline_binary_data_khr
            .expect("vkGetPipelineBinaryDataKHR not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe {
            fp(
                self.handle(),
                p_info,
                p_pipeline_binary_key,
                &mut out,
                p_pipeline_binary_data,
            )
        })?;
        Ok(out)
    }
    ///Wraps [`vkReleaseCapturedPipelineDataKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkReleaseCapturedPipelineDataKHR.html).
    /**
    Provided by **VK_KHR_pipeline_binary**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Releases internal data captured during pipeline creation that was
    ///retained for binary extraction. Call this after you have finished
    ///calling `create_pipeline_binaries_khr` for a pipeline created
    ///with `PIPELINE_CREATE_CAPTURE_DATA`.
    ///
    ///The pipeline itself remains valid after this call, only the
    ///captured internal data is freed. This reduces memory usage when
    ///you no longer need to extract binaries from the pipeline.
    pub unsafe fn release_captured_pipeline_data_khr(
        &self,
        p_info: &ReleaseCapturedPipelineDataInfoKHR,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .release_captured_pipeline_data_khr
            .expect("vkReleaseCapturedPipelineDataKHR not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        check(unsafe { fp(self.handle(), p_info, alloc_ptr) })
    }
    ///Wraps [`vkCreateGraphicsPipelines`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateGraphicsPipelines.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_INVALID_SHADER_NV`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `pipelineCache` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Graphics pipeline creation is the most expensive Vulkan object creation
    ///call. Batch multiple pipelines into a single call when possible,the
    ///driver can often parallelise compilation internally.
    ///
    ///**Pipeline cache**: always pass a `PipelineCache`. Even an empty cache
    ///helps on the first run; on subsequent runs it avoids redundant shader
    ///compilation entirely. Serialize the cache to disk between application
    ///sessions with `get_pipeline_cache_data`.
    ///
    ///**Dynamic state**: mark states like viewport, scissor, and blend
    ///constants as dynamic to reduce the number of pipeline permutations.
    ///Vulkan 1.3 makes viewport and scissor dynamic by default via
    ///`VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT` and
    ///`VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT`.
    ///
    ///If creation fails for one pipeline in a batch, the call returns an
    ///error but may still populate some output handles. Check
    ///`VK_PIPELINE_COMPILE_REQUIRED` when using
    ///`VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT`.
    pub unsafe fn create_graphics_pipelines(
        &self,
        pipeline_cache: PipelineCache,
        p_create_infos: &[GraphicsPipelineCreateInfo],
        allocator: Option<&AllocationCallbacks>,
        p_pipelines: *mut Pipeline,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .create_graphics_pipelines
            .expect("vkCreateGraphicsPipelines not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        check(unsafe {
            fp(
                self.handle(),
                pipeline_cache,
                p_create_infos.len() as u32,
                p_create_infos.as_ptr(),
                alloc_ptr,
                p_pipelines,
            )
        })
    }
    ///Wraps [`vkCreateComputePipelines`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateComputePipelines.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_INVALID_SHADER_NV`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `pipelineCache` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Creates one or more compute pipelines. Compute pipelines are simpler
    ///than graphics pipelines, they only need a single shader stage and a
    ///pipeline layout.
    ///
    ///**Pipeline cache**: pass a `PipelineCache` to speed up creation, the
    ///same way as with graphics pipelines. The cache is shared across both
    ///pipeline types.
    ///
    ///**Specialisation constants**: use `SpecializationInfo` on the shader
    ///stage to bake compile-time constants into the shader (e.g. workgroup
    ///size, algorithm variant). This produces optimised code without
    ///duplicating shader source.
    ///
    ///Batch multiple compute pipelines in a single call when possible.
    ///
    ///Compute pipelines can be created at any time and are not tied to a
    ///render pass. They are bound with `cmd_bind_pipeline` using
    ///`PIPELINE_BIND_POINT_COMPUTE` and dispatched with `cmd_dispatch`.
    pub unsafe fn create_compute_pipelines(
        &self,
        pipeline_cache: PipelineCache,
        p_create_infos: &[ComputePipelineCreateInfo],
        allocator: Option<&AllocationCallbacks>,
        p_pipelines: *mut Pipeline,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .create_compute_pipelines
            .expect("vkCreateComputePipelines not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        check(unsafe {
            fp(
                self.handle(),
                pipeline_cache,
                p_create_infos.len() as u32,
                p_create_infos.as_ptr(),
                alloc_ptr,
                p_pipelines,
            )
        })
    }
    ///Wraps [`vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI.html).
    /**
    Provided by **VK_HUAWEI_subpass_shading**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the maximum workgroup size supported for subpass shading
    ///on the given render pass. Returns an `Extent2D` with the max
    ///width and height. Use this to configure subpass shading dispatch
    ///parameters.
    ///
    ///Requires `VK_HUAWEI_subpass_shading`.
    pub unsafe fn get_device_subpass_shading_max_workgroup_size_huawei(
        &self,
        renderpass: RenderPass,
        p_max_workgroup_size: *mut Extent2D,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_device_subpass_shading_max_workgroup_size_huawei
            .expect("vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI not loaded");
        check(unsafe { fp(self.handle(), renderpass, p_max_workgroup_size) })
    }
    ///Wraps [`vkDestroyPipeline`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyPipeline.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `pipeline` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a graphics, compute, or ray tracing pipeline. The pipeline
    ///must not be bound in any command buffer that is still pending
    ///execution.
    ///
    ///Pipelines are expensive to create but cheap to keep around. Only
    ///destroy them when you are certain they will not be needed again
    ///(e.g. during level transitions or application shutdown).
    ///
    ///Shader modules used to create the pipeline can be destroyed
    ///independently, the pipeline retains its own copy of the compiled
    ///state.
    pub unsafe fn destroy_pipeline(
        &self,
        pipeline: Pipeline,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_pipeline
            .expect("vkDestroyPipeline not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), pipeline, alloc_ptr) };
    }
    ///Wraps [`vkCreatePipelineLayout`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreatePipelineLayout.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///A pipeline layout defines the interface between shader stages and
    ///the descriptor sets and push constants that feed them. It specifies:
    ///
    ///- **Descriptor set layouts**: which bindings are available at each
    ///  set index (0, 1, 2, ...).
    ///- **Push constant ranges**: byte ranges per shader stage for small,
    ///  frequently-updated data.
    ///
    ///**Set layout ordering convention**: a common pattern is:
    ///
    ///- Set 0: per-frame data (camera, time).
    ///- Set 1: per-material data (textures, material params).
    ///- Set 2: per-object data (transforms).
    ///
    ///This lets you bind set 0 once per frame and only rebind sets 1–2
    ///as materials and objects change, minimising descriptor set switches.
    ///
    ///Pipeline layouts are immutable after creation. Two pipelines that
    ///share the same layout can share descriptor sets without rebinding.
    ///
    ///Push constants are limited to `max_push_constants_size` bytes
    ///(guaranteed at least 128). Use them for small per-draw data like
    ///transform matrices or material indices.
    pub unsafe fn create_pipeline_layout(
        &self,
        p_create_info: &PipelineLayoutCreateInfo,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<PipelineLayout> {
        let fp = self
            .commands()
            .create_pipeline_layout
            .expect("vkCreatePipelineLayout not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyPipelineLayout`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyPipelineLayout.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `pipelineLayout` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a pipeline layout. All pipelines and descriptor sets that
    ///were created with this layout must no longer be in use.
    ///
    ///In practice, pipeline layouts are typically created once and live for
    ///the duration of the application or a major rendering context. There
    ///is little reason to destroy them early.
    pub unsafe fn destroy_pipeline_layout(
        &self,
        pipeline_layout: PipelineLayout,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_pipeline_layout
            .expect("vkDestroyPipelineLayout not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), pipeline_layout, alloc_ptr) };
    }
    ///Wraps [`vkCreateSampler`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateSampler.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a sampler that controls how shaders read image data:
    ///filtering, addressing, mip level selection, and anisotropy.
    ///
    ///**Common configurations**:
    ///
    ///- **Nearest/point**: `MIN_FILTER_NEAREST`, `MAG_FILTER_NEAREST`.
    ///  No interpolation, pixel art, data textures, or shadow map
    ///  comparison.
    ///- **Bilinear**: `MIN_FILTER_LINEAR`, `MAG_FILTER_LINEAR`,
    ///  `MIPMAP_MODE_NEAREST`. Smooth within a mip level but snaps
    ///  between levels.
    ///- **Trilinear**: same as bilinear but with `MIPMAP_MODE_LINEAR`.
    ///  Smooth transitions between mip levels. The default choice for
    ///  most 3D textures.
    ///- **Anisotropic**: enable `anisotropy_enable` and set
    ///  `max_anisotropy` (commonly 4–16). Improves quality at oblique
    ///  viewing angles at a small GPU cost.
    ///
    ///**Address modes** (`REPEAT`, `MIRRORED_REPEAT`, `CLAMP_TO_EDGE`,
    ///`CLAMP_TO_BORDER`) control what happens when UVs go outside [0, 1].
    ///
    ///Samplers are immutable after creation and can be shared across any
    ///number of descriptor sets. Most applications need only a handful of
    ///samplers.
    pub unsafe fn create_sampler(
        &self,
        p_create_info: &SamplerCreateInfo,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<Sampler> {
        let fp = self
            .commands()
            .create_sampler
            .expect("vkCreateSampler not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroySampler`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroySampler.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `sampler` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a sampler. The sampler must not be referenced by any
    ///descriptor set that is bound in a pending command buffer.
    ///
    ///Since most applications use a small fixed set of samplers, they are
    ///typically created once at startup and destroyed only during
    ///application shutdown.
    pub unsafe fn destroy_sampler(
        &self,
        sampler: Sampler,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_sampler
            .expect("vkDestroySampler not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), sampler, alloc_ptr) };
    }
    ///Wraps [`vkCreateDescriptorSetLayout`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateDescriptorSetLayout.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///A descriptor set layout defines the shape of a descriptor set: which
    ///binding numbers exist, what descriptor type each binding holds, and
    ///at which shader stages each binding is visible.
    ///
    ///**Binding tips**:
    ///
    ///- Keep `stage_flags` as narrow as possible. Declaring a binding
    ///  visible to all stages when only the fragment shader uses it wastes
    ///  driver resources on some implementations.
    ///- Use `DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER` for simple texture
    ///  sampling. Separate sampler + sampled image bindings offer more
    ///  flexibility when you want to reuse samplers across many textures.
    ///- Array descriptors (`descriptor_count > 1`) map to GLSL arrays.
    ///  Useful for bindless or material-table patterns.
    ///
    ///Layouts are immutable after creation and can be shared across
    ///multiple pipeline layouts and descriptor set allocations.
    ///
    ///Destroy with `destroy_descriptor_set_layout` when no pipeline layout
    ///or pending descriptor set allocation still references it.
    pub unsafe fn create_descriptor_set_layout(
        &self,
        p_create_info: &DescriptorSetLayoutCreateInfo,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<DescriptorSetLayout> {
        let fp = self
            .commands()
            .create_descriptor_set_layout
            .expect("vkCreateDescriptorSetLayout not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyDescriptorSetLayout`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyDescriptorSetLayout.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `descriptorSetLayout` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a descriptor set layout. The layout must not be referenced
    ///by any pipeline layout or pending descriptor set allocation that is
    ///still in use.
    ///
    ///Descriptor set layouts are lightweight and typically long-lived.
    ///Destroy them during application shutdown after all dependent
    ///pipeline layouts and descriptor pools have been destroyed.
    pub unsafe fn destroy_descriptor_set_layout(
        &self,
        descriptor_set_layout: DescriptorSetLayout,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_descriptor_set_layout
            .expect("vkDestroyDescriptorSetLayout not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), descriptor_set_layout, alloc_ptr) };
    }
    ///Wraps [`vkCreateDescriptorPool`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateDescriptorPool.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_FRAGMENTATION_EXT`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a pool from which descriptor sets are allocated. The pool
    ///must be sized to accommodate all the descriptor sets and individual
    ///descriptor types your application needs.
    ///
    ///**Sizing**: specify `max_sets` (total descriptor sets) and a list of
    ///`DescriptorPoolSize` entries that declare how many descriptors of
    ///each type the pool holds. Under-sizing causes
    ///`VK_ERROR_OUT_OF_POOL_MEMORY` at allocation time.
    ///
    ///**Flags**:
    ///
    ///- `FREE_DESCRIPTOR_SET`: allows individual sets to be freed with
    ///  `free_descriptor_sets`. Without this flag, sets can only be
    ///  reclaimed by resetting the entire pool.
    ///- `UPDATE_AFTER_BIND`: required if any allocated set uses
    ///  update-after-bind bindings.
    ///
    ///**Common pattern**: create one pool per frame-in-flight. At the
    ///start of each frame, `reset_descriptor_pool` reclaims all sets at
    ///once, no individual tracking or freeing needed.
    pub unsafe fn create_descriptor_pool(
        &self,
        p_create_info: &DescriptorPoolCreateInfo,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<DescriptorPool> {
        let fp = self
            .commands()
            .create_descriptor_pool
            .expect("vkCreateDescriptorPool not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyDescriptorPool`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyDescriptorPool.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `descriptorPool` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a descriptor pool and implicitly frees all descriptor sets
    ///allocated from it. You do not need to free individual sets before
    ///destroying the pool.
    ///
    ///Ensure no command buffer that references any set from this pool is
    ///still pending execution.
    pub unsafe fn destroy_descriptor_pool(
        &self,
        descriptor_pool: DescriptorPool,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_descriptor_pool
            .expect("vkDestroyDescriptorPool not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), descriptor_pool, alloc_ptr) };
    }
    ///Wraps [`vkResetDescriptorPool`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkResetDescriptorPool.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `descriptorPool` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Recycles all descriptor sets allocated from this pool back to the
    ///pool, without destroying the pool itself. After a reset, all
    ///previously allocated sets are invalid and must not be used.
    ///
    ///This is the fastest way to reclaim descriptor sets, much cheaper
    ///than freeing them individually. Ideal for the per-frame pool pattern
    ///where you allocate fresh sets every frame and reset the pool at the
    ///start of the next frame.
    ///
    ///No command buffer that references any set from this pool may be
    ///pending execution when you reset.
    pub unsafe fn reset_descriptor_pool(
        &self,
        descriptor_pool: DescriptorPool,
        flags: DescriptorPoolResetFlags,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .reset_descriptor_pool
            .expect("vkResetDescriptorPool not loaded");
        check(unsafe { fp(self.handle(), descriptor_pool, flags) })
    }
    ///Wraps [`vkAllocateDescriptorSets`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkAllocateDescriptorSets.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_FRAGMENTED_POOL`
    ///- `VK_ERROR_OUT_OF_POOL_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Allocates descriptor sets from a descriptor pool. Each set is backed
    ///by one of the `set_layouts` in the allocate info.
    ///
    ///Common failure causes:
    ///
    ///- **Pool exhaustion**: the pool does not have enough descriptors of
    ///  the required types, or the maximum set count has been reached.
    ///  Returns `VK_ERROR_OUT_OF_POOL_MEMORY`. Pre-calculate pool sizes
    ///  carefully or use `VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT`
    ///  to allow freeing and reallocation.
    ///- **Fragmentation**: even with enough total descriptors, internal
    ///  fragmentation can cause allocation failure. Resetting the entire
    ///  pool with `reset_descriptor_pool` defragments it.
    ///
    ///Descriptor sets become invalid when their parent pool is destroyed
    ///or reset. Do not submit command buffers that reference descriptor
    ///sets from a pool that has been reset.
    ///
    ///For frequently-updated descriptors, consider
    ///`VK_KHR_push_descriptor` which avoids set allocation entirely.
    pub unsafe fn allocate_descriptor_sets(
        &self,
        p_allocate_info: &DescriptorSetAllocateInfo,
        p_descriptor_sets: *mut DescriptorSet,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .allocate_descriptor_sets
            .expect("vkAllocateDescriptorSets not loaded");
        check(unsafe { fp(self.handle(), p_allocate_info, p_descriptor_sets) })
    }
    ///Wraps [`vkFreeDescriptorSets`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkFreeDescriptorSets.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `descriptorPool` must be externally synchronized.
    ///- `pDescriptorSets` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Returns individual descriptor sets back to their parent pool. The
    ///pool must have been created with `FREE_DESCRIPTOR_SET`, without
    ///that flag this call is invalid.
    ///
    ///For most applications, resetting the entire pool with
    ///`reset_descriptor_pool` is simpler and faster than tracking and
    ///freeing individual sets. Use `free_descriptor_sets` only when you
    ///need fine-grained lifetime control over specific sets.
    ///
    ///Freed sets must not be referenced by any pending command buffer.
    pub unsafe fn free_descriptor_sets(
        &self,
        descriptor_pool: DescriptorPool,
        p_descriptor_sets: &[DescriptorSet],
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .free_descriptor_sets
            .expect("vkFreeDescriptorSets not loaded");
        check(unsafe {
            fp(
                self.handle(),
                descriptor_pool,
                p_descriptor_sets.len() as u32,
                p_descriptor_sets.as_ptr(),
            )
        })
    }
    ///Wraps [`vkUpdateDescriptorSets`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkUpdateDescriptorSets.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `pDescriptorWrites` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Writes or copies resource bindings into descriptor sets. This is
    ///how you connect actual buffers, images, and samplers to the
    ///descriptor slots that shaders read from.
    ///
    ///**Writes** (`WriteDescriptorSet`): bind concrete resources to a
    ///specific set + binding + array element. Each write targets one
    ///descriptor type (uniform buffer, combined image sampler, storage
    ///buffer, etc.).
    ///
    ///**Copies** (`CopyDescriptorSet`): duplicate bindings from one set
    ///to another. Rarely used, writes cover nearly all cases.
    ///
    ///Updates take effect immediately and are visible to any command buffer
    ///recorded after the update. However, updating a set that is currently
    ///bound in a pending command buffer is undefined behaviour unless the
    ///set was allocated from a pool with `UPDATE_AFTER_BIND` and the
    ///binding is marked as update-after-bind in the layout.
    ///
    ///Batch multiple writes into a single call when possible, the driver
    ///can often process them more efficiently.
    pub unsafe fn update_descriptor_sets(
        &self,
        p_descriptor_writes: &[WriteDescriptorSet],
        p_descriptor_copies: &[CopyDescriptorSet],
    ) {
        let fp = self
            .commands()
            .update_descriptor_sets
            .expect("vkUpdateDescriptorSets not loaded");
        unsafe {
            fp(
                self.handle(),
                p_descriptor_writes.len() as u32,
                p_descriptor_writes.as_ptr(),
                p_descriptor_copies.len() as u32,
                p_descriptor_copies.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCreateFramebuffer`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateFramebuffer.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///A framebuffer binds concrete image views to the attachment slots
    ///defined by a render pass. The number and format of attachments must
    ///match the render pass exactly, mismatches cause validation errors.
    ///
    ///**Dimensions**: `width`, `height`, and `layers` must be less than
    ///or equal to the corresponding dimensions of every attached image
    ///view. They define the renderable area for `cmd_begin_render_pass`.
    ///
    ///**Lifetime**: the framebuffer must stay alive for the entire
    ///duration of any render pass instance that uses it. In practice,
    ///framebuffers are typically recreated when the swapchain is resized.
    ///
    ///**Imageless framebuffers** (Vulkan 1.2+): create the framebuffer
    ///with `FRAMEBUFFER_CREATE_IMAGELESS` and no attachments. Concrete
    ///image views are then supplied at `cmd_begin_render_pass` time via
    ///`RenderPassAttachmentBeginInfo`. This avoids recreating framebuffers
    ///on swapchain resize.
    pub unsafe fn create_framebuffer(
        &self,
        p_create_info: &FramebufferCreateInfo,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<Framebuffer> {
        let fp = self
            .commands()
            .create_framebuffer
            .expect("vkCreateFramebuffer not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyFramebuffer`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyFramebuffer.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `framebuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a framebuffer. No render pass instance using this
    ///framebuffer may be pending execution.
    ///
    ///Framebuffers are typically recreated whenever the swapchain is
    ///resized, so they tend to have shorter lifetimes than most Vulkan
    ///objects. With imageless framebuffers (Vulkan 1.2+) you can avoid
    ///this churn entirely.
    pub unsafe fn destroy_framebuffer(
        &self,
        framebuffer: Framebuffer,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_framebuffer
            .expect("vkDestroyFramebuffer not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), framebuffer, alloc_ptr) };
    }
    ///Wraps [`vkCreateRenderPass`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateRenderPass.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///A render pass describes the attachments, subpasses, and dependencies
    ///used during rendering. It does not reference actual images, those
    ///are bound later via a framebuffer.
    ///
    ///Key design points:
    ///
    ///- **`load_op` / `store_op`**: use `DONT_CARE` for attachments whose
    ///  prior contents are irrelevant (e.g. a transient depth buffer). This
    ///  lets tile-based GPUs skip loads/stores, which is significant on
    ///  mobile.
    ///- **`initial_layout` / `final_layout`**: Vulkan inserts implicit layout
    ///  transitions at render pass boundaries. Set these to match your actual
    ///  usage to avoid unnecessary transitions. `UNDEFINED` for `initial_layout`
    ///  is fine when `load_op` is `CLEAR` or `DONT_CARE`.
    ///- **Subpass dependencies**: the implicit external dependencies
    ///  (`VK_SUBPASS_EXTERNAL`) are often insufficient. Add explicit
    ///  dependencies when subsequent passes read the output.
    ///
    ///For dynamic rendering (Vulkan 1.3+), consider `cmd_begin_rendering`
    ///instead, which avoids the need for render pass and framebuffer objects.
    pub unsafe fn create_render_pass(
        &self,
        p_create_info: &RenderPassCreateInfo,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<RenderPass> {
        let fp = self
            .commands()
            .create_render_pass
            .expect("vkCreateRenderPass not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyRenderPass`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyRenderPass.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `renderPass` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a render pass. All framebuffers and pipelines created with
    ///this render pass must no longer be in use.
    ///
    ///Render passes are typically long-lived, created once at startup
    ///and destroyed during shutdown. Destroying a render pass does not
    ///affect pipelines that were created with a compatible render pass
    ///(same attachment count, formats, and sample counts).
    pub unsafe fn destroy_render_pass(
        &self,
        render_pass: RenderPass,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_render_pass
            .expect("vkDestroyRenderPass not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), render_pass, alloc_ptr) };
    }
    ///Wraps [`vkGetRenderAreaGranularity`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetRenderAreaGranularity.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns the granularity (in pixels) at which the render area should
    ///be aligned for optimal performance with this render pass.
    ///
    ///The render area passed to `cmd_begin_render_pass` should have its
    ///`offset` be a multiple of this granularity and its `extent` should
    ///either cover the full framebuffer or be rounded up to a multiple.
    ///
    ///On most desktop GPUs this returns (1, 1), meaning any alignment is
    ///fine. On tile-based GPUs this may return the tile size (e.g. 32×32),
    ///and misalignment can cause partial-tile overhead.
    ///
    ///In practice, most applications render to the full framebuffer extent
    ///and never need to worry about this.
    pub unsafe fn get_render_area_granularity(&self, render_pass: RenderPass) -> Extent2D {
        let fp = self
            .commands()
            .get_render_area_granularity
            .expect("vkGetRenderAreaGranularity not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(self.handle(), render_pass, &mut out) };
        out
    }
    ///Wraps [`vkGetRenderingAreaGranularity`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetRenderingAreaGranularity.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_4**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.4 command that queries the optimal render area granularity
    ///for a given dynamic rendering configuration, without needing a
    ///render pass object.
    ///
    ///This is the dynamic rendering equivalent of
    ///`get_render_area_granularity`. Pass a `RenderingAreaInfo` describing
    ///the attachment formats and sample count, and receive the granularity
    ///(in pixels) at which the render area should be aligned.
    ///
    ///On most desktop GPUs this returns (1, 1). On tile-based GPUs the
    ///granularity may match the tile size.
    pub unsafe fn get_rendering_area_granularity(
        &self,
        p_rendering_area_info: &RenderingAreaInfo,
    ) -> Extent2D {
        let fp = self
            .commands()
            .get_rendering_area_granularity
            .expect("vkGetRenderingAreaGranularity not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(self.handle(), p_rendering_area_info, &mut out) };
        out
    }
    ///Wraps [`vkCreateCommandPool`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateCommandPool.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///A command pool provides the memory backing for command buffers
    ///allocated from it. Pools are tied to a single queue family, command
    ///buffers allocated from the pool can only be submitted to queues of
    ///that family.
    ///
    ///**Flags**:
    ///
    ///- `TRANSIENT`: hint that command buffers are short-lived and reset or
    ///  freed frequently. Lets the driver use a faster allocation strategy.
    ///- `RESET_COMMAND_BUFFER`: allows individual command buffers to be
    ///  reset via `reset_command_buffer`. Without this flag you must reset
    ///  the entire pool with `reset_command_pool`.
    ///
    ///A common pattern is one pool per frame-in-flight per thread: reset the
    ///whole pool at the start of each frame instead of managing individual
    ///command buffer lifetimes.
    ///
    ///Command pools are **not thread-safe**. If multiple threads record
    ///commands concurrently, each thread needs its own pool.
    pub unsafe fn create_command_pool(
        &self,
        p_create_info: &CommandPoolCreateInfo,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<CommandPool> {
        let fp = self
            .commands()
            .create_command_pool
            .expect("vkCreateCommandPool not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyCommandPool`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyCommandPool.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `commandPool` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a command pool and implicitly frees all command buffers
    ///allocated from it. You do not need to free individual command buffers
    ///before destroying the pool.
    ///
    ///All command buffers allocated from this pool must have completed
    ///execution before the pool is destroyed. Call `device_wait_idle` or
    ///wait on the relevant fences first.
    pub unsafe fn destroy_command_pool(
        &self,
        command_pool: CommandPool,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_command_pool
            .expect("vkDestroyCommandPool not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), command_pool, alloc_ptr) };
    }
    ///Wraps [`vkResetCommandPool`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkResetCommandPool.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `commandPool` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Resets a command pool, recycling all command buffers allocated from
    ///it back to the initial state. This is faster than resetting
    ///individual command buffers.
    ///
    ///**Flags**:
    ///
    ///- `RELEASE_RESOURCES`: return memory to the system. Without this
    ///  flag, the pool keeps its internal memory for reuse by future
    ///  allocations, usually what you want in a frame loop.
    ///
    ///**Per-frame pattern**: reset the pool at the start of each frame
    ///(without `RELEASE_RESOURCES`), then re-record command buffers from
    ///the same pool. Memory is reused without reallocation overhead.
    ///
    ///All command buffers from this pool must have completed execution
    ///before resetting.
    pub unsafe fn reset_command_pool(
        &self,
        command_pool: CommandPool,
        flags: CommandPoolResetFlags,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .reset_command_pool
            .expect("vkResetCommandPool not loaded");
        check(unsafe { fp(self.handle(), command_pool, flags) })
    }
    ///Wraps [`vkAllocateCommandBuffers`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkAllocateCommandBuffers.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Allocates one or more command buffers from a command pool. The
    ///`command_buffer_count` and output slice length must match.
    ///
    ///**Level**:
    ///
    ///- `PRIMARY`: submitted directly to a queue via `queue_submit`.
    ///- `SECONDARY`: recorded separately and executed inside a primary
    ///  buffer via `cmd_execute_commands`. Useful for pre-recording
    ///  draw calls that are reused across frames.
    ///
    ///All allocated command buffers start in the *initial* state and must
    ///be recorded with `begin_command_buffer` before submission.
    ///
    ///Command buffers are freed either individually with
    ///`free_command_buffers` or implicitly when the parent pool is
    ///destroyed or reset.
    pub unsafe fn allocate_command_buffers(
        &self,
        p_allocate_info: &CommandBufferAllocateInfo,
        p_command_buffers: *mut CommandBuffer,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .allocate_command_buffers
            .expect("vkAllocateCommandBuffers not loaded");
        check(unsafe { fp(self.handle(), p_allocate_info, p_command_buffers) })
    }
    ///Wraps [`vkFreeCommandBuffers`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkFreeCommandBuffers.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `commandPool` must be externally synchronized.
    ///- `pCommandBuffers` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Returns individual command buffers to their parent pool. The command
    ///buffers must not be pending execution.
    ///
    ///For most applications, resetting the entire pool with
    ///`reset_command_pool` is simpler. Use `free_command_buffers` only
    ///when you need fine-grained lifetime control, for example, freeing
    ///a one-shot transfer command buffer immediately after use while
    ///keeping other buffers from the same pool alive.
    ///
    ///Freed command buffer handles become invalid. Do not resubmit them.
    pub unsafe fn free_command_buffers(
        &self,
        command_pool: CommandPool,
        p_command_buffers: &[CommandBuffer],
    ) {
        let fp = self
            .commands()
            .free_command_buffers
            .expect("vkFreeCommandBuffers not loaded");
        unsafe {
            fp(
                self.handle(),
                command_pool,
                p_command_buffers.len() as u32,
                p_command_buffers.as_ptr(),
            )
        };
    }
    ///Wraps [`vkBeginCommandBuffer`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkBeginCommandBuffer.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Begins recording commands into a command buffer. The command buffer
    ///must be in the *initial* state, either freshly allocated, or reset
    ///via `reset_command_buffer` / `reset_command_pool`.
    ///
    ///**Flags**:
    ///
    ///- `ONE_TIME_SUBMIT`: the command buffer will be submitted once and
    ///  then reset or freed. Lets the driver skip internal tracking it
    ///  would otherwise need for resubmission.
    ///- `SIMULTANEOUS_USE`: the command buffer can be pending execution on
    ///  multiple queues simultaneously. Required for secondary command
    ///  buffers reused across multiple primary buffers.
    ///
    ///**Inheritance info**: only required for secondary command buffers.
    ///When recording a secondary buffer that will execute inside a render
    ///pass, set `render_pass`, `subpass`, and optionally `framebuffer` in
    ///the `CommandBufferInheritanceInfo`. For primary buffers the
    ///inheritance info is ignored.
    ///
    ///Calling `begin_command_buffer` on a buffer that is already recording
    ///is an error. Calling it on a buffer in the *executable* state
    ///implicitly resets it first (if the pool allows it).
    pub unsafe fn begin_command_buffer(
        &self,
        command_buffer: CommandBuffer,
        p_begin_info: &CommandBufferBeginInfo,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .begin_command_buffer
            .expect("vkBeginCommandBuffer not loaded");
        check(unsafe { fp(command_buffer, p_begin_info) })
    }
    ///Wraps [`vkEndCommandBuffer`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkEndCommandBuffer.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Finishes recording a command buffer. After this call the command
    ///buffer moves from the *recording* state to the *executable* state
    ///and can be submitted via `queue_submit`.
    ///
    ///If an error occurred during recording (e.g. out of memory), this
    ///call returns the error. Always check the return value, a failed
    ///`end_command_buffer` means the command buffer is in an invalid state
    ///and must be reset before reuse.
    ///
    ///A command buffer that is inside a render pass must end the render
    ///pass with `cmd_end_render_pass` before calling `end_command_buffer`.
    pub unsafe fn end_command_buffer(&self, command_buffer: CommandBuffer) -> VkResult<()> {
        let fp = self
            .commands()
            .end_command_buffer
            .expect("vkEndCommandBuffer not loaded");
        check(unsafe { fp(command_buffer) })
    }
    ///Wraps [`vkResetCommandBuffer`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkResetCommandBuffer.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Resets a single command buffer back to the initial state so it can
    ///be re-recorded. The command pool must have been created with
    ///`RESET_COMMAND_BUFFER` for this to be valid.
    ///
    ///**Flags**:
    ///
    ///- `RELEASE_RESOURCES`: return the command buffer's memory to the
    ///  pool. Without this flag, memory is retained for reuse during the
    ///  next recording, usually preferred in a frame loop.
    ///
    ///For bulk resets, `reset_command_pool` is more efficient than
    ///resetting buffers individually.
    ///
    ///The command buffer must not be pending execution when reset.
    pub unsafe fn reset_command_buffer(
        &self,
        command_buffer: CommandBuffer,
        flags: CommandBufferResetFlags,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .reset_command_buffer
            .expect("vkResetCommandBuffer not loaded");
        check(unsafe { fp(command_buffer, flags) })
    }
    ///Wraps [`vkCmdBindPipeline`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBindPipeline.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Binds a pipeline to a command buffer for subsequent draw or dispatch
    ///calls. The `pipeline_bind_point` must match the pipeline type:
    ///
    ///- `PIPELINE_BIND_POINT_GRAPHICS` for graphics pipelines.
    ///- `PIPELINE_BIND_POINT_COMPUTE` for compute pipelines.
    ///- `PIPELINE_BIND_POINT_RAY_TRACING_KHR` for ray tracing pipelines.
    ///
    ///Binding a pipeline invalidates any incompatible dynamic state. For
    ///example, binding a new graphics pipeline that uses dynamic viewport
    ///requires you to call `cmd_set_viewport` again before drawing.
    ///
    ///Pipeline binds are relatively cheap, the driver patches command
    ///state internally. Minimise binds by sorting draw calls by pipeline
    ///when possible, but do not over-optimise at the expense of code
    ///clarity.
    ///
    ///Graphics pipelines can only be bound inside a render pass (or
    ///dynamic rendering). Compute pipelines can be bound anywhere.
    pub unsafe fn cmd_bind_pipeline(
        &self,
        command_buffer: CommandBuffer,
        pipeline_bind_point: PipelineBindPoint,
        pipeline: Pipeline,
    ) {
        let fp = self
            .commands()
            .cmd_bind_pipeline
            .expect("vkCmdBindPipeline not loaded");
        unsafe { fp(command_buffer, pipeline_bind_point, pipeline) };
    }
    ///Wraps [`vkCmdSetAttachmentFeedbackLoopEnableEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetAttachmentFeedbackLoopEnableEXT.html).
    /**
    Provided by **VK_EXT_attachment_feedback_loop_dynamic_state**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically enables or disables attachment feedback loops for
    ///specific image aspects (color, depth, stencil). When enabled,
    ///shaders can both read from and write to the same attachment
    ///within a render pass.
    ///
    ///Use with care, feedback loops create read-after-write hazards.
    ///The implementation handles coherency when this flag is set.
    ///
    ///Requires `VK_EXT_attachment_feedback_loop_dynamic_state`.
    pub unsafe fn cmd_set_attachment_feedback_loop_enable_ext(
        &self,
        command_buffer: CommandBuffer,
        aspect_mask: ImageAspectFlags,
    ) {
        let fp = self
            .commands()
            .cmd_set_attachment_feedback_loop_enable_ext
            .expect("vkCmdSetAttachmentFeedbackLoopEnableEXT not loaded");
        unsafe { fp(command_buffer, aspect_mask) };
    }
    ///Wraps [`vkCmdSetViewport`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetViewport.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Sets the viewport transform dynamically. Only takes effect if the
    ///pipeline was created with `DYNAMIC_STATE_VIEWPORT`.
    ///
    ///The viewport defines the mapping from normalised device coordinates
    ///to framebuffer coordinates. Most applications use a single viewport
    ///covering the full render target:
    ///
    ///```text
    ///Viewport { x: 0.0, y: 0.0, width: w, height: h, min_depth: 0.0, max_depth: 1.0 }
    ///```
    ///
    ///**Flipped Y**: Vulkan's clip space has Y pointing downward (unlike
    ///OpenGL). To match OpenGL conventions, use a negative height and
    ///offset Y by the framebuffer height:
    ///`Viewport { y: h, height: -h, ... }`. This requires Vulkan 1.1 or
    ///`VK_KHR_maintenance1`.
    ///
    ///Multiple viewports are supported for multi-view rendering (e.g.
    ///VR). The `first_viewport` parameter selects which viewport index to
    ///start writing at.
    pub unsafe fn cmd_set_viewport(
        &self,
        command_buffer: CommandBuffer,
        first_viewport: u32,
        p_viewports: &[Viewport],
    ) {
        let fp = self
            .commands()
            .cmd_set_viewport
            .expect("vkCmdSetViewport not loaded");
        unsafe {
            fp(
                command_buffer,
                first_viewport,
                p_viewports.len() as u32,
                p_viewports.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdSetScissor`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetScissor.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Sets the scissor rectangle dynamically. Only takes effect if the
    ///pipeline was created with `DYNAMIC_STATE_SCISSOR`.
    ///
    ///The scissor test discards fragments outside the rectangle. Unlike
    ///the viewport (which transforms coordinates), the scissor is a
    ///hard clip in framebuffer pixel coordinates.
    ///
    ///A common default is a scissor covering the full framebuffer:
    ///
    ///```text
    ///Rect2D { offset: { x: 0, y: 0 }, extent: { width: w, height: h } }
    ///```
    ///
    ///Scissor rectangles are useful for UI rendering, split-screen, or
    ///any case where you want to restrict rendering to a sub-region
    ///without changing the viewport transform.
    ///
    ///The scissor must be set before any draw call when using dynamic
    ///scissor state, even if it covers the full framebuffer.
    pub unsafe fn cmd_set_scissor(
        &self,
        command_buffer: CommandBuffer,
        first_scissor: u32,
        p_scissors: &[Rect2D],
    ) {
        let fp = self
            .commands()
            .cmd_set_scissor
            .expect("vkCmdSetScissor not loaded");
        unsafe {
            fp(
                command_buffer,
                first_scissor,
                p_scissors.len() as u32,
                p_scissors.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdSetLineWidth`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetLineWidth.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Sets the width of rasterised line primitives dynamically. Only takes
    ///effect if the pipeline was created with `DYNAMIC_STATE_LINE_WIDTH`.
    ///
    ///The default line width is 1.0. Wide lines (width > 1.0) require the
    ///`wide_lines` device feature, check
    ///`physical_device_features.wide_lines` before using.
    ///
    ///The supported range is device-dependent (query
    ///`physical_device_limits.line_width_range`). If `wide_lines` is not
    ///supported, only 1.0 is valid.
    ///
    ///Line width is specified in framebuffer pixels. Anti-aliased lines
    ///may be rendered slightly wider than the specified width due to
    ///coverage calculations.
    pub unsafe fn cmd_set_line_width(&self, command_buffer: CommandBuffer, line_width: f32) {
        let fp = self
            .commands()
            .cmd_set_line_width
            .expect("vkCmdSetLineWidth not loaded");
        unsafe { fp(command_buffer, line_width) };
    }
    ///Wraps [`vkCmdSetDepthBias`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetDepthBias.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Sets depth bias parameters dynamically. Only takes effect if the
    ///pipeline was created with `DYNAMIC_STATE_DEPTH_BIAS` and depth bias
    ///is enabled in the rasterisation state.
    ///
    ///Depth bias adds a computed offset to each fragment's depth value
    ///before the depth test. The primary use case is **shadow mapping**,
    ///biasing shadow caster geometry slightly away from the light prevents
    ///self-shadowing artifacts (shadow acne).
    ///
    ///The final bias is computed as:
    ///
    ///```text
    ///bias = constant_factor * r + slope_factor * max_slope
    ///```
    ///
    ///where `r` is the minimum resolvable depth difference and `max_slope`
    ///is the maximum depth slope of the triangle.
    ///
    ///**`depth_bias_clamp`** limits the maximum bias value. Requires the
    ///`depth_bias_clamp` device feature. A clamp of 0.0 disables clamping.
    ///
    ///Typical shadow map values: `constant_factor` = 1.25,
    ///`slope_factor` = 1.75, `clamp` = 0.0. Tune per scene.
    pub unsafe fn cmd_set_depth_bias(
        &self,
        command_buffer: CommandBuffer,
        depth_bias_constant_factor: f32,
        depth_bias_clamp: f32,
        depth_bias_slope_factor: f32,
    ) {
        let fp = self
            .commands()
            .cmd_set_depth_bias
            .expect("vkCmdSetDepthBias not loaded");
        unsafe {
            fp(
                command_buffer,
                depth_bias_constant_factor,
                depth_bias_clamp,
                depth_bias_slope_factor,
            )
        };
    }
    ///Wraps [`vkCmdSetBlendConstants`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetBlendConstants.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Sets the constant blend colour used when a blend factor references
    ///`BLEND_FACTOR_CONSTANT_COLOR`, `BLEND_FACTOR_CONSTANT_ALPHA`, or
    ///their one-minus variants. Only takes effect if the pipeline was
    ///created with `DYNAMIC_STATE_BLEND_CONSTANTS`.
    ///
    ///The four values are RGBA in [0.0, 1.0]. A common use is fading
    ///geometry by setting a constant alpha and blending with
    ///`BLEND_FACTOR_CONSTANT_ALPHA`.
    ///
    ///If your pipeline does not use any constant blend factors, you do not
    ///need to set this state. The values are ignored for blend modes that
    ///do not reference them.
    pub unsafe fn cmd_set_blend_constants(
        &self,
        command_buffer: CommandBuffer,
        blend_constants: f32,
    ) {
        let fp = self
            .commands()
            .cmd_set_blend_constants
            .expect("vkCmdSetBlendConstants not loaded");
        unsafe { fp(command_buffer, blend_constants) };
    }
    ///Wraps [`vkCmdSetDepthBounds`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetDepthBounds.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Sets the depth bounds test range dynamically. Only takes effect if
    ///the pipeline was created with `DYNAMIC_STATE_DEPTH_BOUNDS` and
    ///depth bounds testing is enabled in the depth-stencil state.
    ///
    ///The depth bounds test discards fragments whose depth buffer value
    ///falls outside [`min_depth_bounds`, `max_depth_bounds`]. Note that
    ///this tests the **existing** depth buffer value, not the fragment's
    ///incoming depth.
    ///
    ///Use cases are niche:
    ///
    ///- **Stencil shadow volumes**: reject fragments that are clearly
    ///  outside the shadow volume's depth range.
    ///- **Deferred shading light volumes**: skip fragments outside the
    ///  light's depth range.
    ///
    ///Requires the `depth_bounds` device feature. Not supported on all
    ///hardware, check `physical_device_features.depth_bounds` before
    ///enabling.
    pub unsafe fn cmd_set_depth_bounds(
        &self,
        command_buffer: CommandBuffer,
        min_depth_bounds: f32,
        max_depth_bounds: f32,
    ) {
        let fp = self
            .commands()
            .cmd_set_depth_bounds
            .expect("vkCmdSetDepthBounds not loaded");
        unsafe { fp(command_buffer, min_depth_bounds, max_depth_bounds) };
    }
    ///Wraps [`vkCmdSetStencilCompareMask`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetStencilCompareMask.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Sets the stencil compare mask dynamically for front-facing,
    ///back-facing, or both face sets. Only takes effect if the pipeline
    ///was created with `DYNAMIC_STATE_STENCIL_COMPARE_MASK`.
    ///
    ///The compare mask is ANDed with both the reference value and the
    ///stencil buffer value before the stencil comparison. This lets you
    ///use individual bits of the stencil buffer for different purposes
    ///(e.g. bit 0 for portals, bits 1–3 for decal layers).
    ///
    ///A mask of `0xFF` (the default) uses all 8 bits of the stencil
    ///buffer. Narrower masks isolate specific bit planes for multi-purpose
    ///stencil schemes.
    pub unsafe fn cmd_set_stencil_compare_mask(
        &self,
        command_buffer: CommandBuffer,
        face_mask: StencilFaceFlags,
        compare_mask: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_stencil_compare_mask
            .expect("vkCmdSetStencilCompareMask not loaded");
        unsafe { fp(command_buffer, face_mask, compare_mask) };
    }
    ///Wraps [`vkCmdSetStencilWriteMask`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetStencilWriteMask.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Sets the stencil write mask dynamically for front-facing,
    ///back-facing, or both face sets. Only takes effect if the pipeline
    ///was created with `DYNAMIC_STATE_STENCIL_WRITE_MASK`.
    ///
    ///The write mask controls which bits of the stencil buffer are updated
    ///by stencil operations (`KEEP`, `REPLACE`, `INCREMENT`, etc.). Bits
    ///that are zero in the mask are left unchanged.
    ///
    ///A mask of `0xFF` writes all 8 bits. Use narrower masks when
    ///different rendering passes own different stencil bit planes.
    pub unsafe fn cmd_set_stencil_write_mask(
        &self,
        command_buffer: CommandBuffer,
        face_mask: StencilFaceFlags,
        write_mask: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_stencil_write_mask
            .expect("vkCmdSetStencilWriteMask not loaded");
        unsafe { fp(command_buffer, face_mask, write_mask) };
    }
    ///Wraps [`vkCmdSetStencilReference`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetStencilReference.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Sets the stencil reference value dynamically for front-facing,
    ///back-facing, or both face sets. Only takes effect if the pipeline
    ///was created with `DYNAMIC_STATE_STENCIL_REFERENCE`.
    ///
    ///The reference value is used in stencil comparison operations (e.g.
    ///`COMPARE_OP_EQUAL` compares the masked buffer value against the
    ///masked reference) and as the source value for `STENCIL_OP_REPLACE`.
    ///
    ///Common patterns:
    ///
    ///- **Portal/mirror rendering**: set reference to a unique ID per
    ///  portal, write it with `STENCIL_OP_REPLACE`, then test with
    ///  `COMPARE_OP_EQUAL` to mask subsequent draws to that portal's
    ///  region.
    ///- **Decal layering**: increment the reference per layer.
    pub unsafe fn cmd_set_stencil_reference(
        &self,
        command_buffer: CommandBuffer,
        face_mask: StencilFaceFlags,
        reference: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_stencil_reference
            .expect("vkCmdSetStencilReference not loaded");
        unsafe { fp(command_buffer, face_mask, reference) };
    }
    ///Wraps [`vkCmdBindDescriptorSets`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBindDescriptorSets.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Binds one or more descriptor sets to a command buffer at specified
    ///set indices. Subsequent draw or dispatch calls read resources from
    ///these bound sets.
    ///
    ///**`first_set`**: the set index at which binding starts. Sets at
    ///lower indices are not disturbed. This lets you bind per-frame data
    ///at set 0 once and rebind only per-material or per-object sets at
    ///higher indices.
    ///
    ///**Dynamic offsets**: for descriptors of type
    ///`UNIFORM_BUFFER_DYNAMIC` or `STORAGE_BUFFER_DYNAMIC`, the
    ///`dynamic_offsets` slice provides byte offsets applied at bind time.
    ///This lets multiple draw calls share a single large buffer with
    ///different sub-regions without updating the descriptor set.
    ///
    ///The bound pipeline layout and the descriptor set layouts must be
    ///compatible, same binding layout at each set index. Binding a set
    ///with an incompatible layout is undefined behaviour.
    pub unsafe fn cmd_bind_descriptor_sets(
        &self,
        command_buffer: CommandBuffer,
        pipeline_bind_point: PipelineBindPoint,
        layout: PipelineLayout,
        first_set: u32,
        p_descriptor_sets: &[DescriptorSet],
        p_dynamic_offsets: &[u32],
    ) {
        let fp = self
            .commands()
            .cmd_bind_descriptor_sets
            .expect("vkCmdBindDescriptorSets not loaded");
        unsafe {
            fp(
                command_buffer,
                pipeline_bind_point,
                layout,
                first_set,
                p_descriptor_sets.len() as u32,
                p_descriptor_sets.as_ptr(),
                p_dynamic_offsets.len() as u32,
                p_dynamic_offsets.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdBindIndexBuffer`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBindIndexBuffer.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Binds an index buffer for subsequent indexed draw calls
    ///(`cmd_draw_indexed`, `cmd_draw_indexed_indirect`).
    ///
    ///**Index type**:
    ///
    ///- `INDEX_TYPE_UINT16`: 2 bytes per index. Good for meshes with
    ///  fewer than 65536 vertices, saves memory bandwidth.
    ///- `INDEX_TYPE_UINT32`: 4 bytes per index. Required for large meshes.
    ///- `INDEX_TYPE_UINT8_KHR` (extension): 1 byte per index for very
    ///  small meshes.
    ///
    ///The buffer must have been created with `BUFFER_USAGE_INDEX_BUFFER`.
    ///The `offset` must be a multiple of the index type size (2 for
    ///UINT16, 4 for UINT32).
    ///
    ///Only one index buffer can be bound at a time, binding a new one
    ///replaces the previous binding.
    pub unsafe fn cmd_bind_index_buffer(
        &self,
        command_buffer: CommandBuffer,
        buffer: Buffer,
        offset: u64,
        index_type: IndexType,
    ) {
        let fp = self
            .commands()
            .cmd_bind_index_buffer
            .expect("vkCmdBindIndexBuffer not loaded");
        unsafe { fp(command_buffer, buffer, offset, index_type) };
    }
    ///Wraps [`vkCmdBindVertexBuffers`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBindVertexBuffers.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Binds one or more vertex buffers to input binding slots for
    ///subsequent draw calls.
    ///
    ///**`first_binding`**: the binding slot index to start at. Binding
    ///slots are defined in the pipeline's vertex input state. Multiple
    ///buffers can be bound to consecutive slots in a single call.
    ///
    ///**Interleaved vs separate**: a single buffer with interleaved
    ///attributes (position + normal + UV) uses one binding slot. Separate
    ///attribute streams (one buffer per attribute) use multiple slots.
    ///Interleaved is generally more cache-friendly.
    ///
    ///Buffers must have been created with `BUFFER_USAGE_VERTEX_BUFFER`.
    ///The `offsets` array specifies the byte offset within each buffer
    ///where vertex data starts.
    ///
    ///For Vulkan 1.3+, `cmd_bind_vertex_buffers2` also lets you set
    ///buffer sizes and strides dynamically.
    pub unsafe fn cmd_bind_vertex_buffers(
        &self,
        command_buffer: CommandBuffer,
        first_binding: u32,
        p_buffers: &[Buffer],
        p_offsets: &u64,
    ) {
        let fp = self
            .commands()
            .cmd_bind_vertex_buffers
            .expect("vkCmdBindVertexBuffers not loaded");
        unsafe {
            fp(
                command_buffer,
                first_binding,
                p_buffers.len() as u32,
                p_buffers.as_ptr(),
                p_offsets,
            )
        };
    }
    ///Wraps [`vkCmdDraw`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDraw.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Records a non-indexed draw call. Vertices are generated sequentially
    ///from `first_vertex` to `first_vertex + vertex_count - 1`.
    ///
    ///**Parameters**:
    ///
    ///- `vertex_count`: number of vertices to draw.
    ///- `instance_count`: number of instances. Use 1 for non-instanced
    ///  drawing.
    ///- `first_vertex`: offset into the vertex buffer (added to
    ///  `gl_VertexIndex` in the shader).
    ///- `first_instance`: offset into instance data (added to
    ///  `gl_InstanceIndex`). Requires the `first_instance` feature if
    ///  non-zero.
    ///
    ///For indexed geometry (the common case for meshes), use
    ///`cmd_draw_indexed` instead. `cmd_draw` is typically used for
    ///full-screen quads, procedural geometry, or particle systems where
    ///vertices are generated in the shader.
    pub unsafe fn cmd_draw(
        &self,
        command_buffer: CommandBuffer,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        let fp = self.commands().cmd_draw.expect("vkCmdDraw not loaded");
        unsafe {
            fp(
                command_buffer,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            )
        };
    }
    ///Wraps [`vkCmdDrawIndexed`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDrawIndexed.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Records an indexed draw call, the standard way to draw meshes.
    ///Indices are read from the bound index buffer, and each index is used
    ///to fetch vertex attributes from the bound vertex buffers.
    ///
    ///**Parameters**:
    ///
    ///- `index_count`: number of indices to read.
    ///- `instance_count`: number of instances. Use 1 for non-instanced.
    ///- `first_index`: offset into the index buffer (in units of indices,
    ///  not bytes).
    ///- `vertex_offset`: added to each index value before fetching the
    ///  vertex. Useful for packing multiple meshes into a single vertex
    ///  buffer at different offsets.
    ///- `first_instance`: offset into instance data.
    ///
    ///Indexed drawing reuses vertices via the index buffer, saving memory
    ///and bandwidth compared to non-indexed draws for any mesh with shared
    ///vertices (which is nearly all of them).
    pub unsafe fn cmd_draw_indexed(
        &self,
        command_buffer: CommandBuffer,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        let fp = self
            .commands()
            .cmd_draw_indexed
            .expect("vkCmdDrawIndexed not loaded");
        unsafe {
            fp(
                command_buffer,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            )
        };
    }
    ///Wraps [`vkCmdDrawMultiEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDrawMultiEXT.html).
    /**
    Provided by **VK_EXT_multi_draw**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Draws multiple non-indexed draw calls from an array of
    ///`MultiDrawInfoEXT` (first_vertex, vertex_count pairs). More
    ///efficient than issuing separate `cmd_draw` calls because the
    ///driver can batch them.
    ///
    ///Requires `VK_EXT_multi_draw` and the `multiDraw` feature.
    pub unsafe fn cmd_draw_multi_ext(
        &self,
        command_buffer: CommandBuffer,
        p_vertex_info: &[MultiDrawInfoEXT],
        instance_count: u32,
        first_instance: u32,
        stride: u32,
    ) {
        let fp = self
            .commands()
            .cmd_draw_multi_ext
            .expect("vkCmdDrawMultiEXT not loaded");
        unsafe {
            fp(
                command_buffer,
                p_vertex_info.len() as u32,
                p_vertex_info.as_ptr(),
                instance_count,
                first_instance,
                stride,
            )
        };
    }
    ///Wraps [`vkCmdDrawMultiIndexedEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDrawMultiIndexedEXT.html).
    /**
    Provided by **VK_EXT_multi_draw**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Draws multiple indexed draw calls from an array of
    ///`MultiDrawIndexedInfoEXT` (first_index, index_count,
    ///vertex_offset triples). More efficient than separate
    ///`cmd_draw_indexed` calls.
    ///
    ///An optional `p_vertex_offset` overrides all per-draw vertex
    ///offsets with a single value.
    ///
    ///Requires `VK_EXT_multi_draw` and the `multiDraw` feature.
    pub unsafe fn cmd_draw_multi_indexed_ext(
        &self,
        command_buffer: CommandBuffer,
        p_index_info: &[MultiDrawIndexedInfoEXT],
        instance_count: u32,
        first_instance: u32,
        stride: u32,
        p_vertex_offset: *const i32,
    ) {
        let fp = self
            .commands()
            .cmd_draw_multi_indexed_ext
            .expect("vkCmdDrawMultiIndexedEXT not loaded");
        unsafe {
            fp(
                command_buffer,
                p_index_info.len() as u32,
                p_index_info.as_ptr(),
                instance_count,
                first_instance,
                stride,
                p_vertex_offset,
            )
        };
    }
    ///Wraps [`vkCmdDrawIndirect`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDrawIndirect.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Records one or more non-indexed draw calls whose parameters are read
    ///from a GPU buffer at execution time rather than baked into the
    ///command buffer.
    ///
    ///Each `DrawIndirectCommand` in the buffer contains `vertex_count`,
    ///`instance_count`, `first_vertex`, and `first_instance`, the same
    ///parameters as `cmd_draw`.
    ///
    ///**Use cases**:
    ///
    ///- **GPU-driven rendering**: a compute shader fills the indirect
    ///  buffer with draw parameters (e.g. after culling), and the GPU
    ///  draws without CPU round-trips.
    ///- **Conditional draw counts**: pair with
    ///  `cmd_draw_indirect_count` (Vulkan 1.2) to have the GPU also
    ///  determine how many draws to execute.
    ///
    ///The buffer must have been created with
    ///`BUFFER_USAGE_INDIRECT_BUFFER`. The `stride` between commands must
    ///be at least `sizeof(DrawIndirectCommand)` (16 bytes) and a
    ///multiple of 4.
    pub unsafe fn cmd_draw_indirect(
        &self,
        command_buffer: CommandBuffer,
        buffer: Buffer,
        offset: u64,
        draw_count: u32,
        stride: u32,
    ) {
        let fp = self
            .commands()
            .cmd_draw_indirect
            .expect("vkCmdDrawIndirect not loaded");
        unsafe { fp(command_buffer, buffer, offset, draw_count, stride) };
    }
    ///Wraps [`vkCmdDrawIndexedIndirect`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDrawIndexedIndirect.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Records one or more indexed draw calls whose parameters are read
    ///from a GPU buffer. Each `DrawIndexedIndirectCommand` contains
    ///`index_count`, `instance_count`, `first_index`, `vertex_offset`,
    ///and `first_instance`.
    ///
    ///This is the indexed counterpart to `cmd_draw_indirect` and the most
    ///common indirect draw call for GPU-driven rendering pipelines. A
    ///compute shader performs culling and writes surviving draw commands
    ///into the buffer; the GPU then draws them without CPU involvement.
    ///
    ///The buffer must have `BUFFER_USAGE_INDIRECT_BUFFER`. The `stride`
    ///must be at least `sizeof(DrawIndexedIndirectCommand)` (20 bytes)
    ///and a multiple of 4.
    ///
    ///For dynamic draw counts, use `cmd_draw_indexed_indirect_count`
    ///(Vulkan 1.2).
    pub unsafe fn cmd_draw_indexed_indirect(
        &self,
        command_buffer: CommandBuffer,
        buffer: Buffer,
        offset: u64,
        draw_count: u32,
        stride: u32,
    ) {
        let fp = self
            .commands()
            .cmd_draw_indexed_indirect
            .expect("vkCmdDrawIndexedIndirect not loaded");
        unsafe { fp(command_buffer, buffer, offset, draw_count, stride) };
    }
    ///Wraps [`vkCmdDispatch`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDispatch.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Launches a compute shader with the given number of workgroups in
    ///X, Y, and Z dimensions. The total number of invocations is
    ///`group_count * local_size` (defined in the shader's `local_size_x`,
    ///`local_size_y`, `local_size_z`).
    ///
    ///A compute pipeline must be bound before calling this. Compute
    ///dispatches can be recorded outside a render pass.
    ///
    ///**Sizing**: to process an image of `width × height` pixels with a
    ///local size of 16×16, dispatch
    ///`ceil(width / 16) × ceil(height / 16) × 1` workgroups.
    ///
    ///**Limits**: each dimension is capped by
    ///`max_compute_work_group_count` (at least 65535 per axis). The total
    ///invocations per workgroup are capped by
    ///`max_compute_work_group_invocations` (at least 128).
    ///
    ///For GPU-driven dispatch counts, use `cmd_dispatch_indirect`.
    pub unsafe fn cmd_dispatch(
        &self,
        command_buffer: CommandBuffer,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) {
        let fp = self
            .commands()
            .cmd_dispatch
            .expect("vkCmdDispatch not loaded");
        unsafe { fp(command_buffer, group_count_x, group_count_y, group_count_z) };
    }
    ///Wraps [`vkCmdDispatchIndirect`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDispatchIndirect.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Launches a compute shader with workgroup counts read from a GPU
    ///buffer. The buffer contains a `DispatchIndirectCommand` with
    ///`group_count_x`, `group_count_y`, `group_count_z`.
    ///
    ///Use this when a prior compute pass determines how much work to do,
    ///for example, a culling pass writes the surviving workgroup count
    ///for a subsequent processing pass.
    ///
    ///The buffer must have `BUFFER_USAGE_INDIRECT_BUFFER`. The offset must
    ///be a multiple of 4.
    ///
    ///The same workgroup count limits apply as for `cmd_dispatch`.
    pub unsafe fn cmd_dispatch_indirect(
        &self,
        command_buffer: CommandBuffer,
        buffer: Buffer,
        offset: u64,
    ) {
        let fp = self
            .commands()
            .cmd_dispatch_indirect
            .expect("vkCmdDispatchIndirect not loaded");
        unsafe { fp(command_buffer, buffer, offset) };
    }
    ///Wraps [`vkCmdSubpassShadingHUAWEI`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSubpassShadingHUAWEI.html).
    /**
    Provided by **VK_HUAWEI_subpass_shading**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dispatches subpass shading work within the current subpass.
    ///Subpass shading runs compute-like shaders that have access to
    ///input attachments at the fragment's location, combining the
    ///efficiency of compute with the data locality of subpasses.
    ///
    ///Requires `VK_HUAWEI_subpass_shading`.
    pub unsafe fn cmd_subpass_shading_huawei(&self, command_buffer: CommandBuffer) {
        let fp = self
            .commands()
            .cmd_subpass_shading_huawei
            .expect("vkCmdSubpassShadingHUAWEI not loaded");
        unsafe { fp(command_buffer) };
    }
    ///Wraps [`vkCmdDrawClusterHUAWEI`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDrawClusterHUAWEI.html).
    /**
    Provided by **VK_HUAWEI_cluster_culling_shader**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dispatches cluster culling shader workgroups using the Huawei
    ///cluster culling shader pipeline. The group counts specify the
    ///3D dispatch dimensions, similar to `cmd_dispatch`.
    ///
    ///Requires `VK_HUAWEI_cluster_culling_shader`.
    pub unsafe fn cmd_draw_cluster_huawei(
        &self,
        command_buffer: CommandBuffer,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) {
        let fp = self
            .commands()
            .cmd_draw_cluster_huawei
            .expect("vkCmdDrawClusterHUAWEI not loaded");
        unsafe { fp(command_buffer, group_count_x, group_count_y, group_count_z) };
    }
    ///Wraps [`vkCmdDrawClusterIndirectHUAWEI`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDrawClusterIndirectHUAWEI.html).
    /**
    Provided by **VK_HUAWEI_cluster_culling_shader**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Indirect variant of `cmd_draw_cluster_huawei`. Reads the cluster
    ///dispatch parameters from a buffer at the given offset, allowing
    ///GPU-driven cluster culling without CPU readback.
    ///
    ///Requires `VK_HUAWEI_cluster_culling_shader`.
    pub unsafe fn cmd_draw_cluster_indirect_huawei(
        &self,
        command_buffer: CommandBuffer,
        buffer: Buffer,
        offset: u64,
    ) {
        let fp = self
            .commands()
            .cmd_draw_cluster_indirect_huawei
            .expect("vkCmdDrawClusterIndirectHUAWEI not loaded");
        unsafe { fp(command_buffer, buffer, offset) };
    }
    ///Wraps [`vkCmdUpdatePipelineIndirectBufferNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdUpdatePipelineIndirectBufferNV.html).
    /**
    Provided by **VK_NV_device_generated_commands_compute**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Updates the indirect dispatch parameters for a compute pipeline
    ///in a GPU buffer. Used with device-generated compute commands so
    ///the GPU can dispatch compute pipelines indirectly.
    ///
    ///Requires `VK_NV_device_generated_commands_compute`.
    pub unsafe fn cmd_update_pipeline_indirect_buffer_nv(
        &self,
        command_buffer: CommandBuffer,
        pipeline_bind_point: PipelineBindPoint,
        pipeline: Pipeline,
    ) {
        let fp = self
            .commands()
            .cmd_update_pipeline_indirect_buffer_nv
            .expect("vkCmdUpdatePipelineIndirectBufferNV not loaded");
        unsafe { fp(command_buffer, pipeline_bind_point, pipeline) };
    }
    ///Wraps [`vkCmdCopyBuffer`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyBuffer.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Copies data between two buffers. Multiple regions can be copied in
    ///a single call. Must be recorded outside a render pass.
    ///
    ///Common patterns:
    ///
    ///- **Staging upload**: copy from a host-visible staging buffer to a
    ///  device-local buffer. This is the standard way to get vertex,
    ///  index, and uniform data onto the GPU.
    ///- **Buffer-to-buffer transfers**: defragment or reorganise GPU data.
    ///
    ///Source and destination regions must not overlap within the same
    ///buffer. Use a temporary staging buffer if you need to shift data
    ///within a single buffer.
    ///
    ///For Vulkan 1.3+, prefer `cmd_copy_buffer2` which uses an extensible
    ///`CopyBufferInfo2` struct.
    pub unsafe fn cmd_copy_buffer(
        &self,
        command_buffer: CommandBuffer,
        src_buffer: Buffer,
        dst_buffer: Buffer,
        p_regions: &[BufferCopy],
    ) {
        let fp = self
            .commands()
            .cmd_copy_buffer
            .expect("vkCmdCopyBuffer not loaded");
        unsafe {
            fp(
                command_buffer,
                src_buffer,
                dst_buffer,
                p_regions.len() as u32,
                p_regions.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdCopyImage`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyImage.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Copies texel data between two images. Both images must have been
    ///created with `TRANSFER_SRC` and `TRANSFER_DST` usage respectively,
    ///and must be in compatible layouts (`TRANSFER_SRC_OPTIMAL` /
    ///`TRANSFER_DST_OPTIMAL` or `GENERAL`).
    ///
    ///The source and destination formats must be identical, or both must
    ///be in the same size-compatibility class. For format conversion, use
    ///`cmd_blit_image` instead.
    ///
    ///Copy operates on raw texel blocks, no filtering or scaling. The
    ///extent must be aligned to the texel block size for compressed
    ///formats.
    ///
    ///Must be recorded outside a render pass. For Vulkan 1.3+, prefer
    ///`cmd_copy_image2`.
    pub unsafe fn cmd_copy_image(
        &self,
        command_buffer: CommandBuffer,
        src_image: Image,
        src_image_layout: ImageLayout,
        dst_image: Image,
        dst_image_layout: ImageLayout,
        p_regions: &[ImageCopy],
    ) {
        let fp = self
            .commands()
            .cmd_copy_image
            .expect("vkCmdCopyImage not loaded");
        unsafe {
            fp(
                command_buffer,
                src_image,
                src_image_layout,
                dst_image,
                dst_image_layout,
                p_regions.len() as u32,
                p_regions.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdBlitImage`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBlitImage.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Copies a region between two images with optional scaling and format
    ///conversion. Unlike `cmd_copy_image`, blit supports different source
    ///and destination extents (scaling) and applies a filter.
    ///
    ///**Filters**:
    ///
    ///- `FILTER_NEAREST`: no interpolation. Fast, but produces blocky
    ///  results when scaling.
    ///- `FILTER_LINEAR`: bilinear interpolation. Smooth scaling. Requires
    ///  the format to support `FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR`.
    ///
    ///Common uses:
    ///
    ///- **Mipmap generation**: blit each mip level from the previous one,
    ///  halving dimensions each step.
    ///- **Resolve or downscale**: blit a high-resolution offscreen image
    ///  to a smaller swapchain image.
    ///
    ///Both images must be in appropriate transfer layouts. Must be recorded
    ///outside a render pass. For Vulkan 1.3+, prefer `cmd_blit_image2`.
    ///
    ///Not supported for depth/stencil or compressed formats. Use
    ///`cmd_copy_image` or `cmd_resolve_image` for those.
    pub unsafe fn cmd_blit_image(
        &self,
        command_buffer: CommandBuffer,
        src_image: Image,
        src_image_layout: ImageLayout,
        dst_image: Image,
        dst_image_layout: ImageLayout,
        p_regions: &[ImageBlit],
        filter: Filter,
    ) {
        let fp = self
            .commands()
            .cmd_blit_image
            .expect("vkCmdBlitImage not loaded");
        unsafe {
            fp(
                command_buffer,
                src_image,
                src_image_layout,
                dst_image,
                dst_image_layout,
                p_regions.len() as u32,
                p_regions.as_ptr(),
                filter,
            )
        };
    }
    ///Wraps [`vkCmdCopyBufferToImage`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyBufferToImage.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Copies data from a buffer to an image, the primary way to upload
    ///texture data from CPU to GPU.
    ///
    ///**Typical upload workflow**:
    ///
    ///1. Write pixel data into a host-visible staging buffer.
    ///2. Transition the target image to `TRANSFER_DST_OPTIMAL`.
    ///3. `cmd_copy_buffer_to_image` with the appropriate
    ///   `BufferImageCopy` regions.
    ///4. Transition the image to `SHADER_READ_ONLY_OPTIMAL` for sampling.
    ///
    ///**Buffer layout**: `buffer_row_length` and `buffer_image_height`
    ///control the row and slice pitch of the source data in the buffer.
    ///Set both to zero to use a tightly packed layout matching the image
    ///extent.
    ///
    ///Multiple regions can be copied in a single call (e.g. all mip
    ///levels of a texture). Must be recorded outside a render pass.
    ///
    ///For Vulkan 1.3+, prefer `cmd_copy_buffer_to_image2`.
    pub unsafe fn cmd_copy_buffer_to_image(
        &self,
        command_buffer: CommandBuffer,
        src_buffer: Buffer,
        dst_image: Image,
        dst_image_layout: ImageLayout,
        p_regions: &[BufferImageCopy],
    ) {
        let fp = self
            .commands()
            .cmd_copy_buffer_to_image
            .expect("vkCmdCopyBufferToImage not loaded");
        unsafe {
            fp(
                command_buffer,
                src_buffer,
                dst_image,
                dst_image_layout,
                p_regions.len() as u32,
                p_regions.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdCopyImageToBuffer`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyImageToBuffer.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Copies data from an image to a buffer, used for GPU readback of
    ///rendered images, screenshots, or compute shader output.
    ///
    ///**Typical readback workflow**:
    ///
    ///1. Transition the source image to `TRANSFER_SRC_OPTIMAL`.
    ///2. `cmd_copy_image_to_buffer` into a host-visible buffer.
    ///3. Submit and wait for the fence.
    ///4. Map the buffer and read the pixel data on the CPU.
    ///
    ///The `buffer_row_length` and `buffer_image_height` fields in
    ///`BufferImageCopy` control the destination layout. Set both to zero
    ///for tightly packed output.
    ///
    ///Be aware that readback is not instantaneous, it requires a full
    ///GPU round-trip. Avoid reading back in the render loop unless you
    ///are double- or triple-buffering the readback to hide latency.
    ///
    ///Must be recorded outside a render pass. For Vulkan 1.3+, prefer
    ///`cmd_copy_image_to_buffer2`.
    pub unsafe fn cmd_copy_image_to_buffer(
        &self,
        command_buffer: CommandBuffer,
        src_image: Image,
        src_image_layout: ImageLayout,
        dst_buffer: Buffer,
        p_regions: &[BufferImageCopy],
    ) {
        let fp = self
            .commands()
            .cmd_copy_image_to_buffer
            .expect("vkCmdCopyImageToBuffer not loaded");
        unsafe {
            fp(
                command_buffer,
                src_image,
                src_image_layout,
                dst_buffer,
                p_regions.len() as u32,
                p_regions.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdCopyMemoryIndirectNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyMemoryIndirectNV.html).
    /**
    Provided by **VK_NV_copy_memory_indirect**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Copies memory regions using indirect parameters stored in a GPU
    ///buffer at the given device address. Enables GPU-driven memory
    ///copy workflows where the copy descriptors are generated on the
    ///device.
    ///
    ///Requires `VK_NV_copy_memory_indirect`.
    pub unsafe fn cmd_copy_memory_indirect_nv(
        &self,
        command_buffer: CommandBuffer,
        copy_buffer_address: u64,
        copy_count: u32,
        stride: u32,
    ) {
        let fp = self
            .commands()
            .cmd_copy_memory_indirect_nv
            .expect("vkCmdCopyMemoryIndirectNV not loaded");
        unsafe { fp(command_buffer, copy_buffer_address, copy_count, stride) };
    }
    ///Wraps [`vkCmdCopyMemoryIndirectKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyMemoryIndirectKHR.html).
    /**
    Provided by **VK_KHR_copy_memory_indirect**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Copies memory regions using parameters read from a device-side
    ///info structure. Enables GPU-driven memory copies without CPU
    ///involvement in specifying source/destination addresses.
    ///
    ///Requires `VK_KHR_copy_memory_indirect`.
    pub unsafe fn cmd_copy_memory_indirect_khr(
        &self,
        command_buffer: CommandBuffer,
        p_copy_memory_indirect_info: &CopyMemoryIndirectInfoKHR,
    ) {
        let fp = self
            .commands()
            .cmd_copy_memory_indirect_khr
            .expect("vkCmdCopyMemoryIndirectKHR not loaded");
        unsafe { fp(command_buffer, p_copy_memory_indirect_info) };
    }
    ///Wraps [`vkCmdCopyMemoryToImageIndirectNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyMemoryToImageIndirectNV.html).
    /**
    Provided by **VK_NV_copy_memory_indirect**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Copies data from memory to an image using indirect parameters
    ///stored at a device address. Enables GPU-driven texture uploads
    ///where the copy regions are generated on the device.
    ///
    ///Requires `VK_NV_copy_memory_indirect`.
    pub unsafe fn cmd_copy_memory_to_image_indirect_nv(
        &self,
        command_buffer: CommandBuffer,
        copy_buffer_address: u64,
        stride: u32,
        dst_image: Image,
        dst_image_layout: ImageLayout,
        p_image_subresources: &[ImageSubresourceLayers],
    ) {
        let fp = self
            .commands()
            .cmd_copy_memory_to_image_indirect_nv
            .expect("vkCmdCopyMemoryToImageIndirectNV not loaded");
        unsafe {
            fp(
                command_buffer,
                copy_buffer_address,
                p_image_subresources.len() as u32,
                stride,
                dst_image,
                dst_image_layout,
                p_image_subresources.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdCopyMemoryToImageIndirectKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyMemoryToImageIndirectKHR.html).
    /**
    Provided by **VK_KHR_copy_memory_indirect**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Copies data from memory to an image using parameters read from a
    ///device-side info structure. Enables GPU-driven memory-to-image
    ///transfers without CPU readback of copy parameters.
    ///
    ///Requires `VK_KHR_copy_memory_indirect`.
    pub unsafe fn cmd_copy_memory_to_image_indirect_khr(
        &self,
        command_buffer: CommandBuffer,
        p_copy_memory_to_image_indirect_info: &CopyMemoryToImageIndirectInfoKHR,
    ) {
        let fp = self
            .commands()
            .cmd_copy_memory_to_image_indirect_khr
            .expect("vkCmdCopyMemoryToImageIndirectKHR not loaded");
        unsafe { fp(command_buffer, p_copy_memory_to_image_indirect_info) };
    }
    ///Wraps [`vkCmdUpdateBuffer`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdUpdateBuffer.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Writes a small amount of data inline into a buffer from the command
    ///stream. The data is embedded directly in the command buffer, no
    ///staging buffer needed.
    ///
    ///**Size limit**: the data size must be ≤ 65536 bytes and a multiple
    ///of 4. For larger uploads, use `cmd_copy_buffer` with a staging
    ///buffer instead.
    ///
    ///This is convenient for small per-frame updates (e.g. a uniform
    ///buffer with a single matrix) but should not be used for bulk data,
    ///it inflates command buffer size and the data is uploaded through the
    ///command stream, which is slower than a DMA transfer.
    ///
    ///Must be recorded outside a render pass. The destination buffer must
    ///have `BUFFER_USAGE_TRANSFER_DST`.
    pub unsafe fn cmd_update_buffer(
        &self,
        command_buffer: CommandBuffer,
        dst_buffer: Buffer,
        dst_offset: u64,
        data_size: u64,
        p_data: *const core::ffi::c_void,
    ) {
        let fp = self
            .commands()
            .cmd_update_buffer
            .expect("vkCmdUpdateBuffer not loaded");
        unsafe { fp(command_buffer, dst_buffer, dst_offset, data_size, p_data) };
    }
    ///Wraps [`vkCmdFillBuffer`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdFillBuffer.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Fills a region of a buffer with a repeating 4-byte value. Useful
    ///for clearing GPU buffers to zero (or any uniform value) without a
    ///staging upload.
    ///
    ///Common uses:
    ///
    ///- **Zero-initialise** an indirect draw count buffer before a compute
    ///  culling pass.
    ///- **Clear** a storage buffer used as a histogram or counter.
    ///
    ///The `offset` and `size` must be multiples of 4. Use `VK_WHOLE_SIZE`
    ///to fill from the offset to the end of the buffer.
    ///
    ///Must be recorded outside a render pass. The buffer must have
    ///`BUFFER_USAGE_TRANSFER_DST`.
    pub unsafe fn cmd_fill_buffer(
        &self,
        command_buffer: CommandBuffer,
        dst_buffer: Buffer,
        dst_offset: u64,
        size: u64,
        data: u32,
    ) {
        let fp = self
            .commands()
            .cmd_fill_buffer
            .expect("vkCmdFillBuffer not loaded");
        unsafe { fp(command_buffer, dst_buffer, dst_offset, size, data) };
    }
    ///Wraps [`vkCmdClearColorImage`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdClearColorImage.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Clears one or more regions of a colour image to a specified value.
    ///The image must be in `TRANSFER_DST_OPTIMAL` or `GENERAL` layout.
    ///
    ///This is an explicit clear outside a render pass. For clears inside
    ///a render pass, use `load_op = CLEAR` in the attachment description
    ///or `cmd_clear_attachments`, both are typically faster because the
    ///driver can integrate them with tile-based rendering.
    ///
    ///The clear value is a `ClearColorValue` union: either four `float32`,
    ///`int32`, or `uint32` values depending on the image format.
    ///
    ///Must be recorded outside a render pass.
    pub unsafe fn cmd_clear_color_image(
        &self,
        command_buffer: CommandBuffer,
        image: Image,
        image_layout: ImageLayout,
        p_color: &ClearColorValue,
        p_ranges: &[ImageSubresourceRange],
    ) {
        let fp = self
            .commands()
            .cmd_clear_color_image
            .expect("vkCmdClearColorImage not loaded");
        unsafe {
            fp(
                command_buffer,
                image,
                image_layout,
                p_color,
                p_ranges.len() as u32,
                p_ranges.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdClearDepthStencilImage`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdClearDepthStencilImage.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Clears one or more regions of a depth/stencil image to a specified
    ///depth and stencil value. The image must be in
    ///`TRANSFER_DST_OPTIMAL` or `GENERAL` layout.
    ///
    ///For most rendering, clearing via `load_op = CLEAR` on the
    ///depth/stencil attachment is preferred, it lets tile-based GPUs
    ///avoid a separate clear pass. Use this command only when you need to
    ///clear a depth/stencil image outside a render pass.
    ///
    ///The `image_subresource_range` must reference the appropriate aspect
    ///flags (`DEPTH`, `STENCIL`, or both).
    ///
    ///Must be recorded outside a render pass.
    pub unsafe fn cmd_clear_depth_stencil_image(
        &self,
        command_buffer: CommandBuffer,
        image: Image,
        image_layout: ImageLayout,
        p_depth_stencil: &ClearDepthStencilValue,
        p_ranges: &[ImageSubresourceRange],
    ) {
        let fp = self
            .commands()
            .cmd_clear_depth_stencil_image
            .expect("vkCmdClearDepthStencilImage not loaded");
        unsafe {
            fp(
                command_buffer,
                image,
                image_layout,
                p_depth_stencil,
                p_ranges.len() as u32,
                p_ranges.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdClearAttachments`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdClearAttachments.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Clears one or more attachment regions **inside** an active render
    ///pass. Unlike `load_op = CLEAR` (which clears the entire attachment
    ///at render pass begin), this clears arbitrary rectangular regions
    ///mid-render-pass.
    ///
    ///Use cases:
    ///
    ///- Clear a sub-region of a colour attachment (e.g. a UI panel
    ///  background).
    ///- Clear the stencil buffer for a specific screen region.
    ///
    ///Each `ClearAttachment` specifies which attachment to clear (colour
    ///index, depth, or stencil) and the clear value. Each `ClearRect`
    ///defines the pixel rectangle and layer range.
    ///
    ///For whole-attachment clears, prefer `load_op = CLEAR`, it is
    ///always at least as fast and often faster on tile-based hardware.
    pub unsafe fn cmd_clear_attachments(
        &self,
        command_buffer: CommandBuffer,
        p_attachments: &[ClearAttachment],
        p_rects: &[ClearRect],
    ) {
        let fp = self
            .commands()
            .cmd_clear_attachments
            .expect("vkCmdClearAttachments not loaded");
        unsafe {
            fp(
                command_buffer,
                p_attachments.len() as u32,
                p_attachments.as_ptr(),
                p_rects.len() as u32,
                p_rects.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdResolveImage`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdResolveImage.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Resolves (downsamples) a multisample image into a single-sample
    ///image. Typically used to produce the final single-sample result from
    ///a multisampled colour attachment.
    ///
    ///Both images must be in appropriate transfer layouts
    ///(`TRANSFER_SRC_OPTIMAL` and `TRANSFER_DST_OPTIMAL` respectively).
    ///The source must be multisampled; the destination must be
    ///single-sample. Formats must be identical.
    ///
    ///For resolving inside a render pass, use `resolve_attachment` in the
    ///subpass description instead, it is more efficient on tile-based
    ///GPUs because the resolve happens in-tile.
    ///
    ///Must be recorded outside a render pass. For Vulkan 1.3+, prefer
    ///`cmd_resolve_image2`.
    pub unsafe fn cmd_resolve_image(
        &self,
        command_buffer: CommandBuffer,
        src_image: Image,
        src_image_layout: ImageLayout,
        dst_image: Image,
        dst_image_layout: ImageLayout,
        p_regions: &[ImageResolve],
    ) {
        let fp = self
            .commands()
            .cmd_resolve_image
            .expect("vkCmdResolveImage not loaded");
        unsafe {
            fp(
                command_buffer,
                src_image,
                src_image_layout,
                dst_image,
                dst_image_layout,
                p_regions.len() as u32,
                p_regions.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdSetEvent`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetEvent.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Signals an event from the GPU at a specific pipeline stage. A later
    ///`cmd_wait_events` call can wait for this signal to synchronise work
    ///within the same queue.
    ///
    ///Events provide finer-grained synchronisation than pipeline barriers
    ///when you want to split a dependency into a "signal" point and a
    ///"wait" point separated by other commands. This lets the GPU execute
    ///interleaving work between the signal and wait.
    ///
    ///The `stage_mask` specifies at which pipeline stage the event is
    ///signaled. The event becomes signaled once all commands prior to this
    ///call have completed that stage.
    ///
    ///Events must only be used within a single queue. For cross-queue
    ///synchronisation, use semaphores.
    ///
    ///For Vulkan 1.3+, prefer `cmd_set_event2` which supports more
    ///precise stage and access masks.
    pub unsafe fn cmd_set_event(
        &self,
        command_buffer: CommandBuffer,
        event: Event,
        stage_mask: PipelineStageFlags,
    ) {
        let fp = self
            .commands()
            .cmd_set_event
            .expect("vkCmdSetEvent not loaded");
        unsafe { fp(command_buffer, event, stage_mask) };
    }
    ///Wraps [`vkCmdResetEvent`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdResetEvent.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Resets an event to the unsignaled state from the GPU at a specific
    ///pipeline stage. The event can then be signaled again by a
    ///subsequent `cmd_set_event`.
    ///
    ///Must not be called while a `cmd_wait_events` that waits on this
    ///event is between its wait and the completion of the dependent work.
    ///
    ///For Vulkan 1.3+, prefer `cmd_reset_event2`.
    pub unsafe fn cmd_reset_event(
        &self,
        command_buffer: CommandBuffer,
        event: Event,
        stage_mask: PipelineStageFlags,
    ) {
        let fp = self
            .commands()
            .cmd_reset_event
            .expect("vkCmdResetEvent not loaded");
        unsafe { fp(command_buffer, event, stage_mask) };
    }
    ///Wraps [`vkCmdWaitEvents`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdWaitEvents.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Waits for one or more events to be signaled and then inserts memory
    ///and execution dependencies. This is the "wait" half of the
    ///signal/wait pattern started by `cmd_set_event`.
    ///
    ///The wait blocks execution at the specified destination pipeline
    ///stages until all events are signaled. Memory barriers provided in
    ///this call make the specified source writes visible to the
    ///destination stages.
    ///
    ///**Split barriers**: the main advantage over `cmd_pipeline_barrier`
    ///is that you can interleave unrelated commands between the signal and
    ///wait, giving the GPU more opportunity for parallel execution.
    ///
    ///Events must not be waited on across different queues. For
    ///cross-queue synchronisation, use semaphores.
    ///
    ///For Vulkan 1.3+, prefer `cmd_wait_events2`.
    pub unsafe fn cmd_wait_events(
        &self,
        command_buffer: CommandBuffer,
        p_events: &[Event],
        src_stage_mask: PipelineStageFlags,
        dst_stage_mask: PipelineStageFlags,
        p_memory_barriers: &[MemoryBarrier],
        p_buffer_memory_barriers: &[BufferMemoryBarrier],
        p_image_memory_barriers: &[ImageMemoryBarrier],
    ) {
        let fp = self
            .commands()
            .cmd_wait_events
            .expect("vkCmdWaitEvents not loaded");
        unsafe {
            fp(
                command_buffer,
                p_events.len() as u32,
                p_events.as_ptr(),
                src_stage_mask,
                dst_stage_mask,
                p_memory_barriers.len() as u32,
                p_memory_barriers.as_ptr(),
                p_buffer_memory_barriers.len() as u32,
                p_buffer_memory_barriers.as_ptr(),
                p_image_memory_barriers.len() as u32,
                p_image_memory_barriers.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdPipelineBarrier`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdPipelineBarrier.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Inserts an execution and memory dependency between commands recorded
    ///before and after the barrier. This is the primary synchronisation
    ///tool inside a command buffer.
    ///
    ///**Three types of barrier**:
    ///
    ///- **Memory barriers** (`MemoryBarrier`): global, affects all
    ///  resources. Rarely needed, prefer the more specific variants.
    ///- **Buffer memory barriers** (`BufferMemoryBarrier`): targets a
    ///  specific buffer region. Use for storage buffer read-after-write.
    ///- **Image memory barriers** (`ImageMemoryBarrier`): targets a
    ///  specific image subresource range. Also performs layout transitions.
    ///
    ///**Layout transitions**: image memory barriers are the primary way to
    ///transition images between layouts (e.g.
    ///`TRANSFER_DST_OPTIMAL` → `SHADER_READ_ONLY_OPTIMAL`). The old and
    ///new layouts in the barrier define the transition.
    ///
    ///**Stage masks**: `src_stage_mask` is the set of stages that must
    ///complete before the barrier. `dst_stage_mask` is the set of stages
    ///that must wait for the barrier. Choose the narrowest stages possible
    ///to minimise stalls.
    ///
    ///Inside a render pass, only self-dependencies are allowed (barriers
    ///within a single subpass). Outside a render pass, there are no
    ///restrictions.
    ///
    ///For Vulkan 1.3+, prefer `cmd_pipeline_barrier2` which uses
    ///extensible structs.
    pub unsafe fn cmd_pipeline_barrier(
        &self,
        command_buffer: CommandBuffer,
        src_stage_mask: PipelineStageFlags,
        dst_stage_mask: PipelineStageFlags,
        dependency_flags: DependencyFlags,
        p_memory_barriers: &[MemoryBarrier],
        p_buffer_memory_barriers: &[BufferMemoryBarrier],
        p_image_memory_barriers: &[ImageMemoryBarrier],
    ) {
        let fp = self
            .commands()
            .cmd_pipeline_barrier
            .expect("vkCmdPipelineBarrier not loaded");
        unsafe {
            fp(
                command_buffer,
                src_stage_mask,
                dst_stage_mask,
                dependency_flags,
                p_memory_barriers.len() as u32,
                p_memory_barriers.as_ptr(),
                p_buffer_memory_barriers.len() as u32,
                p_buffer_memory_barriers.as_ptr(),
                p_image_memory_barriers.len() as u32,
                p_image_memory_barriers.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdBeginQuery`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBeginQuery.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Begins a query at the specified index in a query pool. All
    ///rendering or compute commands recorded between `cmd_begin_query` and
    ///`cmd_end_query` are measured by the query.
    ///
    ///**Flags**:
    ///
    ///- `QUERY_CONTROL_PRECISE`: for occlusion queries, return an exact
    ///  sample count instead of a boolean. More expensive on some
    ///  hardware. Requires the `occlusion_query_precise` device feature.
    ///
    ///The query slot must have been reset with `cmd_reset_query_pool` (or
    ///`reset_query_pool` on Vulkan 1.2+) before beginning.
    ///
    ///Pipeline statistics queries must be begun and ended outside a render
    ///pass. Occlusion queries can span draw calls within a render pass.
    pub unsafe fn cmd_begin_query(
        &self,
        command_buffer: CommandBuffer,
        query_pool: QueryPool,
        query: u32,
        flags: QueryControlFlags,
    ) {
        let fp = self
            .commands()
            .cmd_begin_query
            .expect("vkCmdBeginQuery not loaded");
        unsafe { fp(command_buffer, query_pool, query, flags) };
    }
    ///Wraps [`vkCmdEndQuery`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdEndQuery.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Ends an active query at the specified index. The query results
    ///become available for retrieval via `get_query_pool_results` or
    ///`cmd_copy_query_pool_results` once the command buffer has completed
    ///execution.
    ///
    ///Must be paired with a preceding `cmd_begin_query` on the same
    ///query index. Beginning a query without ending it, or ending one
    ///that was not begun, is an error.
    pub unsafe fn cmd_end_query(
        &self,
        command_buffer: CommandBuffer,
        query_pool: QueryPool,
        query: u32,
    ) {
        let fp = self
            .commands()
            .cmd_end_query
            .expect("vkCmdEndQuery not loaded");
        unsafe { fp(command_buffer, query_pool, query) };
    }
    ///Wraps [`vkCmdBeginConditionalRenderingEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBeginConditionalRenderingEXT.html).
    /**
    Provided by **VK_EXT_conditional_rendering**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Begins a conditional rendering block. Subsequent rendering and
    ///dispatch commands are discarded if the 32-bit value at the
    ///specified buffer offset is zero (or non-zero if `INVERTED` is
    ///set).
    ///
    ///End with `cmd_end_conditional_rendering_ext`.
    ///
    ///Useful for GPU-driven occlusion culling, write visibility
    ///results to a buffer, then conditionally skip draw calls.
    ///
    ///Requires `VK_EXT_conditional_rendering`.
    pub unsafe fn cmd_begin_conditional_rendering_ext(
        &self,
        command_buffer: CommandBuffer,
        p_conditional_rendering_begin: &ConditionalRenderingBeginInfoEXT,
    ) {
        let fp = self
            .commands()
            .cmd_begin_conditional_rendering_ext
            .expect("vkCmdBeginConditionalRenderingEXT not loaded");
        unsafe { fp(command_buffer, p_conditional_rendering_begin) };
    }
    ///Wraps [`vkCmdEndConditionalRenderingEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdEndConditionalRenderingEXT.html).
    /**
    Provided by **VK_EXT_conditional_rendering**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Ends a conditional rendering block started with
    ///`cmd_begin_conditional_rendering_ext`. Commands after this call
    ///execute unconditionally.
    ///
    ///Requires `VK_EXT_conditional_rendering`.
    pub unsafe fn cmd_end_conditional_rendering_ext(&self, command_buffer: CommandBuffer) {
        let fp = self
            .commands()
            .cmd_end_conditional_rendering_ext
            .expect("vkCmdEndConditionalRenderingEXT not loaded");
        unsafe { fp(command_buffer) };
    }
    ///Wraps [`vkCmdBeginCustomResolveEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBeginCustomResolveEXT.html).
    /**
    Provided by **VK_EXT_custom_resolve**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Begins a custom resolve region, allowing the application to use
    ///its own fragment shader for MSAA resolve instead of the fixed-
    ///function resolve. End with `cmd_end_custom_resolve_ext`.
    ///
    ///Useful for tone-mapped or weighted resolves that the built-in
    ///resolve operations cannot express.
    ///
    ///Requires `VK_EXT_custom_resolve`.
    pub unsafe fn cmd_begin_custom_resolve_ext(
        &self,
        command_buffer: CommandBuffer,
        p_begin_custom_resolve_info: Option<&BeginCustomResolveInfoEXT>,
    ) {
        let fp = self
            .commands()
            .cmd_begin_custom_resolve_ext
            .expect("vkCmdBeginCustomResolveEXT not loaded");
        let p_begin_custom_resolve_info_ptr =
            p_begin_custom_resolve_info.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(command_buffer, p_begin_custom_resolve_info_ptr) };
    }
    ///Wraps [`vkCmdResetQueryPool`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdResetQueryPool.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Resets a range of queries in a pool from the GPU command stream.
    ///Queries must be reset before they can be used in `cmd_begin_query`
    ///or `cmd_write_timestamp`.
    ///
    ///This is the pre-1.2 way to reset queries. For Vulkan 1.2+,
    ///`reset_query_pool` (host-side) is often more convenient and avoids
    ///adding the reset to the command buffer.
    ///
    ///Must be recorded outside a render pass.
    pub unsafe fn cmd_reset_query_pool(
        &self,
        command_buffer: CommandBuffer,
        query_pool: QueryPool,
        first_query: u32,
        query_count: u32,
    ) {
        let fp = self
            .commands()
            .cmd_reset_query_pool
            .expect("vkCmdResetQueryPool not loaded");
        unsafe { fp(command_buffer, query_pool, first_query, query_count) };
    }
    ///Wraps [`vkCmdWriteTimestamp`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdWriteTimestamp.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Writes the current GPU timestamp into a query pool slot when the
    ///specified pipeline stage completes. Use two timestamps to measure
    ///elapsed GPU time:
    ///
    ///```text
    ///cmd_write_timestamp(PIPELINE_STAGE_TOP_OF_PIPE, pool, 0);
    #[doc = "// ... commands to measure ..."]
    ///cmd_write_timestamp(PIPELINE_STAGE_BOTTOM_OF_PIPE, pool, 1);
    ///```
    ///
    ///After the command buffer completes, read the values with
    ///`get_query_pool_results` (with `QUERY_RESULT_64`) and compute:
    ///
    ///```text
    ///elapsed_ns = (timestamp[1] - timestamp[0]) * timestamp_period
    ///```
    ///
    ///`timestamp_period` is in nanoseconds per tick, available from
    ///`physical_device_limits`.
    ///
    ///Not all queue families support timestamps, check
    ///`timestamp_valid_bits` in the queue family properties. A value of 0
    ///means timestamps are not supported on that queue.
    ///
    ///For Vulkan 1.3+, prefer `cmd_write_timestamp2`.
    pub unsafe fn cmd_write_timestamp(
        &self,
        command_buffer: CommandBuffer,
        pipeline_stage: PipelineStageFlagBits,
        query_pool: QueryPool,
        query: u32,
    ) {
        let fp = self
            .commands()
            .cmd_write_timestamp
            .expect("vkCmdWriteTimestamp not loaded");
        unsafe { fp(command_buffer, pipeline_stage, query_pool, query) };
    }
    ///Wraps [`vkCmdCopyQueryPoolResults`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyQueryPoolResults.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Copies query results directly into a GPU buffer. This is the
    ///GPU-side counterpart to `get_query_pool_results` and avoids a CPU
    ///round-trip when the results are consumed by subsequent GPU work
    ///(e.g. conditional rendering or indirect dispatch).
    ///
    ///The same flags apply as for `get_query_pool_results`:
    ///`QUERY_RESULT_64`, `QUERY_RESULT_WAIT`,
    ///`QUERY_RESULT_WITH_AVAILABILITY`, and `QUERY_RESULT_PARTIAL`.
    ///
    ///The destination buffer must have `BUFFER_USAGE_TRANSFER_DST`. The
    ///stride must be large enough to hold the result (and availability
    ///value, if requested).
    ///
    ///Must be recorded outside a render pass.
    pub unsafe fn cmd_copy_query_pool_results(
        &self,
        command_buffer: CommandBuffer,
        query_pool: QueryPool,
        first_query: u32,
        query_count: u32,
        dst_buffer: Buffer,
        dst_offset: u64,
        stride: u64,
        flags: QueryResultFlags,
    ) {
        let fp = self
            .commands()
            .cmd_copy_query_pool_results
            .expect("vkCmdCopyQueryPoolResults not loaded");
        unsafe {
            fp(
                command_buffer,
                query_pool,
                first_query,
                query_count,
                dst_buffer,
                dst_offset,
                stride,
                flags,
            )
        };
    }
    ///Wraps [`vkCmdPushConstants`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdPushConstants.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Updates push constant values for the bound pipeline layout. Push
    ///constants are a fast path for small, frequently-changing data that
    ///avoids descriptor set updates entirely.
    ///
    ///**Size limit**: the total push constant range is at least 128 bytes
    ///(device limit `max_push_constants_size`). Use push constants for
    ///per-draw data like transform matrices, material indices, or time
    ///values.
    ///
    ///**Stage flags**: the `stage_flags` parameter must match the stages
    ///declared in the pipeline layout's push constant range. You can
    ///update different stage ranges separately (e.g. update the vertex
    ///shader's range without touching the fragment shader's range).
    ///
    ///Push constant data persists across draw/dispatch calls until the
    ///pipeline layout is changed or the values are overwritten.
    ///
    ///For Vulkan 1.4+, `cmd_push_constants2` uses an extensible struct.
    pub unsafe fn cmd_push_constants(
        &self,
        command_buffer: CommandBuffer,
        layout: PipelineLayout,
        stage_flags: ShaderStageFlags,
        offset: u32,
        p_values: &[core::ffi::c_void],
    ) {
        let fp = self
            .commands()
            .cmd_push_constants
            .expect("vkCmdPushConstants not loaded");
        unsafe {
            fp(
                command_buffer,
                layout,
                stage_flags,
                offset,
                p_values.len() as u32,
                p_values.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdBeginRenderPass`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBeginRenderPass.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Begins a render pass instance. All subsequent drawing commands are
    ///recorded within this render pass until `cmd_end_render_pass`.
    ///
    ///**`render_pass_begin_info`** specifies:
    ///
    ///- **Render pass and framebuffer**: which render pass to use and
    ///  which concrete image views are bound.
    ///- **Render area**: the pixel region to render. Should match the
    ///  framebuffer extent for best performance. Misalignment with the
    ///  render area granularity can cause overhead on tile-based GPUs.
    ///- **Clear values**: one per attachment with `load_op = CLEAR`. The
    ///  array must include entries for all attachments (use a dummy value
    ///  for non-cleared attachments).
    ///
    ///**`contents`**:
    ///
    ///- `SUBPASS_CONTENTS_INLINE`: draw commands are recorded directly
    ///  in this command buffer.
    ///- `SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS`: draw commands come
    ///  from secondary command buffers via `cmd_execute_commands`.
    ///
    ///For Vulkan 1.2+, `cmd_begin_render_pass2` accepts a `SubpassBeginInfo`.
    ///For Vulkan 1.3+, consider dynamic rendering (`cmd_begin_rendering`)
    ///which avoids render pass and framebuffer objects entirely.
    pub unsafe fn cmd_begin_render_pass(
        &self,
        command_buffer: CommandBuffer,
        p_render_pass_begin: &RenderPassBeginInfo,
        contents: SubpassContents,
    ) {
        let fp = self
            .commands()
            .cmd_begin_render_pass
            .expect("vkCmdBeginRenderPass not loaded");
        unsafe { fp(command_buffer, p_render_pass_begin, contents) };
    }
    ///Wraps [`vkCmdNextSubpass`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdNextSubpass.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Advances to the next subpass within a render pass. Subpass
    ///transitions allow the driver to resolve dependencies between
    ///subpasses, for example, reading a colour attachment written in
    ///the previous subpass as an input attachment.
    ///
    ///The `contents` parameter has the same meaning as in
    ///`cmd_begin_render_pass`: `INLINE` or `SECONDARY_COMMAND_BUFFERS`.
    ///
    ///Multi-subpass render passes are an optimisation for tile-based GPUs
    ///where they can keep data on-chip between subpasses. On desktop GPUs
    ///the benefit is smaller. Many applications use a single subpass and
    ///handle inter-pass dependencies with explicit pipeline barriers.
    ///
    ///For Vulkan 1.2+, prefer `cmd_next_subpass2`.
    pub unsafe fn cmd_next_subpass(
        &self,
        command_buffer: CommandBuffer,
        contents: SubpassContents,
    ) {
        let fp = self
            .commands()
            .cmd_next_subpass
            .expect("vkCmdNextSubpass not loaded");
        unsafe { fp(command_buffer, contents) };
    }
    ///Wraps [`vkCmdEndRenderPass`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdEndRenderPass.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Ends the current render pass instance. After this call, the
    ///implicit layout transitions specified by each attachment's
    ///`final_layout` are applied.
    ///
    ///No draw commands may be recorded after this until a new render pass
    ///is begun (or dynamic rendering is started).
    ///
    ///For Vulkan 1.2+, prefer `cmd_end_render_pass2`.
    pub unsafe fn cmd_end_render_pass(&self, command_buffer: CommandBuffer) {
        let fp = self
            .commands()
            .cmd_end_render_pass
            .expect("vkCmdEndRenderPass not loaded");
        unsafe { fp(command_buffer) };
    }
    ///Wraps [`vkCmdExecuteCommands`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdExecuteCommands.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Executes one or more secondary command buffers from a primary
    ///command buffer. The secondary buffers are inlined into the primary
    ///buffer's execution stream in array order.
    ///
    ///**Use cases**:
    ///
    ///- **Multi-threaded recording**: each thread records a secondary
    ///  command buffer, and the main thread assembles them with a single
    ///  `cmd_execute_commands` call. This is the primary scaling strategy
    ///  for CPU-bound recording.
    ///- **Reusable draw sequences**: record a secondary buffer once and
    ///  execute it in multiple frames or from multiple primary buffers
    ///  (requires `SIMULTANEOUS_USE` on the secondary buffer).
    ///
    ///Secondary command buffers inherit certain state from the primary
    ///buffer (viewport, scissor, etc.) only if declared in the
    ///`CommandBufferInheritanceInfo`. The render pass and subpass must
    ///match what the primary buffer is currently in.
    ///
    ///This command can only be called from a primary command buffer.
    pub unsafe fn cmd_execute_commands(
        &self,
        command_buffer: CommandBuffer,
        p_command_buffers: &[CommandBuffer],
    ) {
        let fp = self
            .commands()
            .cmd_execute_commands
            .expect("vkCmdExecuteCommands not loaded");
        unsafe {
            fp(
                command_buffer,
                p_command_buffers.len() as u32,
                p_command_buffers.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCreateSharedSwapchainsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateSharedSwapchainsKHR.html).
    /**
    Provided by **VK_KHR_display_swapchain**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_INCOMPATIBLE_DISPLAY_KHR`
    ///- `VK_ERROR_DEVICE_LOST`
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates one or more swapchains that share presentable images with
    ///their `old_swapchain`. Provided by `VK_KHR_display_swapchain`.
    ///
    ///This is primarily used for direct-to-display rendering where
    ///multiple swapchains share the same display plane. For window-based
    ///rendering, use `create_swapchain_khr` instead.
    pub unsafe fn create_shared_swapchains_khr(
        &self,
        p_create_infos: &[SwapchainCreateInfoKHR],
        allocator: Option<&AllocationCallbacks>,
        p_swapchains: *mut SwapchainKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .create_shared_swapchains_khr
            .expect("vkCreateSharedSwapchainsKHR not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        check(unsafe {
            fp(
                self.handle(),
                p_create_infos.len() as u32,
                p_create_infos.as_ptr(),
                alloc_ptr,
                p_swapchains,
            )
        })
    }
    ///Wraps [`vkCreateSwapchainKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateSwapchainKHR.html).
    /**
    Provided by **VK_KHR_swapchain**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_DEVICE_LOST`
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_NATIVE_WINDOW_IN_USE_KHR`
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_COMPRESSION_EXHAUSTED_EXT`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a swapchain, a queue of presentable images tied to a
    ///surface (window). The swapchain is the bridge between rendering and
    ///display.
    ///
    ///**Key parameters**:
    ///
    ///- **`min_image_count`**: request at least this many images. For
    ///  double buffering use 2, for triple buffering use 3. Query
    ///  `get_physical_device_surface_capabilities_khr` for the supported
    ///  range.
    ///- **`image_format` / `image_color_space`**: pick a pair from
    ///  `get_physical_device_surface_formats_khr`. `B8G8R8A8_SRGB` +
    ///  `SRGB_NONLINEAR` is the most portable.
    ///- **`present_mode`**: `FIFO` (vsync, always supported), `MAILBOX`
    ///  (low-latency triple buffering), `IMMEDIATE` (no vsync, tearing).
    ///- **`pre_transform`**: set to `current_transform` from surface
    ///  capabilities to avoid an extra composition blit.
    ///- **`old_swapchain`**: when recreating after a resize, pass the old
    ///  swapchain here. The driver can reuse internal resources.
    ///
    ///**Swapchain recreation** is required when the surface size changes
    ///(window resize) or when `acquire_next_image_khr` returns
    ///`VK_ERROR_OUT_OF_DATE_KHR`. Destroy the old swapchain after
    ///creating the new one.
    pub unsafe fn create_swapchain_khr(
        &self,
        p_create_info: &SwapchainCreateInfoKHR,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<SwapchainKHR> {
        let fp = self
            .commands()
            .create_swapchain_khr
            .expect("vkCreateSwapchainKHR not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroySwapchainKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroySwapchainKHR.html).
    /**
    Provided by **VK_KHR_swapchain**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `swapchain` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a swapchain. All images obtained from
    ///`get_swapchain_images_khr` become invalid, destroy any image views
    ///and framebuffers referencing them first.
    ///
    ///Wait for all rendering to complete (`device_wait_idle`) before
    ///destroying. Do not destroy a swapchain while an acquired image has
    ///not been presented.
    ///
    ///When recreating a swapchain (e.g. on resize), create the new one
    ///first (passing the old as `old_swapchain`), then destroy the old
    ///one.
    pub unsafe fn destroy_swapchain_khr(
        &self,
        swapchain: SwapchainKHR,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_swapchain_khr
            .expect("vkDestroySwapchainKHR not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), swapchain, alloc_ptr) };
    }
    ///Wraps [`vkGetSwapchainImagesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetSwapchainImagesKHR.html).
    /**
    Provided by **VK_KHR_swapchain**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns the array of presentable images owned by the swapchain. You
    ///do not create or destroy these images, they are managed by the
    ///swapchain.
    ///
    ///The returned image count may be greater than `min_image_count`
    ///requested at swapchain creation.
    ///
    ///Create an `ImageView` for each swapchain image to use them as
    ///render targets. These views (and any framebuffers using them) must
    ///be destroyed before the swapchain is destroyed.
    ///
    ///The images start in an undefined layout. Transition them to the
    ///appropriate layout (e.g. `COLOR_ATTACHMENT_OPTIMAL`) during the
    ///first render pass or via a pipeline barrier.
    pub unsafe fn get_swapchain_images_khr(&self, swapchain: SwapchainKHR) -> VkResult<Vec<Image>> {
        let fp = self
            .commands()
            .get_swapchain_images_khr
            .expect("vkGetSwapchainImagesKHR not loaded");
        enumerate_two_call(|count, data| unsafe { fp(self.handle(), swapchain, count, data) })
    }
    ///Wraps [`vkAcquireNextImageKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkAcquireNextImageKHR.html).
    /**
    Provided by **VK_KHR_swapchain**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_DEVICE_LOST`
    ///- `VK_ERROR_OUT_OF_DATE_KHR`
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `swapchain` must be externally synchronized.
    ///- `semaphore` must be externally synchronized.
    ///- `fence` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Acquires the next available image from the swapchain for rendering.
    ///Returns the index into the array from `get_swapchain_images_khr`.
    ///
    ///**Synchronisation**: pass a semaphore, a fence, or both. The
    ///semaphore/fence is signaled when the image is ready to be rendered
    ///to. Do not start rendering until the semaphore is waited on in
    ///`queue_submit`.
    ///
    ///**Timeout**: in nanoseconds. `u64::MAX` blocks indefinitely.
    ///
    ///**Special return values**:
    ///
    ///- `VK_SUBOPTIMAL_KHR`: the swapchain still works but no longer
    ///  matches the surface perfectly (e.g. after a resize). You can
    ///  continue rendering but should recreate the swapchain soon.
    ///- `VK_ERROR_OUT_OF_DATE_KHR`: the swapchain is incompatible with
    ///  the surface and must be recreated before rendering. Do not present
    ///  the acquired image.
    ///
    ///A common frame loop:
    ///
    ///```text
    ///acquire_next_image_khr(swapchain, u64::MAX, image_available_sem, null)
    #[doc = "// wait on image_available_sem in queue_submit"]
    #[doc = "// render to swapchain_images[index]"]
    #[doc = "// signal render_finished_sem in queue_submit"]
    ///queue_present_khr(render_finished_sem, swapchain, index)
    ///```
    pub unsafe fn acquire_next_image_khr(
        &self,
        swapchain: SwapchainKHR,
        timeout: u64,
        semaphore: Semaphore,
        fence: Fence,
    ) -> VkResult<u32> {
        let fp = self
            .commands()
            .acquire_next_image_khr
            .expect("vkAcquireNextImageKHR not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe {
            fp(
                self.handle(),
                swapchain,
                timeout,
                semaphore,
                fence,
                &mut out,
            )
        })?;
        Ok(out)
    }
    ///Wraps [`vkQueuePresentKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkQueuePresentKHR.html).
    /**
    Provided by **VK_KHR_swapchain**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_DEVICE_LOST`
    ///- `VK_ERROR_OUT_OF_DATE_KHR`
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///- `VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT`
    ///
    ///# Safety
    ///- `queue` (self) must be valid and not destroyed.
    ///- `queue` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Presents a rendered swapchain image to the display. This is the
    ///final step in the frame loop, after rendering is complete, present
    ///the image to make it visible.
    ///
    ///**Wait semaphores**: the present waits on these semaphores before
    ///presenting. Pass the semaphore that your render submission signals
    ///to ensure the image is fully rendered before it goes to the display.
    ///
    ///**Multiple swapchains**: a single present call can present to
    ///multiple swapchains simultaneously (e.g. for multi-window or
    ///multi-monitor rendering).
    ///
    ///**Return values**:
    ///
    ///- `VK_SUBOPTIMAL_KHR`: presented successfully but the swapchain
    ///  should be recreated.
    ///- `VK_ERROR_OUT_OF_DATE_KHR`: presentation failed, the swapchain
    ///  must be recreated.
    ///
    ///The present queue does not need to be the same as the graphics
    ///queue, but the semaphore synchronisation must be correct if they
    ///differ.
    pub unsafe fn queue_present_khr(
        &self,
        queue: Queue,
        p_present_info: &PresentInfoKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .queue_present_khr
            .expect("vkQueuePresentKHR not loaded");
        check(unsafe { fp(queue, p_present_info) })
    }
    ///Wraps [`vkDebugMarkerSetObjectNameEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDebugMarkerSetObjectNameEXT.html).
    /**
    Provided by **VK_EXT_debug_marker**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Assigns a name to a Vulkan object for debugging. This is the
    ///legacy `VK_EXT_debug_marker` equivalent of
    ///`set_debug_utils_object_name_ext`.
    ///
    ///`DebugMarkerObjectNameInfoEXT` uses the old
    ///`DebugReportObjectTypeEXT` enum to identify object types.
    ///
    ///Superseded by `VK_EXT_debug_utils`. Prefer
    ///`set_debug_utils_object_name_ext` for new code.
    pub unsafe fn debug_marker_set_object_name_ext(
        &self,
        p_name_info: &DebugMarkerObjectNameInfoEXT,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .debug_marker_set_object_name_ext
            .expect("vkDebugMarkerSetObjectNameEXT not loaded");
        check(unsafe { fp(self.handle(), p_name_info) })
    }
    ///Wraps [`vkDebugMarkerSetObjectTagEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDebugMarkerSetObjectTagEXT.html).
    /**
    Provided by **VK_EXT_debug_marker**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Attaches arbitrary binary data to a Vulkan object. This is the
    ///legacy `VK_EXT_debug_marker` equivalent of
    ///`set_debug_utils_object_tag_ext`.
    ///
    ///Superseded by `VK_EXT_debug_utils`. Prefer
    ///`set_debug_utils_object_tag_ext` for new code.
    pub unsafe fn debug_marker_set_object_tag_ext(
        &self,
        p_tag_info: &DebugMarkerObjectTagInfoEXT,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .debug_marker_set_object_tag_ext
            .expect("vkDebugMarkerSetObjectTagEXT not loaded");
        check(unsafe { fp(self.handle(), p_tag_info) })
    }
    ///Wraps [`vkCmdDebugMarkerBeginEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDebugMarkerBeginEXT.html).
    /**
    Provided by **VK_EXT_debug_marker**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Opens a debug marker region in a command buffer. This is the
    ///legacy `VK_EXT_debug_marker` equivalent of
    ///`cmd_begin_debug_utils_label_ext`.
    ///
    ///`DebugMarkerMarkerInfoEXT` specifies a name and optional RGBA
    ///color. Close with `cmd_debug_marker_end_ext`.
    ///
    ///Superseded by `VK_EXT_debug_utils`. Prefer
    ///`cmd_begin_debug_utils_label_ext` for new code.
    pub unsafe fn cmd_debug_marker_begin_ext(
        &self,
        command_buffer: CommandBuffer,
        p_marker_info: &DebugMarkerMarkerInfoEXT,
    ) {
        let fp = self
            .commands()
            .cmd_debug_marker_begin_ext
            .expect("vkCmdDebugMarkerBeginEXT not loaded");
        unsafe { fp(command_buffer, p_marker_info) };
    }
    ///Wraps [`vkCmdDebugMarkerEndEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDebugMarkerEndEXT.html).
    /**
    Provided by **VK_EXT_debug_marker**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Closes the most recently opened debug marker region in the command
    ///buffer. Must be paired with `cmd_debug_marker_begin_ext`.
    ///
    ///This is the legacy `VK_EXT_debug_marker` equivalent of
    ///`cmd_end_debug_utils_label_ext`. Prefer `VK_EXT_debug_utils`
    ///for new code.
    pub unsafe fn cmd_debug_marker_end_ext(&self, command_buffer: CommandBuffer) {
        let fp = self
            .commands()
            .cmd_debug_marker_end_ext
            .expect("vkCmdDebugMarkerEndEXT not loaded");
        unsafe { fp(command_buffer) };
    }
    ///Wraps [`vkCmdDebugMarkerInsertEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDebugMarkerInsertEXT.html).
    /**
    Provided by **VK_EXT_debug_marker**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Inserts a single-point debug marker into the command buffer.
    ///This is the legacy `VK_EXT_debug_marker` equivalent of
    ///`cmd_insert_debug_utils_label_ext`.
    ///
    ///`DebugMarkerMarkerInfoEXT` specifies a name and optional RGBA
    ///color.
    ///
    ///Superseded by `VK_EXT_debug_utils`. Prefer
    ///`cmd_insert_debug_utils_label_ext` for new code.
    pub unsafe fn cmd_debug_marker_insert_ext(
        &self,
        command_buffer: CommandBuffer,
        p_marker_info: &DebugMarkerMarkerInfoEXT,
    ) {
        let fp = self
            .commands()
            .cmd_debug_marker_insert_ext
            .expect("vkCmdDebugMarkerInsertEXT not loaded");
        unsafe { fp(command_buffer, p_marker_info) };
    }
    ///Wraps [`vkGetMemoryWin32HandleNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetMemoryWin32HandleNV.html).
    /**
    Provided by **VK_NV_external_memory_win32**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_TOO_MANY_OBJECTS`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Exports a Vulkan device memory allocation as a Win32 handle
    ///(HANDLE) for sharing with other APIs or processes. This is the
    ///legacy NV path; prefer `get_memory_win32_handle_khr` for new
    ///code.
    ///
    ///Requires `VK_NV_external_memory_win32`. Windows only.
    pub unsafe fn get_memory_win32_handle_nv(
        &self,
        memory: DeviceMemory,
        handle_type: ExternalMemoryHandleTypeFlagsNV,
    ) -> VkResult<isize> {
        let fp = self
            .commands()
            .get_memory_win32_handle_nv
            .expect("vkGetMemoryWin32HandleNV not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), memory, handle_type, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkCmdExecuteGeneratedCommandsNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdExecuteGeneratedCommandsNV.html).
    /**
    Provided by **VK_NV_device_generated_commands**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Executes commands that were generated on the GPU. If
    ///`is_preprocessed` is set, the commands must have been
    ///preprocessed with `cmd_preprocess_generated_commands_nv` first.
    ///
    ///Requires `VK_NV_device_generated_commands`.
    pub unsafe fn cmd_execute_generated_commands_nv(
        &self,
        command_buffer: CommandBuffer,
        is_preprocessed: u32,
        p_generated_commands_info: &GeneratedCommandsInfoNV,
    ) {
        let fp = self
            .commands()
            .cmd_execute_generated_commands_nv
            .expect("vkCmdExecuteGeneratedCommandsNV not loaded");
        unsafe { fp(command_buffer, is_preprocessed, p_generated_commands_info) };
    }
    ///Wraps [`vkCmdPreprocessGeneratedCommandsNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdPreprocessGeneratedCommandsNV.html).
    /**
    Provided by **VK_NV_device_generated_commands**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Preprocesses device-generated commands into a form suitable for
    ///fast execution. The preprocessing result is stored in a
    ///preprocess buffer and later consumed by
    ///`cmd_execute_generated_commands_nv` with `is_preprocessed` set.
    ///
    ///Requires `VK_NV_device_generated_commands`.
    pub unsafe fn cmd_preprocess_generated_commands_nv(
        &self,
        command_buffer: CommandBuffer,
        p_generated_commands_info: &GeneratedCommandsInfoNV,
    ) {
        let fp = self
            .commands()
            .cmd_preprocess_generated_commands_nv
            .expect("vkCmdPreprocessGeneratedCommandsNV not loaded");
        unsafe { fp(command_buffer, p_generated_commands_info) };
    }
    ///Wraps [`vkCmdBindPipelineShaderGroupNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBindPipelineShaderGroupNV.html).
    /**
    Provided by **VK_NV_device_generated_commands**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Binds a shader group from a pipeline that was created with
    ///multiple shader groups. Used with device-generated commands to
    ///switch between pre-compiled shader variants without rebinding
    ///the entire pipeline.
    ///
    ///Requires `VK_NV_device_generated_commands`.
    pub unsafe fn cmd_bind_pipeline_shader_group_nv(
        &self,
        command_buffer: CommandBuffer,
        pipeline_bind_point: PipelineBindPoint,
        pipeline: Pipeline,
        group_index: u32,
    ) {
        let fp = self
            .commands()
            .cmd_bind_pipeline_shader_group_nv
            .expect("vkCmdBindPipelineShaderGroupNV not loaded");
        unsafe { fp(command_buffer, pipeline_bind_point, pipeline, group_index) };
    }
    ///Wraps [`vkGetGeneratedCommandsMemoryRequirementsNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetGeneratedCommandsMemoryRequirementsNV.html).
    /**
    Provided by **VK_NV_device_generated_commands**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the memory requirements for preprocessing device-generated
    ///commands. The returned size determines how large the preprocess
    ///buffer must be for `cmd_preprocess_generated_commands_nv`.
    ///
    ///Requires `VK_NV_device_generated_commands`.
    pub unsafe fn get_generated_commands_memory_requirements_nv(
        &self,
        p_info: &GeneratedCommandsMemoryRequirementsInfoNV,
        p_memory_requirements: *mut MemoryRequirements2,
    ) {
        let fp = self
            .commands()
            .get_generated_commands_memory_requirements_nv
            .expect("vkGetGeneratedCommandsMemoryRequirementsNV not loaded");
        unsafe { fp(self.handle(), p_info, p_memory_requirements) };
    }
    ///Wraps [`vkCreateIndirectCommandsLayoutNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateIndirectCommandsLayoutNV.html).
    /**
    Provided by **VK_NV_device_generated_commands**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a layout that describes the structure of indirect command
    ///sequences for device-generated commands. The layout defines which
    ///tokens (draw, dispatch, push constants, etc.) appear in the
    ///command stream and their order.
    ///
    ///Destroy with `destroy_indirect_commands_layout_nv`.
    ///
    ///Requires `VK_NV_device_generated_commands`.
    pub unsafe fn create_indirect_commands_layout_nv(
        &self,
        p_create_info: &IndirectCommandsLayoutCreateInfoNV,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<IndirectCommandsLayoutNV> {
        let fp = self
            .commands()
            .create_indirect_commands_layout_nv
            .expect("vkCreateIndirectCommandsLayoutNV not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyIndirectCommandsLayoutNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyIndirectCommandsLayoutNV.html).
    /**
    Provided by **VK_NV_device_generated_commands**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `indirectCommandsLayout` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys an indirect commands layout created with
    ///`create_indirect_commands_layout_nv`.
    ///
    ///Requires `VK_NV_device_generated_commands`.
    pub unsafe fn destroy_indirect_commands_layout_nv(
        &self,
        indirect_commands_layout: IndirectCommandsLayoutNV,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_indirect_commands_layout_nv
            .expect("vkDestroyIndirectCommandsLayoutNV not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), indirect_commands_layout, alloc_ptr) };
    }
    ///Wraps [`vkCmdExecuteGeneratedCommandsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdExecuteGeneratedCommandsEXT.html).
    /**
    Provided by **VK_EXT_device_generated_commands**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Executes commands generated on the GPU from an indirect commands
    ///layout. The `GeneratedCommandsInfoEXT` specifies the indirect
    ///commands layout, pipeline/shader objects, and the buffer
    ///containing the generated command data.
    ///
    ///If `is_preprocessed` is true, the command data was prepared by
    ///a prior `cmd_preprocess_generated_commands_ext` call. Otherwise,
    ///preprocessing and execution happen in one step.
    ///
    ///Requires `VK_EXT_device_generated_commands`.
    pub unsafe fn cmd_execute_generated_commands_ext(
        &self,
        command_buffer: CommandBuffer,
        is_preprocessed: u32,
        p_generated_commands_info: &GeneratedCommandsInfoEXT,
    ) {
        let fp = self
            .commands()
            .cmd_execute_generated_commands_ext
            .expect("vkCmdExecuteGeneratedCommandsEXT not loaded");
        unsafe { fp(command_buffer, is_preprocessed, p_generated_commands_info) };
    }
    ///Wraps [`vkCmdPreprocessGeneratedCommandsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdPreprocessGeneratedCommandsEXT.html).
    /**
    Provided by **VK_EXT_device_generated_commands**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///- `stateCommandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Preprocesses device-generated commands into an intermediate
    ///format. This can be done in a separate command buffer or pass,
    ///then executed later with `cmd_execute_generated_commands_ext`
    ///(with `is_preprocessed` = true).
    ///
    ///Separating preprocessing from execution allows overlapping the
    ///preprocessing work with other GPU tasks.
    ///
    ///Requires `VK_EXT_device_generated_commands`.
    pub unsafe fn cmd_preprocess_generated_commands_ext(
        &self,
        command_buffer: CommandBuffer,
        p_generated_commands_info: &GeneratedCommandsInfoEXT,
        state_command_buffer: CommandBuffer,
    ) {
        let fp = self
            .commands()
            .cmd_preprocess_generated_commands_ext
            .expect("vkCmdPreprocessGeneratedCommandsEXT not loaded");
        unsafe {
            fp(
                command_buffer,
                p_generated_commands_info,
                state_command_buffer,
            )
        };
    }
    ///Wraps [`vkGetGeneratedCommandsMemoryRequirementsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetGeneratedCommandsMemoryRequirementsEXT.html).
    /**
    Provided by **VK_EXT_device_generated_commands**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the memory requirements for preprocessing and executing
    ///device-generated commands. Returns a `MemoryRequirements2` with
    ///the size and alignment needed for the preprocess buffer.
    ///
    ///Call this before allocating the preprocess buffer used by
    ///`cmd_preprocess_generated_commands_ext` and
    ///`cmd_execute_generated_commands_ext`.
    ///
    ///Requires `VK_EXT_device_generated_commands`.
    pub unsafe fn get_generated_commands_memory_requirements_ext(
        &self,
        p_info: &GeneratedCommandsMemoryRequirementsInfoEXT,
        p_memory_requirements: *mut MemoryRequirements2,
    ) {
        let fp = self
            .commands()
            .get_generated_commands_memory_requirements_ext
            .expect("vkGetGeneratedCommandsMemoryRequirementsEXT not loaded");
        unsafe { fp(self.handle(), p_info, p_memory_requirements) };
    }
    ///Wraps [`vkCreateIndirectCommandsLayoutEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateIndirectCommandsLayoutEXT.html).
    /**
    Provided by **VK_EXT_device_generated_commands**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates an indirect commands layout that defines the structure of
    ///GPU-generated command sequences. Each token in the layout
    ///describes one command element (draw, dispatch, push constant,
    ///vertex buffer bind, index buffer bind, etc.).
    ///
    ///The layout is used with `cmd_execute_generated_commands_ext`.
    ///
    ///Destroy with `destroy_indirect_commands_layout_ext`.
    ///
    ///Requires `VK_EXT_device_generated_commands`.
    pub unsafe fn create_indirect_commands_layout_ext(
        &self,
        p_create_info: &IndirectCommandsLayoutCreateInfoEXT,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<IndirectCommandsLayoutEXT> {
        let fp = self
            .commands()
            .create_indirect_commands_layout_ext
            .expect("vkCreateIndirectCommandsLayoutEXT not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyIndirectCommandsLayoutEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyIndirectCommandsLayoutEXT.html).
    /**
    Provided by **VK_EXT_device_generated_commands**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `indirectCommandsLayout` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys an indirect commands layout created with
    ///`create_indirect_commands_layout_ext`.
    ///
    ///Requires `VK_EXT_device_generated_commands`.
    pub unsafe fn destroy_indirect_commands_layout_ext(
        &self,
        indirect_commands_layout: IndirectCommandsLayoutEXT,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_indirect_commands_layout_ext
            .expect("vkDestroyIndirectCommandsLayoutEXT not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), indirect_commands_layout, alloc_ptr) };
    }
    ///Wraps [`vkCreateIndirectExecutionSetEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateIndirectExecutionSetEXT.html).
    /**
    Provided by **VK_EXT_device_generated_commands**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates an indirect execution set, a table of pipelines or
    ///shader objects that can be indexed at execution time by
    ///device-generated commands.
    ///
    ///The GPU selects which pipeline/shader to use based on an index
    ///in the generated command stream, enabling fully GPU-driven
    ///material/shader selection.
    ///
    ///Destroy with `destroy_indirect_execution_set_ext`.
    ///
    ///Requires `VK_EXT_device_generated_commands`.
    pub unsafe fn create_indirect_execution_set_ext(
        &self,
        p_create_info: &IndirectExecutionSetCreateInfoEXT,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<IndirectExecutionSetEXT> {
        let fp = self
            .commands()
            .create_indirect_execution_set_ext
            .expect("vkCreateIndirectExecutionSetEXT not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyIndirectExecutionSetEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyIndirectExecutionSetEXT.html).
    /**
    Provided by **VK_EXT_device_generated_commands**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `indirectExecutionSet` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys an indirect execution set created with
    ///`create_indirect_execution_set_ext`.
    ///
    ///Requires `VK_EXT_device_generated_commands`.
    pub unsafe fn destroy_indirect_execution_set_ext(
        &self,
        indirect_execution_set: IndirectExecutionSetEXT,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_indirect_execution_set_ext
            .expect("vkDestroyIndirectExecutionSetEXT not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), indirect_execution_set, alloc_ptr) };
    }
    ///Wraps [`vkUpdateIndirectExecutionSetPipelineEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkUpdateIndirectExecutionSetPipelineEXT.html).
    /**
    Provided by **VK_EXT_device_generated_commands**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `indirectExecutionSet` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Updates entries in an indirect execution set that holds pipelines.
    ///Each `WriteIndirectExecutionSetPipelineEXT` maps an index to a
    ///pipeline handle.
    ///
    ///The pipelines must be compatible with the initial pipeline used
    ///to create the execution set.
    ///
    ///Requires `VK_EXT_device_generated_commands`.
    pub unsafe fn update_indirect_execution_set_pipeline_ext(
        &self,
        indirect_execution_set: IndirectExecutionSetEXT,
        p_execution_set_writes: &[WriteIndirectExecutionSetPipelineEXT],
    ) {
        let fp = self
            .commands()
            .update_indirect_execution_set_pipeline_ext
            .expect("vkUpdateIndirectExecutionSetPipelineEXT not loaded");
        unsafe {
            fp(
                self.handle(),
                indirect_execution_set,
                p_execution_set_writes.len() as u32,
                p_execution_set_writes.as_ptr(),
            )
        };
    }
    ///Wraps [`vkUpdateIndirectExecutionSetShaderEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkUpdateIndirectExecutionSetShaderEXT.html).
    /**
    Provided by **VK_EXT_device_generated_commands**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `indirectExecutionSet` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Updates entries in an indirect execution set that holds shader
    ///objects. Each `WriteIndirectExecutionSetShaderEXT` maps an index
    ///to a shader object handle.
    ///
    ///The shaders must be compatible with the initial shader used to
    ///create the execution set.
    ///
    ///Requires `VK_EXT_device_generated_commands`.
    pub unsafe fn update_indirect_execution_set_shader_ext(
        &self,
        indirect_execution_set: IndirectExecutionSetEXT,
        p_execution_set_writes: &[WriteIndirectExecutionSetShaderEXT],
    ) {
        let fp = self
            .commands()
            .update_indirect_execution_set_shader_ext
            .expect("vkUpdateIndirectExecutionSetShaderEXT not loaded");
        unsafe {
            fp(
                self.handle(),
                indirect_execution_set,
                p_execution_set_writes.len() as u32,
                p_execution_set_writes.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdPushDescriptorSet`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdPushDescriptorSet.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_4**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Pushes descriptor updates directly into the command buffer without
    ///allocating a descriptor set from a pool. The descriptors are embedded
    ///in the command stream and only live for the duration of the current
    ///command buffer recording.
    ///
    ///**Advantages**:
    ///
    ///- No descriptor pool allocation or management.
    ///- No need to track descriptor set lifetimes.
    ///- Ideal for per-draw data that changes every frame.
    ///
    ///**Trade-offs**:
    ///
    ///- Inflates command buffer size (descriptors are stored inline).
    ///- Not suitable for large descriptor sets, use conventional
    ///  allocated sets for sets with many bindings.
    ///
    ///The pipeline layout must have been created with
    ///`DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR` on the target set
    ///index.
    ///
    ///Core in Vulkan 1.4. Previously available via `VK_KHR_push_descriptor`.
    pub unsafe fn cmd_push_descriptor_set(
        &self,
        command_buffer: CommandBuffer,
        pipeline_bind_point: PipelineBindPoint,
        layout: PipelineLayout,
        set: u32,
        p_descriptor_writes: &[WriteDescriptorSet],
    ) {
        let fp = self
            .commands()
            .cmd_push_descriptor_set
            .expect("vkCmdPushDescriptorSet not loaded");
        unsafe {
            fp(
                command_buffer,
                pipeline_bind_point,
                layout,
                set,
                p_descriptor_writes.len() as u32,
                p_descriptor_writes.as_ptr(),
            )
        };
    }
    ///Wraps [`vkTrimCommandPool`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkTrimCommandPool.html).
    /**
    Provided by **VK_BASE_VERSION_1_1**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `commandPool` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Returns unused memory from a command pool back to the system. This
    ///is a hint to the driver, it may or may not actually release memory.
    ///
    ///Call this after a period of high command buffer allocation followed
    ///by a return to lower usage (e.g. after loading screens or level
    ///transitions). It does not affect any allocated command buffers.
    ///
    ///Unlike `reset_command_pool`, trimming does not reset or invalidate
    ///command buffers. It only reclaims excess internal memory that the
    ///pool pre-allocated.
    ///
    ///In a steady-state frame loop where you reset the pool every frame,
    ///trimming is unnecessary, the pool reuses its memory naturally.
    pub unsafe fn trim_command_pool(&self, command_pool: CommandPool, flags: CommandPoolTrimFlags) {
        let fp = self
            .commands()
            .trim_command_pool
            .expect("vkTrimCommandPool not loaded");
        unsafe { fp(self.handle(), command_pool, flags) };
    }
    ///Wraps [`vkGetMemoryWin32HandleKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetMemoryWin32HandleKHR.html).
    /**
    Provided by **VK_KHR_external_memory_win32**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_TOO_MANY_OBJECTS`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Exports a device memory allocation as a Windows HANDLE. The
    ///handle can be shared with other processes or APIs (D3D12, CUDA)
    ///for GPU memory interop.
    ///
    ///`MemoryGetWin32HandleInfoKHR` specifies the `DeviceMemory` and
    ///handle type (`OPAQUE_WIN32` or `OPAQUE_WIN32_KMT`).
    ///
    ///For `OPAQUE_WIN32`, the handle must be closed with `CloseHandle`
    ///when done. `OPAQUE_WIN32_KMT` handles are kernel-managed and do
    ///not need explicit cleanup.
    ///
    ///Windows only. Use `get_memory_fd_khr` on Linux.
    pub unsafe fn get_memory_win32_handle_khr(
        &self,
        p_get_win32_handle_info: &MemoryGetWin32HandleInfoKHR,
    ) -> VkResult<isize> {
        let fp = self
            .commands()
            .get_memory_win32_handle_khr
            .expect("vkGetMemoryWin32HandleKHR not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_get_win32_handle_info, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetMemoryWin32HandlePropertiesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetMemoryWin32HandlePropertiesKHR.html).
    /**
    Provided by **VK_KHR_external_memory_win32**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_INVALID_EXTERNAL_HANDLE`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries which memory types are compatible with an external
    ///Windows HANDLE. Use this when importing memory from a handle
    ///received from another process or API (e.g., D3D11 shared
    ///textures).
    ///
    ///Returns `MemoryWin32HandlePropertiesKHR` with `memory_type_bits`
    ///indicating compatible memory type indices for import.
    ///
    ///Not valid for `OPAQUE_WIN32` or `OPAQUE_WIN32_KMT` handle types,
    ///those have their memory type determined by the exporting
    ///allocation.
    pub unsafe fn get_memory_win32_handle_properties_khr(
        &self,
        handle_type: ExternalMemoryHandleTypeFlagBits,
        handle: isize,
        p_memory_win32_handle_properties: *mut MemoryWin32HandlePropertiesKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_memory_win32_handle_properties_khr
            .expect("vkGetMemoryWin32HandlePropertiesKHR not loaded");
        check(unsafe {
            fp(
                self.handle(),
                handle_type,
                handle,
                p_memory_win32_handle_properties,
            )
        })
    }
    ///Wraps [`vkGetMemoryFdKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetMemoryFdKHR.html).
    /**
    Provided by **VK_KHR_external_memory_fd**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_TOO_MANY_OBJECTS`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Exports a device memory allocation as a POSIX file descriptor.
    ///The fd can be sent to another process (via Unix domain sockets)
    ///or another Vulkan device to share GPU memory.
    ///
    ///`MemoryGetFdInfoKHR` specifies the `DeviceMemory` and the handle
    ///type (`OPAQUE_FD` or `DMA_BUF`). The memory must have been
    ///allocated with `ExportMemoryAllocateInfo` requesting the
    ///corresponding handle type.
    ///
    ///The caller owns the returned fd and must close it when done.
    ///Each call returns a new fd, duplicates are independent.
    ///
    ///Linux/Android only. Use `get_memory_win32_handle_khr` on Windows.
    pub unsafe fn get_memory_fd_khr(
        &self,
        p_get_fd_info: &MemoryGetFdInfoKHR,
    ) -> VkResult<core::ffi::c_int> {
        let fp = self
            .commands()
            .get_memory_fd_khr
            .expect("vkGetMemoryFdKHR not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_get_fd_info, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetMemoryFdPropertiesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetMemoryFdPropertiesKHR.html).
    /**
    Provided by **VK_KHR_external_memory_fd**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_INVALID_EXTERNAL_HANDLE`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries which memory types are compatible with an external file
    ///descriptor. Use this when importing memory from an fd received
    ///from another process or API.
    ///
    ///Returns `MemoryFdPropertiesKHR` with a `memory_type_bits` bitmask
    ///indicating which memory type indices can be used when allocating
    ///memory to import this fd.
    ///
    ///Only valid for `DMA_BUF` handle types. `OPAQUE_FD` handles don't
    ///need this query, their memory type is determined by the
    ///exporting allocation.
    pub unsafe fn get_memory_fd_properties_khr(
        &self,
        handle_type: ExternalMemoryHandleTypeFlagBits,
        fd: core::ffi::c_int,
        p_memory_fd_properties: *mut MemoryFdPropertiesKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_memory_fd_properties_khr
            .expect("vkGetMemoryFdPropertiesKHR not loaded");
        check(unsafe { fp(self.handle(), handle_type, fd, p_memory_fd_properties) })
    }
    ///Wraps [`vkGetMemoryZirconHandleFUCHSIA`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetMemoryZirconHandleFUCHSIA.html).
    /**
    Provided by **VK_FUCHSIA_external_memory**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_TOO_MANY_OBJECTS`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Exports a Vulkan device memory allocation as a Fuchsia Zircon
    ///VMO handle. The returned handle can be shared with other Fuchsia
    ///processes. Fuchsia OS only.
    ///
    ///Requires `VK_FUCHSIA_external_memory`.
    pub unsafe fn get_memory_zircon_handle_fuchsia(
        &self,
        p_get_zircon_handle_info: &MemoryGetZirconHandleInfoFUCHSIA,
    ) -> VkResult<u32> {
        let fp = self
            .commands()
            .get_memory_zircon_handle_fuchsia
            .expect("vkGetMemoryZirconHandleFUCHSIA not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_get_zircon_handle_info, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetMemoryZirconHandlePropertiesFUCHSIA`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetMemoryZirconHandlePropertiesFUCHSIA.html).
    /**
    Provided by **VK_FUCHSIA_external_memory**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_INVALID_EXTERNAL_HANDLE`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the memory properties (compatible memory type bits) for
    ///a Zircon VMO handle. Use before importing external memory to
    ///determine which memory type to allocate. Fuchsia OS only.
    ///
    ///Requires `VK_FUCHSIA_external_memory`.
    pub unsafe fn get_memory_zircon_handle_properties_fuchsia(
        &self,
        handle_type: ExternalMemoryHandleTypeFlagBits,
        zircon_handle: u32,
        p_memory_zircon_handle_properties: *mut MemoryZirconHandlePropertiesFUCHSIA,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_memory_zircon_handle_properties_fuchsia
            .expect("vkGetMemoryZirconHandlePropertiesFUCHSIA not loaded");
        check(unsafe {
            fp(
                self.handle(),
                handle_type,
                zircon_handle,
                p_memory_zircon_handle_properties,
            )
        })
    }
    ///Wraps [`vkGetMemoryRemoteAddressNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetMemoryRemoteAddressNV.html).
    /**
    Provided by **VK_NV_external_memory_rdma**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_INVALID_EXTERNAL_HANDLE`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves a remote device address for a Vulkan memory allocation,
    ///enabling RDMA (Remote Direct Memory Access) between devices. The
    ///returned address can be used by another device to directly access
    ///this memory over a high-speed interconnect.
    ///
    ///Requires `VK_NV_external_memory_rdma`.
    pub unsafe fn get_memory_remote_address_nv(
        &self,
        p_memory_get_remote_address_info: &MemoryGetRemoteAddressInfoNV,
    ) -> VkResult<*mut core::ffi::c_void> {
        let fp = self
            .commands()
            .get_memory_remote_address_nv
            .expect("vkGetMemoryRemoteAddressNV not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_memory_get_remote_address_info, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetMemorySciBufNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetMemorySciBufNV.html).
    ///
    ///# Errors
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Exports a Vulkan device memory allocation as a QNX SCI buffer
    ///for cross-process or cross-API sharing on QNX platforms.
    ///
    ///Requires `VK_NV_external_memory_sci_buf`. QNX only.
    pub unsafe fn get_memory_sci_buf_nv(
        &self,
        p_get_sci_buf_info: &MemoryGetSciBufInfoNV,
    ) -> VkResult<core::ffi::c_void> {
        let fp = self
            .commands()
            .get_memory_sci_buf_nv
            .expect("vkGetMemorySciBufNV not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_get_sci_buf_info, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetSemaphoreWin32HandleKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetSemaphoreWin32HandleKHR.html).
    /**
    Provided by **VK_KHR_external_semaphore_win32**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_TOO_MANY_OBJECTS`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Exports a semaphore's synchronization state as a Windows HANDLE.
    ///The handle can be shared with other processes or APIs for
    ///cross-process GPU synchronization.
    ///
    ///`SemaphoreGetWin32HandleInfoKHR` specifies the semaphore and
    ///handle type (`OPAQUE_WIN32`, `OPAQUE_WIN32_KMT`, or
    ///`D3D12_FENCE`).
    ///
    ///For `OPAQUE_WIN32` and `D3D12_FENCE`, close the handle with
    ///`CloseHandle` when done. `OPAQUE_WIN32_KMT` handles are
    ///kernel-managed.
    ///
    ///Windows only. Use `get_semaphore_fd_khr` on Linux.
    pub unsafe fn get_semaphore_win32_handle_khr(
        &self,
        p_get_win32_handle_info: &SemaphoreGetWin32HandleInfoKHR,
    ) -> VkResult<isize> {
        let fp = self
            .commands()
            .get_semaphore_win32_handle_khr
            .expect("vkGetSemaphoreWin32HandleKHR not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_get_win32_handle_info, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkImportSemaphoreWin32HandleKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkImportSemaphoreWin32HandleKHR.html).
    /**
    Provided by **VK_KHR_external_semaphore_win32**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_INVALID_EXTERNAL_HANDLE`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Imports a synchronization payload from a Windows HANDLE into a
    ///semaphore. After import, the semaphore uses the external
    ///synchronization state.
    ///
    ///`ImportSemaphoreWin32HandleInfoKHR` specifies the target
    ///semaphore, handle type, handle (or name for named handles), and
    ///whether the import is temporary or permanent.
    ///
    ///The handle is duplicated internally, the caller retains
    ///ownership and should close it when no longer needed.
    ///
    ///Windows only. Use `import_semaphore_fd_khr` on Linux.
    pub unsafe fn import_semaphore_win32_handle_khr(
        &self,
        p_import_semaphore_win32_handle_info: &ImportSemaphoreWin32HandleInfoKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .import_semaphore_win32_handle_khr
            .expect("vkImportSemaphoreWin32HandleKHR not loaded");
        check(unsafe { fp(self.handle(), p_import_semaphore_win32_handle_info) })
    }
    ///Wraps [`vkGetSemaphoreFdKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetSemaphoreFdKHR.html).
    /**
    Provided by **VK_KHR_external_semaphore_fd**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_TOO_MANY_OBJECTS`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Exports a semaphore's synchronization state as a POSIX file
    ///descriptor. The fd can be transferred to another process for
    ///cross-process GPU synchronization.
    ///
    ///`SemaphoreGetFdInfoKHR` specifies the semaphore and handle type
    ///(`OPAQUE_FD` or `SYNC_FD`). For `SYNC_FD`, the semaphore must
    ///be signaled or have a pending signal operation, exporting
    ///transfers ownership and resets the semaphore to unsignaled.
    ///
    ///The caller owns the returned fd. For `OPAQUE_FD`, each export
    ///creates a new reference. For `SYNC_FD`, the export is a
    ///move, the semaphore payload is transferred.
    ///
    ///Linux/Android only. Use `get_semaphore_win32_handle_khr` on
    ///Windows.
    pub unsafe fn get_semaphore_fd_khr(
        &self,
        p_get_fd_info: &SemaphoreGetFdInfoKHR,
    ) -> VkResult<core::ffi::c_int> {
        let fp = self
            .commands()
            .get_semaphore_fd_khr
            .expect("vkGetSemaphoreFdKHR not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_get_fd_info, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkImportSemaphoreFdKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkImportSemaphoreFdKHR.html).
    /**
    Provided by **VK_KHR_external_semaphore_fd**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_INVALID_EXTERNAL_HANDLE`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Imports a synchronization payload from a POSIX file descriptor
    ///into a semaphore. After import, the semaphore uses the external
    ///synchronization state.
    ///
    ///`ImportSemaphoreFdInfoKHR` specifies the target semaphore, handle
    ///type, fd, and whether the import is temporary (payload consumed
    ///on first wait) or permanent.
    ///
    ///For `SYNC_FD`, the import takes ownership of the fd, do not
    ///close it afterward. For `OPAQUE_FD`, the fd is duplicated
    ///internally and can be closed after the call.
    ///
    ///Linux/Android only. Use `import_semaphore_win32_handle_khr` on
    ///Windows.
    pub unsafe fn import_semaphore_fd_khr(
        &self,
        p_import_semaphore_fd_info: &ImportSemaphoreFdInfoKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .import_semaphore_fd_khr
            .expect("vkImportSemaphoreFdKHR not loaded");
        check(unsafe { fp(self.handle(), p_import_semaphore_fd_info) })
    }
    ///Wraps [`vkGetSemaphoreZirconHandleFUCHSIA`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetSemaphoreZirconHandleFUCHSIA.html).
    /**
    Provided by **VK_FUCHSIA_external_semaphore**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_TOO_MANY_OBJECTS`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Exports a Vulkan semaphore as a Fuchsia Zircon event handle for
    ///cross-process synchronisation. Fuchsia OS only.
    ///
    ///Requires `VK_FUCHSIA_external_semaphore`.
    pub unsafe fn get_semaphore_zircon_handle_fuchsia(
        &self,
        p_get_zircon_handle_info: &SemaphoreGetZirconHandleInfoFUCHSIA,
    ) -> VkResult<u32> {
        let fp = self
            .commands()
            .get_semaphore_zircon_handle_fuchsia
            .expect("vkGetSemaphoreZirconHandleFUCHSIA not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_get_zircon_handle_info, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkImportSemaphoreZirconHandleFUCHSIA`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkImportSemaphoreZirconHandleFUCHSIA.html).
    /**
    Provided by **VK_FUCHSIA_external_semaphore**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_INVALID_EXTERNAL_HANDLE`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Imports a Fuchsia Zircon event handle into an existing Vulkan
    ///semaphore for cross-process synchronisation. Fuchsia OS only.
    ///
    ///Requires `VK_FUCHSIA_external_semaphore`.
    pub unsafe fn import_semaphore_zircon_handle_fuchsia(
        &self,
        p_import_semaphore_zircon_handle_info: &ImportSemaphoreZirconHandleInfoFUCHSIA,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .import_semaphore_zircon_handle_fuchsia
            .expect("vkImportSemaphoreZirconHandleFUCHSIA not loaded");
        check(unsafe { fp(self.handle(), p_import_semaphore_zircon_handle_info) })
    }
    ///Wraps [`vkGetFenceWin32HandleKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetFenceWin32HandleKHR.html).
    /**
    Provided by **VK_KHR_external_fence_win32**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_TOO_MANY_OBJECTS`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Exports a fence's synchronization state as a Windows HANDLE
    ///for cross-process fence synchronization.
    ///
    ///`FenceGetWin32HandleInfoKHR` specifies the fence and handle type
    ///(`OPAQUE_WIN32` or `OPAQUE_WIN32_KMT`).
    ///
    ///For `OPAQUE_WIN32`, close the handle with `CloseHandle` when
    ///done. `OPAQUE_WIN32_KMT` handles are kernel-managed.
    ///
    ///Windows only. Use `get_fence_fd_khr` on Linux.
    pub unsafe fn get_fence_win32_handle_khr(
        &self,
        p_get_win32_handle_info: &FenceGetWin32HandleInfoKHR,
    ) -> VkResult<isize> {
        let fp = self
            .commands()
            .get_fence_win32_handle_khr
            .expect("vkGetFenceWin32HandleKHR not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_get_win32_handle_info, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkImportFenceWin32HandleKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkImportFenceWin32HandleKHR.html).
    /**
    Provided by **VK_KHR_external_fence_win32**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_INVALID_EXTERNAL_HANDLE`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Imports a synchronization payload from a Windows HANDLE into a
    ///fence.
    ///
    ///`ImportFenceWin32HandleInfoKHR` specifies the target fence,
    ///handle type, handle (or name), and whether the import is
    ///temporary or permanent.
    ///
    ///The handle is duplicated internally, the caller retains
    ///ownership and should close it when no longer needed.
    ///
    ///Windows only. Use `import_fence_fd_khr` on Linux.
    pub unsafe fn import_fence_win32_handle_khr(
        &self,
        p_import_fence_win32_handle_info: &ImportFenceWin32HandleInfoKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .import_fence_win32_handle_khr
            .expect("vkImportFenceWin32HandleKHR not loaded");
        check(unsafe { fp(self.handle(), p_import_fence_win32_handle_info) })
    }
    ///Wraps [`vkGetFenceFdKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetFenceFdKHR.html).
    /**
    Provided by **VK_KHR_external_fence_fd**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_TOO_MANY_OBJECTS`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Exports a fence's synchronization state as a POSIX file
    ///descriptor. The fd enables cross-process fence synchronization.
    ///
    ///`FenceGetFdInfoKHR` specifies the fence and handle type
    ///(`OPAQUE_FD` or `SYNC_FD`). For `SYNC_FD`, the fence must be
    ///signaled or have a pending signal, exporting transfers the
    ///payload and resets the fence.
    ///
    ///The caller owns the returned fd and must close it when done.
    ///
    ///Linux/Android only. Use `get_fence_win32_handle_khr` on Windows.
    pub unsafe fn get_fence_fd_khr(
        &self,
        p_get_fd_info: &FenceGetFdInfoKHR,
    ) -> VkResult<core::ffi::c_int> {
        let fp = self
            .commands()
            .get_fence_fd_khr
            .expect("vkGetFenceFdKHR not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_get_fd_info, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkImportFenceFdKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkImportFenceFdKHR.html).
    /**
    Provided by **VK_KHR_external_fence_fd**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_INVALID_EXTERNAL_HANDLE`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Imports a synchronization payload from a POSIX file descriptor
    ///into a fence.
    ///
    ///`ImportFenceFdInfoKHR` specifies the target fence, handle type,
    ///fd, and whether the import is temporary (payload consumed on
    ///first wait/reset) or permanent.
    ///
    ///For `SYNC_FD`, ownership of the fd transfers to the
    ///implementation, do not close it. For `OPAQUE_FD`, the fd is
    ///duplicated and can be closed after the call.
    ///
    ///Linux/Android only. Use `import_fence_win32_handle_khr` on
    ///Windows.
    pub unsafe fn import_fence_fd_khr(
        &self,
        p_import_fence_fd_info: &ImportFenceFdInfoKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .import_fence_fd_khr
            .expect("vkImportFenceFdKHR not loaded");
        check(unsafe { fp(self.handle(), p_import_fence_fd_info) })
    }
    ///Wraps [`vkGetFenceSciSyncFenceNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetFenceSciSyncFenceNV.html).
    ///
    ///# Errors
    ///- `VK_ERROR_INVALID_EXTERNAL_HANDLE`
    ///- `VK_ERROR_NOT_PERMITTED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Exports a Vulkan fence as a SciSync fence handle for
    ///cross-process synchronisation on NV safety-critical platforms.
    ///QNX/NVIDIA Safety only.
    ///
    ///Requires `VK_NV_external_sci_sync`.
    pub unsafe fn get_fence_sci_sync_fence_nv(
        &self,
        p_get_sci_sync_handle_info: &FenceGetSciSyncInfoNV,
    ) -> VkResult<core::ffi::c_void> {
        let fp = self
            .commands()
            .get_fence_sci_sync_fence_nv
            .expect("vkGetFenceSciSyncFenceNV not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_get_sci_sync_handle_info, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetFenceSciSyncObjNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetFenceSciSyncObjNV.html).
    ///
    ///# Errors
    ///- `VK_ERROR_INVALID_EXTERNAL_HANDLE`
    ///- `VK_ERROR_NOT_PERMITTED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Exports a Vulkan fence as a SciSync object handle for
    ///cross-process synchronisation on NV safety-critical platforms.
    ///QNX/NVIDIA Safety only.
    ///
    ///Requires `VK_NV_external_sci_sync`.
    pub unsafe fn get_fence_sci_sync_obj_nv(
        &self,
        p_get_sci_sync_handle_info: &FenceGetSciSyncInfoNV,
    ) -> VkResult<core::ffi::c_void> {
        let fp = self
            .commands()
            .get_fence_sci_sync_obj_nv
            .expect("vkGetFenceSciSyncObjNV not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_get_sci_sync_handle_info, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkImportFenceSciSyncFenceNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkImportFenceSciSyncFenceNV.html).
    ///
    ///# Errors
    ///- `VK_ERROR_INVALID_EXTERNAL_HANDLE`
    ///- `VK_ERROR_NOT_PERMITTED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Imports a SciSync fence handle into an existing Vulkan fence for
    ///cross-process synchronisation on NV safety-critical platforms.
    ///QNX/NVIDIA Safety only.
    ///
    ///Requires `VK_NV_external_sci_sync`.
    pub unsafe fn import_fence_sci_sync_fence_nv(
        &self,
        p_import_fence_sci_sync_info: &ImportFenceSciSyncInfoNV,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .import_fence_sci_sync_fence_nv
            .expect("vkImportFenceSciSyncFenceNV not loaded");
        check(unsafe { fp(self.handle(), p_import_fence_sci_sync_info) })
    }
    ///Wraps [`vkImportFenceSciSyncObjNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkImportFenceSciSyncObjNV.html).
    ///
    ///# Errors
    ///- `VK_ERROR_INVALID_EXTERNAL_HANDLE`
    ///- `VK_ERROR_NOT_PERMITTED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Imports a SciSync object handle into an existing Vulkan fence
    ///for cross-process synchronisation on NV safety-critical
    ///platforms. QNX/NVIDIA Safety only.
    ///
    ///Requires `VK_NV_external_sci_sync`.
    pub unsafe fn import_fence_sci_sync_obj_nv(
        &self,
        p_import_fence_sci_sync_info: &ImportFenceSciSyncInfoNV,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .import_fence_sci_sync_obj_nv
            .expect("vkImportFenceSciSyncObjNV not loaded");
        check(unsafe { fp(self.handle(), p_import_fence_sci_sync_info) })
    }
    ///Wraps [`vkGetSemaphoreSciSyncObjNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetSemaphoreSciSyncObjNV.html).
    ///
    ///# Errors
    ///- `VK_ERROR_INVALID_EXTERNAL_HANDLE`
    ///- `VK_ERROR_NOT_PERMITTED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Exports a Vulkan semaphore as a SciSync object handle for
    ///cross-process synchronisation on NV safety-critical platforms.
    ///QNX/NVIDIA Safety only.
    ///
    ///Requires `VK_NV_external_sci_sync`.
    pub unsafe fn get_semaphore_sci_sync_obj_nv(
        &self,
        p_get_sci_sync_info: &SemaphoreGetSciSyncInfoNV,
    ) -> VkResult<core::ffi::c_void> {
        let fp = self
            .commands()
            .get_semaphore_sci_sync_obj_nv
            .expect("vkGetSemaphoreSciSyncObjNV not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_get_sci_sync_info, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkImportSemaphoreSciSyncObjNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkImportSemaphoreSciSyncObjNV.html).
    ///
    ///# Errors
    ///- `VK_ERROR_INVALID_EXTERNAL_HANDLE`
    ///- `VK_ERROR_NOT_PERMITTED`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Imports a SciSync object handle into an existing Vulkan
    ///semaphore for cross-process synchronisation on NV
    ///safety-critical platforms. QNX/NVIDIA Safety only.
    ///
    ///Requires `VK_NV_external_sci_sync`.
    pub unsafe fn import_semaphore_sci_sync_obj_nv(
        &self,
        p_import_semaphore_sci_sync_info: &ImportSemaphoreSciSyncInfoNV,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .import_semaphore_sci_sync_obj_nv
            .expect("vkImportSemaphoreSciSyncObjNV not loaded");
        check(unsafe { fp(self.handle(), p_import_semaphore_sci_sync_info) })
    }
    ///Wraps [`vkCreateSemaphoreSciSyncPoolNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateSemaphoreSciSyncPoolNV.html).
    ///
    ///# Errors
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a pool of SciSync-backed semaphores for efficient
    ///cross-engine synchronisation on NV safety-critical platforms.
    ///QNX/NVIDIA Safety only.
    ///
    ///Requires `VK_NV_external_sci_sync2`.
    pub unsafe fn create_semaphore_sci_sync_pool_nv(
        &self,
        p_create_info: &SemaphoreSciSyncPoolCreateInfoNV,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<SemaphoreSciSyncPoolNV> {
        let fp = self
            .commands()
            .create_semaphore_sci_sync_pool_nv
            .expect("vkCreateSemaphoreSciSyncPoolNV not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroySemaphoreSciSyncPoolNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroySemaphoreSciSyncPoolNV.html).
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `semaphorePool` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a SciSync semaphore pool. The pool must not be in use
    ///by any pending operations. QNX/NVIDIA Safety only.
    ///
    ///Requires `VK_NV_external_sci_sync2`.
    pub unsafe fn destroy_semaphore_sci_sync_pool_nv(
        &self,
        semaphore_pool: SemaphoreSciSyncPoolNV,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_semaphore_sci_sync_pool_nv
            .expect("vkDestroySemaphoreSciSyncPoolNV not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), semaphore_pool, alloc_ptr) };
    }
    ///Wraps [`vkDisplayPowerControlEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDisplayPowerControlEXT.html).
    /**
    Provided by **VK_EXT_display_control**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Controls the power state of a display (e.g., standby, suspend,
    ///off, on). `DisplayPowerInfoEXT` specifies the desired power state.
    ///
    ///Requires `VK_EXT_display_control`.
    pub unsafe fn display_power_control_ext(
        &self,
        display: DisplayKHR,
        p_display_power_info: &DisplayPowerInfoEXT,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .display_power_control_ext
            .expect("vkDisplayPowerControlEXT not loaded");
        check(unsafe { fp(self.handle(), display, p_display_power_info) })
    }
    ///Wraps [`vkRegisterDeviceEventEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkRegisterDeviceEventEXT.html).
    /**
    Provided by **VK_EXT_display_control**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Registers a fence to be signaled when a device event occurs.
    ///`DeviceEventInfoEXT` specifies the event type (e.g.,
    ///`DISPLAY_HOTPLUG`).
    ///
    ///Returns a fence that will be signaled when the event fires.
    ///
    ///Requires `VK_EXT_display_control`.
    pub unsafe fn register_device_event_ext(
        &self,
        p_device_event_info: &DeviceEventInfoEXT,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<Fence> {
        let fp = self
            .commands()
            .register_device_event_ext
            .expect("vkRegisterDeviceEventEXT not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_device_event_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkRegisterDisplayEventEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkRegisterDisplayEventEXT.html).
    /**
    Provided by **VK_EXT_display_control**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Registers a fence to be signaled when a display event occurs.
    ///`DisplayEventInfoEXT` specifies the event type (e.g.,
    ///`FIRST_PIXEL_OUT`, signaled at the start of the first scanline
    ///after a present).
    ///
    ///Returns a fence. Useful for frame pacing and display timing.
    ///
    ///Requires `VK_EXT_display_control`.
    pub unsafe fn register_display_event_ext(
        &self,
        display: DisplayKHR,
        p_display_event_info: &DisplayEventInfoEXT,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<Fence> {
        let fp = self
            .commands()
            .register_display_event_ext
            .expect("vkRegisterDisplayEventEXT not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe {
            fp(
                self.handle(),
                display,
                p_display_event_info,
                alloc_ptr,
                &mut out,
            )
        })?;
        Ok(out)
    }
    ///Wraps [`vkGetSwapchainCounterEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetSwapchainCounterEXT.html).
    /**
    Provided by **VK_EXT_display_control**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_DEVICE_LOST`
    ///- `VK_ERROR_OUT_OF_DATE_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries a performance counter associated with a swapchain (e.g.,
    ///vertical blanks). Returns the counter value as a `u64`.
    ///
    ///The counter type is specified as a `SurfaceCounterFlagBitsEXT`
    ///(typically `VBLANK`).
    ///
    ///Requires `VK_EXT_display_control`.
    pub unsafe fn get_swapchain_counter_ext(
        &self,
        swapchain: SwapchainKHR,
        counter: SurfaceCounterFlagBitsEXT,
    ) -> VkResult<u64> {
        let fp = self
            .commands()
            .get_swapchain_counter_ext
            .expect("vkGetSwapchainCounterEXT not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), swapchain, counter, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetDeviceGroupPeerMemoryFeatures`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDeviceGroupPeerMemoryFeatures.html).
    /**
    Provided by **VK_BASE_VERSION_1_1**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the memory access capabilities between two physical devices
    ///in a device group. Returns flags indicating whether memory allocated
    ///on one device can be copied to, accessed generically, or accessed
    ///natively from the other device.
    ///
    ///Only relevant for multi-GPU device groups. On single-GPU systems
    ///this is not needed.
    ///
    ///Use the returned flags to decide how to share resources across
    ///devices, for example, whether a texture on GPU 0 can be sampled
    ///directly by GPU 1, or whether it must be copied.
    pub unsafe fn get_device_group_peer_memory_features(
        &self,
        heap_index: u32,
        local_device_index: u32,
        remote_device_index: u32,
    ) -> PeerMemoryFeatureFlags {
        let fp = self
            .commands()
            .get_device_group_peer_memory_features
            .expect("vkGetDeviceGroupPeerMemoryFeatures not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe {
            fp(
                self.handle(),
                heap_index,
                local_device_index,
                remote_device_index,
                &mut out,
            )
        };
        out
    }
    ///Wraps [`vkBindBufferMemory2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkBindBufferMemory2.html).
    /**
    Provided by **VK_BASE_VERSION_1_1**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Binds memory to one or more buffers in a single call. This is the
    ///Vulkan 1.1 batch version of `bind_buffer_memory`.
    ///
    ///Each `BindBufferMemoryInfo` specifies a buffer, memory object, and
    ///offset, the same parameters as `bind_buffer_memory`, but batched.
    ///
    ///Use `BindBufferMemoryDeviceGroupInfo` in the pNext chain to bind
    ///memory for specific devices in a device group (multi-GPU). For
    ///single-GPU usage, `bind_buffer_memory` and `bind_buffer_memory2`
    ///are equivalent.
    pub unsafe fn bind_buffer_memory2(
        &self,
        p_bind_infos: &[BindBufferMemoryInfo],
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .bind_buffer_memory2
            .expect("vkBindBufferMemory2 not loaded");
        check(unsafe {
            fp(
                self.handle(),
                p_bind_infos.len() as u32,
                p_bind_infos.as_ptr(),
            )
        })
    }
    ///Wraps [`vkBindImageMemory2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkBindImageMemory2.html).
    /**
    Provided by **VK_BASE_VERSION_1_1**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Binds memory to one or more images in a single call. This is the
    ///Vulkan 1.1 batch version of `bind_image_memory`.
    ///
    ///Also required when binding memory to images created with
    ///disjoint multi-planar formats, each plane is bound separately via
    ///`BindImagePlaneMemoryInfo` in the pNext chain.
    ///
    ///For device groups (multi-GPU), chain
    ///`BindImageMemoryDeviceGroupInfo` to assign memory per device and
    ///specify split-instance bind regions.
    pub unsafe fn bind_image_memory2(&self, p_bind_infos: &[BindImageMemoryInfo]) -> VkResult<()> {
        let fp = self
            .commands()
            .bind_image_memory2
            .expect("vkBindImageMemory2 not loaded");
        check(unsafe {
            fp(
                self.handle(),
                p_bind_infos.len() as u32,
                p_bind_infos.as_ptr(),
            )
        })
    }
    ///Wraps [`vkCmdSetDeviceMask`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetDeviceMask.html).
    /**
    Provided by **VK_BASE_VERSION_1_1**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Sets the device mask for subsequent commands in a command buffer
    ///when using device groups (multi-GPU with
    ///`VK_KHR_device_group` / Vulkan 1.1).
    ///
    ///The device mask is a bitmask where bit *i* indicates that subsequent
    ///commands execute on physical device *i* in the device group.
    ///
    ///For single-GPU systems (the common case), the device mask is always
    ///`0x1` and this command is not needed.
    ///
    ///Must only be called outside a render pass, or inside a render pass
    ///that was begun with `DEVICE_GROUP_BEGIN_INFO` and has the
    ///`DEVICE_GROUP` flag set.
    pub unsafe fn cmd_set_device_mask(&self, command_buffer: CommandBuffer, device_mask: u32) {
        let fp = self
            .commands()
            .cmd_set_device_mask
            .expect("vkCmdSetDeviceMask not loaded");
        unsafe { fp(command_buffer, device_mask) };
    }
    ///Wraps [`vkGetDeviceGroupPresentCapabilitiesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDeviceGroupPresentCapabilitiesKHR.html).
    /**
    Provided by **VK_KHR_swapchain**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the present capabilities of a device group, which physical
    ///devices can present to which surfaces, and what presentation modes
    ///are supported.
    ///
    ///Only relevant for multi-GPU device groups. On single-GPU systems,
    ///only `DEVICE_GROUP_PRESENT_MODE_LOCAL` is supported (each device
    ///presents its own images).
    pub unsafe fn get_device_group_present_capabilities_khr(
        &self,
        p_device_group_present_capabilities: *mut DeviceGroupPresentCapabilitiesKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_device_group_present_capabilities_khr
            .expect("vkGetDeviceGroupPresentCapabilitiesKHR not loaded");
        check(unsafe { fp(self.handle(), p_device_group_present_capabilities) })
    }
    ///Wraps [`vkGetDeviceGroupSurfacePresentModesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDeviceGroupSurfacePresentModesKHR.html).
    /**
    Provided by **VK_KHR_swapchain**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `surface` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Queries which device group present modes a surface supports. The
    ///returned bitmask indicates whether `LOCAL`, `REMOTE`, `SUM`, or
    ///`LOCAL_MULTI_DEVICE` modes are available.
    ///
    ///Only relevant for multi-GPU device groups. On single-GPU systems,
    ///only `LOCAL` is supported.
    pub unsafe fn get_device_group_surface_present_modes_khr(
        &self,
        surface: SurfaceKHR,
    ) -> VkResult<DeviceGroupPresentModeFlagsKHR> {
        let fp = self
            .commands()
            .get_device_group_surface_present_modes_khr
            .expect("vkGetDeviceGroupSurfacePresentModesKHR not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), surface, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkAcquireNextImage2KHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkAcquireNextImage2KHR.html).
    /**
    Provided by **VK_KHR_swapchain**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_DEVICE_LOST`
    ///- `VK_ERROR_OUT_OF_DATE_KHR`
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Extended version of `acquire_next_image_khr` that takes an
    ///`AcquireNextImageInfoKHR` struct with pNext support.
    ///
    ///The key addition is `device_mask` for device groups (multi-GPU),
    ///specifying which physical devices the acquired image will be used
    ///on.
    ///
    ///For single-GPU usage, this is functionally identical to
    ///`acquire_next_image_khr`.
    pub unsafe fn acquire_next_image2_khr(
        &self,
        p_acquire_info: &AcquireNextImageInfoKHR,
    ) -> VkResult<u32> {
        let fp = self
            .commands()
            .acquire_next_image2_khr
            .expect("vkAcquireNextImage2KHR not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_acquire_info, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkCmdDispatchBase`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDispatchBase.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_1**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dispatches a compute shader with a non-zero base workgroup offset.
    ///The shader's `gl_WorkGroupID` starts at (`base_group_x`,
    ///`base_group_y`, `base_group_z`) instead of (0, 0, 0).
    ///
    ///This is useful for splitting a large dispatch across multiple
    ///submissions or for device groups where different physical devices
    ///handle different regions of the workgroup space.
    ///
    ///`gl_NumWorkGroups` reflects the `group_count` parameters, not the
    ///total. The shader sees workgroup IDs in the range
    ///[`base`, `base + count`).
    ///
    ///For single-GPU, zero-base dispatches, use `cmd_dispatch` instead.
    pub unsafe fn cmd_dispatch_base(
        &self,
        command_buffer: CommandBuffer,
        base_group_x: u32,
        base_group_y: u32,
        base_group_z: u32,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) {
        let fp = self
            .commands()
            .cmd_dispatch_base
            .expect("vkCmdDispatchBase not loaded");
        unsafe {
            fp(
                command_buffer,
                base_group_x,
                base_group_y,
                base_group_z,
                group_count_x,
                group_count_y,
                group_count_z,
            )
        };
    }
    ///Wraps [`vkCreateDescriptorUpdateTemplate`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateDescriptorUpdateTemplate.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_1**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a template that describes how to update a descriptor set
    ///from a compact block of host memory. Instead of building an array
    ///of `WriteDescriptorSet` structs, you define the layout once in the
    ///template and then call `update_descriptor_set_with_template` with
    ///a raw pointer to your data.
    ///
    ///**Benefits**:
    ///
    ///- Reduces CPU overhead for frequent descriptor updates.
    ///- Avoids allocating `WriteDescriptorSet` arrays every frame.
    ///- Pairs well with `cmd_push_descriptor_set_with_template` for
    ///  push descriptors.
    ///
    ///Each entry in the template maps an offset in your host data block to
    ///a descriptor binding, array element, and type. The driver compiles
    ///this into an optimised update path.
    ///
    ///Templates are immutable after creation and can be reused across
    ///frames.
    pub unsafe fn create_descriptor_update_template(
        &self,
        p_create_info: &DescriptorUpdateTemplateCreateInfo,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<DescriptorUpdateTemplate> {
        let fp = self
            .commands()
            .create_descriptor_update_template
            .expect("vkCreateDescriptorUpdateTemplate not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyDescriptorUpdateTemplate`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyDescriptorUpdateTemplate.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_1**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `descriptorUpdateTemplate` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a descriptor update template. The template must not be in
    ///use by any pending `update_descriptor_set_with_template` or
    ///`cmd_push_descriptor_set_with_template` call.
    pub unsafe fn destroy_descriptor_update_template(
        &self,
        descriptor_update_template: DescriptorUpdateTemplate,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_descriptor_update_template
            .expect("vkDestroyDescriptorUpdateTemplate not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), descriptor_update_template, alloc_ptr) };
    }
    ///Wraps [`vkUpdateDescriptorSetWithTemplate`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkUpdateDescriptorSetWithTemplate.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_1**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `descriptorSet` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Updates a descriptor set using a descriptor update template and a
    ///raw pointer to a block of host data. The template defines the mapping
    ///from the data block to descriptor bindings.
    ///
    ///This is faster than `update_descriptor_sets` for repeated updates
    ///with the same layout, the driver has pre-compiled the update path.
    ///
    ///The `data` pointer must point to a block of memory laid out according
    ///to the template's entry offsets and strides. The data is consumed
    ///immediately; the pointer does not need to remain valid after the
    ///call returns.
    pub unsafe fn update_descriptor_set_with_template(
        &self,
        descriptor_set: DescriptorSet,
        descriptor_update_template: DescriptorUpdateTemplate,
        p_data: *const core::ffi::c_void,
    ) {
        let fp = self
            .commands()
            .update_descriptor_set_with_template
            .expect("vkUpdateDescriptorSetWithTemplate not loaded");
        unsafe {
            fp(
                self.handle(),
                descriptor_set,
                descriptor_update_template,
                p_data,
            )
        };
    }
    ///Wraps [`vkCmdPushDescriptorSetWithTemplate`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdPushDescriptorSetWithTemplate.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_4**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Combines push descriptors with descriptor update templates for
    ///maximum efficiency. Updates are pushed directly into the command
    ///buffer using the compact host data format defined by the template.
    ///
    ///This is the fastest path for per-draw descriptor updates: no pool
    ///allocation, no `WriteDescriptorSet` array construction, and the
    ///driver has a pre-compiled update path from the template.
    ///
    ///The template must have been created with
    ///`DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS`.
    ///
    ///Core in Vulkan 1.4. Previously via `VK_KHR_push_descriptor` +
    ///`VK_KHR_descriptor_update_template`.
    pub unsafe fn cmd_push_descriptor_set_with_template(
        &self,
        command_buffer: CommandBuffer,
        descriptor_update_template: DescriptorUpdateTemplate,
        layout: PipelineLayout,
        set: u32,
        p_data: *const core::ffi::c_void,
    ) {
        let fp = self
            .commands()
            .cmd_push_descriptor_set_with_template
            .expect("vkCmdPushDescriptorSetWithTemplate not loaded");
        unsafe {
            fp(
                command_buffer,
                descriptor_update_template,
                layout,
                set,
                p_data,
            )
        };
    }
    ///Wraps [`vkSetHdrMetadataEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkSetHdrMetadataEXT.html).
    /**
    Provided by **VK_EXT_hdr_metadata**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Sets HDR metadata (mastering display color volume, content light
    ///levels) for one or more swapchains. The compositor uses this to
    ///tone-map content appropriately for the connected display.
    ///
    ///Call whenever the content characteristics change (e.g., switching
    ///between SDR UI and HDR scene rendering).
    ///
    ///Requires `VK_EXT_hdr_metadata`.
    pub unsafe fn set_hdr_metadata_ext(
        &self,
        p_swapchains: &[SwapchainKHR],
        p_metadata: &HdrMetadataEXT,
    ) {
        let fp = self
            .commands()
            .set_hdr_metadata_ext
            .expect("vkSetHdrMetadataEXT not loaded");
        unsafe {
            fp(
                self.handle(),
                p_swapchains.len() as u32,
                p_swapchains.as_ptr(),
                p_metadata,
            )
        };
    }
    ///Wraps [`vkGetSwapchainStatusKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetSwapchainStatusKHR.html).
    /**
    Provided by **VK_KHR_shared_presentable_image**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_DEVICE_LOST`
    ///- `VK_ERROR_OUT_OF_DATE_KHR`
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `swapchain` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Queries the current status of a shared presentable swapchain
    ///(created with `PRESENT_MODE_SHARED_DEMAND_REFRESH` or
    ///`PRESENT_MODE_SHARED_CONTINUOUS_REFRESH`).
    ///
    ///Returns `VK_SUCCESS` if the swapchain is usable, or
    ///`VK_SUBOPTIMAL_KHR` / `VK_ERROR_OUT_OF_DATE_KHR` / surface-lost
    ///errors if the swapchain needs recreation.
    ///
    ///Only relevant for shared presentable images
    ///(`VK_KHR_shared_presentable_image`). For regular swapchains, status
    ///is communicated through `acquire_next_image_khr` and
    ///`queue_present_khr` return values.
    pub unsafe fn get_swapchain_status_khr(&self, swapchain: SwapchainKHR) -> VkResult<()> {
        let fp = self
            .commands()
            .get_swapchain_status_khr
            .expect("vkGetSwapchainStatusKHR not loaded");
        check(unsafe { fp(self.handle(), swapchain) })
    }
    ///Wraps [`vkGetRefreshCycleDurationGOOGLE`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetRefreshCycleDurationGOOGLE.html).
    /**
    Provided by **VK_GOOGLE_display_timing**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_DEVICE_LOST`
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `swapchain` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Queries the duration of a single refresh cycle (vsync interval)
    ///for the display associated with the given swapchain. Returns
    ///the period in nanoseconds. Essential for accurate frame pacing.
    ///
    ///Requires `VK_GOOGLE_display_timing`.
    pub unsafe fn get_refresh_cycle_duration_google(
        &self,
        swapchain: SwapchainKHR,
    ) -> VkResult<RefreshCycleDurationGOOGLE> {
        let fp = self
            .commands()
            .get_refresh_cycle_duration_google
            .expect("vkGetRefreshCycleDurationGOOGLE not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), swapchain, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetPastPresentationTimingGOOGLE`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPastPresentationTimingGOOGLE.html).
    /**
    Provided by **VK_GOOGLE_display_timing**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_DEVICE_LOST`
    ///- `VK_ERROR_OUT_OF_DATE_KHR`
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `swapchain` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Returns timing data for past presentations on a swapchain,
    ///including actual present time, earliest possible present time,
    ///and finish time. Uses the two-call idiom (call once to get the
    ///count, again to fill the buffer). Useful for frame pacing and
    ///latency analysis.
    ///
    ///Requires `VK_GOOGLE_display_timing`.
    pub unsafe fn get_past_presentation_timing_google(
        &self,
        swapchain: SwapchainKHR,
    ) -> VkResult<Vec<PastPresentationTimingGOOGLE>> {
        let fp = self
            .commands()
            .get_past_presentation_timing_google
            .expect("vkGetPastPresentationTimingGOOGLE not loaded");
        enumerate_two_call(|count, data| unsafe { fp(self.handle(), swapchain, count, data) })
    }
    ///Wraps [`vkCmdSetViewportWScalingNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetViewportWScalingNV.html).
    /**
    Provided by **VK_NV_clip_space_w_scaling**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Sets per-viewport W scaling factors for lens-matched shading.
    ///The W scaling modifies the clip-space W coordinate to account
    ///for lens distortion in VR headsets, enabling more efficient
    ///shading by varying pixel density across the viewport.
    ///
    ///Requires `VK_NV_clip_space_w_scaling`.
    pub unsafe fn cmd_set_viewport_w_scaling_nv(
        &self,
        command_buffer: CommandBuffer,
        first_viewport: u32,
        p_viewport_w_scalings: &[ViewportWScalingNV],
    ) {
        let fp = self
            .commands()
            .cmd_set_viewport_w_scaling_nv
            .expect("vkCmdSetViewportWScalingNV not loaded");
        unsafe {
            fp(
                command_buffer,
                first_viewport,
                p_viewport_w_scalings.len() as u32,
                p_viewport_w_scalings.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdSetDiscardRectangleEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetDiscardRectangleEXT.html).
    /**
    Provided by **VK_EXT_discard_rectangles**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets the discard rectangles for the current command
    ///buffer. Fragments inside (or outside, depending on mode) these
    ///rectangles are discarded before the fragment shader runs.
    ///
    ///Useful for multi-view or split-screen rendering to cheaply cull
    ///fragments that belong to a different viewport.
    ///
    ///Requires `VK_EXT_discard_rectangles`.
    pub unsafe fn cmd_set_discard_rectangle_ext(
        &self,
        command_buffer: CommandBuffer,
        first_discard_rectangle: u32,
        p_discard_rectangles: &[Rect2D],
    ) {
        let fp = self
            .commands()
            .cmd_set_discard_rectangle_ext
            .expect("vkCmdSetDiscardRectangleEXT not loaded");
        unsafe {
            fp(
                command_buffer,
                first_discard_rectangle,
                p_discard_rectangles.len() as u32,
                p_discard_rectangles.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdSetDiscardRectangleEnableEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetDiscardRectangleEnableEXT.html).
    /**
    Provided by **VK_EXT_discard_rectangles**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically enables or disables discard rectangles for subsequent
    ///draw commands. When disabled, no fragments are discarded regardless
    ///of the configured rectangles.
    ///
    ///Requires `VK_EXT_discard_rectangles`.
    pub unsafe fn cmd_set_discard_rectangle_enable_ext(
        &self,
        command_buffer: CommandBuffer,
        discard_rectangle_enable: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_discard_rectangle_enable_ext
            .expect("vkCmdSetDiscardRectangleEnableEXT not loaded");
        unsafe { fp(command_buffer, discard_rectangle_enable) };
    }
    ///Wraps [`vkCmdSetDiscardRectangleModeEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetDiscardRectangleModeEXT.html).
    /**
    Provided by **VK_EXT_discard_rectangles**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets whether fragments inside or outside the discard
    ///rectangles are discarded. `INCLUSIVE` discards fragments inside
    ///the rectangles; `EXCLUSIVE` discards fragments outside.
    ///
    ///Requires `VK_EXT_discard_rectangles`.
    pub unsafe fn cmd_set_discard_rectangle_mode_ext(
        &self,
        command_buffer: CommandBuffer,
        discard_rectangle_mode: DiscardRectangleModeEXT,
    ) {
        let fp = self
            .commands()
            .cmd_set_discard_rectangle_mode_ext
            .expect("vkCmdSetDiscardRectangleModeEXT not loaded");
        unsafe { fp(command_buffer, discard_rectangle_mode) };
    }
    ///Wraps [`vkCmdSetSampleLocationsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetSampleLocationsEXT.html).
    /**
    Provided by **VK_EXT_sample_locations**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Sets custom sample locations for multisampled rasterization.
    ///`SampleLocationsInfoEXT` specifies the grid size, sample count,
    ///and per-sample (x, y) positions within each pixel.
    ///
    ///Custom sample locations enable techniques like temporal AA
    ///jittering and programmable MSAA patterns.
    ///
    ///Must be called when custom sample locations are enabled (via
    ///`cmd_set_sample_locations_enable_ext` or pipeline state).
    ///
    ///Requires `VK_EXT_sample_locations`.
    pub unsafe fn cmd_set_sample_locations_ext(
        &self,
        command_buffer: CommandBuffer,
        p_sample_locations_info: &SampleLocationsInfoEXT,
    ) {
        let fp = self
            .commands()
            .cmd_set_sample_locations_ext
            .expect("vkCmdSetSampleLocationsEXT not loaded");
        unsafe { fp(command_buffer, p_sample_locations_info) };
    }
    ///Wraps [`vkGetBufferMemoryRequirements2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetBufferMemoryRequirements2.html).
    /**
    Provided by **VK_BASE_VERSION_1_1**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.1 version of `get_buffer_memory_requirements` that supports
    ///extensible output structs via pNext.
    ///
    ///Chain `MemoryDedicatedRequirements` to query whether the driver
    ///prefers or requires a dedicated allocation for this buffer. If
    ///`prefers_dedicated_allocation` is true, allocating a dedicated
    ///`DeviceMemory` for this buffer may improve performance.
    ///
    ///The base `MemoryRequirements` (size, alignment, memory type bits) is
    ///identical to what `get_buffer_memory_requirements` returns.
    pub unsafe fn get_buffer_memory_requirements2(
        &self,
        p_info: &BufferMemoryRequirementsInfo2,
        p_memory_requirements: *mut MemoryRequirements2,
    ) {
        let fp = self
            .commands()
            .get_buffer_memory_requirements2
            .expect("vkGetBufferMemoryRequirements2 not loaded");
        unsafe { fp(self.handle(), p_info, p_memory_requirements) };
    }
    ///Wraps [`vkGetImageMemoryRequirements2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetImageMemoryRequirements2.html).
    /**
    Provided by **VK_BASE_VERSION_1_1**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.1 version of `get_image_memory_requirements` that supports
    ///extensible output structs via pNext.
    ///
    ///Chain `MemoryDedicatedRequirements` to query whether the driver
    ///prefers or requires a dedicated allocation for this image. Dedicated
    ///allocations are common for large render targets and swapchain-sized
    ///images, some drivers require them.
    ///
    ///For multi-planar images, chain `ImagePlaneMemoryRequirementsInfo` in
    ///the input to query requirements for a specific plane.
    ///
    ///The base `MemoryRequirements` is identical to what
    ///`get_image_memory_requirements` returns.
    pub unsafe fn get_image_memory_requirements2(
        &self,
        p_info: &ImageMemoryRequirementsInfo2,
        p_memory_requirements: *mut MemoryRequirements2,
    ) {
        let fp = self
            .commands()
            .get_image_memory_requirements2
            .expect("vkGetImageMemoryRequirements2 not loaded");
        unsafe { fp(self.handle(), p_info, p_memory_requirements) };
    }
    ///Wraps [`vkGetImageSparseMemoryRequirements2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetImageSparseMemoryRequirements2.html).
    /**
    Provided by **VK_BASE_VERSION_1_1**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.1 version of `get_image_sparse_memory_requirements` that
    ///supports extensible output structs via pNext. Returns the sparse
    ///memory requirements for an image created with sparse flags.
    ///
    ///For non-sparse images, returns an empty list. Only relevant if you
    ///are using sparse resources.
    pub unsafe fn get_image_sparse_memory_requirements2(
        &self,
        p_info: &ImageSparseMemoryRequirementsInfo2,
    ) -> Vec<SparseImageMemoryRequirements2> {
        let fp = self
            .commands()
            .get_image_sparse_memory_requirements2
            .expect("vkGetImageSparseMemoryRequirements2 not loaded");
        fill_two_call(|count, data| unsafe { fp(self.handle(), p_info, count, data) })
    }
    ///Wraps [`vkGetDeviceBufferMemoryRequirements`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDeviceBufferMemoryRequirements.html).
    /**
    Provided by **VK_BASE_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.3 command that queries memory requirements for a buffer
    ///**without creating it first**. Pass a `DeviceBufferMemoryRequirements`
    ///containing the hypothetical `BufferCreateInfo`.
    ///
    ///This lets you pre-plan memory allocations before creating any
    ///objects, useful for memory allocation strategies that need to know
    ///sizes and alignments up front.
    ///
    ///The returned requirements are identical to what
    ///`get_buffer_memory_requirements2` would return for an actual buffer
    ///created with the same parameters.
    pub unsafe fn get_device_buffer_memory_requirements(
        &self,
        p_info: &DeviceBufferMemoryRequirements,
        p_memory_requirements: *mut MemoryRequirements2,
    ) {
        let fp = self
            .commands()
            .get_device_buffer_memory_requirements
            .expect("vkGetDeviceBufferMemoryRequirements not loaded");
        unsafe { fp(self.handle(), p_info, p_memory_requirements) };
    }
    ///Wraps [`vkGetDeviceImageMemoryRequirements`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDeviceImageMemoryRequirements.html).
    /**
    Provided by **VK_BASE_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.3 command that queries memory requirements for an image
    ///**without creating it first**. Pass a `DeviceImageMemoryRequirements`
    ///containing the hypothetical `ImageCreateInfo`.
    ///
    ///Useful for pre-planning memory allocations or estimating VRAM usage
    ///before committing to image creation.
    ///
    ///For multi-planar images, set `plane_aspect` to query requirements
    ///for a specific plane.
    ///
    ///The returned requirements are identical to what
    ///`get_image_memory_requirements2` would return for an actual image
    ///created with the same parameters.
    pub unsafe fn get_device_image_memory_requirements(
        &self,
        p_info: &DeviceImageMemoryRequirements,
        p_memory_requirements: *mut MemoryRequirements2,
    ) {
        let fp = self
            .commands()
            .get_device_image_memory_requirements
            .expect("vkGetDeviceImageMemoryRequirements not loaded");
        unsafe { fp(self.handle(), p_info, p_memory_requirements) };
    }
    ///Wraps [`vkGetDeviceImageSparseMemoryRequirements`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDeviceImageSparseMemoryRequirements.html).
    /**
    Provided by **VK_BASE_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.3 command that queries sparse memory requirements for an
    ///image **without creating it first**. The counterpart to
    ///`get_device_image_memory_requirements` for sparse images.
    ///
    ///Only relevant if you are using sparse resources with the hypothetical
    ///image creation parameters.
    pub unsafe fn get_device_image_sparse_memory_requirements(
        &self,
        p_info: &DeviceImageMemoryRequirements,
    ) -> Vec<SparseImageMemoryRequirements2> {
        let fp = self
            .commands()
            .get_device_image_sparse_memory_requirements
            .expect("vkGetDeviceImageSparseMemoryRequirements not loaded");
        fill_two_call(|count, data| unsafe { fp(self.handle(), p_info, count, data) })
    }
    ///Wraps [`vkCreateSamplerYcbcrConversion`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateSamplerYcbcrConversion.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_1**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a sampler YCBCR conversion object that describes how to
    ///convert YCBCR-encoded image data to RGB during sampling. Required
    ///for multi-planar formats like `G8_B8_R8_3PLANE_420_UNORM` commonly
    ///used in video decoding and camera capture.
    ///
    ///The conversion parameters specify:
    ///
    ///- **Format**: the multi-planar format being converted.
    ///- **YCBCR model**: `RGB_IDENTITY`, `YCBCR_IDENTITY`,
    ///  `YCBCR_709`, `YCBCR_601`, `YCBCR_2020`.
    ///- **Range**: `ITU_FULL` or `ITU_NARROW`.
    ///- **Chroma location**: where subsampled chroma samples are located
    ///  relative to luma samples.
    ///- **Chroma filter**: `NEAREST` or `LINEAR` for chroma upsampling.
    ///
    ///The conversion object is attached to a sampler via
    ///`SamplerYcbcrConversionInfo` in the pNext chain, and that sampler
    ///must be used as an immutable sampler in the descriptor set layout.
    pub unsafe fn create_sampler_ycbcr_conversion(
        &self,
        p_create_info: &SamplerYcbcrConversionCreateInfo,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<SamplerYcbcrConversion> {
        let fp = self
            .commands()
            .create_sampler_ycbcr_conversion
            .expect("vkCreateSamplerYcbcrConversion not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroySamplerYcbcrConversion`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroySamplerYcbcrConversion.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_1**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `ycbcrConversion` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a sampler YCBCR conversion object. Any sampler that was
    ///created with this conversion must already be destroyed.
    ///
    ///After destruction, any descriptor set layout that used the associated
    ///sampler as an immutable sampler remains valid but cannot be used to
    ///allocate new descriptor sets.
    pub unsafe fn destroy_sampler_ycbcr_conversion(
        &self,
        ycbcr_conversion: SamplerYcbcrConversion,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_sampler_ycbcr_conversion
            .expect("vkDestroySamplerYcbcrConversion not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), ycbcr_conversion, alloc_ptr) };
    }
    ///Wraps [`vkGetDeviceQueue2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDeviceQueue2.html).
    /**
    Provided by **VK_BASE_VERSION_1_1**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves a queue handle for a queue created with specific flags.
    ///This is the Vulkan 1.1 version of `get_device_queue` that supports
    ///`DeviceQueueInfo2` with queue creation flags.
    ///
    ///Use this instead of `get_device_queue` when you created queues with
    ///non-zero `DeviceQueueCreateFlags` (e.g. `PROTECTED` for protected
    ///content processing). For queues created without flags, both
    ///`get_device_queue` and `get_device_queue2` work.
    pub unsafe fn get_device_queue2(&self, p_queue_info: &DeviceQueueInfo2) -> Queue {
        let fp = self
            .commands()
            .get_device_queue2
            .expect("vkGetDeviceQueue2 not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(self.handle(), p_queue_info, &mut out) };
        out
    }
    ///Wraps [`vkCreateValidationCacheEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateValidationCacheEXT.html).
    /**
    Provided by **VK_EXT_validation_cache**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a validation cache that stores the results of validation
    ///layer checks. Subsequent pipeline creations with the same shaders
    ///can skip redundant validation, improving pipeline creation time.
    ///
    ///Provide previously saved cache data to warm-start the cache.
    ///Retrieve data with `get_validation_cache_data_ext`.
    ///
    ///Destroy with `destroy_validation_cache_ext`.
    ///
    ///Requires `VK_EXT_validation_cache`.
    pub unsafe fn create_validation_cache_ext(
        &self,
        p_create_info: &ValidationCacheCreateInfoEXT,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<ValidationCacheEXT> {
        let fp = self
            .commands()
            .create_validation_cache_ext
            .expect("vkCreateValidationCacheEXT not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyValidationCacheEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyValidationCacheEXT.html).
    /**
    Provided by **VK_EXT_validation_cache**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `validationCache` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a validation cache created with
    ///`create_validation_cache_ext`.
    ///
    ///Requires `VK_EXT_validation_cache`.
    pub unsafe fn destroy_validation_cache_ext(
        &self,
        validation_cache: ValidationCacheEXT,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_validation_cache_ext
            .expect("vkDestroyValidationCacheEXT not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), validation_cache, alloc_ptr) };
    }
    ///Wraps [`vkGetValidationCacheDataEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetValidationCacheDataEXT.html).
    /**
    Provided by **VK_EXT_validation_cache**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves the data from a validation cache for serialization to
    ///disk. Call once with a null buffer to query the size, then again
    ///with an appropriately sized buffer.
    ///
    ///Feed the saved data back into `create_validation_cache_ext` on
    ///the next run to avoid redundant validation.
    ///
    ///Requires `VK_EXT_validation_cache`.
    pub unsafe fn get_validation_cache_data_ext(
        &self,
        validation_cache: ValidationCacheEXT,
        p_data: *mut core::ffi::c_void,
    ) -> VkResult<usize> {
        let fp = self
            .commands()
            .get_validation_cache_data_ext
            .expect("vkGetValidationCacheDataEXT not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), validation_cache, &mut out, p_data) })?;
        Ok(out)
    }
    ///Wraps [`vkMergeValidationCachesEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkMergeValidationCachesEXT.html).
    /**
    Provided by **VK_EXT_validation_cache**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `dstCache` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Merges one or more source validation caches into a destination
    ///cache. Useful for combining caches from parallel pipeline creation
    ///threads.
    ///
    ///Requires `VK_EXT_validation_cache`.
    pub unsafe fn merge_validation_caches_ext(
        &self,
        dst_cache: ValidationCacheEXT,
        p_src_caches: &[ValidationCacheEXT],
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .merge_validation_caches_ext
            .expect("vkMergeValidationCachesEXT not loaded");
        check(unsafe {
            fp(
                self.handle(),
                dst_cache,
                p_src_caches.len() as u32,
                p_src_caches.as_ptr(),
            )
        })
    }
    ///Wraps [`vkGetDescriptorSetLayoutSupport`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDescriptorSetLayoutSupport.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_1**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries whether a descriptor set layout with the given bindings can
    ///be created on this device, and returns information about the
    ///variable descriptor count limit if applicable.
    ///
    ///Use this before creating layouts with very large descriptor counts
    ///or update-after-bind bindings to verify they are within device
    ///limits. The call is lightweight and does not allocate anything.
    ///
    ///Chain `DescriptorSetVariableDescriptorCountLayoutSupport` in the
    ///output to query the maximum variable descriptor count for layouts
    ///that use `DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT`.
    pub unsafe fn get_descriptor_set_layout_support(
        &self,
        p_create_info: &DescriptorSetLayoutCreateInfo,
        p_support: *mut DescriptorSetLayoutSupport,
    ) {
        let fp = self
            .commands()
            .get_descriptor_set_layout_support
            .expect("vkGetDescriptorSetLayoutSupport not loaded");
        unsafe { fp(self.handle(), p_create_info, p_support) };
    }
    ///Wraps [`vkGetSwapchainGrallocUsageANDROID`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetSwapchainGrallocUsageANDROID.html).
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the Android gralloc usage flags needed for swapchain
    ///images with the given format and Vulkan image usage. Used
    ///internally by the Android WSI implementation. Android only.
    ///
    ///Requires `VK_ANDROID_native_buffer`.
    pub unsafe fn get_swapchain_gralloc_usage_android(
        &self,
        format: Format,
        image_usage: ImageUsageFlags,
    ) -> VkResult<core::ffi::c_int> {
        let fp = self
            .commands()
            .get_swapchain_gralloc_usage_android
            .expect("vkGetSwapchainGrallocUsageANDROID not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), format, image_usage, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetSwapchainGrallocUsage2ANDROID`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetSwapchainGrallocUsage2ANDROID.html).
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Extended version of `get_swapchain_gralloc_usage_android` that
    ///additionally accepts swapchain-specific image usage flags and
    ///returns both producer and consumer gralloc usage. Android only.
    ///
    ///Requires `VK_ANDROID_native_buffer`.
    pub unsafe fn get_swapchain_gralloc_usage2_android(
        &self,
        format: Format,
        image_usage: ImageUsageFlags,
        swapchain_image_usage: SwapchainImageUsageFlagsANDROID,
        gralloc_consumer_usage: *mut u64,
    ) -> VkResult<u64> {
        let fp = self
            .commands()
            .get_swapchain_gralloc_usage2_android
            .expect("vkGetSwapchainGrallocUsage2ANDROID not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe {
            fp(
                self.handle(),
                format,
                image_usage,
                swapchain_image_usage,
                gralloc_consumer_usage,
                &mut out,
            )
        })?;
        Ok(out)
    }
    ///Wraps [`vkAcquireImageANDROID`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkAcquireImageANDROID.html).
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Acquires ownership of a swapchain image on Android. Takes a
    ///native fence FD for synchronisation and can signal a Vulkan
    ///semaphore or fence on completion. Android only.
    ///
    ///Requires `VK_ANDROID_native_buffer`.
    pub unsafe fn acquire_image_android(
        &self,
        image: Image,
        native_fence_fd: core::ffi::c_int,
        semaphore: Semaphore,
        fence: Fence,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .acquire_image_android
            .expect("vkAcquireImageANDROID not loaded");
        check(unsafe { fp(self.handle(), image, native_fence_fd, semaphore, fence) })
    }
    ///Wraps [`vkQueueSignalReleaseImageANDROID`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkQueueSignalReleaseImageANDROID.html).
    ///
    ///# Safety
    ///- `queue` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Releases a swapchain image back to the Android compositor after
    ///rendering. Waits on the given semaphores and returns a native
    ///fence FD for external synchronisation. Android only.
    ///
    ///Requires `VK_ANDROID_native_buffer`.
    pub unsafe fn queue_signal_release_image_android(
        &self,
        queue: Queue,
        p_wait_semaphores: &[Semaphore],
        image: Image,
    ) -> VkResult<core::ffi::c_int> {
        let fp = self
            .commands()
            .queue_signal_release_image_android
            .expect("vkQueueSignalReleaseImageANDROID not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe {
            fp(
                queue,
                p_wait_semaphores.len() as u32,
                p_wait_semaphores.as_ptr(),
                image,
                &mut out,
            )
        })?;
        Ok(out)
    }
    ///Wraps [`vkGetShaderInfoAMD`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetShaderInfoAMD.html).
    /**
    Provided by **VK_AMD_shader_info**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_FEATURE_NOT_PRESENT`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries AMD-specific shader information such as compiled binary
    ///statistics, disassembly, or resource usage. Call once with a null
    ///buffer to query the size, then again with an appropriately sized
    ///buffer.
    ///
    ///Requires `VK_AMD_shader_info`.
    pub unsafe fn get_shader_info_amd(
        &self,
        pipeline: Pipeline,
        shader_stage: ShaderStageFlagBits,
        info_type: ShaderInfoTypeAMD,
        p_info: *mut core::ffi::c_void,
    ) -> VkResult<usize> {
        let fp = self
            .commands()
            .get_shader_info_amd
            .expect("vkGetShaderInfoAMD not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe {
            fp(
                self.handle(),
                pipeline,
                shader_stage,
                info_type,
                &mut out,
                p_info,
            )
        })?;
        Ok(out)
    }
    ///Wraps [`vkSetLocalDimmingAMD`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkSetLocalDimmingAMD.html).
    /**
    Provided by **VK_AMD_display_native_hdr**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Enables or disables local dimming on an AMD display with native
    ///HDR support. Local dimming improves HDR contrast by adjusting
    ///backlight zones independently.
    ///
    ///Requires `VK_AMD_display_native_hdr`.
    pub unsafe fn set_local_dimming_amd(
        &self,
        swap_chain: SwapchainKHR,
        local_dimming_enable: u32,
    ) {
        let fp = self
            .commands()
            .set_local_dimming_amd
            .expect("vkSetLocalDimmingAMD not loaded");
        unsafe { fp(self.handle(), swap_chain, local_dimming_enable) };
    }
    ///Wraps [`vkGetCalibratedTimestampsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetCalibratedTimestampsKHR.html).
    /**
    Provided by **VK_KHR_calibrated_timestamps**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Samples multiple time domains simultaneously and returns
    ///calibrated timestamps. This allows correlating GPU timestamps
    ///(from `cmd_write_timestamp2`) with CPU time.
    ///
    ///Each `CalibratedTimestampInfoKHR` specifies a time domain
    ///(e.g., `DEVICE`, `CLOCK_MONOTONIC`, `CLOCK_MONOTONIC_RAW`,
    ///`QUERY_PERFORMANCE_COUNTER`). All requested timestamps are
    ///sampled as close together as possible.
    ///
    ///The returned `max_deviation` (in nanoseconds) bounds how far
    ///apart the samples could be, smaller is better. If deviation
    ///is too large, retry the call.
    ///
    ///Query available time domains first with
    ///`get_physical_device_calibrateable_time_domains_khr`.
    pub unsafe fn get_calibrated_timestamps_khr(
        &self,
        p_timestamp_infos: &[CalibratedTimestampInfoKHR],
        p_timestamps: *mut u64,
    ) -> VkResult<u64> {
        let fp = self
            .commands()
            .get_calibrated_timestamps_khr
            .expect("vkGetCalibratedTimestampsKHR not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe {
            fp(
                self.handle(),
                p_timestamp_infos.len() as u32,
                p_timestamp_infos.as_ptr(),
                p_timestamps,
                &mut out,
            )
        })?;
        Ok(out)
    }
    ///Wraps [`vkSetDebugUtilsObjectNameEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkSetDebugUtilsObjectNameEXT.html).
    /**
    Provided by **VK_EXT_debug_utils**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `pNameInfo` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Assigns a human-readable name to any Vulkan object. The name
    ///appears in validation layer messages, GPU debuggers (RenderDoc,
    ///Nsight), and crash reports.
    ///
    ///`DebugUtilsObjectNameInfoEXT` specifies the object type, handle,
    ///and a null-terminated UTF-8 name string.
    ///
    ///Set the name to null to remove a previously assigned name.
    ///
    ///This is the most impactful debugging tool in Vulkan, name
    ///every object you create.
    ///
    ///Requires `VK_EXT_debug_utils`.
    pub unsafe fn set_debug_utils_object_name_ext(
        &self,
        p_name_info: &DebugUtilsObjectNameInfoEXT,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .set_debug_utils_object_name_ext
            .expect("vkSetDebugUtilsObjectNameEXT not loaded");
        check(unsafe { fp(self.handle(), p_name_info) })
    }
    ///Wraps [`vkSetDebugUtilsObjectTagEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkSetDebugUtilsObjectTagEXT.html).
    /**
    Provided by **VK_EXT_debug_utils**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Attaches arbitrary binary data to a Vulkan object. Unlike
    ///`set_debug_utils_object_name_ext` (which sets a string),
    ///tags carry opaque byte data identified by a `tag_name` (u64).
    ///
    ///Tags are consumed by debugging tools and layers that understand
    ///the specific `tag_name` value. Most applications only need
    ///`set_debug_utils_object_name_ext`, use tags for tool-specific
    ///metadata.
    ///
    ///Requires `VK_EXT_debug_utils`.
    pub unsafe fn set_debug_utils_object_tag_ext(
        &self,
        p_tag_info: &DebugUtilsObjectTagInfoEXT,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .set_debug_utils_object_tag_ext
            .expect("vkSetDebugUtilsObjectTagEXT not loaded");
        check(unsafe { fp(self.handle(), p_tag_info) })
    }
    ///Wraps [`vkQueueBeginDebugUtilsLabelEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkQueueBeginDebugUtilsLabelEXT.html).
    /**
    Provided by **VK_EXT_debug_utils**.*/
    ///
    ///# Safety
    ///- `queue` (self) must be valid and not destroyed.
    ///- `queue` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Opens a debug label region on a queue. All submissions between
    ///this call and the matching `queue_end_debug_utils_label_ext` are
    ///grouped under the label in GPU debuggers.
    ///
    ///Unlike `cmd_begin_debug_utils_label_ext` (which operates inside
    ///a command buffer), this groups entire queue submissions.
    ///
    ///Requires `VK_EXT_debug_utils`.
    pub unsafe fn queue_begin_debug_utils_label_ext(
        &self,
        queue: Queue,
        p_label_info: &DebugUtilsLabelEXT,
    ) {
        let fp = self
            .commands()
            .queue_begin_debug_utils_label_ext
            .expect("vkQueueBeginDebugUtilsLabelEXT not loaded");
        unsafe { fp(queue, p_label_info) };
    }
    ///Wraps [`vkQueueEndDebugUtilsLabelEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkQueueEndDebugUtilsLabelEXT.html).
    /**
    Provided by **VK_EXT_debug_utils**.*/
    ///
    ///# Safety
    ///- `queue` (self) must be valid and not destroyed.
    ///- `queue` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Closes the most recently opened debug label region on the queue.
    ///Must be paired with a prior `queue_begin_debug_utils_label_ext`
    ///on the same queue.
    ///
    ///Requires `VK_EXT_debug_utils`.
    pub unsafe fn queue_end_debug_utils_label_ext(&self, queue: Queue) {
        let fp = self
            .commands()
            .queue_end_debug_utils_label_ext
            .expect("vkQueueEndDebugUtilsLabelEXT not loaded");
        unsafe { fp(queue) };
    }
    ///Wraps [`vkQueueInsertDebugUtilsLabelEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkQueueInsertDebugUtilsLabelEXT.html).
    /**
    Provided by **VK_EXT_debug_utils**.*/
    ///
    ///# Safety
    ///- `queue` (self) must be valid and not destroyed.
    ///- `queue` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Inserts a single-point debug label on a queue. Marks a specific
    ///moment in the queue's submission timeline without opening a
    ///begin/end region.
    ///
    ///Requires `VK_EXT_debug_utils`.
    pub unsafe fn queue_insert_debug_utils_label_ext(
        &self,
        queue: Queue,
        p_label_info: &DebugUtilsLabelEXT,
    ) {
        let fp = self
            .commands()
            .queue_insert_debug_utils_label_ext
            .expect("vkQueueInsertDebugUtilsLabelEXT not loaded");
        unsafe { fp(queue, p_label_info) };
    }
    ///Wraps [`vkCmdBeginDebugUtilsLabelEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBeginDebugUtilsLabelEXT.html).
    /**
    Provided by **VK_EXT_debug_utils**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Opens a debug label region in a command buffer. All commands
    ///recorded between this call and the matching
    ///`cmd_end_debug_utils_label_ext` are grouped under the label in
    ///GPU debuggers (RenderDoc, Nsight).
    ///
    ///`DebugUtilsLabelEXT` specifies a name and optional RGBA color
    ///for the region.
    ///
    ///Labels can nest. Every begin must have a matching end within the
    ///same command buffer.
    ///
    ///Requires `VK_EXT_debug_utils`.
    pub unsafe fn cmd_begin_debug_utils_label_ext(
        &self,
        command_buffer: CommandBuffer,
        p_label_info: &DebugUtilsLabelEXT,
    ) {
        let fp = self
            .commands()
            .cmd_begin_debug_utils_label_ext
            .expect("vkCmdBeginDebugUtilsLabelEXT not loaded");
        unsafe { fp(command_buffer, p_label_info) };
    }
    ///Wraps [`vkCmdEndDebugUtilsLabelEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdEndDebugUtilsLabelEXT.html).
    /**
    Provided by **VK_EXT_debug_utils**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Closes the most recently opened debug label region in the command
    ///buffer. Must be paired with a prior `cmd_begin_debug_utils_label_ext`
    ///in the same command buffer.
    ///
    ///Requires `VK_EXT_debug_utils`.
    pub unsafe fn cmd_end_debug_utils_label_ext(&self, command_buffer: CommandBuffer) {
        let fp = self
            .commands()
            .cmd_end_debug_utils_label_ext
            .expect("vkCmdEndDebugUtilsLabelEXT not loaded");
        unsafe { fp(command_buffer) };
    }
    ///Wraps [`vkCmdInsertDebugUtilsLabelEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdInsertDebugUtilsLabelEXT.html).
    /**
    Provided by **VK_EXT_debug_utils**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Inserts a single-point debug label into the command buffer
    ///(as opposed to a begin/end region). Useful for marking specific
    ///events like "shadow pass complete" or "post-process start" that
    ///don't span a range of commands.
    ///
    ///Appears as a marker in GPU debuggers (RenderDoc, Nsight).
    ///
    ///Requires `VK_EXT_debug_utils`.
    pub unsafe fn cmd_insert_debug_utils_label_ext(
        &self,
        command_buffer: CommandBuffer,
        p_label_info: &DebugUtilsLabelEXT,
    ) {
        let fp = self
            .commands()
            .cmd_insert_debug_utils_label_ext
            .expect("vkCmdInsertDebugUtilsLabelEXT not loaded");
        unsafe { fp(command_buffer, p_label_info) };
    }
    ///Wraps [`vkGetMemoryHostPointerPropertiesEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetMemoryHostPointerPropertiesEXT.html).
    /**
    Provided by **VK_EXT_external_memory_host**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_INVALID_EXTERNAL_HANDLE`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries which memory types are compatible with importing a host
    ///pointer as external memory. Use this before allocating device
    ///memory backed by a host-allocated buffer to determine valid
    ///memory type bits.
    ///
    ///Requires `VK_EXT_external_memory_host`.
    pub unsafe fn get_memory_host_pointer_properties_ext(
        &self,
        handle_type: ExternalMemoryHandleTypeFlagBits,
        p_host_pointer: *const core::ffi::c_void,
        p_memory_host_pointer_properties: *mut MemoryHostPointerPropertiesEXT,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_memory_host_pointer_properties_ext
            .expect("vkGetMemoryHostPointerPropertiesEXT not loaded");
        check(unsafe {
            fp(
                self.handle(),
                handle_type,
                p_host_pointer,
                p_memory_host_pointer_properties,
            )
        })
    }
    ///Wraps [`vkCmdWriteBufferMarkerAMD`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdWriteBufferMarkerAMD.html).
    /**
    Provided by **VK_AMD_buffer_marker**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Writes a 32-bit marker value into a buffer after a specified
    ///pipeline stage completes. Useful for fine-grained GPU progress
    ///tracking and debugging GPU hangs by identifying the last
    ///completed stage.
    ///
    ///Requires `VK_AMD_buffer_marker`.
    pub unsafe fn cmd_write_buffer_marker_amd(
        &self,
        command_buffer: CommandBuffer,
        pipeline_stage: PipelineStageFlagBits,
        dst_buffer: Buffer,
        dst_offset: u64,
        marker: u32,
    ) {
        let fp = self
            .commands()
            .cmd_write_buffer_marker_amd
            .expect("vkCmdWriteBufferMarkerAMD not loaded");
        unsafe {
            fp(
                command_buffer,
                pipeline_stage,
                dst_buffer,
                dst_offset,
                marker,
            )
        };
    }
    ///Wraps [`vkCreateRenderPass2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateRenderPass2.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_2**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.2 version of `create_render_pass` that uses extensible
    ///`RenderPassCreateInfo2`, `AttachmentDescription2`,
    ///`SubpassDescription2`, and `SubpassDependency2` structs.
    ///
    ///Key additions over the 1.0 version:
    ///
    ///- **View masks**: `SubpassDescription2::view_mask` enables multiview
    ///  rendering where a single draw call renders to multiple layers
    ///  simultaneously (e.g. VR left/right eyes).
    ///- **Fragment density map**: chain
    ///  `RenderPassFragmentDensityMapCreateInfoEXT` for variable-rate
    ///  shading via density maps.
    ///- **Depth/stencil resolve**: `SubpassDescriptionDepthStencilResolve`
    ///  enables automatic depth/stencil resolve at the end of a subpass.
    ///
    ///Prefer this over `create_render_pass` when targeting Vulkan 1.2+.
    ///For Vulkan 1.3+, consider dynamic rendering (`cmd_begin_rendering`)
    ///which avoids render pass objects entirely.
    pub unsafe fn create_render_pass2(
        &self,
        p_create_info: &RenderPassCreateInfo2,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<RenderPass> {
        let fp = self
            .commands()
            .create_render_pass2
            .expect("vkCreateRenderPass2 not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkCmdBeginRenderPass2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBeginRenderPass2.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_2**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.2 version of `cmd_begin_render_pass` that takes an
    ///additional `SubpassBeginInfo` parameter specifying the subpass
    ///contents mode.
    ///
    ///The extensible structs allow chaining `RenderPassAttachmentBeginInfo`
    ///for imageless framebuffers, concrete image views are supplied at
    ///begin time rather than at framebuffer creation time.
    ///
    ///Prefer this over `cmd_begin_render_pass` when targeting Vulkan 1.2+.
    ///For Vulkan 1.3+, consider `cmd_begin_rendering` (dynamic rendering)
    ///which eliminates render pass and framebuffer objects entirely.
    pub unsafe fn cmd_begin_render_pass2(
        &self,
        command_buffer: CommandBuffer,
        p_render_pass_begin: &RenderPassBeginInfo,
        p_subpass_begin_info: &SubpassBeginInfo,
    ) {
        let fp = self
            .commands()
            .cmd_begin_render_pass2
            .expect("vkCmdBeginRenderPass2 not loaded");
        unsafe { fp(command_buffer, p_render_pass_begin, p_subpass_begin_info) };
    }
    ///Wraps [`vkCmdNextSubpass2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdNextSubpass2.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_2**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.2 version of `cmd_next_subpass` that takes extensible
    ///`SubpassBeginInfo` and `SubpassEndInfo` structs.
    ///
    ///Functionally identical to `cmd_next_subpass`. Prefer this when
    ///targeting Vulkan 1.2+.
    pub unsafe fn cmd_next_subpass2(
        &self,
        command_buffer: CommandBuffer,
        p_subpass_begin_info: &SubpassBeginInfo,
        p_subpass_end_info: &SubpassEndInfo,
    ) {
        let fp = self
            .commands()
            .cmd_next_subpass2
            .expect("vkCmdNextSubpass2 not loaded");
        unsafe { fp(command_buffer, p_subpass_begin_info, p_subpass_end_info) };
    }
    ///Wraps [`vkCmdEndRenderPass2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdEndRenderPass2.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_2**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.2 version of `cmd_end_render_pass` that takes an extensible
    ///`SubpassEndInfo` struct.
    ///
    ///Functionally identical to `cmd_end_render_pass`. Prefer this when
    ///targeting Vulkan 1.2+.
    pub unsafe fn cmd_end_render_pass2(
        &self,
        command_buffer: CommandBuffer,
        p_subpass_end_info: &SubpassEndInfo,
    ) {
        let fp = self
            .commands()
            .cmd_end_render_pass2
            .expect("vkCmdEndRenderPass2 not loaded");
        unsafe { fp(command_buffer, p_subpass_end_info) };
    }
    ///Wraps [`vkGetSemaphoreCounterValue`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetSemaphoreCounterValue.html).
    /**
    Provided by **VK_BASE_VERSION_1_2**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_DEVICE_LOST`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns the current counter value of a timeline semaphore. Timeline
    ///semaphores (Vulkan 1.2) use a monotonically increasing 64-bit
    ///counter instead of binary signaled/unsignaled state.
    ///
    ///Use this for non-blocking progress checks:
    ///
    ///```text
    ///let value = get_semaphore_counter_value(semaphore);
    ///if value >= expected_frame {
    ///    // GPU has finished frame N, safe to reuse resources
    ///}
    ///```
    ///
    ///For blocking waits, use `wait_semaphores`. For signaling from the
    ///CPU, use `signal_semaphore`.
    ///
    ///Only valid for semaphores created with `SEMAPHORE_TYPE_TIMELINE`.
    ///Calling this on a binary semaphore is an error.
    pub unsafe fn get_semaphore_counter_value(&self, semaphore: Semaphore) -> VkResult<u64> {
        let fp = self
            .commands()
            .get_semaphore_counter_value
            .expect("vkGetSemaphoreCounterValue not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), semaphore, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkWaitSemaphores`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkWaitSemaphores.html).
    /**
    Provided by **VK_BASE_VERSION_1_2**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_DEVICE_LOST`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Blocks the calling thread until one or all of the specified timeline
    ///semaphores reach their target values, or until the timeout expires.
    ///
    ///**`SemaphoreWaitInfo` flags**:
    ///
    ///- `SEMAPHORE_WAIT_ANY`: return when *any* semaphore reaches its
    ///  target. Without this flag, waits for *all* semaphores.
    ///
    ///**Timeout**: in nanoseconds. `u64::MAX` for indefinite. Zero for a
    ///non-blocking poll.
    ///
    ///Timeline semaphore waits are the CPU-side counterpart to
    ///`queue_submit` timeline waits. They replace many fence-based
    ///synchronisation patterns with a single, more flexible primitive.
    ///
    ///```text
    #[doc = "// Wait for frame N to complete on the GPU"]
    ///let info = SemaphoreWaitInfo::builder()
    ///    .semaphores(&[timeline_sem])
    ///    .values(&[frame_number]);
    ///wait_semaphores(&info, u64::MAX);
    ///```
    ///
    ///Only valid for semaphores created with `SEMAPHORE_TYPE_TIMELINE`.
    pub unsafe fn wait_semaphores(
        &self,
        p_wait_info: &SemaphoreWaitInfo,
        timeout: u64,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .wait_semaphores
            .expect("vkWaitSemaphores not loaded");
        check(unsafe { fp(self.handle(), p_wait_info, timeout) })
    }
    ///Wraps [`vkSignalSemaphore`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkSignalSemaphore.html).
    /**
    Provided by **VK_BASE_VERSION_1_2**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Signals a timeline semaphore from the host (CPU), advancing its
    ///counter to the specified value. The value must be greater than the
    ///current counter value.
    ///
    ///Use this to unblock GPU work that is waiting on the semaphore via
    ///`queue_submit`. For example, a CPU-side data preparation step can
    ///signal a timeline semaphore when data is ready, and the GPU waits on
    ///it before processing.
    ///
    ///Only valid for semaphores created with `SEMAPHORE_TYPE_TIMELINE`.
    ///
    ///Timeline semaphores replace many use cases that previously required
    ///fences, they can be waited on from both the CPU (`wait_semaphores`)
    ///and the GPU (`queue_submit`).
    pub unsafe fn signal_semaphore(&self, p_signal_info: &SemaphoreSignalInfo) -> VkResult<()> {
        let fp = self
            .commands()
            .signal_semaphore
            .expect("vkSignalSemaphore not loaded");
        check(unsafe { fp(self.handle(), p_signal_info) })
    }
    ///Wraps [`vkGetAndroidHardwareBufferPropertiesANDROID`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetAndroidHardwareBufferPropertiesANDROID.html).
    /**
    Provided by **VK_ANDROID_external_memory_android_hardware_buffer**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_INVALID_EXTERNAL_HANDLE_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the Vulkan memory properties (memory type bits, size)
    ///for an Android `AHardwareBuffer`. Use before importing the
    ///buffer as Vulkan memory. Android only.
    ///
    ///Requires `VK_ANDROID_external_memory_android_hardware_buffer`.
    pub unsafe fn get_android_hardware_buffer_properties_android(
        &self,
        buffer: *const core::ffi::c_void,
        p_properties: *mut AndroidHardwareBufferPropertiesANDROID,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_android_hardware_buffer_properties_android
            .expect("vkGetAndroidHardwareBufferPropertiesANDROID not loaded");
        check(unsafe { fp(self.handle(), buffer, p_properties) })
    }
    ///Wraps [`vkGetMemoryAndroidHardwareBufferANDROID`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetMemoryAndroidHardwareBufferANDROID.html).
    /**
    Provided by **VK_ANDROID_external_memory_android_hardware_buffer**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_TOO_MANY_OBJECTS`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Exports a Vulkan device memory allocation as an Android
    ///`AHardwareBuffer` for sharing with other Android APIs (camera,
    ///media codec, SurfaceFlinger). Android only.
    ///
    ///Requires `VK_ANDROID_external_memory_android_hardware_buffer`.
    pub unsafe fn get_memory_android_hardware_buffer_android(
        &self,
        p_info: &MemoryGetAndroidHardwareBufferInfoANDROID,
        p_buffer: *mut *mut core::ffi::c_void,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_memory_android_hardware_buffer_android
            .expect("vkGetMemoryAndroidHardwareBufferANDROID not loaded");
        check(unsafe { fp(self.handle(), p_info, p_buffer) })
    }
    ///Wraps [`vkCmdDrawIndirectCount`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDrawIndirectCount.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_2**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Draws non-indexed geometry with both the draw parameters and the
    ///draw **count** read from GPU buffers. The non-indexed counterpart to
    ///`cmd_draw_indexed_indirect_count`.
    ///
    ///See `cmd_draw_indexed_indirect_count` for a full explanation of the
    ///GPU-driven rendering pattern. The only difference is that this
    ///command reads `DrawIndirectCommand` entries instead of
    ///`DrawIndexedIndirectCommand`.
    ///
    ///Core in Vulkan 1.2. Previously available via
    ///`VK_KHR_draw_indirect_count`.
    pub unsafe fn cmd_draw_indirect_count(
        &self,
        command_buffer: CommandBuffer,
        buffer: Buffer,
        offset: u64,
        count_buffer: Buffer,
        count_buffer_offset: u64,
        max_draw_count: u32,
        stride: u32,
    ) {
        let fp = self
            .commands()
            .cmd_draw_indirect_count
            .expect("vkCmdDrawIndirectCount not loaded");
        unsafe {
            fp(
                command_buffer,
                buffer,
                offset,
                count_buffer,
                count_buffer_offset,
                max_draw_count,
                stride,
            )
        };
    }
    ///Wraps [`vkCmdDrawIndexedIndirectCount`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDrawIndexedIndirectCount.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_2**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Draws indexed geometry with both the draw parameters and the draw
    ///**count** read from GPU buffers. The `count_buffer` contains a
    ///`u32` that limits how many `DrawIndexedIndirectCommand` entries from
    ///the main buffer are actually executed, up to `max_draw_count`.
    ///
    ///This is the key primitive for GPU-driven rendering pipelines: a
    ///compute shader fills an indirect buffer and writes the surviving
    ///draw count after culling. The CPU does not need to know the count.
    ///
    ///The main buffer must have `BUFFER_USAGE_INDIRECT_BUFFER`. The count
    ///buffer must also have `BUFFER_USAGE_INDIRECT_BUFFER`. The count
    ///offset must be a multiple of 4.
    ///
    ///Core in Vulkan 1.2. Previously available via
    ///`VK_KHR_draw_indirect_count`.
    pub unsafe fn cmd_draw_indexed_indirect_count(
        &self,
        command_buffer: CommandBuffer,
        buffer: Buffer,
        offset: u64,
        count_buffer: Buffer,
        count_buffer_offset: u64,
        max_draw_count: u32,
        stride: u32,
    ) {
        let fp = self
            .commands()
            .cmd_draw_indexed_indirect_count
            .expect("vkCmdDrawIndexedIndirectCount not loaded");
        unsafe {
            fp(
                command_buffer,
                buffer,
                offset,
                count_buffer,
                count_buffer_offset,
                max_draw_count,
                stride,
            )
        };
    }
    ///Wraps [`vkCmdSetCheckpointNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetCheckpointNV.html).
    /**
    Provided by **VK_NV_device_diagnostic_checkpoints**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Inserts a checkpoint marker into the command buffer for
    ///diagnostic purposes. If the device is lost, the most recently
    ///executed checkpoint can be retrieved with
    ///`get_queue_checkpoint_data_nv` to identify which commands
    ///completed before the failure.
    ///
    ///Requires `VK_NV_device_diagnostic_checkpoints`.
    pub unsafe fn cmd_set_checkpoint_nv(
        &self,
        command_buffer: CommandBuffer,
        p_checkpoint_marker: *const core::ffi::c_void,
    ) {
        let fp = self
            .commands()
            .cmd_set_checkpoint_nv
            .expect("vkCmdSetCheckpointNV not loaded");
        unsafe { fp(command_buffer, p_checkpoint_marker) };
    }
    ///Wraps [`vkGetQueueCheckpointDataNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetQueueCheckpointDataNV.html).
    /**
    Provided by **VK_NV_device_diagnostic_checkpoints**.*/
    ///
    ///# Safety
    ///- `queue` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves the checkpoint markers that were most recently executed
    ///on a queue before a device-lost event. Use with
    ///`cmd_set_checkpoint_nv` for post-mortem debugging.
    ///
    ///Requires `VK_NV_device_diagnostic_checkpoints`.
    pub unsafe fn get_queue_checkpoint_data_nv(&self, queue: Queue) -> Vec<CheckpointDataNV> {
        let fp = self
            .commands()
            .get_queue_checkpoint_data_nv
            .expect("vkGetQueueCheckpointDataNV not loaded");
        fill_two_call(|count, data| unsafe { fp(queue, count, data) })
    }
    ///Wraps [`vkCmdBindTransformFeedbackBuffersEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBindTransformFeedbackBuffersEXT.html).
    /**
    Provided by **VK_EXT_transform_feedback**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Binds buffers for transform feedback output. Each binding slot
    ///receives vertex data streamed from the vertex or geometry shader
    ///during a transform feedback pass.
    ///
    ///`first_binding` is the first binding index. Arrays of buffers,
    ///offsets, and sizes specify the output targets.
    ///
    ///Must be called before `cmd_begin_transform_feedback_ext`.
    ///
    ///Requires `VK_EXT_transform_feedback`.
    pub unsafe fn cmd_bind_transform_feedback_buffers_ext(
        &self,
        command_buffer: CommandBuffer,
        first_binding: u32,
        p_buffers: &[Buffer],
        p_offsets: &u64,
        p_sizes: Option<&u64>,
    ) {
        let fp = self
            .commands()
            .cmd_bind_transform_feedback_buffers_ext
            .expect("vkCmdBindTransformFeedbackBuffersEXT not loaded");
        let p_sizes_ptr = p_sizes.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe {
            fp(
                command_buffer,
                first_binding,
                p_buffers.len() as u32,
                p_buffers.as_ptr(),
                p_offsets,
                p_sizes_ptr,
            )
        };
    }
    ///Wraps [`vkCmdBeginTransformFeedbackEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBeginTransformFeedbackEXT.html).
    /**
    Provided by **VK_EXT_transform_feedback**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Begins a transform feedback pass. Vertex shader outputs (or
    ///geometry shader outputs) are written to transform feedback buffers
    ///previously bound with `cmd_bind_transform_feedback_buffers_ext`.
    ///
    ///`first_counter_buffer` and `p_counter_buffer_offsets` specify
    ///where to resume writing from (pass null offsets to start from
    ///scratch).
    ///
    ///End with `cmd_end_transform_feedback_ext`.
    ///
    ///Requires `VK_EXT_transform_feedback` and the
    ///`transformFeedback` feature.
    pub unsafe fn cmd_begin_transform_feedback_ext(
        &self,
        command_buffer: CommandBuffer,
        first_counter_buffer: u32,
        p_counter_buffers: &[Buffer],
        p_counter_buffer_offsets: Option<&u64>,
    ) {
        let fp = self
            .commands()
            .cmd_begin_transform_feedback_ext
            .expect("vkCmdBeginTransformFeedbackEXT not loaded");
        let p_counter_buffer_offsets_ptr =
            p_counter_buffer_offsets.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe {
            fp(
                command_buffer,
                first_counter_buffer,
                p_counter_buffers.len() as u32,
                p_counter_buffers.as_ptr(),
                p_counter_buffer_offsets_ptr,
            )
        };
    }
    ///Wraps [`vkCmdEndTransformFeedbackEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdEndTransformFeedbackEXT.html).
    /**
    Provided by **VK_EXT_transform_feedback**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Ends a transform feedback pass started with
    ///`cmd_begin_transform_feedback_ext`. Counter values are written
    ///back to the counter buffers so that a subsequent pass can resume
    ///where this one left off.
    ///
    ///Requires `VK_EXT_transform_feedback`.
    pub unsafe fn cmd_end_transform_feedback_ext(
        &self,
        command_buffer: CommandBuffer,
        first_counter_buffer: u32,
        p_counter_buffers: &[Buffer],
        p_counter_buffer_offsets: Option<&u64>,
    ) {
        let fp = self
            .commands()
            .cmd_end_transform_feedback_ext
            .expect("vkCmdEndTransformFeedbackEXT not loaded");
        let p_counter_buffer_offsets_ptr =
            p_counter_buffer_offsets.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe {
            fp(
                command_buffer,
                first_counter_buffer,
                p_counter_buffers.len() as u32,
                p_counter_buffers.as_ptr(),
                p_counter_buffer_offsets_ptr,
            )
        };
    }
    ///Wraps [`vkCmdBeginQueryIndexedEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBeginQueryIndexedEXT.html).
    /**
    Provided by **VK_EXT_transform_feedback**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Begins an indexed query, like `cmd_begin_query` but with an
    ///additional `index` parameter that selects which vertex stream
    ///to query when used with transform feedback statistics queries.
    ///
    ///For `QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT`, the index
    ///selects the stream (0–3). For other query types, index must be 0.
    ///
    ///End with `cmd_end_query_indexed_ext`.
    ///
    ///Requires `VK_EXT_transform_feedback`.
    pub unsafe fn cmd_begin_query_indexed_ext(
        &self,
        command_buffer: CommandBuffer,
        query_pool: QueryPool,
        query: u32,
        flags: QueryControlFlags,
        index: u32,
    ) {
        let fp = self
            .commands()
            .cmd_begin_query_indexed_ext
            .expect("vkCmdBeginQueryIndexedEXT not loaded");
        unsafe { fp(command_buffer, query_pool, query, flags, index) };
    }
    ///Wraps [`vkCmdEndQueryIndexedEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdEndQueryIndexedEXT.html).
    /**
    Provided by **VK_EXT_transform_feedback**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Ends an indexed query started with `cmd_begin_query_indexed_ext`.
    ///The `index` parameter must match the one used in the begin call.
    ///
    ///Requires `VK_EXT_transform_feedback`.
    pub unsafe fn cmd_end_query_indexed_ext(
        &self,
        command_buffer: CommandBuffer,
        query_pool: QueryPool,
        query: u32,
        index: u32,
    ) {
        let fp = self
            .commands()
            .cmd_end_query_indexed_ext
            .expect("vkCmdEndQueryIndexedEXT not loaded");
        unsafe { fp(command_buffer, query_pool, query, index) };
    }
    ///Wraps [`vkCmdDrawIndirectByteCountEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDrawIndirectByteCountEXT.html).
    /**
    Provided by **VK_EXT_transform_feedback**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Draws vertices using a byte count stored in a counter buffer
    ///(typically written by a transform feedback pass). The vertex
    ///count is computed as `counter_value / vertex_stride`.
    ///
    ///`counter_offset` accounts for any header bytes before the
    ///counter value in the buffer.
    ///
    ///Requires `VK_EXT_transform_feedback`.
    pub unsafe fn cmd_draw_indirect_byte_count_ext(
        &self,
        command_buffer: CommandBuffer,
        instance_count: u32,
        first_instance: u32,
        counter_buffer: Buffer,
        counter_buffer_offset: u64,
        counter_offset: u32,
        vertex_stride: u32,
    ) {
        let fp = self
            .commands()
            .cmd_draw_indirect_byte_count_ext
            .expect("vkCmdDrawIndirectByteCountEXT not loaded");
        unsafe {
            fp(
                command_buffer,
                instance_count,
                first_instance,
                counter_buffer,
                counter_buffer_offset,
                counter_offset,
                vertex_stride,
            )
        };
    }
    ///Wraps [`vkCmdSetExclusiveScissorNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetExclusiveScissorNV.html).
    /**
    Provided by **VK_NV_scissor_exclusive**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets the exclusive scissor rectangles for one or
    ///more viewports. Fragments outside these rectangles are discarded
    ///when exclusive scissor testing is enabled.
    ///
    ///Requires `VK_NV_scissor_exclusive`.
    pub unsafe fn cmd_set_exclusive_scissor_nv(
        &self,
        command_buffer: CommandBuffer,
        first_exclusive_scissor: u32,
        p_exclusive_scissors: &[Rect2D],
    ) {
        let fp = self
            .commands()
            .cmd_set_exclusive_scissor_nv
            .expect("vkCmdSetExclusiveScissorNV not loaded");
        unsafe {
            fp(
                command_buffer,
                first_exclusive_scissor,
                p_exclusive_scissors.len() as u32,
                p_exclusive_scissors.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdSetExclusiveScissorEnableNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetExclusiveScissorEnableNV.html).
    /**
    Provided by **VK_NV_scissor_exclusive**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically enables or disables the exclusive scissor test for
    ///one or more viewports. When enabled, fragments outside the
    ///exclusive scissor rectangle are discarded.
    ///
    ///Requires `VK_NV_scissor_exclusive`.
    pub unsafe fn cmd_set_exclusive_scissor_enable_nv(
        &self,
        command_buffer: CommandBuffer,
        first_exclusive_scissor: u32,
        p_exclusive_scissor_enables: &[u32],
    ) {
        let fp = self
            .commands()
            .cmd_set_exclusive_scissor_enable_nv
            .expect("vkCmdSetExclusiveScissorEnableNV not loaded");
        unsafe {
            fp(
                command_buffer,
                first_exclusive_scissor,
                p_exclusive_scissor_enables.len() as u32,
                p_exclusive_scissor_enables.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdBindShadingRateImageNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBindShadingRateImageNV.html).
    /**
    Provided by **VK_NV_shading_rate_image**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Binds a shading rate image that controls per-region fragment
    ///shading rate. Each texel in the image maps to a tile of the
    ///framebuffer and specifies the coarse shading rate for that tile.
    ///
    ///Requires `VK_NV_shading_rate_image`.
    pub unsafe fn cmd_bind_shading_rate_image_nv(
        &self,
        command_buffer: CommandBuffer,
        image_view: ImageView,
        image_layout: ImageLayout,
    ) {
        let fp = self
            .commands()
            .cmd_bind_shading_rate_image_nv
            .expect("vkCmdBindShadingRateImageNV not loaded");
        unsafe { fp(command_buffer, image_view, image_layout) };
    }
    ///Wraps [`vkCmdSetViewportShadingRatePaletteNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetViewportShadingRatePaletteNV.html).
    /**
    Provided by **VK_NV_shading_rate_image**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Sets the shading rate palette for one or more viewports. The
    ///palette maps shading rate image texel values to actual shading
    ///rates (e.g., 1x1, 2x2, 4x4).
    ///
    ///Requires `VK_NV_shading_rate_image`.
    pub unsafe fn cmd_set_viewport_shading_rate_palette_nv(
        &self,
        command_buffer: CommandBuffer,
        first_viewport: u32,
        p_shading_rate_palettes: &[ShadingRatePaletteNV],
    ) {
        let fp = self
            .commands()
            .cmd_set_viewport_shading_rate_palette_nv
            .expect("vkCmdSetViewportShadingRatePaletteNV not loaded");
        unsafe {
            fp(
                command_buffer,
                first_viewport,
                p_shading_rate_palettes.len() as u32,
                p_shading_rate_palettes.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdSetCoarseSampleOrderNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetCoarseSampleOrderNV.html).
    /**
    Provided by **VK_NV_shading_rate_image**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Sets the sample ordering for coarse shading rate fragments. By
    ///default, the driver chooses sample order; use this to specify a
    ///custom order when the fragment shader relies on specific sample
    ///positions within coarse fragments.
    ///
    ///Requires `VK_NV_shading_rate_image`.
    pub unsafe fn cmd_set_coarse_sample_order_nv(
        &self,
        command_buffer: CommandBuffer,
        sample_order_type: CoarseSampleOrderTypeNV,
        p_custom_sample_orders: &[CoarseSampleOrderCustomNV],
    ) {
        let fp = self
            .commands()
            .cmd_set_coarse_sample_order_nv
            .expect("vkCmdSetCoarseSampleOrderNV not loaded");
        unsafe {
            fp(
                command_buffer,
                sample_order_type,
                p_custom_sample_orders.len() as u32,
                p_custom_sample_orders.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdDrawMeshTasksNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDrawMeshTasksNV.html).
    /**
    Provided by **VK_NV_mesh_shader**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dispatches mesh shader work using the NV mesh shader model.
    ///Launches `task_count` task shader workgroups starting at
    ///`first_task`.
    ///
    ///This is the legacy NV path; prefer `cmd_draw_mesh_tasks_ext`
    ///for new code.
    ///
    ///Requires `VK_NV_mesh_shader`.
    pub unsafe fn cmd_draw_mesh_tasks_nv(
        &self,
        command_buffer: CommandBuffer,
        task_count: u32,
        first_task: u32,
    ) {
        let fp = self
            .commands()
            .cmd_draw_mesh_tasks_nv
            .expect("vkCmdDrawMeshTasksNV not loaded");
        unsafe { fp(command_buffer, task_count, first_task) };
    }
    ///Wraps [`vkCmdDrawMeshTasksIndirectNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDrawMeshTasksIndirectNV.html).
    /**
    Provided by **VK_NV_mesh_shader**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Indirect variant of `cmd_draw_mesh_tasks_nv`. Reads draw
    ///parameters from a buffer, enabling GPU-driven mesh shader
    ///dispatch.
    ///
    ///Requires `VK_NV_mesh_shader`.
    pub unsafe fn cmd_draw_mesh_tasks_indirect_nv(
        &self,
        command_buffer: CommandBuffer,
        buffer: Buffer,
        offset: u64,
        draw_count: u32,
        stride: u32,
    ) {
        let fp = self
            .commands()
            .cmd_draw_mesh_tasks_indirect_nv
            .expect("vkCmdDrawMeshTasksIndirectNV not loaded");
        unsafe { fp(command_buffer, buffer, offset, draw_count, stride) };
    }
    ///Wraps [`vkCmdDrawMeshTasksIndirectCountNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDrawMeshTasksIndirectCountNV.html).
    /**
    Provided by **VK_NV_mesh_shader**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Indirect-count variant of `cmd_draw_mesh_tasks_nv`. Reads both
    ///the draw parameters and the draw count from GPU buffers, enabling
    ///fully GPU-driven mesh shader dispatch where the number of draws
    ///is determined at runtime.
    ///
    ///Requires `VK_NV_mesh_shader`.
    pub unsafe fn cmd_draw_mesh_tasks_indirect_count_nv(
        &self,
        command_buffer: CommandBuffer,
        buffer: Buffer,
        offset: u64,
        count_buffer: Buffer,
        count_buffer_offset: u64,
        max_draw_count: u32,
        stride: u32,
    ) {
        let fp = self
            .commands()
            .cmd_draw_mesh_tasks_indirect_count_nv
            .expect("vkCmdDrawMeshTasksIndirectCountNV not loaded");
        unsafe {
            fp(
                command_buffer,
                buffer,
                offset,
                count_buffer,
                count_buffer_offset,
                max_draw_count,
                stride,
            )
        };
    }
    ///Wraps [`vkCmdDrawMeshTasksEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDrawMeshTasksEXT.html).
    /**
    Provided by **VK_EXT_mesh_shader**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dispatches mesh shader work groups. `group_count_x/y/z` specify
    ///the number of task or mesh shader work groups to launch.
    ///
    ///Each work group runs the mesh shader (or task shader, if bound)
    ///which emits primitives directly without traditional vertex/index
    ///buffers.
    ///
    ///Requires `VK_EXT_mesh_shader` and the `meshShader` feature.
    pub unsafe fn cmd_draw_mesh_tasks_ext(
        &self,
        command_buffer: CommandBuffer,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) {
        let fp = self
            .commands()
            .cmd_draw_mesh_tasks_ext
            .expect("vkCmdDrawMeshTasksEXT not loaded");
        unsafe { fp(command_buffer, group_count_x, group_count_y, group_count_z) };
    }
    ///Wraps [`vkCmdDrawMeshTasksIndirectEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDrawMeshTasksIndirectEXT.html).
    /**
    Provided by **VK_EXT_mesh_shader**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Indirect version of `cmd_draw_mesh_tasks_ext`. Reads mesh shader
    ///dispatch parameters from a buffer. Each indirect command contains
    ///group counts (x, y, z).
    ///
    ///`draw_count` specifies how many indirect commands to execute.
    ///`stride` is the byte stride between commands in the buffer.
    ///
    ///Requires `VK_EXT_mesh_shader`.
    pub unsafe fn cmd_draw_mesh_tasks_indirect_ext(
        &self,
        command_buffer: CommandBuffer,
        buffer: Buffer,
        offset: u64,
        draw_count: u32,
        stride: u32,
    ) {
        let fp = self
            .commands()
            .cmd_draw_mesh_tasks_indirect_ext
            .expect("vkCmdDrawMeshTasksIndirectEXT not loaded");
        unsafe { fp(command_buffer, buffer, offset, draw_count, stride) };
    }
    ///Wraps [`vkCmdDrawMeshTasksIndirectCountEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDrawMeshTasksIndirectCountEXT.html).
    /**
    Provided by **VK_EXT_mesh_shader**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Count-indirect version of mesh shader dispatch. Both the dispatch
    ///parameters and the draw count are read from GPU buffers, enabling
    ///fully GPU-driven mesh shader workloads.
    ///
    ///`max_draw_count` caps the count read from the count buffer.
    ///
    ///Requires `VK_EXT_mesh_shader`.
    pub unsafe fn cmd_draw_mesh_tasks_indirect_count_ext(
        &self,
        command_buffer: CommandBuffer,
        buffer: Buffer,
        offset: u64,
        count_buffer: Buffer,
        count_buffer_offset: u64,
        max_draw_count: u32,
        stride: u32,
    ) {
        let fp = self
            .commands()
            .cmd_draw_mesh_tasks_indirect_count_ext
            .expect("vkCmdDrawMeshTasksIndirectCountEXT not loaded");
        unsafe {
            fp(
                command_buffer,
                buffer,
                offset,
                count_buffer,
                count_buffer_offset,
                max_draw_count,
                stride,
            )
        };
    }
    ///Wraps [`vkCompileDeferredNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCompileDeferredNV.html).
    /**
    Provided by **VK_NV_ray_tracing**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Triggers compilation of a deferred shader in an NV ray tracing
    ///pipeline. When a pipeline is created with
    ///`PIPELINE_CREATE_DEFER_COMPILE_BIT_NV`, individual shaders can
    ///be compiled on demand using this command.
    ///
    ///Useful for spreading compilation cost across frames or threads.
    ///
    ///Requires `VK_NV_ray_tracing`.
    pub unsafe fn compile_deferred_nv(&self, pipeline: Pipeline, shader: u32) -> VkResult<()> {
        let fp = self
            .commands()
            .compile_deferred_nv
            .expect("vkCompileDeferredNV not loaded");
        check(unsafe { fp(self.handle(), pipeline, shader) })
    }
    ///Wraps [`vkCreateAccelerationStructureNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateAccelerationStructureNV.html).
    /**
    Provided by **VK_NV_ray_tracing**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates an NV ray tracing acceleration structure. This is the
    ///legacy NV path; prefer `create_acceleration_structure_khr` for
    ///new code.
    ///
    ///The NV acceleration structure owns its memory implicitly, bind
    ///memory with `bind_acceleration_structure_memory_nv`. Destroy with
    ///`destroy_acceleration_structure_nv`.
    ///
    ///Requires `VK_NV_ray_tracing`.
    pub unsafe fn create_acceleration_structure_nv(
        &self,
        p_create_info: &AccelerationStructureCreateInfoNV,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<AccelerationStructureNV> {
        let fp = self
            .commands()
            .create_acceleration_structure_nv
            .expect("vkCreateAccelerationStructureNV not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkCmdBindInvocationMaskHUAWEI`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBindInvocationMaskHUAWEI.html).
    /**
    Provided by **VK_HUAWEI_invocation_mask**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Binds an image view as an invocation mask for ray tracing. The
    ///mask selectively enables or disables ray generation shader
    ///invocations per pixel, allowing efficient partial-screen ray
    ///tracing on Huawei GPUs.
    ///
    ///Requires `VK_HUAWEI_invocation_mask`.
    pub unsafe fn cmd_bind_invocation_mask_huawei(
        &self,
        command_buffer: CommandBuffer,
        image_view: ImageView,
        image_layout: ImageLayout,
    ) {
        let fp = self
            .commands()
            .cmd_bind_invocation_mask_huawei
            .expect("vkCmdBindInvocationMaskHUAWEI not loaded");
        unsafe { fp(command_buffer, image_view, image_layout) };
    }
    ///Wraps [`vkDestroyAccelerationStructureKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyAccelerationStructureKHR.html).
    /**
    Provided by **VK_KHR_acceleration_structure**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `accelerationStructure` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys an acceleration structure. The structure must not be
    ///referenced by any pending ray tracing command or TLAS build.
    ///
    ///Destroying the acceleration structure does not free the backing
    ///buffer, destroy or reclaim it separately.
    pub unsafe fn destroy_acceleration_structure_khr(
        &self,
        acceleration_structure: AccelerationStructureKHR,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_acceleration_structure_khr
            .expect("vkDestroyAccelerationStructureKHR not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), acceleration_structure, alloc_ptr) };
    }
    ///Wraps [`vkDestroyAccelerationStructureNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyAccelerationStructureNV.html).
    /**
    Provided by **VK_NV_ray_tracing**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `accelerationStructure` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys an NV acceleration structure created with
    ///`create_acceleration_structure_nv`. The structure must not be
    ///referenced by any pending command.
    ///
    ///Requires `VK_NV_ray_tracing`.
    pub unsafe fn destroy_acceleration_structure_nv(
        &self,
        acceleration_structure: AccelerationStructureNV,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_acceleration_structure_nv
            .expect("vkDestroyAccelerationStructureNV not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), acceleration_structure, alloc_ptr) };
    }
    ///Wraps [`vkGetAccelerationStructureMemoryRequirementsNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetAccelerationStructureMemoryRequirementsNV.html).
    /**
    Provided by **VK_NV_ray_tracing**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the memory requirements for an NV acceleration structure,
    ///including the scratch memory needed for builds and updates. Use
    ///the result to allocate and bind memory before building.
    ///
    ///Requires `VK_NV_ray_tracing`.
    pub unsafe fn get_acceleration_structure_memory_requirements_nv(
        &self,
        p_info: &AccelerationStructureMemoryRequirementsInfoNV,
    ) -> MemoryRequirements2KHR {
        let fp = self
            .commands()
            .get_acceleration_structure_memory_requirements_nv
            .expect("vkGetAccelerationStructureMemoryRequirementsNV not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(self.handle(), p_info, &mut out) };
        out
    }
    ///Wraps [`vkBindAccelerationStructureMemoryNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkBindAccelerationStructureMemoryNV.html).
    /**
    Provided by **VK_NV_ray_tracing**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Binds device memory to one or more NV acceleration structures.
    ///Must be called before the structure can be used in a build or
    ///trace command.
    ///
    ///Requires `VK_NV_ray_tracing`.
    pub unsafe fn bind_acceleration_structure_memory_nv(
        &self,
        p_bind_infos: &[BindAccelerationStructureMemoryInfoNV],
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .bind_acceleration_structure_memory_nv
            .expect("vkBindAccelerationStructureMemoryNV not loaded");
        check(unsafe {
            fp(
                self.handle(),
                p_bind_infos.len() as u32,
                p_bind_infos.as_ptr(),
            )
        })
    }
    ///Wraps [`vkCmdCopyAccelerationStructureNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyAccelerationStructureNV.html).
    /**
    Provided by **VK_NV_ray_tracing**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Copies or compacts an NV acceleration structure. Use `CLONE` mode
    ///for a full copy, or `COMPACT` to produce a smaller copy after
    ///querying the compacted size.
    ///
    ///Requires `VK_NV_ray_tracing`.
    pub unsafe fn cmd_copy_acceleration_structure_nv(
        &self,
        command_buffer: CommandBuffer,
        dst: AccelerationStructureNV,
        src: AccelerationStructureNV,
        mode: CopyAccelerationStructureModeKHR,
    ) {
        let fp = self
            .commands()
            .cmd_copy_acceleration_structure_nv
            .expect("vkCmdCopyAccelerationStructureNV not loaded");
        unsafe { fp(command_buffer, dst, src, mode) };
    }
    ///Wraps [`vkCmdCopyAccelerationStructureKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyAccelerationStructureKHR.html).
    /**
    Provided by **VK_KHR_acceleration_structure**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Copies an acceleration structure. Modes:
    ///
    ///- `COPY_ACCELERATION_STRUCTURE_MODE_CLONE`: full copy. The
    ///  destination must be at least as large as the source.
    ///- `COPY_ACCELERATION_STRUCTURE_MODE_COMPACT`: copies into a smaller
    ///  buffer after a compaction query. Use this to reclaim memory after
    ///  building.
    ///
    ///**Compaction workflow**:
    ///
    ///1. Build with `BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION`.
    ///2. Query compacted size with
    ///   `cmd_write_acceleration_structures_properties_khr`.
    ///3. Create a new, smaller backing buffer.
    ///4. Copy with `MODE_COMPACT`.
    ///
    ///Compaction typically saves 50–70% of BLAS memory.
    pub unsafe fn cmd_copy_acceleration_structure_khr(
        &self,
        command_buffer: CommandBuffer,
        p_info: &CopyAccelerationStructureInfoKHR,
    ) {
        let fp = self
            .commands()
            .cmd_copy_acceleration_structure_khr
            .expect("vkCmdCopyAccelerationStructureKHR not loaded");
        unsafe { fp(command_buffer, p_info) };
    }
    ///Wraps [`vkCopyAccelerationStructureKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCopyAccelerationStructureKHR.html).
    /**
    Provided by **VK_KHR_acceleration_structure**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Host-side acceleration structure copy. The CPU counterpart to
    ///`cmd_copy_acceleration_structure_khr`. Supports the same clone and
    ///compact modes.
    ///
    ///Requires the `acceleration_structure_host_commands` feature.
    pub unsafe fn copy_acceleration_structure_khr(
        &self,
        deferred_operation: DeferredOperationKHR,
        p_info: &CopyAccelerationStructureInfoKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .copy_acceleration_structure_khr
            .expect("vkCopyAccelerationStructureKHR not loaded");
        check(unsafe { fp(self.handle(), deferred_operation, p_info) })
    }
    ///Wraps [`vkCmdCopyAccelerationStructureToMemoryKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyAccelerationStructureToMemoryKHR.html).
    /**
    Provided by **VK_KHR_acceleration_structure**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Serializes an acceleration structure into a buffer for storage or
    ///transfer. The serialized format is opaque and driver-specific, it
    ///can only be deserialized on the same driver and hardware.
    ///
    ///Use this for:
    ///
    ///- Saving acceleration structures to disk for faster subsequent loads.
    ///- Transferring structures between devices in a device group.
    ///
    ///Deserialize with `cmd_copy_memory_to_acceleration_structure_khr`.
    pub unsafe fn cmd_copy_acceleration_structure_to_memory_khr(
        &self,
        command_buffer: CommandBuffer,
        p_info: &CopyAccelerationStructureToMemoryInfoKHR,
    ) {
        let fp = self
            .commands()
            .cmd_copy_acceleration_structure_to_memory_khr
            .expect("vkCmdCopyAccelerationStructureToMemoryKHR not loaded");
        unsafe { fp(command_buffer, p_info) };
    }
    ///Wraps [`vkCopyAccelerationStructureToMemoryKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCopyAccelerationStructureToMemoryKHR.html).
    /**
    Provided by **VK_KHR_acceleration_structure**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Host-side acceleration structure serialization. The CPU counterpart
    ///to `cmd_copy_acceleration_structure_to_memory_khr`.
    ///
    ///Requires the `acceleration_structure_host_commands` feature.
    pub unsafe fn copy_acceleration_structure_to_memory_khr(
        &self,
        deferred_operation: DeferredOperationKHR,
        p_info: &CopyAccelerationStructureToMemoryInfoKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .copy_acceleration_structure_to_memory_khr
            .expect("vkCopyAccelerationStructureToMemoryKHR not loaded");
        check(unsafe { fp(self.handle(), deferred_operation, p_info) })
    }
    ///Wraps [`vkCmdCopyMemoryToAccelerationStructureKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyMemoryToAccelerationStructureKHR.html).
    /**
    Provided by **VK_KHR_acceleration_structure**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Deserializes an acceleration structure from a buffer that was
    ///previously written by
    ///`cmd_copy_acceleration_structure_to_memory_khr`.
    ///
    ///The data must have been serialized on the same driver and hardware
    ///(check `acceleration_structure_uuid` compatibility before loading).
    ///
    ///After deserialization the acceleration structure is ready for use
    ///in ray tracing commands.
    pub unsafe fn cmd_copy_memory_to_acceleration_structure_khr(
        &self,
        command_buffer: CommandBuffer,
        p_info: &CopyMemoryToAccelerationStructureInfoKHR,
    ) {
        let fp = self
            .commands()
            .cmd_copy_memory_to_acceleration_structure_khr
            .expect("vkCmdCopyMemoryToAccelerationStructureKHR not loaded");
        unsafe { fp(command_buffer, p_info) };
    }
    ///Wraps [`vkCopyMemoryToAccelerationStructureKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCopyMemoryToAccelerationStructureKHR.html).
    /**
    Provided by **VK_KHR_acceleration_structure**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Host-side acceleration structure deserialization. The CPU counterpart
    ///to `cmd_copy_memory_to_acceleration_structure_khr`.
    ///
    ///Requires the `acceleration_structure_host_commands` feature.
    pub unsafe fn copy_memory_to_acceleration_structure_khr(
        &self,
        deferred_operation: DeferredOperationKHR,
        p_info: &CopyMemoryToAccelerationStructureInfoKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .copy_memory_to_acceleration_structure_khr
            .expect("vkCopyMemoryToAccelerationStructureKHR not loaded");
        check(unsafe { fp(self.handle(), deferred_operation, p_info) })
    }
    ///Wraps [`vkCmdWriteAccelerationStructuresPropertiesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdWriteAccelerationStructuresPropertiesKHR.html).
    /**
    Provided by **VK_KHR_acceleration_structure**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Writes acceleration structure properties into a query pool. The
    ///primary use is querying `QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE`
    ///after a build to determine the compacted size before copying.
    ///
    ///**Compaction workflow**:
    ///
    ///1. Build with `ALLOW_COMPACTION`.
    ///2. `cmd_write_acceleration_structures_properties_khr` with
    ///   `COMPACTED_SIZE` query type.
    ///3. Read the result from the query pool.
    ///4. Create a smaller buffer and copy with `MODE_COMPACT`.
    ///
    ///Also supports `QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE`
    ///for estimating serialization buffer requirements.
    pub unsafe fn cmd_write_acceleration_structures_properties_khr(
        &self,
        command_buffer: CommandBuffer,
        p_acceleration_structures: &[AccelerationStructureKHR],
        query_type: QueryType,
        query_pool: QueryPool,
        first_query: u32,
    ) {
        let fp = self
            .commands()
            .cmd_write_acceleration_structures_properties_khr
            .expect("vkCmdWriteAccelerationStructuresPropertiesKHR not loaded");
        unsafe {
            fp(
                command_buffer,
                p_acceleration_structures.len() as u32,
                p_acceleration_structures.as_ptr(),
                query_type,
                query_pool,
                first_query,
            )
        };
    }
    ///Wraps [`vkCmdWriteAccelerationStructuresPropertiesNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdWriteAccelerationStructuresPropertiesNV.html).
    /**
    Provided by **VK_NV_ray_tracing**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Writes acceleration structure properties (such as compacted size)
    ///into a query pool. Use this after a build to determine the
    ///compacted size before calling `cmd_copy_acceleration_structure_nv`
    ///with `COMPACT` mode.
    ///
    ///Requires `VK_NV_ray_tracing`.
    pub unsafe fn cmd_write_acceleration_structures_properties_nv(
        &self,
        command_buffer: CommandBuffer,
        p_acceleration_structures: &[AccelerationStructureNV],
        query_type: QueryType,
        query_pool: QueryPool,
        first_query: u32,
    ) {
        let fp = self
            .commands()
            .cmd_write_acceleration_structures_properties_nv
            .expect("vkCmdWriteAccelerationStructuresPropertiesNV not loaded");
        unsafe {
            fp(
                command_buffer,
                p_acceleration_structures.len() as u32,
                p_acceleration_structures.as_ptr(),
                query_type,
                query_pool,
                first_query,
            )
        };
    }
    ///Wraps [`vkCmdBuildAccelerationStructureNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBuildAccelerationStructureNV.html).
    /**
    Provided by **VK_NV_ray_tracing**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Builds or updates an NV acceleration structure from geometry data.
    ///Set `update` to non-zero to refit an existing structure in place
    ///(faster but lower quality than a full rebuild).
    ///
    ///A scratch buffer is required; query its size with
    ///`get_acceleration_structure_memory_requirements_nv`.
    ///
    ///Requires `VK_NV_ray_tracing`.
    pub unsafe fn cmd_build_acceleration_structure_nv(
        &self,
        command_buffer: CommandBuffer,
        p_info: &AccelerationStructureInfoNV,
        instance_data: Buffer,
        instance_offset: u64,
        update: u32,
        dst: AccelerationStructureNV,
        src: AccelerationStructureNV,
        scratch: Buffer,
        scratch_offset: u64,
    ) {
        let fp = self
            .commands()
            .cmd_build_acceleration_structure_nv
            .expect("vkCmdBuildAccelerationStructureNV not loaded");
        unsafe {
            fp(
                command_buffer,
                p_info,
                instance_data,
                instance_offset,
                update,
                dst,
                src,
                scratch,
                scratch_offset,
            )
        };
    }
    ///Wraps [`vkWriteAccelerationStructuresPropertiesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkWriteAccelerationStructuresPropertiesKHR.html).
    /**
    Provided by **VK_KHR_acceleration_structure**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Host-side query of acceleration structure properties. The CPU
    ///counterpart to `cmd_write_acceleration_structures_properties_khr`.
    ///
    ///Writes results directly to a host buffer rather than a query pool.
    ///Supports the same query types: compacted size and serialization
    ///size.
    ///
    ///Requires the `acceleration_structure_host_commands` feature.
    pub unsafe fn write_acceleration_structures_properties_khr(
        &self,
        p_acceleration_structures: &[AccelerationStructureKHR],
        query_type: QueryType,
        data_size: usize,
        p_data: *mut core::ffi::c_void,
        stride: usize,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .write_acceleration_structures_properties_khr
            .expect("vkWriteAccelerationStructuresPropertiesKHR not loaded");
        check(unsafe {
            fp(
                self.handle(),
                p_acceleration_structures.len() as u32,
                p_acceleration_structures.as_ptr(),
                query_type,
                data_size,
                p_data,
                stride,
            )
        })
    }
    ///Wraps [`vkCmdTraceRaysKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdTraceRaysKHR.html).
    /**
    Provided by **VK_KHR_ray_tracing_pipeline**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dispatches rays into the scene. This is the primary ray tracing
    ///dispatch command, the ray tracing equivalent of `cmd_draw` or
    ///`cmd_dispatch`.
    ///
    ///Each of the four shader binding table (SBT) regions points to a
    ///device memory region containing shader group handles:
    ///
    ///- **Raygen**: exactly one entry, the ray generation shader.
    ///- **Miss**: shaders invoked when a ray hits nothing.
    ///- **Hit**: shader groups invoked on ray-geometry intersection.
    ///- **Callable**: shaders invoked explicitly from other stages.
    ///
    ///The `width`, `height`, and `depth` parameters define the 3D launch
    ///dimensions. Each invocation gets a unique `gl_LaunchIDEXT`. For
    ///a fullscreen ray trace, use the render target resolution with
    ///`depth = 1`.
    ///
    ///The SBT entries must be built from handles retrieved with
    ///`get_ray_tracing_shader_group_handles_khr`, stored in a buffer
    ///with `BUFFER_USAGE_SHADER_BINDING_TABLE`. Each region's `stride`
    ///must be a multiple of `shaderGroupHandleAlignment` and the base
    ///address must be aligned to `shaderGroupBaseAlignment`.
    pub unsafe fn cmd_trace_rays_khr(
        &self,
        command_buffer: CommandBuffer,
        p_raygen_shader_binding_table: &StridedDeviceAddressRegionKHR,
        p_miss_shader_binding_table: &StridedDeviceAddressRegionKHR,
        p_hit_shader_binding_table: &StridedDeviceAddressRegionKHR,
        p_callable_shader_binding_table: &StridedDeviceAddressRegionKHR,
        width: u32,
        height: u32,
        depth: u32,
    ) {
        let fp = self
            .commands()
            .cmd_trace_rays_khr
            .expect("vkCmdTraceRaysKHR not loaded");
        unsafe {
            fp(
                command_buffer,
                p_raygen_shader_binding_table,
                p_miss_shader_binding_table,
                p_hit_shader_binding_table,
                p_callable_shader_binding_table,
                width,
                height,
                depth,
            )
        };
    }
    ///Wraps [`vkCmdTraceRaysNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdTraceRaysNV.html).
    /**
    Provided by **VK_NV_ray_tracing**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dispatches a ray tracing workload using the NV ray tracing
    ///pipeline. Takes shader binding table buffer/offset/stride for
    ///each shader stage (raygen, miss, closest hit, callable) and the
    ///dispatch dimensions.
    ///
    ///This is the legacy NV path; prefer `cmd_trace_rays_khr` for new
    ///code.
    ///
    ///Requires `VK_NV_ray_tracing`.
    pub unsafe fn cmd_trace_rays_nv(
        &self,
        command_buffer: CommandBuffer,
        raygen_shader_binding_table_buffer: Buffer,
        raygen_shader_binding_offset: u64,
        miss_shader_binding_table_buffer: Buffer,
        miss_shader_binding_offset: u64,
        miss_shader_binding_stride: u64,
        hit_shader_binding_table_buffer: Buffer,
        hit_shader_binding_offset: u64,
        hit_shader_binding_stride: u64,
        callable_shader_binding_table_buffer: Buffer,
        callable_shader_binding_offset: u64,
        callable_shader_binding_stride: u64,
        width: u32,
        height: u32,
        depth: u32,
    ) {
        let fp = self
            .commands()
            .cmd_trace_rays_nv
            .expect("vkCmdTraceRaysNV not loaded");
        unsafe {
            fp(
                command_buffer,
                raygen_shader_binding_table_buffer,
                raygen_shader_binding_offset,
                miss_shader_binding_table_buffer,
                miss_shader_binding_offset,
                miss_shader_binding_stride,
                hit_shader_binding_table_buffer,
                hit_shader_binding_offset,
                hit_shader_binding_stride,
                callable_shader_binding_table_buffer,
                callable_shader_binding_offset,
                callable_shader_binding_stride,
                width,
                height,
                depth,
            )
        };
    }
    ///Wraps [`vkGetRayTracingShaderGroupHandlesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetRayTracingShaderGroupHandlesKHR.html).
    /**
    Provided by **VK_KHR_ray_tracing_pipeline**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves opaque shader group handles from a ray tracing pipeline.
    ///These handles are copied into GPU-visible buffers to build the
    ///**shader binding table** (SBT) that `cmd_trace_rays_khr` indexes
    ///into.
    ///
    ///Each handle is `shaderGroupHandleSize` bytes (query from
    ///`PhysicalDeviceRayTracingPipelinePropertiesKHR`, typically 32
    ///bytes). The `p_data` buffer must be at least
    ///`group_count * shaderGroupHandleSize` bytes.
    ///
    ///`first_group` and `group_count` index into the `groups` array
    ///from `RayTracingPipelineCreateInfoKHR`. Handles are written
    ///sequentially, group `first_group` first, then
    ///`first_group + 1`, and so on.
    ///
    ///After retrieving handles, copy them into a buffer with
    ///`BUFFER_USAGE_SHADER_BINDING_TABLE` at the correct stride and
    ///alignment for each SBT region.
    pub unsafe fn get_ray_tracing_shader_group_handles_khr(
        &self,
        pipeline: Pipeline,
        first_group: u32,
        group_count: u32,
        data_size: usize,
        p_data: *mut core::ffi::c_void,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_ray_tracing_shader_group_handles_khr
            .expect("vkGetRayTracingShaderGroupHandlesKHR not loaded");
        check(unsafe {
            fp(
                self.handle(),
                pipeline,
                first_group,
                group_count,
                data_size,
                p_data,
            )
        })
    }
    ///Wraps [`vkGetRayTracingCaptureReplayShaderGroupHandlesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetRayTracingCaptureReplayShaderGroupHandlesKHR.html).
    /**
    Provided by **VK_KHR_ray_tracing_pipeline**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves opaque capture/replay handles for shader groups. These
    ///handles allow recreating a ray tracing pipeline with identical
    ///shader group assignments on a subsequent run, enabling
    ///deterministic replay of GPU traces.
    ///
    ///Use this for tools, profilers, and capture-replay frameworks.
    ///The handles are passed back via
    ///`RayTracingShaderGroupCreateInfoKHR::shader_group_capture_replay_handle`
    ///when recreating the pipeline.
    ///
    ///The pipeline must have been created with
    ///`PIPELINE_CREATE_RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY`.
    ///Requires the `rayTracingPipelineShaderGroupHandleCaptureReplay`
    ///device feature.
    ///
    ///The buffer layout is the same as
    ///`get_ray_tracing_shader_group_handles_khr`, sequential handles
    ///of `shaderGroupHandleSize` bytes each.
    pub unsafe fn get_ray_tracing_capture_replay_shader_group_handles_khr(
        &self,
        pipeline: Pipeline,
        first_group: u32,
        group_count: u32,
        data_size: usize,
        p_data: *mut core::ffi::c_void,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_ray_tracing_capture_replay_shader_group_handles_khr
            .expect("vkGetRayTracingCaptureReplayShaderGroupHandlesKHR not loaded");
        check(unsafe {
            fp(
                self.handle(),
                pipeline,
                first_group,
                group_count,
                data_size,
                p_data,
            )
        })
    }
    ///Wraps [`vkGetAccelerationStructureHandleNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetAccelerationStructureHandleNV.html).
    /**
    Provided by **VK_NV_ray_tracing**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves an opaque handle for an NV acceleration structure. This
    ///handle is used when building a top-level acceleration structure
    ///that references bottom-level structures.
    ///
    ///Requires `VK_NV_ray_tracing`.
    pub unsafe fn get_acceleration_structure_handle_nv(
        &self,
        acceleration_structure: AccelerationStructureNV,
        data_size: usize,
        p_data: *mut core::ffi::c_void,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_acceleration_structure_handle_nv
            .expect("vkGetAccelerationStructureHandleNV not loaded");
        check(unsafe { fp(self.handle(), acceleration_structure, data_size, p_data) })
    }
    ///Wraps [`vkCreateRayTracingPipelinesNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateRayTracingPipelinesNV.html).
    /**
    Provided by **VK_NV_ray_tracing**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_INVALID_SHADER_NV`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `pipelineCache` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Creates one or more ray tracing pipelines using the NV model.
    ///This is the legacy NV path; prefer
    ///`create_ray_tracing_pipelines_khr` for new code.
    ///
    ///Supports a pipeline cache for faster subsequent creation.
    ///
    ///Requires `VK_NV_ray_tracing`.
    pub unsafe fn create_ray_tracing_pipelines_nv(
        &self,
        pipeline_cache: PipelineCache,
        p_create_infos: &[RayTracingPipelineCreateInfoNV],
        allocator: Option<&AllocationCallbacks>,
        p_pipelines: *mut Pipeline,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .create_ray_tracing_pipelines_nv
            .expect("vkCreateRayTracingPipelinesNV not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        check(unsafe {
            fp(
                self.handle(),
                pipeline_cache,
                p_create_infos.len() as u32,
                p_create_infos.as_ptr(),
                alloc_ptr,
                p_pipelines,
            )
        })
    }
    ///Wraps [`vkCreateRayTracingPipelinesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateRayTracingPipelinesKHR.html).
    /**
    Provided by **VK_KHR_ray_tracing_pipeline**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `pipelineCache` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Creates one or more ray tracing pipelines. A ray tracing pipeline
    ///contains the shader stages (ray generation, miss, closest hit,
    ///any hit, intersection, callable) and shader groups that define
    ///how rays interact with geometry.
    ///
    ///Unlike graphics pipelines, ray tracing pipelines organize shaders
    ///into **groups**:
    ///
    ///- **General**: ray generation, miss, or callable shaders.
    ///- **Triangles hit**: closest hit + optional any hit for triangles.
    ///- **Procedural hit**: intersection + closest hit + optional any hit
    ///  for custom geometry (AABBs).
    ///
    ///Pass a `DeferredOperationKHR` handle to compile asynchronously,
    ///the call returns `OPERATION_DEFERRED_KHR` and the pipeline handles
    ///are not valid until the deferred operation completes. Pass a null
    ///handle for synchronous creation.
    ///
    ///Supports `pipeline_cache` for faster creation on subsequent runs
    ///and `base_pipeline_handle` / `base_pipeline_index` for derivative
    ///pipelines when `PIPELINE_CREATE_DERIVATIVE` is set.
    ///
    ///After creation, retrieve shader group handles with
    ///`get_ray_tracing_shader_group_handles_khr` to build the shader
    ///binding table.
    pub unsafe fn create_ray_tracing_pipelines_khr(
        &self,
        deferred_operation: DeferredOperationKHR,
        pipeline_cache: PipelineCache,
        p_create_infos: &[RayTracingPipelineCreateInfoKHR],
        allocator: Option<&AllocationCallbacks>,
        p_pipelines: *mut Pipeline,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .create_ray_tracing_pipelines_khr
            .expect("vkCreateRayTracingPipelinesKHR not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        check(unsafe {
            fp(
                self.handle(),
                deferred_operation,
                pipeline_cache,
                p_create_infos.len() as u32,
                p_create_infos.as_ptr(),
                alloc_ptr,
                p_pipelines,
            )
        })
    }
    ///Wraps [`vkCmdTraceRaysIndirectKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdTraceRaysIndirectKHR.html).
    /**
    Provided by **VK_KHR_ray_tracing_pipeline**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dispatches rays with launch dimensions read from a GPU buffer.
    ///Identical to `cmd_trace_rays_khr` except the `width`, `height`,
    ///and `depth` are sourced from a `TraceRaysIndirectCommandKHR`
    ///struct at `indirect_device_address`.
    ///
    ///This enables the GPU to determine ray dispatch dimensions without
    ///a CPU round-trip, useful when the dispatch size depends on prior
    ///GPU work such as culling, tile classification, or adaptive
    ///sampling.
    ///
    ///The indirect buffer must have been created with
    ///`BUFFER_USAGE_INDIRECT_BUFFER` and the address must be aligned
    ///to 4 bytes. The SBT parameters are still provided directly on
    ///the CPU side.
    ///
    ///Requires the `rayTracingPipelineTraceRaysIndirect` feature.
    pub unsafe fn cmd_trace_rays_indirect_khr(
        &self,
        command_buffer: CommandBuffer,
        p_raygen_shader_binding_table: &StridedDeviceAddressRegionKHR,
        p_miss_shader_binding_table: &StridedDeviceAddressRegionKHR,
        p_hit_shader_binding_table: &StridedDeviceAddressRegionKHR,
        p_callable_shader_binding_table: &StridedDeviceAddressRegionKHR,
        indirect_device_address: u64,
    ) {
        let fp = self
            .commands()
            .cmd_trace_rays_indirect_khr
            .expect("vkCmdTraceRaysIndirectKHR not loaded");
        unsafe {
            fp(
                command_buffer,
                p_raygen_shader_binding_table,
                p_miss_shader_binding_table,
                p_hit_shader_binding_table,
                p_callable_shader_binding_table,
                indirect_device_address,
            )
        };
    }
    ///Wraps [`vkCmdTraceRaysIndirect2KHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdTraceRaysIndirect2KHR.html).
    /**
    Provided by **VK_KHR_ray_tracing_maintenance1**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Fully indirect ray dispatch, both the shader binding table
    ///addresses and the launch dimensions are read from a GPU buffer.
    ///This is the most flexible ray tracing dispatch.
    ///
    ///The `indirect_device_address` points to a
    ///`TraceRaysIndirectCommand2KHR` struct on the device, which
    ///contains all four SBT regions (raygen, miss, hit, callable) plus
    ///the `width`, `height`, and `depth`.
    ///
    ///This allows the GPU to dynamically select which shaders to use
    ///and how many rays to launch, enabling advanced techniques like
    ///GPU-driven material sorting or multi-pass ray tracing without
    ///CPU synchronization.
    ///
    ///Provided by `VK_KHR_ray_tracing_maintenance1`, not the base
    ///ray tracing pipeline extension. Requires the
    ///`rayTracingPipelineTraceRaysIndirect2` feature.
    pub unsafe fn cmd_trace_rays_indirect2_khr(
        &self,
        command_buffer: CommandBuffer,
        indirect_device_address: u64,
    ) {
        let fp = self
            .commands()
            .cmd_trace_rays_indirect2_khr
            .expect("vkCmdTraceRaysIndirect2KHR not loaded");
        unsafe { fp(command_buffer, indirect_device_address) };
    }
    ///Wraps [`vkGetClusterAccelerationStructureBuildSizesNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetClusterAccelerationStructureBuildSizesNV.html).
    /**
    Provided by **VK_NV_cluster_acceleration_structure**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the buffer sizes needed to build a cluster acceleration
    ///structure. Use the returned sizes to allocate the destination
    ///and scratch buffers before building.
    ///
    ///Requires `VK_NV_cluster_acceleration_structure`.
    pub unsafe fn get_cluster_acceleration_structure_build_sizes_nv(
        &self,
        p_info: &ClusterAccelerationStructureInputInfoNV,
        p_size_info: *mut AccelerationStructureBuildSizesInfoKHR,
    ) {
        let fp = self
            .commands()
            .get_cluster_acceleration_structure_build_sizes_nv
            .expect("vkGetClusterAccelerationStructureBuildSizesNV not loaded");
        unsafe { fp(self.handle(), p_info, p_size_info) };
    }
    ///Wraps [`vkCmdBuildClusterAccelerationStructureIndirectNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBuildClusterAccelerationStructureIndirectNV.html).
    /**
    Provided by **VK_NV_cluster_acceleration_structure**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Builds a cluster acceleration structure using indirect parameters.
    ///Cluster acceleration structures organize geometry into spatial
    ///clusters for more efficient ray traversal on NVIDIA hardware.
    ///
    ///Requires `VK_NV_cluster_acceleration_structure`.
    pub unsafe fn cmd_build_cluster_acceleration_structure_indirect_nv(
        &self,
        command_buffer: CommandBuffer,
        p_command_infos: &ClusterAccelerationStructureCommandsInfoNV,
    ) {
        let fp = self
            .commands()
            .cmd_build_cluster_acceleration_structure_indirect_nv
            .expect("vkCmdBuildClusterAccelerationStructureIndirectNV not loaded");
        unsafe { fp(command_buffer, p_command_infos) };
    }
    ///Wraps [`vkGetDeviceAccelerationStructureCompatibilityKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDeviceAccelerationStructureCompatibilityKHR.html).
    /**
    Provided by **VK_KHR_acceleration_structure**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Checks whether a serialized acceleration structure (from
    ///`copy_acceleration_structure_to_memory_khr`) is compatible with
    ///this device and can be deserialized.
    ///
    ///Returns `ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE` if the
    ///data can be loaded, or `INCOMPATIBLE` if not. Incompatibility
    ///typically means the data was serialized on different hardware or a
    ///different driver version.
    ///
    ///Check compatibility before attempting deserialization to avoid
    ///errors.
    pub unsafe fn get_device_acceleration_structure_compatibility_khr(
        &self,
        p_version_info: &AccelerationStructureVersionInfoKHR,
    ) -> AccelerationStructureCompatibilityKHR {
        let fp = self
            .commands()
            .get_device_acceleration_structure_compatibility_khr
            .expect("vkGetDeviceAccelerationStructureCompatibilityKHR not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(self.handle(), p_version_info, &mut out) };
        out
    }
    ///Wraps [`vkGetRayTracingShaderGroupStackSizeKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetRayTracingShaderGroupStackSizeKHR.html).
    /**
    Provided by **VK_KHR_ray_tracing_pipeline**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the stack size contribution of a single shader within a
    ///shader group. The result is used to compute the total pipeline
    ///stack size for `cmd_set_ray_tracing_pipeline_stack_size_khr`.
    ///
    ///`group` indexes into the shader groups array from pipeline
    ///creation. `group_shader` selects which shader within the group:
    ///`GENERAL`, `CLOSEST_HIT`, `ANY_HIT`, or `INTERSECTION`.
    ///
    ///The default pipeline stack size is computed automatically at
    ///creation time, but it assumes worst-case recursion. If you know
    ///your actual `maxPipelineRayRecursionDepth` is lower, query
    ///individual stack sizes and compute a tighter total to reduce
    ///scratch memory usage.
    ///
    ///Stack size computation formula (from spec):
    ///
    ///`raygen + max(closesthit + intersection, miss, callable) * maxRecursionDepth`
    ///
    ///Call this per-shader, aggregate across all groups, then set the
    ///result with `cmd_set_ray_tracing_pipeline_stack_size_khr`.
    pub unsafe fn get_ray_tracing_shader_group_stack_size_khr(
        &self,
        pipeline: Pipeline,
        group: u32,
        group_shader: ShaderGroupShaderKHR,
    ) {
        let fp = self
            .commands()
            .get_ray_tracing_shader_group_stack_size_khr
            .expect("vkGetRayTracingShaderGroupStackSizeKHR not loaded");
        unsafe { fp(self.handle(), pipeline, group, group_shader) };
    }
    ///Wraps [`vkCmdSetRayTracingPipelineStackSizeKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetRayTracingPipelineStackSizeKHR.html).
    /**
    Provided by **VK_KHR_ray_tracing_pipeline**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Overrides the default ray tracing pipeline stack size for the
    ///bound pipeline. The stack is scratch memory used during shader
    ///execution and recursion.
    ///
    ///The default stack size (set at pipeline creation) assumes
    ///worst-case recursion depth across all shader groups. If your
    ///application uses a lower effective recursion depth or only a
    ///subset of shader groups, setting a smaller stack size reduces
    ///per-invocation memory usage and may improve occupancy.
    ///
    ///Compute the required size by querying individual shader
    ///contributions with `get_ray_tracing_shader_group_stack_size_khr`
    ///and applying the recursion formula from the spec.
    ///
    ///This is a dynamic state command, it takes effect for subsequent
    ///`cmd_trace_rays_khr` calls within the same command buffer.
    ///Binding a new pipeline resets the stack size to the pipeline's
    ///default.
    pub unsafe fn cmd_set_ray_tracing_pipeline_stack_size_khr(
        &self,
        command_buffer: CommandBuffer,
        pipeline_stack_size: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_ray_tracing_pipeline_stack_size_khr
            .expect("vkCmdSetRayTracingPipelineStackSizeKHR not loaded");
        unsafe { fp(command_buffer, pipeline_stack_size) };
    }
    ///Wraps [`vkGetImageViewHandleNVX`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetImageViewHandleNVX.html).
    /**
    Provided by **VK_NVX_image_view_handle**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns a 32-bit handle for an image view that can be used as a
    ///bindless descriptor index. Use with `get_image_view_address_nvx`
    ///for fully bindless texture access.
    ///
    ///Requires `VK_NVX_image_view_handle`.
    pub unsafe fn get_image_view_handle_nvx(&self, p_info: &ImageViewHandleInfoNVX) {
        let fp = self
            .commands()
            .get_image_view_handle_nvx
            .expect("vkGetImageViewHandleNVX not loaded");
        unsafe { fp(self.handle(), p_info) };
    }
    ///Wraps [`vkGetImageViewHandle64NVX`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetImageViewHandle64NVX.html).
    /**
    Provided by **VK_NVX_image_view_handle**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns a 64-bit handle for an image view. The 64-bit variant
    ///accommodates larger descriptor heaps than the 32-bit
    ///`get_image_view_handle_nvx`.
    ///
    ///Requires `VK_NVX_image_view_handle`.
    pub unsafe fn get_image_view_handle64_nvx(&self, p_info: &ImageViewHandleInfoNVX) {
        let fp = self
            .commands()
            .get_image_view_handle64_nvx
            .expect("vkGetImageViewHandle64NVX not loaded");
        unsafe { fp(self.handle(), p_info) };
    }
    ///Wraps [`vkGetImageViewAddressNVX`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetImageViewAddressNVX.html).
    /**
    Provided by **VK_NVX_image_view_handle**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the device address and size of an image view's descriptor
    ///data. The address can be used for raw pointer-based descriptor
    ///access in shaders.
    ///
    ///Requires `VK_NVX_image_view_handle`.
    pub unsafe fn get_image_view_address_nvx(
        &self,
        image_view: ImageView,
        p_properties: *mut ImageViewAddressPropertiesNVX,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_image_view_address_nvx
            .expect("vkGetImageViewAddressNVX not loaded");
        check(unsafe { fp(self.handle(), image_view, p_properties) })
    }
    ///Wraps [`vkGetDeviceCombinedImageSamplerIndexNVX`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDeviceCombinedImageSamplerIndexNVX.html).
    /**
    Provided by **VK_NVX_image_view_handle**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns the combined image-sampler descriptor index for a given
    ///image view and sampler pair. Used for bindless combined image-
    ///sampler access.
    ///
    ///Requires `VK_NVX_image_view_handle`.
    pub unsafe fn get_device_combined_image_sampler_index_nvx(
        &self,
        image_view_index: u64,
        sampler_index: u64,
    ) {
        let fp = self
            .commands()
            .get_device_combined_image_sampler_index_nvx
            .expect("vkGetDeviceCombinedImageSamplerIndexNVX not loaded");
        unsafe { fp(self.handle(), image_view_index, sampler_index) };
    }
    ///Wraps [`vkGetDeviceGroupSurfacePresentModes2EXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDeviceGroupSurfacePresentModes2EXT.html).
    /**
    Provided by **VK_EXT_full_screen_exclusive**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the supported present modes for a device group and surface,
    ///using the extended surface info structure. This is the
    ///`VK_EXT_full_screen_exclusive` variant of
    ///`get_device_group_surface_present_modes_khr`, allowing full-screen
    ///exclusive configuration to be factored into the query.
    ///
    ///Requires `VK_EXT_full_screen_exclusive`. Windows only.
    pub unsafe fn get_device_group_surface_present_modes2_ext(
        &self,
        p_surface_info: &PhysicalDeviceSurfaceInfo2KHR,
    ) -> VkResult<DeviceGroupPresentModeFlagsKHR> {
        let fp = self
            .commands()
            .get_device_group_surface_present_modes2_ext
            .expect("vkGetDeviceGroupSurfacePresentModes2EXT not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_surface_info, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkAcquireFullScreenExclusiveModeEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkAcquireFullScreenExclusiveModeEXT.html).
    /**
    Provided by **VK_EXT_full_screen_exclusive**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Acquires full-screen exclusive mode for a swapchain, giving the
    ///application direct control over the display output. This can
    ///reduce latency and enable adaptive sync.
    ///
    ///The swapchain must have been created with
    ///`SurfaceFullScreenExclusiveInfoEXT` in its pNext chain.
    ///Release with `release_full_screen_exclusive_mode_ext`.
    ///
    ///Requires `VK_EXT_full_screen_exclusive`. Windows only.
    pub unsafe fn acquire_full_screen_exclusive_mode_ext(
        &self,
        swapchain: SwapchainKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .acquire_full_screen_exclusive_mode_ext
            .expect("vkAcquireFullScreenExclusiveModeEXT not loaded");
        check(unsafe { fp(self.handle(), swapchain) })
    }
    ///Wraps [`vkReleaseFullScreenExclusiveModeEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkReleaseFullScreenExclusiveModeEXT.html).
    /**
    Provided by **VK_EXT_full_screen_exclusive**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Releases full-screen exclusive mode previously acquired with
    ///`acquire_full_screen_exclusive_mode_ext`. The swapchain returns
    ///to shared/composed presentation.
    ///
    ///Requires `VK_EXT_full_screen_exclusive`. Windows only.
    pub unsafe fn release_full_screen_exclusive_mode_ext(
        &self,
        swapchain: SwapchainKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .release_full_screen_exclusive_mode_ext
            .expect("vkReleaseFullScreenExclusiveModeEXT not loaded");
        check(unsafe { fp(self.handle(), swapchain) })
    }
    ///Wraps [`vkAcquireProfilingLockKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkAcquireProfilingLockKHR.html).
    /**
    Provided by **VK_KHR_performance_query**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_TIMEOUT`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Acquires the device profiling lock, which must be held while
    ///submitting command buffers that contain performance queries.
    ///Only one thread can hold the lock at a time.
    ///
    ///The `AcquireProfilingLockInfoKHR` specifies a timeout in
    ///nanoseconds. Returns `TIMEOUT` if the lock cannot be acquired
    ///within that period.
    ///
    ///Release with `release_profiling_lock_khr` when profiling
    ///submission is complete.
    ///
    ///Requires `VK_KHR_performance_query`.
    pub unsafe fn acquire_profiling_lock_khr(
        &self,
        p_info: &AcquireProfilingLockInfoKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .acquire_profiling_lock_khr
            .expect("vkAcquireProfilingLockKHR not loaded");
        check(unsafe { fp(self.handle(), p_info) })
    }
    ///Wraps [`vkReleaseProfilingLockKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkReleaseProfilingLockKHR.html).
    /**
    Provided by **VK_KHR_performance_query**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Releases the device profiling lock previously acquired with
    ///`acquire_profiling_lock_khr`. Must be called after all command
    ///buffers containing performance queries have been submitted.
    ///
    ///Requires `VK_KHR_performance_query`.
    pub unsafe fn release_profiling_lock_khr(&self) {
        let fp = self
            .commands()
            .release_profiling_lock_khr
            .expect("vkReleaseProfilingLockKHR not loaded");
        unsafe { fp(self.handle()) };
    }
    ///Wraps [`vkGetImageDrmFormatModifierPropertiesEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetImageDrmFormatModifierPropertiesEXT.html).
    /**
    Provided by **VK_EXT_image_drm_format_modifier**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries which DRM format modifier was selected for an image
    ///created with `VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT`. The
    ///chosen modifier determines the memory layout and is needed
    ///when sharing the image with other DRM/KMS clients.
    ///
    ///Requires `VK_EXT_image_drm_format_modifier`.
    pub unsafe fn get_image_drm_format_modifier_properties_ext(
        &self,
        image: Image,
        p_properties: *mut ImageDrmFormatModifierPropertiesEXT,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_image_drm_format_modifier_properties_ext
            .expect("vkGetImageDrmFormatModifierPropertiesEXT not loaded");
        check(unsafe { fp(self.handle(), image, p_properties) })
    }
    ///Wraps [`vkGetBufferOpaqueCaptureAddress`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetBufferOpaqueCaptureAddress.html).
    /**
    Provided by **VK_BASE_VERSION_1_2**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns an opaque capture address for a buffer that was created with
    ///`BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY`. This address is used
    ///to recreate the buffer at the same virtual address in a replay
    ///session (e.g. for GPU crash dump replay or deterministic replay
    ///tools).
    ///
    ///Most applications do not need this, it is primarily for debugging
    ///and profiling tools. Use `get_buffer_device_address` for runtime
    ///buffer address access.
    pub unsafe fn get_buffer_opaque_capture_address(&self, p_info: &BufferDeviceAddressInfo) {
        let fp = self
            .commands()
            .get_buffer_opaque_capture_address
            .expect("vkGetBufferOpaqueCaptureAddress not loaded");
        unsafe { fp(self.handle(), p_info) };
    }
    ///Wraps [`vkGetBufferDeviceAddress`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetBufferDeviceAddress.html).
    /**
    Provided by **VK_BASE_VERSION_1_2**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns a 64-bit GPU virtual address for a buffer. This address can
    ///be passed to shaders via push constants or descriptors, enabling
    ///direct pointer-style access to buffer data from GPU code.
    ///
    ///The buffer must have been created with
    ///`BUFFER_USAGE_SHADER_DEVICE_ADDRESS` and the
    ///`buffer_device_address` feature must be enabled.
    ///
    ///**Use cases**:
    ///
    ///- **Bindless rendering**: pass buffer addresses in a storage buffer
    ///  or push constant instead of binding individual descriptors.
    ///- **Acceleration structures**: ray tracing BLASes and TLASes
    ///  reference geometry buffers by device address.
    ///- **GPU-driven pipelines**: indirect command generators read vertex
    ///  and index data by address.
    ///
    ///The address remains valid for the lifetime of the buffer. If the
    ///buffer was created with
    ///`BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY`, the address can be
    ///captured and replayed across sessions.
    pub unsafe fn get_buffer_device_address(&self, p_info: &BufferDeviceAddressInfo) {
        let fp = self
            .commands()
            .get_buffer_device_address
            .expect("vkGetBufferDeviceAddress not loaded");
        unsafe { fp(self.handle(), p_info) };
    }
    ///Wraps [`vkInitializePerformanceApiINTEL`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkInitializePerformanceApiINTEL.html).
    /**
    Provided by **VK_INTEL_performance_query**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_TOO_MANY_OBJECTS`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Initializes the Intel performance query API on a device. Must be
    ///called before using any other Intel performance query commands.
    ///Uninitialize with `uninitialize_performance_api_intel`.
    ///
    ///Requires `VK_INTEL_performance_query`.
    pub unsafe fn initialize_performance_api_intel(
        &self,
        p_initialize_info: &InitializePerformanceApiInfoINTEL,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .initialize_performance_api_intel
            .expect("vkInitializePerformanceApiINTEL not loaded");
        check(unsafe { fp(self.handle(), p_initialize_info) })
    }
    ///Wraps [`vkUninitializePerformanceApiINTEL`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkUninitializePerformanceApiINTEL.html).
    /**
    Provided by **VK_INTEL_performance_query**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Shuts down the Intel performance query API on a device, releasing
    ///any internal resources. Call when performance profiling is
    ///complete.
    ///
    ///Requires `VK_INTEL_performance_query`.
    pub unsafe fn uninitialize_performance_api_intel(&self) {
        let fp = self
            .commands()
            .uninitialize_performance_api_intel
            .expect("vkUninitializePerformanceApiINTEL not loaded");
        unsafe { fp(self.handle()) };
    }
    ///Wraps [`vkCmdSetPerformanceMarkerINTEL`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetPerformanceMarkerINTEL.html).
    /**
    Provided by **VK_INTEL_performance_query**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_TOO_MANY_OBJECTS`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Sets a performance marker in the command buffer to delimit a
    ///region of interest for Intel GPU profiling. Counters sampled
    ///between markers are attributed to the marked region.
    ///
    ///Requires `VK_INTEL_performance_query`.
    pub unsafe fn cmd_set_performance_marker_intel(
        &self,
        command_buffer: CommandBuffer,
        p_marker_info: &PerformanceMarkerInfoINTEL,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .cmd_set_performance_marker_intel
            .expect("vkCmdSetPerformanceMarkerINTEL not loaded");
        check(unsafe { fp(command_buffer, p_marker_info) })
    }
    ///Wraps [`vkCmdSetPerformanceStreamMarkerINTEL`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetPerformanceStreamMarkerINTEL.html).
    /**
    Provided by **VK_INTEL_performance_query**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_TOO_MANY_OBJECTS`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Sets a performance stream marker in the command buffer. Stream
    ///markers identify points in the command stream for correlating
    ///performance counter data with specific GPU work.
    ///
    ///Requires `VK_INTEL_performance_query`.
    pub unsafe fn cmd_set_performance_stream_marker_intel(
        &self,
        command_buffer: CommandBuffer,
        p_marker_info: &PerformanceStreamMarkerInfoINTEL,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .cmd_set_performance_stream_marker_intel
            .expect("vkCmdSetPerformanceStreamMarkerINTEL not loaded");
        check(unsafe { fp(command_buffer, p_marker_info) })
    }
    ///Wraps [`vkCmdSetPerformanceOverrideINTEL`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetPerformanceOverrideINTEL.html).
    /**
    Provided by **VK_INTEL_performance_query**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_TOO_MANY_OBJECTS`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Overrides hardware performance settings for the remainder of the
    ///command buffer (e.g., forcing specific EU thread counts). Used
    ///for controlled profiling experiments.
    ///
    ///Requires `VK_INTEL_performance_query`.
    pub unsafe fn cmd_set_performance_override_intel(
        &self,
        command_buffer: CommandBuffer,
        p_override_info: &PerformanceOverrideInfoINTEL,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .cmd_set_performance_override_intel
            .expect("vkCmdSetPerformanceOverrideINTEL not loaded");
        check(unsafe { fp(command_buffer, p_override_info) })
    }
    ///Wraps [`vkAcquirePerformanceConfigurationINTEL`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkAcquirePerformanceConfigurationINTEL.html).
    /**
    Provided by **VK_INTEL_performance_query**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_TOO_MANY_OBJECTS`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Acquires a performance configuration that enables specific
    ///hardware counters for profiling. Apply to a queue with
    ///`queue_set_performance_configuration_intel`. Release with
    ///`release_performance_configuration_intel`.
    ///
    ///Requires `VK_INTEL_performance_query`.
    pub unsafe fn acquire_performance_configuration_intel(
        &self,
        p_acquire_info: &PerformanceConfigurationAcquireInfoINTEL,
    ) -> VkResult<PerformanceConfigurationINTEL> {
        let fp = self
            .commands()
            .acquire_performance_configuration_intel
            .expect("vkAcquirePerformanceConfigurationINTEL not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_acquire_info, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkReleasePerformanceConfigurationINTEL`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkReleasePerformanceConfigurationINTEL.html).
    /**
    Provided by **VK_INTEL_performance_query**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_TOO_MANY_OBJECTS`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `configuration` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Releases a performance configuration acquired with
    ///`acquire_performance_configuration_intel`.
    ///
    ///Requires `VK_INTEL_performance_query`.
    pub unsafe fn release_performance_configuration_intel(
        &self,
        configuration: PerformanceConfigurationINTEL,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .release_performance_configuration_intel
            .expect("vkReleasePerformanceConfigurationINTEL not loaded");
        check(unsafe { fp(self.handle(), configuration) })
    }
    ///Wraps [`vkQueueSetPerformanceConfigurationINTEL`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkQueueSetPerformanceConfigurationINTEL.html).
    /**
    Provided by **VK_INTEL_performance_query**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_TOO_MANY_OBJECTS`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `queue` (self) must be valid and not destroyed.
    ///- `queue` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Applies a performance configuration to a queue, enabling the
    ///selected hardware counters for all subsequent submissions to
    ///that queue.
    ///
    ///Requires `VK_INTEL_performance_query`.
    pub unsafe fn queue_set_performance_configuration_intel(
        &self,
        queue: Queue,
        configuration: PerformanceConfigurationINTEL,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .queue_set_performance_configuration_intel
            .expect("vkQueueSetPerformanceConfigurationINTEL not loaded");
        check(unsafe { fp(queue, configuration) })
    }
    ///Wraps [`vkGetPerformanceParameterINTEL`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPerformanceParameterINTEL.html).
    /**
    Provided by **VK_INTEL_performance_query**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_TOO_MANY_OBJECTS`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries a performance parameter value from the Intel driver, such
    ///as GPU clock frequency or EU count. Useful for normalizing
    ///profiling results.
    ///
    ///Requires `VK_INTEL_performance_query`.
    pub unsafe fn get_performance_parameter_intel(
        &self,
        parameter: PerformanceParameterTypeINTEL,
    ) -> VkResult<PerformanceValueINTEL> {
        let fp = self
            .commands()
            .get_performance_parameter_intel
            .expect("vkGetPerformanceParameterINTEL not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), parameter, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetDeviceMemoryOpaqueCaptureAddress`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDeviceMemoryOpaqueCaptureAddress.html).
    /**
    Provided by **VK_BASE_VERSION_1_2**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns an opaque capture address for a device memory allocation
    ///that was created with
    ///`MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY`. Used in conjunction
    ///with `get_buffer_opaque_capture_address` to replay buffer address
    ///assignments.
    ///
    ///This is a debugging/replay tool feature. Most applications do not
    ///need this.
    pub unsafe fn get_device_memory_opaque_capture_address(
        &self,
        p_info: &DeviceMemoryOpaqueCaptureAddressInfo,
    ) {
        let fp = self
            .commands()
            .get_device_memory_opaque_capture_address
            .expect("vkGetDeviceMemoryOpaqueCaptureAddress not loaded");
        unsafe { fp(self.handle(), p_info) };
    }
    ///Wraps [`vkGetPipelineExecutablePropertiesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPipelineExecutablePropertiesKHR.html).
    /**
    Provided by **VK_KHR_pipeline_executable_properties**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Lists the executable components within a pipeline. A single
    ///pipeline may contain multiple executables, for example, a
    ///graphics pipeline typically has separate vertex and fragment
    ///shader executables.
    ///
    ///Each returned `PipelineExecutablePropertiesKHR` contains a name,
    ///description, and shader stage flags identifying the executable.
    ///Use these to index into `get_pipeline_executable_statistics_khr`
    ///and `get_pipeline_executable_internal_representations_khr`.
    ///
    ///The pipeline must have been created with
    ///`PIPELINE_CREATE_CAPTURE_STATISTICS` or
    ///`PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS`. This is a
    ///debugging and profiling tool, not intended for shipping builds.
    pub unsafe fn get_pipeline_executable_properties_khr(
        &self,
        p_pipeline_info: &PipelineInfoKHR,
    ) -> VkResult<Vec<PipelineExecutablePropertiesKHR>> {
        let fp = self
            .commands()
            .get_pipeline_executable_properties_khr
            .expect("vkGetPipelineExecutablePropertiesKHR not loaded");
        enumerate_two_call(|count, data| unsafe { fp(self.handle(), p_pipeline_info, count, data) })
    }
    ///Wraps [`vkGetPipelineExecutableStatisticsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPipelineExecutableStatisticsKHR.html).
    /**
    Provided by **VK_KHR_pipeline_executable_properties**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves compiler statistics for a specific pipeline executable.
    ///Statistics include metrics like register usage, instruction count,
    ///scratch memory, and other driver-specific values.
    ///
    ///Identify the executable by index from
    ///`get_pipeline_executable_properties_khr` via
    ///`PipelineExecutableInfoKHR`.
    ///
    ///Each statistic has a name, description, format (bool, int, float,
    ///or string), and value. The available statistics are
    ///driver-specific, different vendors report different metrics.
    ///
    ///The pipeline must have been created with
    ///`PIPELINE_CREATE_CAPTURE_STATISTICS`. This is a profiling tool
    ///for shader optimization, use it to compare register pressure
    ///or instruction counts across shader variants.
    pub unsafe fn get_pipeline_executable_statistics_khr(
        &self,
        p_executable_info: &PipelineExecutableInfoKHR,
    ) -> VkResult<Vec<PipelineExecutableStatisticKHR>> {
        let fp = self
            .commands()
            .get_pipeline_executable_statistics_khr
            .expect("vkGetPipelineExecutableStatisticsKHR not loaded");
        enumerate_two_call(|count, data| unsafe {
            fp(self.handle(), p_executable_info, count, data)
        })
    }
    ///Wraps [`vkGetPipelineExecutableInternalRepresentationsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPipelineExecutableInternalRepresentationsKHR.html).
    /**
    Provided by **VK_KHR_pipeline_executable_properties**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves internal representations (IR) of a pipeline executable.
    ///These are driver-specific intermediate or final shader
    ///representations, for example, SPIR-V, vendor IR, or GPU ISA
    ///disassembly.
    ///
    ///Each representation has a name, description, and opaque data
    ///blob. Whether the data is human-readable text or binary depends
    ///on `is_text` in the returned structure.
    ///
    ///The pipeline must have been created with
    ///`PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS`. Enabling
    ///this flag may disable optimizations, so only use it for
    ///debugging and shader analysis, not in production.
    ///
    ///Identify the executable by index from
    ///`get_pipeline_executable_properties_khr`.
    pub unsafe fn get_pipeline_executable_internal_representations_khr(
        &self,
        p_executable_info: &PipelineExecutableInfoKHR,
    ) -> VkResult<Vec<PipelineExecutableInternalRepresentationKHR>> {
        let fp = self
            .commands()
            .get_pipeline_executable_internal_representations_khr
            .expect("vkGetPipelineExecutableInternalRepresentationsKHR not loaded");
        enumerate_two_call(|count, data| unsafe {
            fp(self.handle(), p_executable_info, count, data)
        })
    }
    ///Wraps [`vkCmdSetLineStipple`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetLineStipple.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_4**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets the line stipple pattern and repeat factor. Only
    ///takes effect if the pipeline was created with
    ///`DYNAMIC_STATE_LINE_STIPPLE`.
    ///
    ///The `stipple_factor` (1–256) controls how many pixels each bit of
    ///the pattern spans. The `stipple_pattern` is a 16-bit bitmask where
    ///each bit represents a pixel, 1 is drawn, 0 is discarded.
    ///
    ///Line stippling requires `VK_EXT_line_rasterization` and the
    ///`stippled_*_lines` device features, depending on which line
    ///rasterisation mode you use.
    ///
    ///Core dynamic state in Vulkan 1.4.
    pub unsafe fn cmd_set_line_stipple(
        &self,
        command_buffer: CommandBuffer,
        line_stipple_factor: u32,
        line_stipple_pattern: u16,
    ) {
        let fp = self
            .commands()
            .cmd_set_line_stipple
            .expect("vkCmdSetLineStipple not loaded");
        unsafe { fp(command_buffer, line_stipple_factor, line_stipple_pattern) };
    }
    ///Wraps [`vkGetFaultData`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetFaultData.html).
    /**
    Provided by **VKSC_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries recorded fault data from the device. Part of Vulkan SC
    ///(Safety Critical) for fault reporting in safety-certified
    ///environments. Uses the two-call idiom. The
    ///`fault_query_behavior` controls whether queried faults are
    ///cleared. Also reports the count of unrecorded faults that
    ///overflowed the internal buffer.
    ///
    ///Requires Vulkan SC.
    pub unsafe fn get_fault_data(
        &self,
        fault_query_behavior: FaultQueryBehavior,
        p_unrecorded_faults: *mut u32,
    ) -> VkResult<Vec<FaultData>> {
        let fp = self
            .commands()
            .get_fault_data
            .expect("vkGetFaultData not loaded");
        enumerate_two_call(|count, data| unsafe {
            fp(
                self.handle(),
                fault_query_behavior,
                p_unrecorded_faults,
                count,
                data,
            )
        })
    }
    ///Wraps [`vkCreateAccelerationStructureKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateAccelerationStructureKHR.html).
    /**
    Provided by **VK_KHR_acceleration_structure**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates an acceleration structure for hardware ray tracing. An
    ///acceleration structure is a spatial data structure (typically a BVH)
    ///that the GPU traverses during ray intersection tests.
    ///
    ///**Two levels**:
    ///
    ///- **Bottom-level (BLAS)**: contains geometry (triangles or AABBs).
    ///  Create one per mesh or mesh group.
    ///- **Top-level (TLAS)**: contains instances that reference BLASes
    ///  with per-instance transforms. Create one per scene.
    ///
    ///The acceleration structure needs a backing buffer created with
    ///`BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE`. Query the required
    ///size with `get_acceleration_structure_build_sizes_khr` first.
    ///
    ///After creation, build the structure with
    ///`cmd_build_acceleration_structures_khr`. The structure is not usable
    ///for tracing until built.
    pub unsafe fn create_acceleration_structure_khr(
        &self,
        p_create_info: &AccelerationStructureCreateInfoKHR,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<AccelerationStructureKHR> {
        let fp = self
            .commands()
            .create_acceleration_structure_khr
            .expect("vkCreateAccelerationStructureKHR not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkCmdBuildAccelerationStructuresKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBuildAccelerationStructuresKHR.html).
    /**
    Provided by **VK_KHR_acceleration_structure**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Records a GPU-side acceleration structure build or update. This is
    ///the primary way to build BLASes and TLASes for ray tracing.
    ///
    ///**Build vs update**: an initial build creates the structure from
    ///scratch. An update (`BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE`)
    ///modifies an existing structure in-place, which is faster but
    ///produces lower traversal quality. Use updates for dynamic geometry
    ///(e.g. animated characters) and full rebuilds when geometry changes
    ///significantly.
    ///
    ///**Scratch buffer**: builds require a temporary scratch buffer.
    ///Query the required size with
    ///`get_acceleration_structure_build_sizes_khr` and create a buffer
    ///with `BUFFER_USAGE_STORAGE_BUFFER`.
    ///
    ///Multiple builds can be batched in a single call. The driver may
    ///execute them in parallel.
    ///
    ///Must be recorded outside a render pass.
    pub unsafe fn cmd_build_acceleration_structures_khr(
        &self,
        command_buffer: CommandBuffer,
        p_infos: &[AccelerationStructureBuildGeometryInfoKHR],
        pp_build_range_infos: *const *const AccelerationStructureBuildRangeInfoKHR,
    ) {
        let fp = self
            .commands()
            .cmd_build_acceleration_structures_khr
            .expect("vkCmdBuildAccelerationStructuresKHR not loaded");
        unsafe {
            fp(
                command_buffer,
                p_infos.len() as u32,
                p_infos.as_ptr(),
                pp_build_range_infos,
            )
        };
    }
    ///Wraps [`vkCmdBuildAccelerationStructuresIndirectKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBuildAccelerationStructuresIndirectKHR.html).
    /**
    Provided by **VK_KHR_acceleration_structure**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///GPU-side acceleration structure build with indirect parameters. The
    ///primitive counts and build ranges are read from GPU buffers rather
    ///than specified on the CPU.
    ///
    ///This enables fully GPU-driven scene management where a compute
    ///shader determines which geometry to include and writes the build
    ///parameters.
    ///
    ///Requires the `acceleration_structure_indirect_build` feature.
    pub unsafe fn cmd_build_acceleration_structures_indirect_khr(
        &self,
        command_buffer: CommandBuffer,
        p_infos: &[AccelerationStructureBuildGeometryInfoKHR],
        p_indirect_device_addresses: &u64,
        p_indirect_strides: *const u32,
        pp_max_primitive_counts: *const *const u32,
    ) {
        let fp = self
            .commands()
            .cmd_build_acceleration_structures_indirect_khr
            .expect("vkCmdBuildAccelerationStructuresIndirectKHR not loaded");
        unsafe {
            fp(
                command_buffer,
                p_infos.len() as u32,
                p_infos.as_ptr(),
                p_indirect_device_addresses,
                p_indirect_strides,
                pp_max_primitive_counts,
            )
        };
    }
    ///Wraps [`vkBuildAccelerationStructuresKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkBuildAccelerationStructuresKHR.html).
    /**
    Provided by **VK_KHR_acceleration_structure**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Builds or updates acceleration structures on the **host** (CPU).
    ///This is the CPU-side alternative to
    ///`cmd_build_acceleration_structures_khr`.
    ///
    ///Host builds are useful for offline processing, tools, or when GPU
    ///build capacity is limited. However, GPU builds are significantly
    ///faster for real-time applications.
    ///
    ///Requires the `acceleration_structure_host_commands` feature.
    pub unsafe fn build_acceleration_structures_khr(
        &self,
        deferred_operation: DeferredOperationKHR,
        p_infos: &[AccelerationStructureBuildGeometryInfoKHR],
        pp_build_range_infos: *const *const AccelerationStructureBuildRangeInfoKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .build_acceleration_structures_khr
            .expect("vkBuildAccelerationStructuresKHR not loaded");
        check(unsafe {
            fp(
                self.handle(),
                deferred_operation,
                p_infos.len() as u32,
                p_infos.as_ptr(),
                pp_build_range_infos,
            )
        })
    }
    ///Wraps [`vkGetAccelerationStructureDeviceAddressKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetAccelerationStructureDeviceAddressKHR.html).
    /**
    Provided by **VK_KHR_acceleration_structure**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns the GPU device address of an acceleration structure. This
    ///address is used when building a TLAS, each instance in the TLAS
    ///references a BLAS by its device address.
    ///
    ///The address remains valid for the lifetime of the acceleration
    ///structure.
    pub unsafe fn get_acceleration_structure_device_address_khr(
        &self,
        p_info: &AccelerationStructureDeviceAddressInfoKHR,
    ) {
        let fp = self
            .commands()
            .get_acceleration_structure_device_address_khr
            .expect("vkGetAccelerationStructureDeviceAddressKHR not loaded");
        unsafe { fp(self.handle(), p_info) };
    }
    ///Wraps [`vkCreateDeferredOperationKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateDeferredOperationKHR.html).
    /**
    Provided by **VK_KHR_deferred_host_operations**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a deferred operation handle. Deferred operations allow
    ///expensive host-side work (such as ray tracing pipeline compilation)
    ///to be split across multiple CPU threads.
    ///
    ///The typical workflow:
    ///
    ///1. Create a deferred operation with this command.
    ///2. Pass the handle to a deferrable command (e.g.,
    ///   `create_ray_tracing_pipelines_khr`). If deferred, it returns
    ///   `OPERATION_DEFERRED_KHR`.
    ///3. Query `get_deferred_operation_max_concurrency_khr` to learn
    ///   how many threads can contribute.
    ///4. Call `deferred_operation_join_khr` from each worker thread.
    ///5. Once all joins return `SUCCESS`, retrieve the result with
    ///   `get_deferred_operation_result_khr`.
    ///6. Destroy the handle with `destroy_deferred_operation_khr`.
    ///
    ///The handle itself is lightweight, it is just a token for tracking
    ///the deferred work.
    pub unsafe fn create_deferred_operation_khr(
        &self,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<DeferredOperationKHR> {
        let fp = self
            .commands()
            .create_deferred_operation_khr
            .expect("vkCreateDeferredOperationKHR not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyDeferredOperationKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyDeferredOperationKHR.html).
    /**
    Provided by **VK_KHR_deferred_host_operations**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `operation` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a deferred operation handle. The operation must have
    ///completed before destruction, either all joining threads returned
    ///`SUCCESS` or `THREAD_DONE_KHR`, or the operation was never
    ///deferred.
    ///
    ///Do not destroy while threads are still joined to the operation.
    pub unsafe fn destroy_deferred_operation_khr(
        &self,
        operation: DeferredOperationKHR,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_deferred_operation_khr
            .expect("vkDestroyDeferredOperationKHR not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), operation, alloc_ptr) };
    }
    ///Wraps [`vkGetDeferredOperationMaxConcurrencyKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDeferredOperationMaxConcurrencyKHR.html).
    /**
    Provided by **VK_KHR_deferred_host_operations**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns the maximum number of threads that can usefully join this
    ///deferred operation. Spawning more threads than this value wastes
    ///resources, additional joins will return `THREAD_IDLE_KHR`.
    ///
    ///The returned value may decrease over time as work completes, so
    ///query it just before spawning worker threads.
    ///
    ///A return value of zero means the operation is already complete
    ///or requires no additional threads.
    pub unsafe fn get_deferred_operation_max_concurrency_khr(
        &self,
        operation: DeferredOperationKHR,
    ) {
        let fp = self
            .commands()
            .get_deferred_operation_max_concurrency_khr
            .expect("vkGetDeferredOperationMaxConcurrencyKHR not loaded");
        unsafe { fp(self.handle(), operation) };
    }
    ///Wraps [`vkGetDeferredOperationResultKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDeferredOperationResultKHR.html).
    /**
    Provided by **VK_KHR_deferred_host_operations**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns the result of a completed deferred operation. The returned
    ///`VkResult` is the same value that the original deferrable command
    ///would have returned if it had executed synchronously.
    ///
    ///Only call this after the operation has fully completed (all joins
    ///returned `SUCCESS` or `THREAD_DONE_KHR`). Calling on an
    ///in-progress operation returns `NOT_READY`.
    ///
    ///For example, if `create_ray_tracing_pipelines_khr` was deferred,
    ///this returns whether pipeline creation succeeded or failed.
    pub unsafe fn get_deferred_operation_result_khr(
        &self,
        operation: DeferredOperationKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_deferred_operation_result_khr
            .expect("vkGetDeferredOperationResultKHR not loaded");
        check(unsafe { fp(self.handle(), operation) })
    }
    ///Wraps [`vkDeferredOperationJoinKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDeferredOperationJoinKHR.html).
    /**
    Provided by **VK_KHR_deferred_host_operations**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Joins the calling thread to a deferred operation, contributing
    ///CPU time to its completion. Multiple threads can join the same
    ///operation concurrently.
    ///
    ///Return values:
    ///
    ///- `SUCCESS`, the operation completed. The calling thread was
    ///  the last one needed.
    ///- `THREAD_DONE_KHR`, this thread's contribution is finished,
    ///  but the operation may still be in progress on other threads.
    ///- `THREAD_IDLE_KHR`, the operation has enough threads; this
    ///  one was not needed. Retry later or move on.
    ///
    ///Call this in a loop per thread until it returns `SUCCESS` or
    ///`THREAD_DONE_KHR`. After all threads finish, check the final
    ///result with `get_deferred_operation_result_khr`.
    ///
    ///The number of useful threads is bounded by
    ///`get_deferred_operation_max_concurrency_khr`.
    pub unsafe fn deferred_operation_join_khr(
        &self,
        operation: DeferredOperationKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .deferred_operation_join_khr
            .expect("vkDeferredOperationJoinKHR not loaded");
        check(unsafe { fp(self.handle(), operation) })
    }
    ///Wraps [`vkGetPipelineIndirectMemoryRequirementsNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPipelineIndirectMemoryRequirementsNV.html).
    /**
    Provided by **VK_NV_device_generated_commands_compute**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the memory requirements for storing a compute pipeline's
    ///indirect dispatch metadata. Allocate a buffer of this size and
    ///pass it when creating the pipeline for device-generated compute
    ///dispatch.
    ///
    ///Requires `VK_NV_device_generated_commands_compute`.
    pub unsafe fn get_pipeline_indirect_memory_requirements_nv(
        &self,
        p_create_info: &ComputePipelineCreateInfo,
        p_memory_requirements: *mut MemoryRequirements2,
    ) {
        let fp = self
            .commands()
            .get_pipeline_indirect_memory_requirements_nv
            .expect("vkGetPipelineIndirectMemoryRequirementsNV not loaded");
        unsafe { fp(self.handle(), p_create_info, p_memory_requirements) };
    }
    ///Wraps [`vkGetPipelineIndirectDeviceAddressNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPipelineIndirectDeviceAddressNV.html).
    /**
    Provided by **VK_NV_device_generated_commands_compute**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns the device address of a compute pipeline's indirect
    ///dispatch metadata. The GPU writes to this address to select
    ///which pipeline to dispatch in device-generated compute workflows.
    ///
    ///Requires `VK_NV_device_generated_commands_compute`.
    pub unsafe fn get_pipeline_indirect_device_address_nv(
        &self,
        p_info: &PipelineIndirectDeviceAddressInfoNV,
    ) {
        let fp = self
            .commands()
            .get_pipeline_indirect_device_address_nv
            .expect("vkGetPipelineIndirectDeviceAddressNV not loaded");
        unsafe { fp(self.handle(), p_info) };
    }
    ///Wraps [`vkAntiLagUpdateAMD`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkAntiLagUpdateAMD.html).
    /**
    Provided by **VK_AMD_anti_lag**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Submits anti-lag timing data to reduce input-to-display latency.
    ///Called once per frame with presentation timing hints so the driver
    ///can pace GPU work to minimise latency.
    ///
    ///Requires `VK_AMD_anti_lag`.
    pub unsafe fn anti_lag_update_amd(&self, p_data: &AntiLagDataAMD) {
        let fp = self
            .commands()
            .anti_lag_update_amd
            .expect("vkAntiLagUpdateAMD not loaded");
        unsafe { fp(self.handle(), p_data) };
    }
    ///Wraps [`vkCmdSetCullMode`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetCullMode.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets the triangle culling mode. Only takes effect if
    ///the pipeline was created with `DYNAMIC_STATE_CULL_MODE`.
    ///
    ///Values: `CULL_MODE_NONE`, `CULL_MODE_FRONT`, `CULL_MODE_BACK`,
    ///`CULL_MODE_FRONT_AND_BACK`.
    ///
    ///Common pattern: set `CULL_MODE_BACK` for opaque geometry and
    ///`CULL_MODE_NONE` for double-sided or transparent materials, without
    ///needing separate pipelines.
    ///
    ///Core dynamic state in Vulkan 1.3.
    pub unsafe fn cmd_set_cull_mode(
        &self,
        command_buffer: CommandBuffer,
        cull_mode: CullModeFlags,
    ) {
        let fp = self
            .commands()
            .cmd_set_cull_mode
            .expect("vkCmdSetCullMode not loaded");
        unsafe { fp(command_buffer, cull_mode) };
    }
    ///Wraps [`vkCmdSetFrontFace`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetFrontFace.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets which triangle winding order is considered
    ///front-facing. Only takes effect if the pipeline was created with
    ///`DYNAMIC_STATE_FRONT_FACE`.
    ///
    ///Values: `FRONT_FACE_COUNTER_CLOCKWISE` (the Vulkan default) or
    ///`FRONT_FACE_CLOCKWISE`.
    ///
    ///Useful when rendering mirrored or reflected geometry where the
    ///winding order is flipped, without needing a separate pipeline.
    ///
    ///Core dynamic state in Vulkan 1.3.
    pub unsafe fn cmd_set_front_face(&self, command_buffer: CommandBuffer, front_face: FrontFace) {
        let fp = self
            .commands()
            .cmd_set_front_face
            .expect("vkCmdSetFrontFace not loaded");
        unsafe { fp(command_buffer, front_face) };
    }
    ///Wraps [`vkCmdSetPrimitiveTopology`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetPrimitiveTopology.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets the primitive topology. Only takes effect if the
    ///pipeline was created with `DYNAMIC_STATE_PRIMITIVE_TOPOLOGY`.
    ///
    ///Values include `POINT_LIST`, `LINE_LIST`, `LINE_STRIP`,
    ///`TRIANGLE_LIST`, `TRIANGLE_STRIP`, `TRIANGLE_FAN`,
    ///`LINE_LIST_WITH_ADJACENCY`, `PATCH_LIST`, etc.
    ///
    ///The dynamic topology must be in the same topology class as the
    ///pipeline's static topology (e.g. you can switch between
    ///`TRIANGLE_LIST` and `TRIANGLE_STRIP` since both are triangle
    ///topologies, but not between `TRIANGLE_LIST` and `LINE_LIST`).
    ///
    ///Core dynamic state in Vulkan 1.3.
    pub unsafe fn cmd_set_primitive_topology(
        &self,
        command_buffer: CommandBuffer,
        primitive_topology: PrimitiveTopology,
    ) {
        let fp = self
            .commands()
            .cmd_set_primitive_topology
            .expect("vkCmdSetPrimitiveTopology not loaded");
        unsafe { fp(command_buffer, primitive_topology) };
    }
    ///Wraps [`vkCmdSetViewportWithCount`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetViewportWithCount.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets both the viewports and the viewport count. Only
    ///takes effect if the pipeline was created with
    ///`DYNAMIC_STATE_VIEWPORT_WITH_COUNT`.
    ///
    ///Unlike `cmd_set_viewport` (which requires the count to match the
    ///pipeline's static `viewport_count`), this command also sets the
    ///count dynamically. The viewport count must match the scissor count
    ///set by `cmd_set_scissor_with_count`.
    ///
    ///Core dynamic state in Vulkan 1.3.
    pub unsafe fn cmd_set_viewport_with_count(
        &self,
        command_buffer: CommandBuffer,
        p_viewports: &[Viewport],
    ) {
        let fp = self
            .commands()
            .cmd_set_viewport_with_count
            .expect("vkCmdSetViewportWithCount not loaded");
        unsafe {
            fp(
                command_buffer,
                p_viewports.len() as u32,
                p_viewports.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdSetScissorWithCount`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetScissorWithCount.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets both the scissor rectangles and the scissor count.
    ///Only takes effect if the pipeline was created with
    ///`DYNAMIC_STATE_SCISSOR_WITH_COUNT`.
    ///
    ///Unlike `cmd_set_scissor` (which requires the count to match the
    ///pipeline's static `viewport_count`), this command also sets the
    ///count dynamically. The scissor count must match the viewport count
    ///set by `cmd_set_viewport_with_count`.
    ///
    ///Core dynamic state in Vulkan 1.3.
    pub unsafe fn cmd_set_scissor_with_count(
        &self,
        command_buffer: CommandBuffer,
        p_scissors: &[Rect2D],
    ) {
        let fp = self
            .commands()
            .cmd_set_scissor_with_count
            .expect("vkCmdSetScissorWithCount not loaded");
        unsafe { fp(command_buffer, p_scissors.len() as u32, p_scissors.as_ptr()) };
    }
    ///Wraps [`vkCmdBindIndexBuffer2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBindIndexBuffer2.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_4**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.4 version of `cmd_bind_index_buffer` that additionally
    ///accepts a `size` parameter specifying the valid range of the index
    ///buffer. This enables the driver to perform bounds checking.
    ///
    ///Pass `VK_WHOLE_SIZE` for the size to use the remainder of the buffer
    ///from the offset.
    ///
    ///Prefer this over `cmd_bind_index_buffer` when targeting Vulkan 1.4+.
    pub unsafe fn cmd_bind_index_buffer2(
        &self,
        command_buffer: CommandBuffer,
        buffer: Buffer,
        offset: u64,
        size: u64,
        index_type: IndexType,
    ) {
        let fp = self
            .commands()
            .cmd_bind_index_buffer2
            .expect("vkCmdBindIndexBuffer2 not loaded");
        unsafe { fp(command_buffer, buffer, offset, size, index_type) };
    }
    ///Wraps [`vkCmdBindVertexBuffers2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBindVertexBuffers2.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.3 version of `cmd_bind_vertex_buffers` that additionally
    ///accepts optional buffer sizes and strides.
    ///
    ///**Sizes**: if provided, the driver knows where each buffer ends and
    ///can perform bounds checking. Pass null to use the full buffer size.
    ///
    ///**Strides**: if provided, overrides the stride specified in the
    ///pipeline's vertex input state. This enables dynamic vertex stride
    ///without creating separate pipeline permutations. Pass null to use
    ///the pipeline's static stride.
    ///
    ///Prefer this over `cmd_bind_vertex_buffers` when targeting
    ///Vulkan 1.3+.
    pub unsafe fn cmd_bind_vertex_buffers2(
        &self,
        command_buffer: CommandBuffer,
        first_binding: u32,
        p_buffers: &[Buffer],
        p_offsets: &u64,
        p_sizes: Option<&u64>,
        p_strides: Option<&u64>,
    ) {
        let fp = self
            .commands()
            .cmd_bind_vertex_buffers2
            .expect("vkCmdBindVertexBuffers2 not loaded");
        let p_sizes_ptr = p_sizes.map_or(core::ptr::null(), core::ptr::from_ref);
        let p_strides_ptr = p_strides.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe {
            fp(
                command_buffer,
                first_binding,
                p_buffers.len() as u32,
                p_buffers.as_ptr(),
                p_offsets,
                p_sizes_ptr,
                p_strides_ptr,
            )
        };
    }
    ///Wraps [`vkCmdSetDepthTestEnable`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetDepthTestEnable.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically enables or disables depth testing. Only takes effect if
    ///the pipeline was created with `DYNAMIC_STATE_DEPTH_TEST_ENABLE`.
    ///
    ///When disabled, all fragments pass the depth test regardless of the
    ///depth buffer contents. Useful for UI overlays, skyboxes, or
    ///full-screen post-processing passes.
    ///
    ///Core dynamic state in Vulkan 1.3.
    pub unsafe fn cmd_set_depth_test_enable(
        &self,
        command_buffer: CommandBuffer,
        depth_test_enable: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_depth_test_enable
            .expect("vkCmdSetDepthTestEnable not loaded");
        unsafe { fp(command_buffer, depth_test_enable) };
    }
    ///Wraps [`vkCmdSetDepthWriteEnable`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetDepthWriteEnable.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically enables or disables writes to the depth buffer. Only
    ///takes effect if the pipeline was created with
    ///`DYNAMIC_STATE_DEPTH_WRITE_ENABLE`.
    ///
    ///Disable depth writes when:
    ///
    ///- Drawing transparent objects (they should test against depth but
    ///  not write to it).
    ///- Drawing skyboxes after the opaque pass.
    ///- Performing post-processing with depth reads but no depth output.
    ///
    ///Depth testing and depth writing are independent controls, you can
    ///test without writing, or write without testing.
    ///
    ///Core dynamic state in Vulkan 1.3.
    pub unsafe fn cmd_set_depth_write_enable(
        &self,
        command_buffer: CommandBuffer,
        depth_write_enable: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_depth_write_enable
            .expect("vkCmdSetDepthWriteEnable not loaded");
        unsafe { fp(command_buffer, depth_write_enable) };
    }
    ///Wraps [`vkCmdSetDepthCompareOp`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetDepthCompareOp.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets the depth comparison operator. Only takes effect if
    ///the pipeline was created with `DYNAMIC_STATE_DEPTH_COMPARE_OP`.
    ///
    ///Values: `COMPARE_OP_LESS` (the common default for forward rendering),
    ///`COMPARE_OP_GREATER` (for reverse-Z), `COMPARE_OP_LESS_OR_EQUAL`,
    ///`COMPARE_OP_ALWAYS`, etc.
    ///
    ///**Reverse-Z**: using a reversed depth buffer (near=1.0, far=0.0)
    ///with `COMPARE_OP_GREATER` provides better floating-point precision
    ///distribution across the depth range. This is the recommended setup
    ///for modern rendering.
    ///
    ///Core dynamic state in Vulkan 1.3.
    pub unsafe fn cmd_set_depth_compare_op(
        &self,
        command_buffer: CommandBuffer,
        depth_compare_op: CompareOp,
    ) {
        let fp = self
            .commands()
            .cmd_set_depth_compare_op
            .expect("vkCmdSetDepthCompareOp not loaded");
        unsafe { fp(command_buffer, depth_compare_op) };
    }
    ///Wraps [`vkCmdSetDepthBoundsTestEnable`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetDepthBoundsTestEnable.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically enables or disables the depth bounds test. Only takes
    ///effect if the pipeline was created with
    ///`DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE`.
    ///
    ///When enabled, fragments are discarded if the existing depth buffer
    ///value falls outside the range set by `cmd_set_depth_bounds`. When
    ///disabled, the test is skipped.
    ///
    ///Requires the `depth_bounds` device feature.
    ///
    ///Core dynamic state in Vulkan 1.3.
    pub unsafe fn cmd_set_depth_bounds_test_enable(
        &self,
        command_buffer: CommandBuffer,
        depth_bounds_test_enable: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_depth_bounds_test_enable
            .expect("vkCmdSetDepthBoundsTestEnable not loaded");
        unsafe { fp(command_buffer, depth_bounds_test_enable) };
    }
    ///Wraps [`vkCmdSetStencilTestEnable`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetStencilTestEnable.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically enables or disables stencil testing. Only takes effect
    ///if the pipeline was created with
    ///`DYNAMIC_STATE_STENCIL_TEST_ENABLE`.
    ///
    ///When disabled, fragments pass the stencil test unconditionally and
    ///no stencil writes occur.
    ///
    ///Core dynamic state in Vulkan 1.3.
    pub unsafe fn cmd_set_stencil_test_enable(
        &self,
        command_buffer: CommandBuffer,
        stencil_test_enable: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_stencil_test_enable
            .expect("vkCmdSetStencilTestEnable not loaded");
        unsafe { fp(command_buffer, stencil_test_enable) };
    }
    ///Wraps [`vkCmdSetStencilOp`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetStencilOp.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets the stencil operations for front-facing,
    ///back-facing, or both face sets. Only takes effect if the pipeline
    ///was created with `DYNAMIC_STATE_STENCIL_OP`.
    ///
    ///Sets four values per face:
    ///
    ///- **`fail_op`**: action when the stencil test fails.
    ///- **`pass_op`**: action when both stencil and depth tests pass.
    ///- **`depth_fail_op`**: action when stencil passes but depth fails.
    ///- **`compare_op`**: the stencil comparison function.
    ///
    ///Common operations: `KEEP`, `REPLACE`, `INCREMENT_AND_CLAMP`,
    ///`DECREMENT_AND_CLAMP`, `INVERT`, `ZERO`.
    ///
    ///Core dynamic state in Vulkan 1.3.
    pub unsafe fn cmd_set_stencil_op(
        &self,
        command_buffer: CommandBuffer,
        face_mask: StencilFaceFlags,
        fail_op: StencilOp,
        pass_op: StencilOp,
        depth_fail_op: StencilOp,
        compare_op: CompareOp,
    ) {
        let fp = self
            .commands()
            .cmd_set_stencil_op
            .expect("vkCmdSetStencilOp not loaded");
        unsafe {
            fp(
                command_buffer,
                face_mask,
                fail_op,
                pass_op,
                depth_fail_op,
                compare_op,
            )
        };
    }
    ///Wraps [`vkCmdSetPatchControlPointsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetPatchControlPointsEXT.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state2**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets the number of control points per patch for
    ///tessellation. Overrides the value specified at pipeline creation.
    ///
    ///Typical values: 3 (triangles), 4 (quads), 16 (bicubic patches).
    ///The maximum is `maxTessellationPatchSize` (at least 32).
    ///
    ///Requires the `extendedDynamicState2PatchControlPoints` feature.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state2`.
    pub unsafe fn cmd_set_patch_control_points_ext(
        &self,
        command_buffer: CommandBuffer,
        patch_control_points: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_patch_control_points_ext
            .expect("vkCmdSetPatchControlPointsEXT not loaded");
        unsafe { fp(command_buffer, patch_control_points) };
    }
    ///Wraps [`vkCmdSetRasterizerDiscardEnable`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetRasterizerDiscardEnable.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically enables or disables rasterizer discard. Only takes
    ///effect if the pipeline was created with
    ///`DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE`.
    ///
    ///When enabled, primitives are discarded before rasterisation, no
    ///fragments are generated and no colour/depth output is produced. The
    ///vertex and geometry shader stages still execute.
    ///
    ///Use cases:
    ///
    ///- **Transform feedback only**: capture transformed vertices without
    ///  rendering.
    ///- **Occlusion pre-pass**: skip fragment shading when only the depth
    ///  or stencil output matters (though depth writes still require
    ///  rasterisation).
    ///
    ///Core dynamic state in Vulkan 1.3.
    pub unsafe fn cmd_set_rasterizer_discard_enable(
        &self,
        command_buffer: CommandBuffer,
        rasterizer_discard_enable: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_rasterizer_discard_enable
            .expect("vkCmdSetRasterizerDiscardEnable not loaded");
        unsafe { fp(command_buffer, rasterizer_discard_enable) };
    }
    ///Wraps [`vkCmdSetDepthBiasEnable`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetDepthBiasEnable.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically enables or disables depth bias. Only takes effect if the
    ///pipeline was created with `DYNAMIC_STATE_DEPTH_BIAS_ENABLE`.
    ///
    ///When enabled, the depth bias values set by `cmd_set_depth_bias` are
    ///applied to fragment depth values. When disabled, no bias is applied
    ///regardless of the bias values.
    ///
    ///Useful for toggling shadow map bias without separate pipelines.
    ///
    ///Core dynamic state in Vulkan 1.3.
    pub unsafe fn cmd_set_depth_bias_enable(
        &self,
        command_buffer: CommandBuffer,
        depth_bias_enable: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_depth_bias_enable
            .expect("vkCmdSetDepthBiasEnable not loaded");
        unsafe { fp(command_buffer, depth_bias_enable) };
    }
    ///Wraps [`vkCmdSetLogicOpEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetLogicOpEXT.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state2**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets the logic operation used for color blending.
    ///Logic ops (AND, OR, XOR, etc.) operate on the raw integer bit
    ///patterns of the fragment and framebuffer values.
    ///
    ///Only effective when logic op is enabled in the pipeline
    ///(`logicOpEnable`). Requires the `extendedDynamicState2LogicOp`
    ///feature.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state2`.
    pub unsafe fn cmd_set_logic_op_ext(&self, command_buffer: CommandBuffer, logic_op: LogicOp) {
        let fp = self
            .commands()
            .cmd_set_logic_op_ext
            .expect("vkCmdSetLogicOpEXT not loaded");
        unsafe { fp(command_buffer, logic_op) };
    }
    ///Wraps [`vkCmdSetPrimitiveRestartEnable`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetPrimitiveRestartEnable.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically enables or disables primitive restart. Only takes effect
    ///if the pipeline was created with
    ///`DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE`.
    ///
    ///When enabled, a special index value (0xFFFF for UINT16, 0xFFFFFFFF
    ///for UINT32) in the index buffer ends the current primitive and
    ///starts a new one. This lets you draw multiple triangle strips or
    ///line strips from a single draw call without degenerate triangles.
    ///
    ///Only meaningful for strip topologies (`TRIANGLE_STRIP`,
    ///`LINE_STRIP`, `TRIANGLE_FAN`, etc.).
    ///
    ///Core dynamic state in Vulkan 1.3.
    pub unsafe fn cmd_set_primitive_restart_enable(
        &self,
        command_buffer: CommandBuffer,
        primitive_restart_enable: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_primitive_restart_enable
            .expect("vkCmdSetPrimitiveRestartEnable not loaded");
        unsafe { fp(command_buffer, primitive_restart_enable) };
    }
    ///Wraps [`vkCmdSetTessellationDomainOriginEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetTessellationDomainOriginEXT.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets the tessellation domain origin:
    ///- `UPPER_LEFT`: Vulkan default, (0,0) is the upper-left corner.
    ///- `LOWER_LEFT`: OpenGL convention, (0,0) is the lower-left.
    ///
    ///Affects how tessellation coordinates are interpreted. Useful when
    ///porting OpenGL tessellation shaders.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_tessellation_domain_origin_ext(
        &self,
        command_buffer: CommandBuffer,
        domain_origin: TessellationDomainOrigin,
    ) {
        let fp = self
            .commands()
            .cmd_set_tessellation_domain_origin_ext
            .expect("vkCmdSetTessellationDomainOriginEXT not loaded");
        unsafe { fp(command_buffer, domain_origin) };
    }
    ///Wraps [`vkCmdSetDepthClampEnableEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetDepthClampEnableEXT.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically enables or disables depth clamping. When enabled,
    ///fragments outside the near/far depth range are clamped instead
    ///of being clipped.
    ///
    ///Useful for shadow mapping and other techniques where clipping
    ///at the near/far plane is undesirable.
    ///
    ///Requires the `depthClamp` device feature.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_depth_clamp_enable_ext(
        &self,
        command_buffer: CommandBuffer,
        depth_clamp_enable: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_depth_clamp_enable_ext
            .expect("vkCmdSetDepthClampEnableEXT not loaded");
        unsafe { fp(command_buffer, depth_clamp_enable) };
    }
    ///Wraps [`vkCmdSetPolygonModeEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetPolygonModeEXT.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets the polygon rasterization mode:
    ///- `FILL`: normal filled triangles (default).
    ///- `LINE`: wireframe rendering.
    ///- `POINT`: vertices only.
    ///
    ///`LINE` and `POINT` require the `fillModeNonSolid` device feature.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_polygon_mode_ext(
        &self,
        command_buffer: CommandBuffer,
        polygon_mode: PolygonMode,
    ) {
        let fp = self
            .commands()
            .cmd_set_polygon_mode_ext
            .expect("vkCmdSetPolygonModeEXT not loaded");
        unsafe { fp(command_buffer, polygon_mode) };
    }
    ///Wraps [`vkCmdSetRasterizationSamplesEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetRasterizationSamplesEXT.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets the number of rasterization samples
    ///(multisampling level). Overrides the sample count specified
    ///at pipeline creation.
    ///
    ///The sample count must be compatible with the render pass
    ///attachments.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_rasterization_samples_ext(
        &self,
        command_buffer: CommandBuffer,
        rasterization_samples: SampleCountFlagBits,
    ) {
        let fp = self
            .commands()
            .cmd_set_rasterization_samples_ext
            .expect("vkCmdSetRasterizationSamplesEXT not loaded");
        unsafe { fp(command_buffer, rasterization_samples) };
    }
    ///Wraps [`vkCmdSetSampleMaskEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetSampleMaskEXT.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets the sample mask. The sample mask is ANDed with
    ///the coverage mask to determine which samples are written. Bits
    ///that are zero disable the corresponding sample.
    ///
    ///The slice length must match `ceil(rasterizationSamples / 32)`.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_sample_mask_ext(
        &self,
        command_buffer: CommandBuffer,
        samples: SampleCountFlagBits,
        p_sample_mask: Option<&u32>,
    ) {
        let fp = self
            .commands()
            .cmd_set_sample_mask_ext
            .expect("vkCmdSetSampleMaskEXT not loaded");
        let p_sample_mask_ptr = p_sample_mask.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(command_buffer, samples, p_sample_mask_ptr) };
    }
    ///Wraps [`vkCmdSetAlphaToCoverageEnableEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetAlphaToCoverageEnableEXT.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically enables or disables alpha-to-coverage multisample
    ///mode. When enabled, the fragment's alpha value determines which
    ///samples are covered, providing order-independent transparency
    ///for foliage, fences, etc.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_alpha_to_coverage_enable_ext(
        &self,
        command_buffer: CommandBuffer,
        alpha_to_coverage_enable: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_alpha_to_coverage_enable_ext
            .expect("vkCmdSetAlphaToCoverageEnableEXT not loaded");
        unsafe { fp(command_buffer, alpha_to_coverage_enable) };
    }
    ///Wraps [`vkCmdSetAlphaToOneEnableEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetAlphaToOneEnableEXT.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically enables or disables alpha-to-one mode. When enabled,
    ///the fragment's alpha value is replaced with 1.0 after
    ///alpha-to-coverage processing.
    ///
    ///Requires the `alphaToOne` device feature.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_alpha_to_one_enable_ext(
        &self,
        command_buffer: CommandBuffer,
        alpha_to_one_enable: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_alpha_to_one_enable_ext
            .expect("vkCmdSetAlphaToOneEnableEXT not loaded");
        unsafe { fp(command_buffer, alpha_to_one_enable) };
    }
    ///Wraps [`vkCmdSetLogicOpEnableEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetLogicOpEnableEXT.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically enables or disables logic operations for color
    ///blending. When enabled, fragments are combined with framebuffer
    ///values using the logic op set by `cmd_set_logic_op_ext` instead
    ///of standard blend equations.
    ///
    ///Logic ops operate on integer bit patterns. They have no effect
    ///on floating-point color attachments.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_logic_op_enable_ext(
        &self,
        command_buffer: CommandBuffer,
        logic_op_enable: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_logic_op_enable_ext
            .expect("vkCmdSetLogicOpEnableEXT not loaded");
        unsafe { fp(command_buffer, logic_op_enable) };
    }
    ///Wraps [`vkCmdSetColorBlendEnableEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetColorBlendEnableEXT.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically enables or disables color blending for each color
    ///attachment. Pass a slice of `Bool32` values, one per attachment
    ///starting at `first_attachment`.
    ///
    ///When blending is disabled for an attachment, the fragment color
    ///is written directly (no src/dst blending).
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_color_blend_enable_ext(
        &self,
        command_buffer: CommandBuffer,
        first_attachment: u32,
        p_color_blend_enables: &[u32],
    ) {
        let fp = self
            .commands()
            .cmd_set_color_blend_enable_ext
            .expect("vkCmdSetColorBlendEnableEXT not loaded");
        unsafe {
            fp(
                command_buffer,
                first_attachment,
                p_color_blend_enables.len() as u32,
                p_color_blend_enables.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdSetColorBlendEquationEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetColorBlendEquationEXT.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets the blend equation (src factor, dst factor,
    ///blend op) for each color attachment. Each `ColorBlendEquationEXT`
    ///specifies both the color and alpha blend parameters.
    ///
    ///Overrides the values set at pipeline creation. Only effective for
    ///attachments where blending is enabled.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_color_blend_equation_ext(
        &self,
        command_buffer: CommandBuffer,
        first_attachment: u32,
        p_color_blend_equations: &[ColorBlendEquationEXT],
    ) {
        let fp = self
            .commands()
            .cmd_set_color_blend_equation_ext
            .expect("vkCmdSetColorBlendEquationEXT not loaded");
        unsafe {
            fp(
                command_buffer,
                first_attachment,
                p_color_blend_equations.len() as u32,
                p_color_blend_equations.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdSetColorWriteMaskEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetColorWriteMaskEXT.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets the color write mask for each color attachment.
    ///Each `ColorComponentFlags` value controls which channels (R, G,
    ///B, A) are written for the corresponding attachment.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_color_write_mask_ext(
        &self,
        command_buffer: CommandBuffer,
        first_attachment: u32,
        p_color_write_masks: &[ColorComponentFlags],
    ) {
        let fp = self
            .commands()
            .cmd_set_color_write_mask_ext
            .expect("vkCmdSetColorWriteMaskEXT not loaded");
        unsafe {
            fp(
                command_buffer,
                first_attachment,
                p_color_write_masks.len() as u32,
                p_color_write_masks.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdSetRasterizationStreamEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetRasterizationStreamEXT.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets which vertex stream is used for rasterization
    ///when transform feedback is active. Stream 0 is always rasterized
    ///by default; other streams can be selected with this command.
    ///
    ///Requires `VK_EXT_transform_feedback` and the
    ///`geometryStreams` feature.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_rasterization_stream_ext(
        &self,
        command_buffer: CommandBuffer,
        rasterization_stream: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_rasterization_stream_ext
            .expect("vkCmdSetRasterizationStreamEXT not loaded");
        unsafe { fp(command_buffer, rasterization_stream) };
    }
    ///Wraps [`vkCmdSetConservativeRasterizationModeEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetConservativeRasterizationModeEXT.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets the conservative rasterization mode:
    ///- `DISABLED`: normal rasterization.
    ///- `OVERESTIMATE`: a fragment is generated if the primitive
    ///  overlaps any part of the pixel.
    ///- `UNDERESTIMATE`: a fragment is generated only if the pixel
    ///  is fully covered.
    ///
    ///Requires `VK_EXT_conservative_rasterization`.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_conservative_rasterization_mode_ext(
        &self,
        command_buffer: CommandBuffer,
        conservative_rasterization_mode: ConservativeRasterizationModeEXT,
    ) {
        let fp = self
            .commands()
            .cmd_set_conservative_rasterization_mode_ext
            .expect("vkCmdSetConservativeRasterizationModeEXT not loaded");
        unsafe { fp(command_buffer, conservative_rasterization_mode) };
    }
    ///Wraps [`vkCmdSetExtraPrimitiveOverestimationSizeEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetExtraPrimitiveOverestimationSizeEXT.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets the extra overestimation size for conservative
    ///rasterization. This expands the primitive by additional pixels
    ///beyond the minimum overestimation guaranteed by the implementation.
    ///
    ///The value is in units of pixels. 0.0 means no extra
    ///overestimation beyond the implementation minimum.
    ///
    ///Requires `VK_EXT_conservative_rasterization`.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_extra_primitive_overestimation_size_ext(
        &self,
        command_buffer: CommandBuffer,
        extra_primitive_overestimation_size: f32,
    ) {
        let fp = self
            .commands()
            .cmd_set_extra_primitive_overestimation_size_ext
            .expect("vkCmdSetExtraPrimitiveOverestimationSizeEXT not loaded");
        unsafe { fp(command_buffer, extra_primitive_overestimation_size) };
    }
    ///Wraps [`vkCmdSetDepthClipEnableEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetDepthClipEnableEXT.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically enables or disables depth clipping. When disabled,
    ///primitives are not clipped against the near and far planes
    ///(equivalent to `depthClampEnable`, but controlled separately).
    ///
    ///Requires `VK_EXT_depth_clip_enable` and the `depthClipEnable`
    ///feature.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_depth_clip_enable_ext(
        &self,
        command_buffer: CommandBuffer,
        depth_clip_enable: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_depth_clip_enable_ext
            .expect("vkCmdSetDepthClipEnableEXT not loaded");
        unsafe { fp(command_buffer, depth_clip_enable) };
    }
    ///Wraps [`vkCmdSetSampleLocationsEnableEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetSampleLocationsEnableEXT.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically enables or disables custom sample locations. When
    ///enabled, the sample positions used for multisampling are taken
    ///from the locations set by `cmd_set_sample_locations_ext` instead
    ///of the implementation defaults.
    ///
    ///Requires `VK_EXT_sample_locations`.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_sample_locations_enable_ext(
        &self,
        command_buffer: CommandBuffer,
        sample_locations_enable: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_sample_locations_enable_ext
            .expect("vkCmdSetSampleLocationsEnableEXT not loaded");
        unsafe { fp(command_buffer, sample_locations_enable) };
    }
    ///Wraps [`vkCmdSetColorBlendAdvancedEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetColorBlendAdvancedEXT.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets advanced blend parameters per color attachment.
    ///Each `ColorBlendAdvancedEXT` specifies the advanced blend
    ///operation, whether src/dst are premultiplied, and the blend
    ///overlap mode (uncorrelated, disjoint, conjoint).
    ///
    ///Only applies when using advanced blend operations (not the
    ///standard blend factors). Requires `VK_EXT_blend_operation_advanced`.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_color_blend_advanced_ext(
        &self,
        command_buffer: CommandBuffer,
        first_attachment: u32,
        p_color_blend_advanced: &[ColorBlendAdvancedEXT],
    ) {
        let fp = self
            .commands()
            .cmd_set_color_blend_advanced_ext
            .expect("vkCmdSetColorBlendAdvancedEXT not loaded");
        unsafe {
            fp(
                command_buffer,
                first_attachment,
                p_color_blend_advanced.len() as u32,
                p_color_blend_advanced.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdSetProvokingVertexModeEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetProvokingVertexModeEXT.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets which vertex in a primitive is the provoking
    ///vertex (the vertex whose flat-shaded attributes are used):
    ///- `FIRST_VERTEX`: first vertex of the primitive (Vulkan default).
    ///- `LAST_VERTEX`: last vertex (OpenGL convention).
    ///
    ///Requires `VK_EXT_provoking_vertex` and the
    ///`provokingVertexLast` feature for `LAST_VERTEX` mode.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_provoking_vertex_mode_ext(
        &self,
        command_buffer: CommandBuffer,
        provoking_vertex_mode: ProvokingVertexModeEXT,
    ) {
        let fp = self
            .commands()
            .cmd_set_provoking_vertex_mode_ext
            .expect("vkCmdSetProvokingVertexModeEXT not loaded");
        unsafe { fp(command_buffer, provoking_vertex_mode) };
    }
    ///Wraps [`vkCmdSetLineRasterizationModeEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetLineRasterizationModeEXT.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets the line rasterization mode:
    ///- `DEFAULT`: implementation default.
    ///- `RECTANGULAR`: lines are rasterized as parallelograms (Vulkan
    ///  default).
    ///- `BRESENHAM`: pixel-exact Bresenham lines.
    ///- `RECTANGULAR_SMOOTH`: antialiased rectangular lines.
    ///
    ///Requires `VK_EXT_line_rasterization` and the corresponding
    ///device feature for the chosen mode.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_line_rasterization_mode_ext(
        &self,
        command_buffer: CommandBuffer,
        line_rasterization_mode: LineRasterizationModeEXT,
    ) {
        let fp = self
            .commands()
            .cmd_set_line_rasterization_mode_ext
            .expect("vkCmdSetLineRasterizationModeEXT not loaded");
        unsafe { fp(command_buffer, line_rasterization_mode) };
    }
    ///Wraps [`vkCmdSetLineStippleEnableEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetLineStippleEnableEXT.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically enables or disables line stippling. When enabled,
    ///lines are drawn with a repeating pattern defined by
    ///`cmd_set_line_stipple` (core 1.4) or `cmd_set_line_stipple_ext`.
    ///
    ///Requires `VK_EXT_line_rasterization` and the `stippledLineEnable`
    ///feature for the active line rasterization mode.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_line_stipple_enable_ext(
        &self,
        command_buffer: CommandBuffer,
        stippled_line_enable: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_line_stipple_enable_ext
            .expect("vkCmdSetLineStippleEnableEXT not loaded");
        unsafe { fp(command_buffer, stippled_line_enable) };
    }
    ///Wraps [`vkCmdSetDepthClipNegativeOneToOneEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetDepthClipNegativeOneToOneEXT.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets the depth clip range convention:
    ///- `true`: NDC depth range is [-1, 1] (OpenGL convention).
    ///- `false`: NDC depth range is [0, 1] (Vulkan default).
    ///
    ///Useful for porting OpenGL content or using reversed-Z with
    ///OpenGL-style projections.
    ///
    ///Requires `VK_EXT_depth_clip_control` and the
    ///`depthClipControl` feature.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_depth_clip_negative_one_to_one_ext(
        &self,
        command_buffer: CommandBuffer,
        negative_one_to_one: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_depth_clip_negative_one_to_one_ext
            .expect("vkCmdSetDepthClipNegativeOneToOneEXT not loaded");
        unsafe { fp(command_buffer, negative_one_to_one) };
    }
    ///Wraps [`vkCmdSetViewportWScalingEnableNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetViewportWScalingEnableNV.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically enables or disables viewport W scaling. When enabled,
    ///the x and y viewport coordinates are divided by an additional
    ///per-viewport W scaling factor, which can be used for lens-matched
    ///shading in VR.
    ///
    ///Part of `VK_NV_clip_space_w_scaling`.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_viewport_w_scaling_enable_nv(
        &self,
        command_buffer: CommandBuffer,
        viewport_w_scaling_enable: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_viewport_w_scaling_enable_nv
            .expect("vkCmdSetViewportWScalingEnableNV not loaded");
        unsafe { fp(command_buffer, viewport_w_scaling_enable) };
    }
    ///Wraps [`vkCmdSetViewportSwizzleNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetViewportSwizzleNV.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets the viewport swizzle for each viewport. A
    ///swizzle remaps each output component (x, y, z, w) to any input
    ///component, optionally negated.
    ///
    ///Useful for single-pass cubemap rendering and other multi-view
    ///projection tricks.
    ///
    ///Part of `VK_NV_viewport_swizzle`.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_viewport_swizzle_nv(
        &self,
        command_buffer: CommandBuffer,
        first_viewport: u32,
        p_viewport_swizzles: &[ViewportSwizzleNV],
    ) {
        let fp = self
            .commands()
            .cmd_set_viewport_swizzle_nv
            .expect("vkCmdSetViewportSwizzleNV not loaded");
        unsafe {
            fp(
                command_buffer,
                first_viewport,
                p_viewport_swizzles.len() as u32,
                p_viewport_swizzles.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdSetCoverageToColorEnableNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetCoverageToColorEnableNV.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically enables or disables coverage-to-color mode. When
    ///enabled, the fragment coverage mask is written to a specified
    ///color attachment instead of (or in addition to) the normal color
    ///output.
    ///
    ///Part of `VK_NV_fragment_coverage_to_color`.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_coverage_to_color_enable_nv(
        &self,
        command_buffer: CommandBuffer,
        coverage_to_color_enable: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_coverage_to_color_enable_nv
            .expect("vkCmdSetCoverageToColorEnableNV not loaded");
        unsafe { fp(command_buffer, coverage_to_color_enable) };
    }
    ///Wraps [`vkCmdSetCoverageToColorLocationNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetCoverageToColorLocationNV.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets which color attachment receives the coverage
    ///mask when coverage-to-color is enabled.
    ///
    ///Only effective when `cmd_set_coverage_to_color_enable_nv` is true.
    ///
    ///Part of `VK_NV_fragment_coverage_to_color`.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_coverage_to_color_location_nv(
        &self,
        command_buffer: CommandBuffer,
        coverage_to_color_location: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_coverage_to_color_location_nv
            .expect("vkCmdSetCoverageToColorLocationNV not loaded");
        unsafe { fp(command_buffer, coverage_to_color_location) };
    }
    ///Wraps [`vkCmdSetCoverageModulationModeNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetCoverageModulationModeNV.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets how the coverage mask is combined with the
    ///fragment's color. Modes include `NONE`, `RGB`, `ALPHA`, and
    ///`RGBA`.
    ///
    ///Part of the NV coverage modulation feature used with
    ///`VK_NV_framebuffer_mixed_samples`.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_coverage_modulation_mode_nv(
        &self,
        command_buffer: CommandBuffer,
        coverage_modulation_mode: CoverageModulationModeNV,
    ) {
        let fp = self
            .commands()
            .cmd_set_coverage_modulation_mode_nv
            .expect("vkCmdSetCoverageModulationModeNV not loaded");
        unsafe { fp(command_buffer, coverage_modulation_mode) };
    }
    ///Wraps [`vkCmdSetCoverageModulationTableEnableNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetCoverageModulationTableEnableNV.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically enables or disables the coverage modulation lookup
    ///table. When enabled, the coverage value indexes into a table set
    ///by `cmd_set_coverage_modulation_table_nv` instead of using the
    ///default linear modulation.
    ///
    ///Part of `VK_NV_framebuffer_mixed_samples`.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_coverage_modulation_table_enable_nv(
        &self,
        command_buffer: CommandBuffer,
        coverage_modulation_table_enable: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_coverage_modulation_table_enable_nv
            .expect("vkCmdSetCoverageModulationTableEnableNV not loaded");
        unsafe { fp(command_buffer, coverage_modulation_table_enable) };
    }
    ///Wraps [`vkCmdSetCoverageModulationTableNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetCoverageModulationTableNV.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets the coverage modulation lookup table. Each entry
    ///maps a coverage sample count to a modulation factor (0.0–1.0).
    ///
    ///Only used when coverage modulation table is enabled via
    ///`cmd_set_coverage_modulation_table_enable_nv`.
    ///
    ///Part of `VK_NV_framebuffer_mixed_samples`.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_coverage_modulation_table_nv(
        &self,
        command_buffer: CommandBuffer,
        p_coverage_modulation_table: &[f32],
    ) {
        let fp = self
            .commands()
            .cmd_set_coverage_modulation_table_nv
            .expect("vkCmdSetCoverageModulationTableNV not loaded");
        unsafe {
            fp(
                command_buffer,
                p_coverage_modulation_table.len() as u32,
                p_coverage_modulation_table.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdSetShadingRateImageEnableNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetShadingRateImageEnableNV.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically enables or disables the shading rate image. When
    ///enabled, fragment shading rate is read from a shading rate image
    ///attachment instead of using a uniform rate.
    ///
    ///Part of `VK_NV_shading_rate_image`.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_shading_rate_image_enable_nv(
        &self,
        command_buffer: CommandBuffer,
        shading_rate_image_enable: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_shading_rate_image_enable_nv
            .expect("vkCmdSetShadingRateImageEnableNV not loaded");
        unsafe { fp(command_buffer, shading_rate_image_enable) };
    }
    ///Wraps [`vkCmdSetCoverageReductionModeNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetCoverageReductionModeNV.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets the coverage reduction mode, which controls how
    ///the multisampled coverage mask is reduced to a color sample mask
    ///when the number of color samples is less than the rasterization
    ///samples.
    ///
    ///Part of `VK_NV_coverage_reduction_mode`.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_coverage_reduction_mode_nv(
        &self,
        command_buffer: CommandBuffer,
        coverage_reduction_mode: CoverageReductionModeNV,
    ) {
        let fp = self
            .commands()
            .cmd_set_coverage_reduction_mode_nv
            .expect("vkCmdSetCoverageReductionModeNV not loaded");
        unsafe { fp(command_buffer, coverage_reduction_mode) };
    }
    ///Wraps [`vkCmdSetRepresentativeFragmentTestEnableNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetRepresentativeFragmentTestEnableNV.html).
    /**
    Provided by **VK_EXT_extended_dynamic_state3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically enables or disables the representative fragment test.
    ///When enabled, only one fragment per pixel is shaded (the
    ///"representative" fragment), and the rest are discarded early.
    ///
    ///Useful for visibility-only passes (e.g., occlusion culling)
    ///where full shading is unnecessary.
    ///
    ///Part of `VK_NV_representative_fragment_test`.
    ///
    ///Provided by `VK_EXT_extended_dynamic_state3`.
    pub unsafe fn cmd_set_representative_fragment_test_enable_nv(
        &self,
        command_buffer: CommandBuffer,
        representative_fragment_test_enable: u32,
    ) {
        let fp = self
            .commands()
            .cmd_set_representative_fragment_test_enable_nv
            .expect("vkCmdSetRepresentativeFragmentTestEnableNV not loaded");
        unsafe { fp(command_buffer, representative_fragment_test_enable) };
    }
    ///Wraps [`vkCreatePrivateDataSlot`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreatePrivateDataSlot.html).
    /**
    Provided by **VK_BASE_VERSION_1_3**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a private data slot that can be used to attach arbitrary
    ///application-defined `u64` values to any Vulkan object. Private data
    ///is a lightweight alternative to external hash maps for per-object
    ///metadata.
    ///
    ///Each slot represents one "key" that can be set on any object via
    ///`set_private_data` and read back via `get_private_data`.
    ///
    ///Use cases:
    ///
    ///- Tagging objects with debug IDs.
    ///- Associating application-specific state without a separate lookup
    ///  table.
    ///- Layer and tool implementations that need per-object bookkeeping.
    ///
    ///Private data slots are cheap. The slot itself is just an index;
    ///the per-object storage is allocated lazily by the driver.
    pub unsafe fn create_private_data_slot(
        &self,
        p_create_info: &PrivateDataSlotCreateInfo,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<PrivateDataSlot> {
        let fp = self
            .commands()
            .create_private_data_slot
            .expect("vkCreatePrivateDataSlot not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyPrivateDataSlot`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyPrivateDataSlot.html).
    /**
    Provided by **VK_BASE_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `privateDataSlot` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a private data slot. Any data previously stored via
    ///`set_private_data` with this slot is discarded. Objects that had
    ///data set are not affected, they continue to exist normally.
    pub unsafe fn destroy_private_data_slot(
        &self,
        private_data_slot: PrivateDataSlot,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_private_data_slot
            .expect("vkDestroyPrivateDataSlot not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), private_data_slot, alloc_ptr) };
    }
    ///Wraps [`vkSetPrivateData`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkSetPrivateData.html).
    /**
    Provided by **VK_BASE_VERSION_1_3**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Stores a `u64` value on any Vulkan object for the given private data
    ///slot. Overwrites any previously stored value for this object/slot
    ///pair.
    ///
    ///Private data is per-device, the slot must have been created from
    ///the same device that owns the object. Setting data on an object from
    ///a different device is undefined behaviour.
    ///
    ///To clear the association, set the value to 0 or destroy the slot.
    pub unsafe fn set_private_data(
        &self,
        object_type: ObjectType,
        object_handle: u64,
        private_data_slot: PrivateDataSlot,
        data: u64,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .set_private_data
            .expect("vkSetPrivateData not loaded");
        check(unsafe {
            fp(
                self.handle(),
                object_type,
                object_handle,
                private_data_slot,
                data,
            )
        })
    }
    ///Wraps [`vkGetPrivateData`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPrivateData.html).
    /**
    Provided by **VK_BASE_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves the `u64` value previously stored on a Vulkan object with
    ///`set_private_data` for the given private data slot. Returns 0 if no
    ///data was set for this object/slot combination.
    ///
    ///This is a lightweight per-object metadata lookup with no hash table
    ///overhead. See `create_private_data_slot` for usage patterns.
    pub unsafe fn get_private_data(
        &self,
        object_type: ObjectType,
        object_handle: u64,
        private_data_slot: PrivateDataSlot,
    ) -> u64 {
        let fp = self
            .commands()
            .get_private_data
            .expect("vkGetPrivateData not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe {
            fp(
                self.handle(),
                object_type,
                object_handle,
                private_data_slot,
                &mut out,
            )
        };
        out
    }
    ///Wraps [`vkCmdCopyBuffer2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyBuffer2.html).
    /**
    Provided by **VK_BASE_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.3 version of `cmd_copy_buffer` that uses an extensible
    ///`CopyBufferInfo2` struct. Functionally identical to the 1.0 version
    ///but supports future extensions via pNext.
    ///
    ///Prefer this over `cmd_copy_buffer` when targeting Vulkan 1.3+.
    pub unsafe fn cmd_copy_buffer2(
        &self,
        command_buffer: CommandBuffer,
        p_copy_buffer_info: &CopyBufferInfo2,
    ) {
        let fp = self
            .commands()
            .cmd_copy_buffer2
            .expect("vkCmdCopyBuffer2 not loaded");
        unsafe { fp(command_buffer, p_copy_buffer_info) };
    }
    ///Wraps [`vkCmdCopyImage2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyImage2.html).
    /**
    Provided by **VK_BASE_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.3 version of `cmd_copy_image` that uses an extensible
    ///`CopyImageInfo2` struct.
    ///
    ///Functionally identical to the 1.0 version. Prefer this when
    ///targeting Vulkan 1.3+.
    pub unsafe fn cmd_copy_image2(
        &self,
        command_buffer: CommandBuffer,
        p_copy_image_info: &CopyImageInfo2,
    ) {
        let fp = self
            .commands()
            .cmd_copy_image2
            .expect("vkCmdCopyImage2 not loaded");
        unsafe { fp(command_buffer, p_copy_image_info) };
    }
    ///Wraps [`vkCmdBlitImage2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBlitImage2.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.3 version of `cmd_blit_image` that uses an extensible
    ///`BlitImageInfo2` struct.
    ///
    ///Chain `BlitImageCubicWeightsInfoQCOM` for cubic filtering on
    ///Qualcomm hardware. Otherwise functionally identical to
    ///`cmd_blit_image`.
    ///
    ///Prefer this when targeting Vulkan 1.3+.
    pub unsafe fn cmd_blit_image2(
        &self,
        command_buffer: CommandBuffer,
        p_blit_image_info: &BlitImageInfo2,
    ) {
        let fp = self
            .commands()
            .cmd_blit_image2
            .expect("vkCmdBlitImage2 not loaded");
        unsafe { fp(command_buffer, p_blit_image_info) };
    }
    ///Wraps [`vkCmdCopyBufferToImage2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyBufferToImage2.html).
    /**
    Provided by **VK_BASE_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.3 version of `cmd_copy_buffer_to_image` that uses an
    ///extensible `CopyBufferToImageInfo2` struct.
    ///
    ///Functionally identical to the 1.0 version. Prefer this when
    ///targeting Vulkan 1.3+.
    pub unsafe fn cmd_copy_buffer_to_image2(
        &self,
        command_buffer: CommandBuffer,
        p_copy_buffer_to_image_info: &CopyBufferToImageInfo2,
    ) {
        let fp = self
            .commands()
            .cmd_copy_buffer_to_image2
            .expect("vkCmdCopyBufferToImage2 not loaded");
        unsafe { fp(command_buffer, p_copy_buffer_to_image_info) };
    }
    ///Wraps [`vkCmdCopyImageToBuffer2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyImageToBuffer2.html).
    /**
    Provided by **VK_BASE_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.3 version of `cmd_copy_image_to_buffer` that uses an
    ///extensible `CopyImageToBufferInfo2` struct.
    ///
    ///Functionally identical to the 1.0 version. Prefer this when
    ///targeting Vulkan 1.3+.
    pub unsafe fn cmd_copy_image_to_buffer2(
        &self,
        command_buffer: CommandBuffer,
        p_copy_image_to_buffer_info: &CopyImageToBufferInfo2,
    ) {
        let fp = self
            .commands()
            .cmd_copy_image_to_buffer2
            .expect("vkCmdCopyImageToBuffer2 not loaded");
        unsafe { fp(command_buffer, p_copy_image_to_buffer_info) };
    }
    ///Wraps [`vkCmdResolveImage2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdResolveImage2.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.3 version of `cmd_resolve_image` that uses an extensible
    ///`ResolveImageInfo2` struct.
    ///
    ///Functionally identical to `cmd_resolve_image`. Prefer this when
    ///targeting Vulkan 1.3+.
    pub unsafe fn cmd_resolve_image2(
        &self,
        command_buffer: CommandBuffer,
        p_resolve_image_info: &ResolveImageInfo2,
    ) {
        let fp = self
            .commands()
            .cmd_resolve_image2
            .expect("vkCmdResolveImage2 not loaded");
        unsafe { fp(command_buffer, p_resolve_image_info) };
    }
    ///Wraps [`vkCmdRefreshObjectsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdRefreshObjectsKHR.html).
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Refreshes a set of Vulkan objects managed by a refreshable object
    ///type, resetting their internal state. Used in safety-critical
    ///Vulkan SC environments to periodically refresh objects and detect
    ///hardware faults.
    ///
    ///Not relevant for desktop Vulkan. This is part of the
    ///`VK_KHR_object_refresh` extension used in automotive and embedded
    ///safety-critical environments.
    pub unsafe fn cmd_refresh_objects_khr(
        &self,
        command_buffer: CommandBuffer,
        p_refresh_objects: &RefreshObjectListKHR,
    ) {
        let fp = self
            .commands()
            .cmd_refresh_objects_khr
            .expect("vkCmdRefreshObjectsKHR not loaded");
        unsafe { fp(command_buffer, p_refresh_objects) };
    }
    ///Wraps [`vkCmdSetFragmentShadingRateKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetFragmentShadingRateKHR.html).
    /**
    Provided by **VK_KHR_fragment_shading_rate**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Sets the pipeline fragment shading rate and combiner operations
    ///as dynamic state. This controls how many pixels each fragment
    ///shader invocation covers, larger fragment sizes reduce shading
    ///cost at the expense of detail.
    ///
    ///`p_fragment_size` specifies the fragment size (e.g., 1x1, 1x2,
    ///2x2, 2x4, 4x4). Not all sizes are supported, query
    ///`get_physical_device_fragment_shading_rates_khr` for the list.
    ///
    ///`combiner_ops` defines how the pipeline rate, primitive rate
    ///(from the vertex shader), and attachment rate (from a shading
    ///rate image) are combined to produce the final rate.
    ///
    ///Requires `VK_KHR_fragment_shading_rate` and the
    ///`pipelineFragmentShadingRate` device feature.
    pub unsafe fn cmd_set_fragment_shading_rate_khr(
        &self,
        command_buffer: CommandBuffer,
        p_fragment_size: &Extent2D,
        combiner_ops: FragmentShadingRateCombinerOpKHR,
    ) {
        let fp = self
            .commands()
            .cmd_set_fragment_shading_rate_khr
            .expect("vkCmdSetFragmentShadingRateKHR not loaded");
        unsafe { fp(command_buffer, p_fragment_size, combiner_ops) };
    }
    ///Wraps [`vkCmdSetFragmentShadingRateEnumNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetFragmentShadingRateEnumNV.html).
    /**
    Provided by **VK_NV_fragment_shading_rate_enums**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets the fragment shading rate using the NV-specific
    ///enum values. Provides finer control than the KHR variant,
    ///including support for supersample and no-invocations rates.
    ///
    ///Requires `VK_NV_fragment_shading_rate_enums`.
    pub unsafe fn cmd_set_fragment_shading_rate_enum_nv(
        &self,
        command_buffer: CommandBuffer,
        shading_rate: FragmentShadingRateNV,
        combiner_ops: FragmentShadingRateCombinerOpKHR,
    ) {
        let fp = self
            .commands()
            .cmd_set_fragment_shading_rate_enum_nv
            .expect("vkCmdSetFragmentShadingRateEnumNV not loaded");
        unsafe { fp(command_buffer, shading_rate, combiner_ops) };
    }
    ///Wraps [`vkGetAccelerationStructureBuildSizesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetAccelerationStructureBuildSizesKHR.html).
    /**
    Provided by **VK_KHR_acceleration_structure**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the buffer sizes needed to build an acceleration structure:
    ///the final structure size, the scratch buffer size for builds, and
    ///the scratch buffer size for updates.
    ///
    ///Call this before creating the acceleration structure and scratch
    ///buffers. The `max_primitive_counts` parameter specifies the maximum
    ///number of primitives per geometry, the returned sizes are
    ///worst-case guarantees for those counts.
    ///
    ///The actual built size may be smaller. For BLASes, build with
    ///`ALLOW_COMPACTION` and query the compacted size afterwards to
    ///reclaim excess memory.
    pub unsafe fn get_acceleration_structure_build_sizes_khr(
        &self,
        build_type: AccelerationStructureBuildTypeKHR,
        p_build_info: &AccelerationStructureBuildGeometryInfoKHR,
        p_max_primitive_counts: *const u32,
        p_size_info: *mut AccelerationStructureBuildSizesInfoKHR,
    ) {
        let fp = self
            .commands()
            .get_acceleration_structure_build_sizes_khr
            .expect("vkGetAccelerationStructureBuildSizesKHR not loaded");
        unsafe {
            fp(
                self.handle(),
                build_type,
                p_build_info,
                p_max_primitive_counts,
                p_size_info,
            )
        };
    }
    ///Wraps [`vkCmdSetVertexInputEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetVertexInputEXT.html).
    /**
    Provided by **VK_EXT_vertex_input_dynamic_state**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets the complete vertex input state: bindings
    ///(stride, input rate) and attributes (location, format, offset).
    ///
    ///Replaces `VertexInputBindingDescription` and
    ///`VertexInputAttributeDescription` from pipeline creation. Pass
    ///arrays of `VertexInputBindingDescription2EXT` and
    ///`VertexInputAttributeDescription2EXT`.
    ///
    ///Provided by `VK_EXT_vertex_input_dynamic_state`.
    pub unsafe fn cmd_set_vertex_input_ext(
        &self,
        command_buffer: CommandBuffer,
        p_vertex_binding_descriptions: &[VertexInputBindingDescription2EXT],
        p_vertex_attribute_descriptions: &[VertexInputAttributeDescription2EXT],
    ) {
        let fp = self
            .commands()
            .cmd_set_vertex_input_ext
            .expect("vkCmdSetVertexInputEXT not loaded");
        unsafe {
            fp(
                command_buffer,
                p_vertex_binding_descriptions.len() as u32,
                p_vertex_binding_descriptions.as_ptr(),
                p_vertex_attribute_descriptions.len() as u32,
                p_vertex_attribute_descriptions.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdSetColorWriteEnableEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetColorWriteEnableEXT.html).
    /**
    Provided by **VK_EXT_color_write_enable**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically enables or disables color writing for each color
    ///attachment. Pass a slice of `Bool32` values, one per attachment.
    ///
    ///When color write is disabled for an attachment, no color output
    ///is written regardless of blend state.
    ///
    ///Unlike `cmd_set_color_write_mask_ext` (which controls per-channel
    ///masking), this is a simple on/off toggle per attachment.
    ///
    ///Provided by `VK_EXT_color_write_enable`.
    pub unsafe fn cmd_set_color_write_enable_ext(
        &self,
        command_buffer: CommandBuffer,
        p_color_write_enables: &[u32],
    ) {
        let fp = self
            .commands()
            .cmd_set_color_write_enable_ext
            .expect("vkCmdSetColorWriteEnableEXT not loaded");
        unsafe {
            fp(
                command_buffer,
                p_color_write_enables.len() as u32,
                p_color_write_enables.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdSetEvent2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetEvent2.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.3 version of `cmd_set_event` that takes a `DependencyInfo`
    ///describing the memory dependencies, rather than just a stage mask.
    ///
    ///This provides more precise dependency information to the driver and
    ///supports 64-bit stage and access flags. The dependency info specifies
    ///exactly what memory accesses and pipeline stages are involved, which
    ///can reduce unnecessary stalls.
    ///
    ///Prefer this over `cmd_set_event` when targeting Vulkan 1.3+.
    pub unsafe fn cmd_set_event2(
        &self,
        command_buffer: CommandBuffer,
        event: Event,
        p_dependency_info: &DependencyInfo,
    ) {
        let fp = self
            .commands()
            .cmd_set_event2
            .expect("vkCmdSetEvent2 not loaded");
        unsafe { fp(command_buffer, event, p_dependency_info) };
    }
    ///Wraps [`vkCmdResetEvent2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdResetEvent2.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.3 version of `cmd_reset_event` that uses 64-bit pipeline
    ///stage flags. Supports newer stages not available in the original
    ///32-bit field.
    ///
    ///Prefer this over `cmd_reset_event` when targeting Vulkan 1.3+.
    pub unsafe fn cmd_reset_event2(
        &self,
        command_buffer: CommandBuffer,
        event: Event,
        stage_mask: PipelineStageFlags2,
    ) {
        let fp = self
            .commands()
            .cmd_reset_event2
            .expect("vkCmdResetEvent2 not loaded");
        unsafe { fp(command_buffer, event, stage_mask) };
    }
    ///Wraps [`vkCmdWaitEvents2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdWaitEvents2.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.3 version of `cmd_wait_events` that takes per-event
    ///`DependencyInfo` structs instead of global stage masks and barriers.
    ///
    ///Each event can have its own set of memory barriers and stage masks,
    ///giving the driver more precise information about what each event
    ///protects.
    ///
    ///Prefer this over `cmd_wait_events` when targeting Vulkan 1.3+.
    pub unsafe fn cmd_wait_events2(
        &self,
        command_buffer: CommandBuffer,
        p_events: &[Event],
        p_dependency_infos: &DependencyInfo,
    ) {
        let fp = self
            .commands()
            .cmd_wait_events2
            .expect("vkCmdWaitEvents2 not loaded");
        unsafe {
            fp(
                command_buffer,
                p_events.len() as u32,
                p_events.as_ptr(),
                p_dependency_infos,
            )
        };
    }
    ///Wraps [`vkCmdPipelineBarrier2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdPipelineBarrier2.html).
    /**
    Provided by **VK_BASE_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.3 version of `cmd_pipeline_barrier` that uses
    ///`DependencyInfo` with `MemoryBarrier2`, `BufferMemoryBarrier2`, and
    ///`ImageMemoryBarrier2` structs.
    ///
    ///The key improvement over the 1.0 version is that stage and access
    ///masks are specified **per barrier** rather than globally for the
    ///entire call. This gives the driver more precise dependency
    ///information, which can reduce unnecessary stalls.
    ///
    ///The 1.3 barrier structs also use 64-bit stage and access flags,
    ///supporting stages and access types that do not fit in the original
    ///32-bit fields (e.g. ray tracing, mesh shading).
    ///
    ///Prefer this over `cmd_pipeline_barrier` when targeting Vulkan 1.3+.
    pub unsafe fn cmd_pipeline_barrier2(
        &self,
        command_buffer: CommandBuffer,
        p_dependency_info: &DependencyInfo,
    ) {
        let fp = self
            .commands()
            .cmd_pipeline_barrier2
            .expect("vkCmdPipelineBarrier2 not loaded");
        unsafe { fp(command_buffer, p_dependency_info) };
    }
    ///Wraps [`vkQueueSubmit2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkQueueSubmit2.html).
    /**
    Provided by **VK_BASE_VERSION_1_3**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_DEVICE_LOST`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `queue` (self) must be valid and not destroyed.
    ///- `queue` must be externally synchronized.
    ///- `fence` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.3 version of `queue_submit` that uses `SubmitInfo2` with
    ///per-semaphore stage masks and 64-bit pipeline stage flags.
    ///
    ///Key improvements over `queue_submit`:
    ///
    ///- **Per-semaphore stage masks**: each wait semaphore has its own
    ///  stage mask in `SemaphoreSubmitInfo`, instead of a parallel array.
    ///  Clearer and less error-prone.
    ///- **64-bit stages**: supports newer pipeline stages.
    ///- **Timeline semaphores**: timeline values are embedded in
    ///  `SemaphoreSubmitInfo` instead of requiring a separate pNext
    ///  chain.
    ///
    ///Prefer this over `queue_submit` when targeting Vulkan 1.3+. The
    ///fence parameter works identically.
    pub unsafe fn queue_submit2(
        &self,
        queue: Queue,
        p_submits: &[SubmitInfo2],
        fence: Fence,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .queue_submit2
            .expect("vkQueueSubmit2 not loaded");
        check(unsafe { fp(queue, p_submits.len() as u32, p_submits.as_ptr(), fence) })
    }
    ///Wraps [`vkCmdWriteTimestamp2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdWriteTimestamp2.html).
    /**
    Provided by **VK_BASE_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.3 version of `cmd_write_timestamp` that uses 64-bit
    ///pipeline stage flags (`PipelineStageFlags2`).
    ///
    ///The wider stage flags support newer stages like
    ///`PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR` and
    ///`PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT` that do not fit in the
    ///original 32-bit field.
    ///
    ///Prefer this over `cmd_write_timestamp` when targeting Vulkan 1.3+.
    pub unsafe fn cmd_write_timestamp2(
        &self,
        command_buffer: CommandBuffer,
        stage: PipelineStageFlags2,
        query_pool: QueryPool,
        query: u32,
    ) {
        let fp = self
            .commands()
            .cmd_write_timestamp2
            .expect("vkCmdWriteTimestamp2 not loaded");
        unsafe { fp(command_buffer, stage, query_pool, query) };
    }
    ///Wraps [`vkCmdWriteBufferMarker2AMD`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdWriteBufferMarker2AMD.html).
    /**
    Provided by **VK_AMD_buffer_marker**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Extended version of `cmd_write_buffer_marker_amd` that uses
    ///`PipelineStageFlags2` for the stage mask. Supports the full
    ///synchronization2 stage flags.
    ///
    ///Requires `VK_AMD_buffer_marker` + `VK_KHR_synchronization2`.
    pub unsafe fn cmd_write_buffer_marker2_amd(
        &self,
        command_buffer: CommandBuffer,
        stage: PipelineStageFlags2,
        dst_buffer: Buffer,
        dst_offset: u64,
        marker: u32,
    ) {
        let fp = self
            .commands()
            .cmd_write_buffer_marker2_amd
            .expect("vkCmdWriteBufferMarker2AMD not loaded");
        unsafe { fp(command_buffer, stage, dst_buffer, dst_offset, marker) };
    }
    ///Wraps [`vkGetQueueCheckpointData2NV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetQueueCheckpointData2NV.html).
    /**
    Provided by **VK_NV_device_diagnostic_checkpoints**.*/
    ///
    ///# Safety
    ///- `queue` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Extended version of `get_queue_checkpoint_data_nv` that returns
    ///pipeline stage information alongside the checkpoint markers.
    ///Use for finer-grained post-mortem debugging after device loss.
    ///
    ///Requires `VK_NV_device_diagnostic_checkpoints` +
    ///`VK_KHR_synchronization2`.
    pub unsafe fn get_queue_checkpoint_data2_nv(&self, queue: Queue) -> Vec<CheckpointData2NV> {
        let fp = self
            .commands()
            .get_queue_checkpoint_data2_nv
            .expect("vkGetQueueCheckpointData2NV not loaded");
        fill_two_call(|count, data| unsafe { fp(queue, count, data) })
    }
    ///Wraps [`vkCopyMemoryToImage`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCopyMemoryToImage.html).
    /**
    Provided by **VK_BASE_VERSION_1_4**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_MEMORY_MAP_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.4 host-side image upload. Copies texel data from a host
    ///memory pointer directly into an image without a staging buffer or
    ///command buffer.
    ///
    ///The image must be in `GENERAL` layout and must have been created
    ///with `HOST_TRANSFER` usage. The copy happens synchronously.
    ///
    ///This simplifies upload workflows that previously required a staging
    ///buffer + `cmd_copy_buffer_to_image` + layout transitions. However,
    ///the image must support host transfer and be in `GENERAL` layout,
    ///which may not be optimal for subsequent GPU reads.
    pub unsafe fn copy_memory_to_image(
        &self,
        p_copy_memory_to_image_info: &CopyMemoryToImageInfo,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .copy_memory_to_image
            .expect("vkCopyMemoryToImage not loaded");
        check(unsafe { fp(self.handle(), p_copy_memory_to_image_info) })
    }
    ///Wraps [`vkCopyImageToMemory`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCopyImageToMemory.html).
    /**
    Provided by **VK_BASE_VERSION_1_4**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_MEMORY_MAP_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.4 host-side image readback. Copies texel data from an
    ///image directly to a host memory pointer without a staging buffer or
    ///command buffer.
    ///
    ///The image must be in `GENERAL` layout and must have been created
    ///with `HOST_TRANSFER` usage. The copy happens synchronously.
    ///
    ///This simplifies CPU readback workflows that previously required a
    ///staging buffer + `cmd_copy_image_to_buffer` + fence wait + map.
    ///However, it requires the image to support host transfer, which not
    ///all implementations or formats support.
    pub unsafe fn copy_image_to_memory(
        &self,
        p_copy_image_to_memory_info: &CopyImageToMemoryInfo,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .copy_image_to_memory
            .expect("vkCopyImageToMemory not loaded");
        check(unsafe { fp(self.handle(), p_copy_image_to_memory_info) })
    }
    ///Wraps [`vkCopyImageToImage`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCopyImageToImage.html).
    /**
    Provided by **VK_BASE_VERSION_1_4**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_MEMORY_MAP_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.4 host-side image-to-image copy. Copies texel data between
    ///two images from the CPU without recording a command buffer.
    ///
    ///Both images must be in `GENERAL` layout and must have been created
    ///with `HOST_TRANSFER` usage. The copy happens synchronously on the
    ///calling thread.
    ///
    ///Use cases are limited to CPU-side image manipulation (e.g. test
    ///utilities, offline processing). For GPU-side copies in a render
    ///loop, `cmd_copy_image2` is the standard path.
    pub unsafe fn copy_image_to_image(
        &self,
        p_copy_image_to_image_info: &CopyImageToImageInfo,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .copy_image_to_image
            .expect("vkCopyImageToImage not loaded");
        check(unsafe { fp(self.handle(), p_copy_image_to_image_info) })
    }
    ///Wraps [`vkTransitionImageLayout`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkTransitionImageLayout.html).
    /**
    Provided by **VK_BASE_VERSION_1_4**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_MEMORY_MAP_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.4 host-side image layout transition. Transitions an image
    ///between layouts from the CPU without recording a command buffer.
    ///
    ///The image must have been created with `HOST_TRANSFER` usage. The
    ///transition happens synchronously on the calling thread.
    ///
    ///This simplifies workflows where you need to transition an image
    ///layout outside of a command buffer, for example, transitioning a
    ///newly created host-transfer image from `UNDEFINED` to `GENERAL`
    ///before performing host-side copies.
    ///
    ///For GPU-side layout transitions (the common case), use
    ///`cmd_pipeline_barrier2` with an image memory barrier.
    pub unsafe fn transition_image_layout(
        &self,
        p_transitions: &[HostImageLayoutTransitionInfo],
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .transition_image_layout
            .expect("vkTransitionImageLayout not loaded");
        check(unsafe {
            fp(
                self.handle(),
                p_transitions.len() as u32,
                p_transitions.as_ptr(),
            )
        })
    }
    ///Wraps [`vkGetCommandPoolMemoryConsumption`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetCommandPoolMemoryConsumption.html).
    /**
    Provided by **VKSC_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `commandPool` must be externally synchronized.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Queries the memory consumption of a command pool or a specific
    ///command buffer within it. Part of Vulkan SC (Safety Critical)
    ///for tracking resource budgets in safety-certified environments.
    ///Pass a null command buffer to query the entire pool.
    ///
    ///Requires Vulkan SC.
    pub unsafe fn get_command_pool_memory_consumption(
        &self,
        command_pool: CommandPool,
        command_buffer: CommandBuffer,
        p_consumption: *mut CommandPoolMemoryConsumption,
    ) {
        let fp = self
            .commands()
            .get_command_pool_memory_consumption
            .expect("vkGetCommandPoolMemoryConsumption not loaded");
        unsafe { fp(self.handle(), command_pool, command_buffer, p_consumption) };
    }
    ///Wraps [`vkCreateVideoSessionKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateVideoSessionKHR.html).
    /**
    Provided by **VK_KHR_video_queue**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR`
    ///- `VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a video session for hardware-accelerated video decoding
    ///or encoding. The session defines the video codec profile,
    ///resolution range, format, and maximum reference picture count.
    ///
    ///Key fields in `VideoSessionCreateInfoKHR`:
    ///
    ///- `video_profile`: codec (H.264, H.265, AV1) and profile/level.
    ///- `max_coded_extent`: maximum frame resolution.
    ///- `picture_format` / `reference_picture_format`: image formats
    ///  for decoded pictures and DPB (decoded picture buffer) slots.
    ///- `max_dpb_slots` / `max_active_reference_pictures`: reference
    ///  frame capacity.
    ///
    ///After creation, query memory requirements with
    ///`get_video_session_memory_requirements_khr`, allocate and bind
    ///memory with `bind_video_session_memory_khr`, then create session
    ///parameters with `create_video_session_parameters_khr`.
    pub unsafe fn create_video_session_khr(
        &self,
        p_create_info: &VideoSessionCreateInfoKHR,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<VideoSessionKHR> {
        let fp = self
            .commands()
            .create_video_session_khr
            .expect("vkCreateVideoSessionKHR not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyVideoSessionKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyVideoSessionKHR.html).
    /**
    Provided by **VK_KHR_video_queue**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `videoSession` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a video session and releases its internal resources.
    ///Any video session parameters created against this session become
    ///invalid, destroy them first.
    ///
    ///All command buffers referencing this session must have completed
    ///execution before destruction.
    pub unsafe fn destroy_video_session_khr(
        &self,
        video_session: VideoSessionKHR,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_video_session_khr
            .expect("vkDestroyVideoSessionKHR not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), video_session, alloc_ptr) };
    }
    ///Wraps [`vkCreateVideoSessionParametersKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateVideoSessionParametersKHR.html).
    /**
    Provided by **VK_KHR_video_queue**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a video session parameters object that holds codec-specific
    ///parameter sets (SPS, PPS for H.264/H.265, sequence headers for
    ///AV1). These are referenced during decode/encode operations.
    ///
    ///Chain the appropriate codec-specific struct into `pNext`:
    ///
    ///- `VideoDecodeH264SessionParametersCreateInfoKHR` for H.264 decode.
    ///- `VideoDecodeH265SessionParametersCreateInfoKHR` for H.265 decode.
    ///- `VideoEncodeH264SessionParametersCreateInfoKHR` for H.264 encode.
    ///
    ///Parameters can be added incrementally with
    ///`update_video_session_parameters_khr`. A template parameter object
    ///can be specified to inherit existing parameters.
    pub unsafe fn create_video_session_parameters_khr(
        &self,
        p_create_info: &VideoSessionParametersCreateInfoKHR,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<VideoSessionParametersKHR> {
        let fp = self
            .commands()
            .create_video_session_parameters_khr
            .expect("vkCreateVideoSessionParametersKHR not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkUpdateVideoSessionParametersKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkUpdateVideoSessionParametersKHR.html).
    /**
    Provided by **VK_KHR_video_queue**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Adds new codec-specific parameter sets to an existing video
    ///session parameters object. For example, adding new SPS/PPS
    ///entries for H.264 as they are encountered in the bitstream.
    ///
    ///Chain the codec-specific update struct into the `pNext` of
    ///`VideoSessionParametersUpdateInfoKHR`. The `update_sequence_count`
    ///must increment monotonically with each update.
    ///
    ///Parameters cannot be removed or modified, only new entries can
    ///be added. If a parameter set with the same ID already exists,
    ///the update fails.
    pub unsafe fn update_video_session_parameters_khr(
        &self,
        video_session_parameters: VideoSessionParametersKHR,
        p_update_info: &VideoSessionParametersUpdateInfoKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .update_video_session_parameters_khr
            .expect("vkUpdateVideoSessionParametersKHR not loaded");
        check(unsafe { fp(self.handle(), video_session_parameters, p_update_info) })
    }
    ///Wraps [`vkGetEncodedVideoSessionParametersKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetEncodedVideoSessionParametersKHR.html).
    /**
    Provided by **VK_KHR_video_encode_queue**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves the encoded (serialized) form of video session
    ///parameters, typically codec headers (SPS/PPS for H.264/H.265,
    ///sequence header for AV1) that must be prepended to the encoded
    ///bitstream.
    ///
    ///Uses the two-call pattern: call with null `p_data` to query
    ///the size, allocate, then call again to fill the buffer.
    ///
    ///The `p_feedback_info` output indicates whether the driver
    ///modified or overrode any parameters relative to what was
    ///requested (check `has_overrides`).
    ///
    ///This data is the codec parameter payload that decoders need to
    ///initialize before processing encoded frames.
    pub unsafe fn get_encoded_video_session_parameters_khr(
        &self,
        p_video_session_parameters_info: &VideoEncodeSessionParametersGetInfoKHR,
        p_feedback_info: *mut VideoEncodeSessionParametersFeedbackInfoKHR,
        p_data: *mut core::ffi::c_void,
    ) -> VkResult<usize> {
        let fp = self
            .commands()
            .get_encoded_video_session_parameters_khr
            .expect("vkGetEncodedVideoSessionParametersKHR not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe {
            fp(
                self.handle(),
                p_video_session_parameters_info,
                p_feedback_info,
                &mut out,
                p_data,
            )
        })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyVideoSessionParametersKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyVideoSessionParametersKHR.html).
    /**
    Provided by **VK_KHR_video_queue**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `videoSessionParameters` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a video session parameters object. All command buffers
    ///referencing these parameters must have completed execution before
    ///destruction.
    ///
    ///The video session itself is not affected, other parameter objects
    ///associated with the same session remain valid.
    pub unsafe fn destroy_video_session_parameters_khr(
        &self,
        video_session_parameters: VideoSessionParametersKHR,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_video_session_parameters_khr
            .expect("vkDestroyVideoSessionParametersKHR not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), video_session_parameters, alloc_ptr) };
    }
    ///Wraps [`vkGetVideoSessionMemoryRequirementsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetVideoSessionMemoryRequirementsKHR.html).
    /**
    Provided by **VK_KHR_video_queue**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the memory requirements for a video session. A video
    ///session may require multiple memory bindings (each with different
    ///memory type requirements) for internal buffers used during
    ///decode/encode.
    ///
    ///Each returned `VideoSessionMemoryRequirementsKHR` has a
    ///`memory_bind_index` and a `MemoryRequirements` describing the
    ///size, alignment, and compatible memory types.
    ///
    ///Allocate a `DeviceMemory` for each requirement and bind them all
    ///with `bind_video_session_memory_khr` before using the session.
    pub unsafe fn get_video_session_memory_requirements_khr(
        &self,
        video_session: VideoSessionKHR,
    ) -> VkResult<Vec<VideoSessionMemoryRequirementsKHR>> {
        let fp = self
            .commands()
            .get_video_session_memory_requirements_khr
            .expect("vkGetVideoSessionMemoryRequirementsKHR not loaded");
        enumerate_two_call(|count, data| unsafe { fp(self.handle(), video_session, count, data) })
    }
    ///Wraps [`vkBindVideoSessionMemoryKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkBindVideoSessionMemoryKHR.html).
    /**
    Provided by **VK_KHR_video_queue**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `videoSession` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Binds device memory to a video session. Each binding corresponds
    ///to a `memory_bind_index` from
    ///`get_video_session_memory_requirements_khr`.
    ///
    ///All required memory bindings must be satisfied before the session
    ///can be used in video coding operations. Each
    ///`BindVideoSessionMemoryInfoKHR` specifies the bind index, memory
    ///object, offset, and size.
    ///
    ///Memory can only be bound once per index, rebinding is not
    ///allowed.
    pub unsafe fn bind_video_session_memory_khr(
        &self,
        video_session: VideoSessionKHR,
        p_bind_session_memory_infos: &[BindVideoSessionMemoryInfoKHR],
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .bind_video_session_memory_khr
            .expect("vkBindVideoSessionMemoryKHR not loaded");
        check(unsafe {
            fp(
                self.handle(),
                video_session,
                p_bind_session_memory_infos.len() as u32,
                p_bind_session_memory_infos.as_ptr(),
            )
        })
    }
    ///Wraps [`vkCmdDecodeVideoKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDecodeVideoKHR.html).
    /**
    Provided by **VK_KHR_video_decode_queue**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Decodes a single video frame from a compressed bitstream.
    ///Must be recorded within a video coding scope
    ///(`cmd_begin_video_coding_khr` / `cmd_end_video_coding_khr`).
    ///
    ///`VideoDecodeInfoKHR` specifies:
    ///
    ///- `src_buffer` / `src_buffer_offset` / `src_buffer_range`: the
    ///  bitstream data containing the compressed frame.
    ///- `dst_picture_resource`: the output image view for the decoded
    ///  frame.
    ///- `setup_reference_slot`: DPB slot to store this frame for use
    ///  as a reference by future frames.
    ///- `reference_slots`: previously decoded reference frames needed
    ///  to decode this frame.
    ///
    ///Chain codec-specific decode info (e.g.,
    ///`VideoDecodeH264PictureInfoKHR`) into `pNext`.
    pub unsafe fn cmd_decode_video_khr(
        &self,
        command_buffer: CommandBuffer,
        p_decode_info: &VideoDecodeInfoKHR,
    ) {
        let fp = self
            .commands()
            .cmd_decode_video_khr
            .expect("vkCmdDecodeVideoKHR not loaded");
        unsafe { fp(command_buffer, p_decode_info) };
    }
    ///Wraps [`vkCmdBeginVideoCodingKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBeginVideoCodingKHR.html).
    /**
    Provided by **VK_KHR_video_queue**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Begins a video coding scope within a command buffer. All video
    ///decode and encode commands must be recorded between
    ///`cmd_begin_video_coding_khr` and `cmd_end_video_coding_khr`.
    ///
    ///`VideoBeginCodingInfoKHR` specifies:
    ///
    ///- `video_session`: the session to use.
    ///- `video_session_parameters`: codec parameters (SPS/PPS, etc.).
    ///- `reference_slots`: DPB (decoded picture buffer) slots and their
    ///  associated image views for reference pictures.
    ///
    ///The command buffer must be allocated from a queue family that
    ///supports the appropriate video operations (decode or encode),
    ///as reported by `QueueFamilyVideoPropertiesKHR`.
    pub unsafe fn cmd_begin_video_coding_khr(
        &self,
        command_buffer: CommandBuffer,
        p_begin_info: &VideoBeginCodingInfoKHR,
    ) {
        let fp = self
            .commands()
            .cmd_begin_video_coding_khr
            .expect("vkCmdBeginVideoCodingKHR not loaded");
        unsafe { fp(command_buffer, p_begin_info) };
    }
    ///Wraps [`vkCmdControlVideoCodingKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdControlVideoCodingKHR.html).
    /**
    Provided by **VK_KHR_video_queue**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Issues control commands within a video coding scope. Used to
    ///reset the video session state or set encode quality/rate control
    ///parameters.
    ///
    ///`VideoCodingControlInfoKHR` flags include:
    ///
    ///- `RESET`: resets the video session to a clean state, clearing
    ///  all DPB slots and internal codec state.
    ///- `ENCODE_RATE_CONTROL`: applies rate control settings (chain
    ///  `VideoEncodeRateControlInfoKHR` into `pNext`).
    ///- `ENCODE_QUALITY_LEVEL`: sets the encode quality level.
    ///
    ///Must be recorded between `cmd_begin_video_coding_khr` and
    ///`cmd_end_video_coding_khr`.
    pub unsafe fn cmd_control_video_coding_khr(
        &self,
        command_buffer: CommandBuffer,
        p_coding_control_info: &VideoCodingControlInfoKHR,
    ) {
        let fp = self
            .commands()
            .cmd_control_video_coding_khr
            .expect("vkCmdControlVideoCodingKHR not loaded");
        unsafe { fp(command_buffer, p_coding_control_info) };
    }
    ///Wraps [`vkCmdEndVideoCodingKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdEndVideoCodingKHR.html).
    /**
    Provided by **VK_KHR_video_queue**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Ends a video coding scope previously started with
    ///`cmd_begin_video_coding_khr`. After this call, video decode and
    ///encode commands can no longer be recorded until a new scope is
    ///started.
    ///
    ///The `VideoEndCodingInfoKHR` struct is currently reserved for
    ///future use (no flags defined).
    pub unsafe fn cmd_end_video_coding_khr(
        &self,
        command_buffer: CommandBuffer,
        p_end_coding_info: &VideoEndCodingInfoKHR,
    ) {
        let fp = self
            .commands()
            .cmd_end_video_coding_khr
            .expect("vkCmdEndVideoCodingKHR not loaded");
        unsafe { fp(command_buffer, p_end_coding_info) };
    }
    ///Wraps [`vkCmdEncodeVideoKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdEncodeVideoKHR.html).
    /**
    Provided by **VK_KHR_video_encode_queue**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Encodes a single video frame into a compressed bitstream.
    ///Must be recorded within a video coding scope
    ///(`cmd_begin_video_coding_khr` / `cmd_end_video_coding_khr`).
    ///
    ///`VideoEncodeInfoKHR` specifies:
    ///
    ///- `dst_buffer` / `dst_buffer_offset` / `dst_buffer_range`: where
    ///  to write the compressed output.
    ///- `src_picture_resource`: the input image view to encode.
    ///- `setup_reference_slot`: DPB slot to store the reconstructed
    ///  frame for future reference.
    ///- `reference_slots`: reference frames for inter-prediction.
    ///
    ///Chain codec-specific encode info (e.g.,
    ///`VideoEncodeH264PictureInfoKHR`) into `pNext`. Configure rate
    ///control beforehand with `cmd_control_video_coding_khr`.
    pub unsafe fn cmd_encode_video_khr(
        &self,
        command_buffer: CommandBuffer,
        p_encode_info: &VideoEncodeInfoKHR,
    ) {
        let fp = self
            .commands()
            .cmd_encode_video_khr
            .expect("vkCmdEncodeVideoKHR not loaded");
        unsafe { fp(command_buffer, p_encode_info) };
    }
    ///Wraps [`vkCmdDecompressMemoryNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDecompressMemoryNV.html).
    /**
    Provided by **VK_NV_memory_decompression**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Decompresses one or more memory regions on the GPU. Each region
    ///specifies source, destination, size, and decompression method.
    ///
    ///Requires `VK_NV_memory_decompression`.
    pub unsafe fn cmd_decompress_memory_nv(
        &self,
        command_buffer: CommandBuffer,
        p_decompress_memory_regions: &[DecompressMemoryRegionNV],
    ) {
        let fp = self
            .commands()
            .cmd_decompress_memory_nv
            .expect("vkCmdDecompressMemoryNV not loaded");
        unsafe {
            fp(
                command_buffer,
                p_decompress_memory_regions.len() as u32,
                p_decompress_memory_regions.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdDecompressMemoryIndirectCountNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDecompressMemoryIndirectCountNV.html).
    /**
    Provided by **VK_NV_memory_decompression**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Indirect-count variant of `cmd_decompress_memory_nv`. Reads the
    ///decompression region descriptors and count from GPU buffer
    ///addresses, enabling fully GPU-driven decompression.
    ///
    ///Requires `VK_NV_memory_decompression`.
    pub unsafe fn cmd_decompress_memory_indirect_count_nv(
        &self,
        command_buffer: CommandBuffer,
        indirect_commands_address: u64,
        indirect_commands_count_address: u64,
        stride: u32,
    ) {
        let fp = self
            .commands()
            .cmd_decompress_memory_indirect_count_nv
            .expect("vkCmdDecompressMemoryIndirectCountNV not loaded");
        unsafe {
            fp(
                command_buffer,
                indirect_commands_address,
                indirect_commands_count_address,
                stride,
            )
        };
    }
    ///Wraps [`vkGetPartitionedAccelerationStructuresBuildSizesNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPartitionedAccelerationStructuresBuildSizesNV.html).
    /**
    Provided by **VK_NV_partitioned_acceleration_structure**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the buffer sizes needed to build a partitioned
    ///acceleration structure. Use the returned sizes to allocate
    ///destination and scratch buffers.
    ///
    ///Requires `VK_NV_partitioned_acceleration_structure`.
    pub unsafe fn get_partitioned_acceleration_structures_build_sizes_nv(
        &self,
        p_info: &PartitionedAccelerationStructureInstancesInputNV,
        p_size_info: *mut AccelerationStructureBuildSizesInfoKHR,
    ) {
        let fp = self
            .commands()
            .get_partitioned_acceleration_structures_build_sizes_nv
            .expect("vkGetPartitionedAccelerationStructuresBuildSizesNV not loaded");
        unsafe { fp(self.handle(), p_info, p_size_info) };
    }
    ///Wraps [`vkCmdBuildPartitionedAccelerationStructuresNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBuildPartitionedAccelerationStructuresNV.html).
    /**
    Provided by **VK_NV_partitioned_acceleration_structure**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Builds a partitioned acceleration structure where instances are
    ///grouped into independently updatable partitions. This allows
    ///updating subsets of the TLAS without rebuilding the entire
    ///structure.
    ///
    ///Requires `VK_NV_partitioned_acceleration_structure`.
    pub unsafe fn cmd_build_partitioned_acceleration_structures_nv(
        &self,
        command_buffer: CommandBuffer,
        p_build_info: &BuildPartitionedAccelerationStructureInfoNV,
    ) {
        let fp = self
            .commands()
            .cmd_build_partitioned_acceleration_structures_nv
            .expect("vkCmdBuildPartitionedAccelerationStructuresNV not loaded");
        unsafe { fp(command_buffer, p_build_info) };
    }
    ///Wraps [`vkCmdDecompressMemoryEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDecompressMemoryEXT.html).
    /**
    Provided by **VK_EXT_memory_decompression**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Decompresses data from one memory region into another on the GPU.
    ///The decompression algorithm is specified in the info structure.
    ///
    ///Useful for loading compressed assets directly on the GPU without
    ///a CPU round-trip.
    ///
    ///Requires `VK_EXT_memory_decompression`.
    pub unsafe fn cmd_decompress_memory_ext(
        &self,
        command_buffer: CommandBuffer,
        p_decompress_memory_info_ext: &DecompressMemoryInfoEXT,
    ) {
        let fp = self
            .commands()
            .cmd_decompress_memory_ext
            .expect("vkCmdDecompressMemoryEXT not loaded");
        unsafe { fp(command_buffer, p_decompress_memory_info_ext) };
    }
    ///Wraps [`vkCmdDecompressMemoryIndirectCountEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDecompressMemoryIndirectCountEXT.html).
    /**
    Provided by **VK_EXT_memory_decompression**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Indirect variant of `cmd_decompress_memory_ext`. Reads the
    ///decompression parameters and count from GPU-visible buffer
    ///addresses, enabling fully GPU-driven decompression workflows.
    ///
    ///Requires `VK_EXT_memory_decompression`.
    pub unsafe fn cmd_decompress_memory_indirect_count_ext(
        &self,
        command_buffer: CommandBuffer,
        decompression_method: MemoryDecompressionMethodFlagsEXT,
        indirect_commands_address: u64,
        indirect_commands_count_address: u64,
        max_decompression_count: u32,
        stride: u32,
    ) {
        let fp = self
            .commands()
            .cmd_decompress_memory_indirect_count_ext
            .expect("vkCmdDecompressMemoryIndirectCountEXT not loaded");
        unsafe {
            fp(
                command_buffer,
                decompression_method,
                indirect_commands_address,
                indirect_commands_count_address,
                max_decompression_count,
                stride,
            )
        };
    }
    ///Wraps [`vkCreateCuModuleNVX`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateCuModuleNVX.html).
    /**
    Provided by **VK_NVX_binary_import**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a CUDA module from binary data using the legacy NVX
    ///path. Prefer `create_cuda_module_nv` for new code.
    ///
    ///Destroy with `destroy_cu_module_nvx`.
    ///
    ///Requires `VK_NVX_binary_import`.
    pub unsafe fn create_cu_module_nvx(
        &self,
        p_create_info: &CuModuleCreateInfoNVX,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<CuModuleNVX> {
        let fp = self
            .commands()
            .create_cu_module_nvx
            .expect("vkCreateCuModuleNVX not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkCreateCuFunctionNVX`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateCuFunctionNVX.html).
    /**
    Provided by **VK_NVX_binary_import**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a CUDA function handle from an NVX binary module. This
    ///is the legacy NVX path; prefer `create_cuda_function_nv` for
    ///new code.
    ///
    ///Destroy with `destroy_cu_function_nvx`.
    ///
    ///Requires `VK_NVX_binary_import`.
    pub unsafe fn create_cu_function_nvx(
        &self,
        p_create_info: &CuFunctionCreateInfoNVX,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<CuFunctionNVX> {
        let fp = self
            .commands()
            .create_cu_function_nvx
            .expect("vkCreateCuFunctionNVX not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyCuModuleNVX`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyCuModuleNVX.html).
    /**
    Provided by **VK_NVX_binary_import**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Destroys a CUDA module created with `create_cu_module_nvx`.
    ///
    ///Requires `VK_NVX_binary_import`.
    pub unsafe fn destroy_cu_module_nvx(
        &self,
        module: CuModuleNVX,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_cu_module_nvx
            .expect("vkDestroyCuModuleNVX not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), module, alloc_ptr) };
    }
    ///Wraps [`vkDestroyCuFunctionNVX`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyCuFunctionNVX.html).
    /**
    Provided by **VK_NVX_binary_import**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Destroys a CUDA function handle created with
    ///`create_cu_function_nvx`.
    ///
    ///Requires `VK_NVX_binary_import`.
    pub unsafe fn destroy_cu_function_nvx(
        &self,
        function: CuFunctionNVX,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_cu_function_nvx
            .expect("vkDestroyCuFunctionNVX not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), function, alloc_ptr) };
    }
    ///Wraps [`vkCmdCuLaunchKernelNVX`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCuLaunchKernelNVX.html).
    /**
    Provided by **VK_NVX_binary_import**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Launches a CUDA kernel from a Vulkan command buffer using the
    ///legacy NVX binary import path. Prefer `cmd_cuda_launch_kernel_nv`
    ///for new code.
    ///
    ///Requires `VK_NVX_binary_import`.
    pub unsafe fn cmd_cu_launch_kernel_nvx(
        &self,
        command_buffer: CommandBuffer,
        p_launch_info: &CuLaunchInfoNVX,
    ) {
        let fp = self
            .commands()
            .cmd_cu_launch_kernel_nvx
            .expect("vkCmdCuLaunchKernelNVX not loaded");
        unsafe { fp(command_buffer, p_launch_info) };
    }
    ///Wraps [`vkGetDescriptorSetLayoutSizeEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDescriptorSetLayoutSizeEXT.html).
    /**
    Provided by **VK_EXT_descriptor_buffer**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns the total byte size required to store all descriptors for
    ///the given descriptor set layout in a descriptor buffer.
    ///
    ///Use this to allocate the correct amount of buffer memory for each
    ///descriptor set, then write individual descriptors at offsets
    ///obtained from `get_descriptor_set_layout_binding_offset_ext`.
    ///
    ///Requires `VK_EXT_descriptor_buffer`.
    pub unsafe fn get_descriptor_set_layout_size_ext(&self, layout: DescriptorSetLayout) -> u64 {
        let fp = self
            .commands()
            .get_descriptor_set_layout_size_ext
            .expect("vkGetDescriptorSetLayoutSizeEXT not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(self.handle(), layout, &mut out) };
        out
    }
    ///Wraps [`vkGetDescriptorSetLayoutBindingOffsetEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDescriptorSetLayoutBindingOffsetEXT.html).
    /**
    Provided by **VK_EXT_descriptor_buffer**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns the byte offset of a specific binding within the
    ///descriptor buffer layout for the given descriptor set layout.
    ///
    ///Use this to compute where to write a descriptor with
    ///`get_descriptor_ext` within the buffer region for a set.
    ///
    ///Requires `VK_EXT_descriptor_buffer`.
    pub unsafe fn get_descriptor_set_layout_binding_offset_ext(
        &self,
        layout: DescriptorSetLayout,
        binding: u32,
    ) -> u64 {
        let fp = self
            .commands()
            .get_descriptor_set_layout_binding_offset_ext
            .expect("vkGetDescriptorSetLayoutBindingOffsetEXT not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(self.handle(), layout, binding, &mut out) };
        out
    }
    ///Wraps [`vkGetDescriptorEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDescriptorEXT.html).
    /**
    Provided by **VK_EXT_descriptor_buffer**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Writes a descriptor directly into caller-provided memory.
    ///`DescriptorGetInfoEXT` specifies the descriptor type and resource
    ///(buffer, image, sampler, etc.). The descriptor is written to
    ///`p_descriptor` and must be `data_size` bytes.
    ///
    ///Query the required size per descriptor type with
    ///`PhysicalDeviceDescriptorBufferPropertiesEXT`.
    ///
    ///This is the core operation of descriptor buffers, instead of
    ///allocating descriptor sets, you write descriptors directly into
    ///mapped buffer memory.
    ///
    ///Requires `VK_EXT_descriptor_buffer`.
    pub unsafe fn get_descriptor_ext(
        &self,
        p_descriptor_info: &DescriptorGetInfoEXT,
        data_size: usize,
        p_descriptor: *mut core::ffi::c_void,
    ) {
        let fp = self
            .commands()
            .get_descriptor_ext
            .expect("vkGetDescriptorEXT not loaded");
        unsafe { fp(self.handle(), p_descriptor_info, data_size, p_descriptor) };
    }
    ///Wraps [`vkCmdBindDescriptorBuffersEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBindDescriptorBuffersEXT.html).
    /**
    Provided by **VK_EXT_descriptor_buffer**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Binds one or more descriptor buffers to a command buffer. Each
    ///`DescriptorBufferBindingInfoEXT` specifies a buffer address and
    ///usage (resource descriptors, sampler descriptors, or push
    ///descriptors).
    ///
    ///After binding, use `cmd_set_descriptor_buffer_offsets_ext` to
    ///point specific descriptor sets at offsets within the bound buffers.
    ///
    ///Descriptor buffers are an alternative to descriptor sets/pools
    ///that stores descriptors inline in buffer memory.
    ///
    ///Requires `VK_EXT_descriptor_buffer`.
    pub unsafe fn cmd_bind_descriptor_buffers_ext(
        &self,
        command_buffer: CommandBuffer,
        p_binding_infos: &[DescriptorBufferBindingInfoEXT],
    ) {
        let fp = self
            .commands()
            .cmd_bind_descriptor_buffers_ext
            .expect("vkCmdBindDescriptorBuffersEXT not loaded");
        unsafe {
            fp(
                command_buffer,
                p_binding_infos.len() as u32,
                p_binding_infos.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdSetDescriptorBufferOffsetsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetDescriptorBufferOffsetsEXT.html).
    /**
    Provided by **VK_EXT_descriptor_buffer**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Sets the offsets into bound descriptor buffers for one or more
    ///descriptor set slots. Each pair of (buffer_index, offset) maps
    ///a descriptor set to a region of a previously bound descriptor
    ///buffer.
    ///
    ///Must be called after `cmd_bind_descriptor_buffers_ext`.
    ///
    ///For the pNext-extensible variant, see
    ///`cmd_set_descriptor_buffer_offsets2_ext`.
    ///
    ///Requires `VK_EXT_descriptor_buffer`.
    pub unsafe fn cmd_set_descriptor_buffer_offsets_ext(
        &self,
        command_buffer: CommandBuffer,
        pipeline_bind_point: PipelineBindPoint,
        layout: PipelineLayout,
        first_set: u32,
        p_buffer_indices: &[u32],
        p_offsets: &u64,
    ) {
        let fp = self
            .commands()
            .cmd_set_descriptor_buffer_offsets_ext
            .expect("vkCmdSetDescriptorBufferOffsetsEXT not loaded");
        unsafe {
            fp(
                command_buffer,
                pipeline_bind_point,
                layout,
                first_set,
                p_buffer_indices.len() as u32,
                p_buffer_indices.as_ptr(),
                p_offsets,
            )
        };
    }
    ///Wraps [`vkCmdBindDescriptorBufferEmbeddedSamplersEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBindDescriptorBufferEmbeddedSamplersEXT.html).
    /**
    Provided by **VK_EXT_descriptor_buffer**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Binds embedded immutable samplers from a descriptor set layout
    ///that was created with `CREATE_EMBEDDED_IMMUTABLE_SAMPLERS_BIT`.
    ///These samplers are baked into the layout and do not need buffer
    ///memory.
    ///
    ///Specify the `pipeline_bind_point`, `layout`, and `set` index.
    ///
    ///For the pNext-extensible variant, see
    ///`cmd_bind_descriptor_buffer_embedded_samplers2_ext`.
    ///
    ///Requires `VK_EXT_descriptor_buffer`.
    pub unsafe fn cmd_bind_descriptor_buffer_embedded_samplers_ext(
        &self,
        command_buffer: CommandBuffer,
        pipeline_bind_point: PipelineBindPoint,
        layout: PipelineLayout,
        set: u32,
    ) {
        let fp = self
            .commands()
            .cmd_bind_descriptor_buffer_embedded_samplers_ext
            .expect("vkCmdBindDescriptorBufferEmbeddedSamplersEXT not loaded");
        unsafe { fp(command_buffer, pipeline_bind_point, layout, set) };
    }
    ///Wraps [`vkGetBufferOpaqueCaptureDescriptorDataEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetBufferOpaqueCaptureDescriptorDataEXT.html).
    /**
    Provided by **VK_EXT_descriptor_buffer**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves opaque capture data for a buffer descriptor. The
    ///returned data can be used to reconstruct the descriptor in a
    ///replay or capture/replay scenario.
    ///
    ///The buffer must have been created with
    ///`CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT`.
    ///
    ///Requires `VK_EXT_descriptor_buffer` and
    ///`descriptorBufferCaptureReplay`.
    pub unsafe fn get_buffer_opaque_capture_descriptor_data_ext(
        &self,
        p_info: &BufferCaptureDescriptorDataInfoEXT,
    ) -> VkResult<core::ffi::c_void> {
        let fp = self
            .commands()
            .get_buffer_opaque_capture_descriptor_data_ext
            .expect("vkGetBufferOpaqueCaptureDescriptorDataEXT not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_info, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetImageOpaqueCaptureDescriptorDataEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetImageOpaqueCaptureDescriptorDataEXT.html).
    /**
    Provided by **VK_EXT_descriptor_buffer**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves opaque capture data for an image descriptor. The
    ///returned data can be used to reconstruct the descriptor in a
    ///capture/replay scenario.
    ///
    ///The image must have been created with
    ///`CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT`.
    ///
    ///Requires `VK_EXT_descriptor_buffer` and
    ///`descriptorBufferCaptureReplay`.
    pub unsafe fn get_image_opaque_capture_descriptor_data_ext(
        &self,
        p_info: &ImageCaptureDescriptorDataInfoEXT,
    ) -> VkResult<core::ffi::c_void> {
        let fp = self
            .commands()
            .get_image_opaque_capture_descriptor_data_ext
            .expect("vkGetImageOpaqueCaptureDescriptorDataEXT not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_info, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetImageViewOpaqueCaptureDescriptorDataEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetImageViewOpaqueCaptureDescriptorDataEXT.html).
    /**
    Provided by **VK_EXT_descriptor_buffer**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves opaque capture data for an image view descriptor. The
    ///returned data can be used to reconstruct the descriptor in a
    ///capture/replay scenario.
    ///
    ///The image view must have been created with
    ///`CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT`.
    ///
    ///Requires `VK_EXT_descriptor_buffer` and
    ///`descriptorBufferCaptureReplay`.
    pub unsafe fn get_image_view_opaque_capture_descriptor_data_ext(
        &self,
        p_info: &ImageViewCaptureDescriptorDataInfoEXT,
    ) -> VkResult<core::ffi::c_void> {
        let fp = self
            .commands()
            .get_image_view_opaque_capture_descriptor_data_ext
            .expect("vkGetImageViewOpaqueCaptureDescriptorDataEXT not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_info, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetSamplerOpaqueCaptureDescriptorDataEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetSamplerOpaqueCaptureDescriptorDataEXT.html).
    /**
    Provided by **VK_EXT_descriptor_buffer**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves opaque capture data for a sampler descriptor. The
    ///returned data can be used to reconstruct the descriptor in a
    ///capture/replay scenario.
    ///
    ///The sampler must have been created with
    ///`CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT`.
    ///
    ///Requires `VK_EXT_descriptor_buffer` and
    ///`descriptorBufferCaptureReplay`.
    pub unsafe fn get_sampler_opaque_capture_descriptor_data_ext(
        &self,
        p_info: &SamplerCaptureDescriptorDataInfoEXT,
    ) -> VkResult<core::ffi::c_void> {
        let fp = self
            .commands()
            .get_sampler_opaque_capture_descriptor_data_ext
            .expect("vkGetSamplerOpaqueCaptureDescriptorDataEXT not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_info, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT.html).
    /**
    Provided by **VK_EXT_descriptor_buffer**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves opaque capture data for an acceleration structure
    ///descriptor. The returned data can be used to reconstruct the
    ///descriptor in a replay or capture/replay scenario.
    ///
    ///The acceleration structure must have been created with
    ///`CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT`.
    ///
    ///Requires `VK_EXT_descriptor_buffer` and
    ///`descriptorBufferCaptureReplay`.
    pub unsafe fn get_acceleration_structure_opaque_capture_descriptor_data_ext(
        &self,
        p_info: &AccelerationStructureCaptureDescriptorDataInfoEXT,
    ) -> VkResult<core::ffi::c_void> {
        let fp = self
            .commands()
            .get_acceleration_structure_opaque_capture_descriptor_data_ext
            .expect("vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_info, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkSetDeviceMemoryPriorityEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkSetDeviceMemoryPriorityEXT.html).
    /**
    Provided by **VK_EXT_pageable_device_local_memory**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Dynamically updates the priority of a device memory allocation.
    ///Higher-priority allocations are less likely to be evicted under
    ///memory pressure. Use this to promote frequently accessed
    ///resources or demote resources that are no longer critical.
    ///
    ///Requires `VK_EXT_pageable_device_local_memory`.
    pub unsafe fn set_device_memory_priority_ext(&self, memory: DeviceMemory, priority: f32) {
        let fp = self
            .commands()
            .set_device_memory_priority_ext
            .expect("vkSetDeviceMemoryPriorityEXT not loaded");
        unsafe { fp(self.handle(), memory, priority) };
    }
    ///Wraps [`vkWaitForPresent2KHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkWaitForPresent2KHR.html).
    /**
    Provided by **VK_KHR_present_wait2**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_DEVICE_LOST`
    ///- `VK_ERROR_OUT_OF_DATE_KHR`
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `swapchain` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Extensible version of `wait_for_present_khr`. Takes a
    ///`PresentWait2InfoKHR` struct (with `pNext` support) instead of
    ///separate `present_id` and `timeout` parameters.
    ///
    ///Provided by `VK_KHR_present_wait2`. Otherwise identical in
    ///behavior, blocks until the specified present ID completes or
    ///the timeout expires.
    pub unsafe fn wait_for_present2_khr(
        &self,
        swapchain: SwapchainKHR,
        p_present_wait2_info: &PresentWait2InfoKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .wait_for_present2_khr
            .expect("vkWaitForPresent2KHR not loaded");
        check(unsafe { fp(self.handle(), swapchain, p_present_wait2_info) })
    }
    ///Wraps [`vkWaitForPresentKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkWaitForPresentKHR.html).
    /**
    Provided by **VK_KHR_present_wait**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_DEVICE_LOST`
    ///- `VK_ERROR_OUT_OF_DATE_KHR`
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `swapchain` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Blocks the calling thread until a specific present operation
    ///completes on the display. `present_id` identifies which present
    ///to wait for, set it via `PresentIdKHR` chained into
    ///`PresentInfoKHR` during `queue_present_khr`.
    ///
    ///`timeout` is in nanoseconds. Returns `TIMEOUT` if the deadline
    ///expires before the present completes, `SUCCESS` if the present
    ///finished. Use `u64::MAX` for an indefinite wait.
    ///
    ///Requires `VK_KHR_present_wait` and `VK_KHR_present_id`.
    ///
    ///This is useful for frame pacing, wait for the previous frame's
    ///present to complete before starting the next frame's work to
    ///avoid queuing excessive frames.
    pub unsafe fn wait_for_present_khr(
        &self,
        swapchain: SwapchainKHR,
        present_id: u64,
        timeout: u64,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .wait_for_present_khr
            .expect("vkWaitForPresentKHR not loaded");
        check(unsafe { fp(self.handle(), swapchain, present_id, timeout) })
    }
    ///Wraps [`vkCreateBufferCollectionFUCHSIA`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateBufferCollectionFUCHSIA.html).
    /**
    Provided by **VK_FUCHSIA_buffer_collection**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_INVALID_EXTERNAL_HANDLE`
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a Fuchsia buffer collection that negotiates memory
    ///constraints between Vulkan and other Fuchsia services (e.g.
    ///scenic, camera). Fuchsia OS only. After creation, set buffer
    ///or image constraints before allocating.
    ///
    ///Requires `VK_FUCHSIA_buffer_collection`.
    pub unsafe fn create_buffer_collection_fuchsia(
        &self,
        p_create_info: &BufferCollectionCreateInfoFUCHSIA,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<BufferCollectionFUCHSIA> {
        let fp = self
            .commands()
            .create_buffer_collection_fuchsia
            .expect("vkCreateBufferCollectionFUCHSIA not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkSetBufferCollectionBufferConstraintsFUCHSIA`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkSetBufferCollectionBufferConstraintsFUCHSIA.html).
    /**
    Provided by **VK_FUCHSIA_buffer_collection**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_FORMAT_NOT_SUPPORTED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Sets buffer constraints on a Fuchsia buffer collection. The
    ///constraints describe the Vulkan buffer usage requirements that
    ///must be negotiated with other collection participants. Fuchsia
    ///OS only.
    ///
    ///Requires `VK_FUCHSIA_buffer_collection`.
    pub unsafe fn set_buffer_collection_buffer_constraints_fuchsia(
        &self,
        collection: BufferCollectionFUCHSIA,
        p_buffer_constraints_info: &BufferConstraintsInfoFUCHSIA,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .set_buffer_collection_buffer_constraints_fuchsia
            .expect("vkSetBufferCollectionBufferConstraintsFUCHSIA not loaded");
        check(unsafe { fp(self.handle(), collection, p_buffer_constraints_info) })
    }
    ///Wraps [`vkSetBufferCollectionImageConstraintsFUCHSIA`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkSetBufferCollectionImageConstraintsFUCHSIA.html).
    /**
    Provided by **VK_FUCHSIA_buffer_collection**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_FORMAT_NOT_SUPPORTED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Sets image constraints on a Fuchsia buffer collection. The
    ///constraints describe the Vulkan image format and usage
    ///requirements that must be negotiated with other collection
    ///participants. Fuchsia OS only.
    ///
    ///Requires `VK_FUCHSIA_buffer_collection`.
    pub unsafe fn set_buffer_collection_image_constraints_fuchsia(
        &self,
        collection: BufferCollectionFUCHSIA,
        p_image_constraints_info: &ImageConstraintsInfoFUCHSIA,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .set_buffer_collection_image_constraints_fuchsia
            .expect("vkSetBufferCollectionImageConstraintsFUCHSIA not loaded");
        check(unsafe { fp(self.handle(), collection, p_image_constraints_info) })
    }
    ///Wraps [`vkDestroyBufferCollectionFUCHSIA`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyBufferCollectionFUCHSIA.html).
    /**
    Provided by **VK_FUCHSIA_buffer_collection**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Destroys a Fuchsia buffer collection. The collection must not
    ///be in use by any pending operations. Fuchsia OS only.
    ///
    ///Requires `VK_FUCHSIA_buffer_collection`.
    pub unsafe fn destroy_buffer_collection_fuchsia(
        &self,
        collection: BufferCollectionFUCHSIA,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_buffer_collection_fuchsia
            .expect("vkDestroyBufferCollectionFUCHSIA not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), collection, alloc_ptr) };
    }
    ///Wraps [`vkGetBufferCollectionPropertiesFUCHSIA`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetBufferCollectionPropertiesFUCHSIA.html).
    /**
    Provided by **VK_FUCHSIA_buffer_collection**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the negotiated properties of a Fuchsia buffer
    ///collection after constraints have been set. Returns memory
    ///type index, format, and other details needed for allocation.
    ///Fuchsia OS only.
    ///
    ///Requires `VK_FUCHSIA_buffer_collection`.
    pub unsafe fn get_buffer_collection_properties_fuchsia(
        &self,
        collection: BufferCollectionFUCHSIA,
        p_properties: *mut BufferCollectionPropertiesFUCHSIA,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_buffer_collection_properties_fuchsia
            .expect("vkGetBufferCollectionPropertiesFUCHSIA not loaded");
        check(unsafe { fp(self.handle(), collection, p_properties) })
    }
    ///Wraps [`vkCreateCudaModuleNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateCudaModuleNV.html).
    /**
    Provided by **VK_NV_cuda_kernel_launch**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a CUDA module from PTX or cubin data. The module
    ///contains one or more kernel entry points that can be extracted
    ///with `create_cuda_function_nv`.
    ///
    ///Destroy with `destroy_cuda_module_nv`.
    ///
    ///Requires `VK_NV_cuda_kernel_launch`.
    pub unsafe fn create_cuda_module_nv(
        &self,
        p_create_info: &CudaModuleCreateInfoNV,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<CudaModuleNV> {
        let fp = self
            .commands()
            .create_cuda_module_nv
            .expect("vkCreateCudaModuleNV not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetCudaModuleCacheNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetCudaModuleCacheNV.html).
    /**
    Provided by **VK_NV_cuda_kernel_launch**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves the compiled cache data from a CUDA module for
    ///serialization. Call once with a null buffer to query the size,
    ///then again with an appropriately sized buffer. Feed the data
    ///back into `create_cuda_module_nv` on the next run to skip
    ///compilation.
    ///
    ///Requires `VK_NV_cuda_kernel_launch`.
    pub unsafe fn get_cuda_module_cache_nv(
        &self,
        module: CudaModuleNV,
        p_cache_data: *mut core::ffi::c_void,
    ) -> VkResult<usize> {
        let fp = self
            .commands()
            .get_cuda_module_cache_nv
            .expect("vkGetCudaModuleCacheNV not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), module, &mut out, p_cache_data) })?;
        Ok(out)
    }
    ///Wraps [`vkCreateCudaFunctionNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateCudaFunctionNV.html).
    /**
    Provided by **VK_NV_cuda_kernel_launch**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a CUDA function handle from a CUDA module, identifying
    ///a specific kernel entry point. Use with
    ///`cmd_cuda_launch_kernel_nv` to dispatch the kernel.
    ///
    ///Destroy with `destroy_cuda_function_nv`.
    ///
    ///Requires `VK_NV_cuda_kernel_launch`.
    pub unsafe fn create_cuda_function_nv(
        &self,
        p_create_info: &CudaFunctionCreateInfoNV,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<CudaFunctionNV> {
        let fp = self
            .commands()
            .create_cuda_function_nv
            .expect("vkCreateCudaFunctionNV not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyCudaModuleNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyCudaModuleNV.html).
    /**
    Provided by **VK_NV_cuda_kernel_launch**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Destroys a CUDA module created with `create_cuda_module_nv`.
    ///All functions extracted from this module must be destroyed first.
    ///
    ///Requires `VK_NV_cuda_kernel_launch`.
    pub unsafe fn destroy_cuda_module_nv(
        &self,
        module: CudaModuleNV,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_cuda_module_nv
            .expect("vkDestroyCudaModuleNV not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), module, alloc_ptr) };
    }
    ///Wraps [`vkDestroyCudaFunctionNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyCudaFunctionNV.html).
    /**
    Provided by **VK_NV_cuda_kernel_launch**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Destroys a CUDA function handle created with
    ///`create_cuda_function_nv`.
    ///
    ///Requires `VK_NV_cuda_kernel_launch`.
    pub unsafe fn destroy_cuda_function_nv(
        &self,
        function: CudaFunctionNV,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_cuda_function_nv
            .expect("vkDestroyCudaFunctionNV not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), function, alloc_ptr) };
    }
    ///Wraps [`vkCmdCudaLaunchKernelNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCudaLaunchKernelNV.html).
    /**
    Provided by **VK_NV_cuda_kernel_launch**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Launches a CUDA kernel from within a Vulkan command buffer.
    ///The kernel is specified by a CUDA function handle created with
    ///`create_cuda_function_nv`. Grid dimensions and parameters are
    ///provided in the launch info.
    ///
    ///Requires `VK_NV_cuda_kernel_launch`.
    pub unsafe fn cmd_cuda_launch_kernel_nv(
        &self,
        command_buffer: CommandBuffer,
        p_launch_info: &CudaLaunchInfoNV,
    ) {
        let fp = self
            .commands()
            .cmd_cuda_launch_kernel_nv
            .expect("vkCmdCudaLaunchKernelNV not loaded");
        unsafe { fp(command_buffer, p_launch_info) };
    }
    ///Wraps [`vkCmdBeginRendering`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBeginRendering.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Begins dynamic rendering, a Vulkan 1.3 alternative to render pass
    ///objects that specifies attachments inline at command recording time.
    ///
    ///**Advantages over render passes**:
    ///
    ///- No `RenderPass` or `Framebuffer` objects to create and manage.
    ///- Attachments are specified directly as image views in
    ///  `RenderingInfo`.
    ///- Simpler code for applications that do not benefit from tile-based
    ///  subpass optimisations.
    ///
    ///**`RenderingInfo`** specifies:
    ///
    ///- **Colour attachments**: image views, load/store ops, clear values.
    ///- **Depth attachment**: optional, with its own load/store ops.
    ///- **Stencil attachment**: optional, can share the same image view as
    ///  depth.
    ///- **Render area and layer count**.
    ///
    ///**Flags**:
    ///
    ///- `RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS`: draw commands come
    ///  from secondary command buffers.
    ///- `RENDERING_SUSPENDING` / `RENDERING_RESUMING`: split rendering
    ///  across multiple command buffers.
    ///
    ///Graphics pipelines used with dynamic rendering must be created with
    ///`PipelineRenderingCreateInfo` instead of a render pass handle.
    pub unsafe fn cmd_begin_rendering(
        &self,
        command_buffer: CommandBuffer,
        p_rendering_info: &RenderingInfo,
    ) {
        let fp = self
            .commands()
            .cmd_begin_rendering
            .expect("vkCmdBeginRendering not loaded");
        unsafe { fp(command_buffer, p_rendering_info) };
    }
    ///Wraps [`vkCmdEndRendering`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdEndRendering.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_3**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Ends a dynamic rendering instance started by `cmd_begin_rendering`.
    ///Store operations and any resolve operations specified in the
    ///`RenderingInfo` are executed at this point.
    ///
    ///After this call, no draw commands may be recorded until a new
    ///rendering or render pass instance is begun.
    pub unsafe fn cmd_end_rendering(&self, command_buffer: CommandBuffer) {
        let fp = self
            .commands()
            .cmd_end_rendering
            .expect("vkCmdEndRendering not loaded");
        unsafe { fp(command_buffer) };
    }
    ///Wraps [`vkCmdEndRendering2KHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdEndRendering2KHR.html).
    /**
    Provided by **VK_KHR_maintenance10**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Extended version of `cmd_end_rendering` (core 1.3) that accepts
    ///an optional `RenderingEndInfoKHR` with pNext extensibility.
    ///
    ///Ends the current dynamic rendering pass. If `p_rendering_end_info`
    ///is `None`, behaves identically to `cmd_end_rendering`.
    ///
    ///Provided by `VK_KHR_maintenance7`.
    pub unsafe fn cmd_end_rendering2_khr(
        &self,
        command_buffer: CommandBuffer,
        p_rendering_end_info: Option<&RenderingEndInfoKHR>,
    ) {
        let fp = self
            .commands()
            .cmd_end_rendering2_khr
            .expect("vkCmdEndRendering2KHR not loaded");
        let p_rendering_end_info_ptr =
            p_rendering_end_info.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(command_buffer, p_rendering_end_info_ptr) };
    }
    ///Wraps [`vkGetDescriptorSetLayoutHostMappingInfoVALVE`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDescriptorSetLayoutHostMappingInfoVALVE.html).
    /**
    Provided by **VK_VALVE_descriptor_set_host_mapping**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the host memory layout for a specific binding within a
    ///descriptor set layout. Returns the stride and offset needed to
    ///write descriptors directly via the host pointer obtained from
    ///`get_descriptor_set_host_mapping_valve`.
    ///
    ///Requires `VK_VALVE_descriptor_set_host_mapping`.
    pub unsafe fn get_descriptor_set_layout_host_mapping_info_valve(
        &self,
        p_binding_reference: &DescriptorSetBindingReferenceVALVE,
        p_host_mapping: *mut DescriptorSetLayoutHostMappingInfoVALVE,
    ) {
        let fp = self
            .commands()
            .get_descriptor_set_layout_host_mapping_info_valve
            .expect("vkGetDescriptorSetLayoutHostMappingInfoVALVE not loaded");
        unsafe { fp(self.handle(), p_binding_reference, p_host_mapping) };
    }
    ///Wraps [`vkGetDescriptorSetHostMappingVALVE`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDescriptorSetHostMappingVALVE.html).
    /**
    Provided by **VK_VALVE_descriptor_set_host_mapping**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves a host pointer to the internal memory backing a
    ///descriptor set. Allows direct CPU writes to descriptor data,
    ///bypassing the normal `update_descriptor_sets` path for lower
    ///overhead. The layout must match what was queried with
    ///`get_descriptor_set_layout_host_mapping_info_valve`.
    ///
    ///Requires `VK_VALVE_descriptor_set_host_mapping`.
    pub unsafe fn get_descriptor_set_host_mapping_valve(
        &self,
        descriptor_set: DescriptorSet,
        pp_data: *mut *mut core::ffi::c_void,
    ) {
        let fp = self
            .commands()
            .get_descriptor_set_host_mapping_valve
            .expect("vkGetDescriptorSetHostMappingVALVE not loaded");
        unsafe { fp(self.handle(), descriptor_set, pp_data) };
    }
    ///Wraps [`vkCreateMicromapEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateMicromapEXT.html).
    /**
    Provided by **VK_EXT_opacity_micromap**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates an opacity micromap object. Micromaps store per-triangle
    ///opacity or hit data at sub-triangle granularity, enabling the
    ///ray tracing implementation to skip fully transparent micro-
    ///triangles without invoking any-hit shaders.
    ///
    ///The `MicromapCreateInfoEXT` specifies the backing buffer, size,
    ///and type (`OPACITY_MICROMAP`).
    ///
    ///Build with `cmd_build_micromaps_ext` or `build_micromaps_ext`.
    ///Destroy with `destroy_micromap_ext`.
    ///
    ///Requires `VK_EXT_opacity_micromap`.
    pub unsafe fn create_micromap_ext(
        &self,
        p_create_info: &MicromapCreateInfoEXT,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<MicromapEXT> {
        let fp = self
            .commands()
            .create_micromap_ext
            .expect("vkCreateMicromapEXT not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkCmdBuildMicromapsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBuildMicromapsEXT.html).
    /**
    Provided by **VK_EXT_opacity_micromap**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Records a GPU-side micromap build into a command buffer. Each
    ///`MicromapBuildInfoEXT` specifies the source triangle opacity
    ///data, destination micromap, and scratch memory.
    ///
    ///For the CPU-side equivalent, see `build_micromaps_ext`.
    ///
    ///Requires `VK_EXT_opacity_micromap`.
    pub unsafe fn cmd_build_micromaps_ext(
        &self,
        command_buffer: CommandBuffer,
        p_infos: &[MicromapBuildInfoEXT],
    ) {
        let fp = self
            .commands()
            .cmd_build_micromaps_ext
            .expect("vkCmdBuildMicromapsEXT not loaded");
        unsafe { fp(command_buffer, p_infos.len() as u32, p_infos.as_ptr()) };
    }
    ///Wraps [`vkBuildMicromapsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkBuildMicromapsEXT.html).
    /**
    Provided by **VK_EXT_opacity_micromap**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Builds one or more micromaps on the host. This is the CPU-side
    ///equivalent of `cmd_build_micromaps_ext`.
    ///
    ///Each `MicromapBuildInfoEXT` specifies the source triangle data,
    ///destination micromap, and scratch memory.
    ///
    ///Requires `VK_EXT_opacity_micromap` and the
    ///`micromapHostCommands` feature.
    pub unsafe fn build_micromaps_ext(
        &self,
        deferred_operation: DeferredOperationKHR,
        p_infos: &[MicromapBuildInfoEXT],
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .build_micromaps_ext
            .expect("vkBuildMicromapsEXT not loaded");
        check(unsafe {
            fp(
                self.handle(),
                deferred_operation,
                p_infos.len() as u32,
                p_infos.as_ptr(),
            )
        })
    }
    ///Wraps [`vkDestroyMicromapEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyMicromapEXT.html).
    /**
    Provided by **VK_EXT_opacity_micromap**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `micromap` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a micromap created with `create_micromap_ext`. The
    ///backing buffer is not freed, the application must manage buffer
    ///lifetime separately.
    ///
    ///Requires `VK_EXT_opacity_micromap`.
    pub unsafe fn destroy_micromap_ext(
        &self,
        micromap: MicromapEXT,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_micromap_ext
            .expect("vkDestroyMicromapEXT not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), micromap, alloc_ptr) };
    }
    ///Wraps [`vkCmdCopyMicromapEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyMicromapEXT.html).
    /**
    Provided by **VK_EXT_opacity_micromap**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Copies or compacts a micromap. The `CopyMicromapInfoEXT` specifies
    ///source, destination, and mode (`CLONE` or `COMPACT`).
    ///
    ///Use `COMPACT` after building with `BUILD_ALLOW_COMPACTION` to
    ///reduce memory usage. Query the compacted size with
    ///`cmd_write_micromaps_properties_ext`.
    ///
    ///Requires `VK_EXT_opacity_micromap`.
    pub unsafe fn cmd_copy_micromap_ext(
        &self,
        command_buffer: CommandBuffer,
        p_info: &CopyMicromapInfoEXT,
    ) {
        let fp = self
            .commands()
            .cmd_copy_micromap_ext
            .expect("vkCmdCopyMicromapEXT not loaded");
        unsafe { fp(command_buffer, p_info) };
    }
    ///Wraps [`vkCopyMicromapEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCopyMicromapEXT.html).
    /**
    Provided by **VK_EXT_opacity_micromap**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Host-side micromap copy or compaction. This is the CPU equivalent
    ///of `cmd_copy_micromap_ext`.
    ///
    ///Requires `VK_EXT_opacity_micromap` and the
    ///`micromapHostCommands` feature.
    pub unsafe fn copy_micromap_ext(
        &self,
        deferred_operation: DeferredOperationKHR,
        p_info: &CopyMicromapInfoEXT,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .copy_micromap_ext
            .expect("vkCopyMicromapEXT not loaded");
        check(unsafe { fp(self.handle(), deferred_operation, p_info) })
    }
    ///Wraps [`vkCmdCopyMicromapToMemoryEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyMicromapToMemoryEXT.html).
    /**
    Provided by **VK_EXT_opacity_micromap**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Serializes a micromap into a buffer for storage or transfer.
    ///The `CopyMicromapToMemoryInfoEXT` specifies the source micromap
    ///and destination device address.
    ///
    ///Deserialize with `cmd_copy_memory_to_micromap_ext`.
    ///
    ///Requires `VK_EXT_opacity_micromap`.
    pub unsafe fn cmd_copy_micromap_to_memory_ext(
        &self,
        command_buffer: CommandBuffer,
        p_info: &CopyMicromapToMemoryInfoEXT,
    ) {
        let fp = self
            .commands()
            .cmd_copy_micromap_to_memory_ext
            .expect("vkCmdCopyMicromapToMemoryEXT not loaded");
        unsafe { fp(command_buffer, p_info) };
    }
    ///Wraps [`vkCopyMicromapToMemoryEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCopyMicromapToMemoryEXT.html).
    /**
    Provided by **VK_EXT_opacity_micromap**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Host-side serialization of a micromap into a buffer. This is the
    ///CPU equivalent of `cmd_copy_micromap_to_memory_ext`.
    ///
    ///Requires `VK_EXT_opacity_micromap` and the
    ///`micromapHostCommands` feature.
    pub unsafe fn copy_micromap_to_memory_ext(
        &self,
        deferred_operation: DeferredOperationKHR,
        p_info: &CopyMicromapToMemoryInfoEXT,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .copy_micromap_to_memory_ext
            .expect("vkCopyMicromapToMemoryEXT not loaded");
        check(unsafe { fp(self.handle(), deferred_operation, p_info) })
    }
    ///Wraps [`vkCmdCopyMemoryToMicromapEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyMemoryToMicromapEXT.html).
    /**
    Provided by **VK_EXT_opacity_micromap**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Deserializes a micromap from a buffer into a micromap object.
    ///This is the reverse of `cmd_copy_micromap_to_memory_ext`.
    ///
    ///Used for loading previously serialized micromaps from disk or
    ///transferring between devices.
    ///
    ///Requires `VK_EXT_opacity_micromap`.
    pub unsafe fn cmd_copy_memory_to_micromap_ext(
        &self,
        command_buffer: CommandBuffer,
        p_info: &CopyMemoryToMicromapInfoEXT,
    ) {
        let fp = self
            .commands()
            .cmd_copy_memory_to_micromap_ext
            .expect("vkCmdCopyMemoryToMicromapEXT not loaded");
        unsafe { fp(command_buffer, p_info) };
    }
    ///Wraps [`vkCopyMemoryToMicromapEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCopyMemoryToMicromapEXT.html).
    /**
    Provided by **VK_EXT_opacity_micromap**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Host-side deserialization of a micromap from a buffer. This is
    ///the CPU equivalent of `cmd_copy_memory_to_micromap_ext`.
    ///
    ///Requires `VK_EXT_opacity_micromap` and the
    ///`micromapHostCommands` feature.
    pub unsafe fn copy_memory_to_micromap_ext(
        &self,
        deferred_operation: DeferredOperationKHR,
        p_info: &CopyMemoryToMicromapInfoEXT,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .copy_memory_to_micromap_ext
            .expect("vkCopyMemoryToMicromapEXT not loaded");
        check(unsafe { fp(self.handle(), deferred_operation, p_info) })
    }
    ///Wraps [`vkCmdWriteMicromapsPropertiesEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdWriteMicromapsPropertiesEXT.html).
    /**
    Provided by **VK_EXT_opacity_micromap**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Writes micromap properties (e.g., compacted size) to a query pool.
    ///Use `QUERY_TYPE_MICROMAP_COMPACTED_SIZE_EXT` to query the size
    ///needed for a compacted copy.
    ///
    ///Requires `VK_EXT_opacity_micromap`.
    pub unsafe fn cmd_write_micromaps_properties_ext(
        &self,
        command_buffer: CommandBuffer,
        p_micromaps: &[MicromapEXT],
        query_type: QueryType,
        query_pool: QueryPool,
        first_query: u32,
    ) {
        let fp = self
            .commands()
            .cmd_write_micromaps_properties_ext
            .expect("vkCmdWriteMicromapsPropertiesEXT not loaded");
        unsafe {
            fp(
                command_buffer,
                p_micromaps.len() as u32,
                p_micromaps.as_ptr(),
                query_type,
                query_pool,
                first_query,
            )
        };
    }
    ///Wraps [`vkWriteMicromapsPropertiesEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkWriteMicromapsPropertiesEXT.html).
    /**
    Provided by **VK_EXT_opacity_micromap**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Host-side query of micromap properties. This is the CPU equivalent
    ///of `cmd_write_micromaps_properties_ext`.
    ///
    ///Typically used to query compacted size
    ///(`QUERY_TYPE_MICROMAP_COMPACTED_SIZE_EXT`) without going through
    ///a query pool.
    ///
    ///Requires `VK_EXT_opacity_micromap` and the
    ///`micromapHostCommands` feature.
    pub unsafe fn write_micromaps_properties_ext(
        &self,
        p_micromaps: &[MicromapEXT],
        query_type: QueryType,
        data_size: usize,
        p_data: *mut core::ffi::c_void,
        stride: usize,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .write_micromaps_properties_ext
            .expect("vkWriteMicromapsPropertiesEXT not loaded");
        check(unsafe {
            fp(
                self.handle(),
                p_micromaps.len() as u32,
                p_micromaps.as_ptr(),
                query_type,
                data_size,
                p_data,
                stride,
            )
        })
    }
    ///Wraps [`vkGetDeviceMicromapCompatibilityEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDeviceMicromapCompatibilityEXT.html).
    /**
    Provided by **VK_EXT_opacity_micromap**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Checks whether a serialized micromap is compatible with the
    ///current device. Returns `COMPATIBLE` or `INCOMPATIBLE`.
    ///
    ///Use this before deserializing a micromap (via
    ///`cmd_copy_memory_to_micromap_ext` or `copy_memory_to_micromap_ext`)
    ///to verify it will work on this device.
    ///
    ///Requires `VK_EXT_opacity_micromap`.
    pub unsafe fn get_device_micromap_compatibility_ext(
        &self,
        p_version_info: &MicromapVersionInfoEXT,
    ) -> AccelerationStructureCompatibilityKHR {
        let fp = self
            .commands()
            .get_device_micromap_compatibility_ext
            .expect("vkGetDeviceMicromapCompatibilityEXT not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(self.handle(), p_version_info, &mut out) };
        out
    }
    ///Wraps [`vkGetMicromapBuildSizesEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetMicromapBuildSizesEXT.html).
    /**
    Provided by **VK_EXT_opacity_micromap**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the memory requirements for building a micromap. Returns
    ///the destination micromap size and scratch memory size needed.
    ///
    ///Provide a `MicromapBuildInfoEXT` with the triangle counts and
    ///format. The addresses can be null, only the sizes and counts
    ///matter for this query.
    ///
    ///Use the results to allocate the micromap buffer and scratch buffer
    ///before calling `cmd_build_micromaps_ext`.
    ///
    ///Requires `VK_EXT_opacity_micromap`.
    pub unsafe fn get_micromap_build_sizes_ext(
        &self,
        build_type: AccelerationStructureBuildTypeKHR,
        p_build_info: &MicromapBuildInfoEXT,
        p_size_info: *mut MicromapBuildSizesInfoEXT,
    ) {
        let fp = self
            .commands()
            .get_micromap_build_sizes_ext
            .expect("vkGetMicromapBuildSizesEXT not loaded");
        unsafe { fp(self.handle(), build_type, p_build_info, p_size_info) };
    }
    ///Wraps [`vkGetShaderModuleIdentifierEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetShaderModuleIdentifierEXT.html).
    /**
    Provided by **VK_EXT_shader_module_identifier**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves the identifier for an existing shader module. The
    ///identifier can be used for pipeline cache lookups via
    ///`PipelineShaderStageModuleIdentifierCreateInfoEXT`.
    ///
    ///The identifier is deterministic for the same SPIR-V content
    ///on the same implementation.
    ///
    ///Requires `VK_EXT_shader_module_identifier`.
    pub unsafe fn get_shader_module_identifier_ext(
        &self,
        shader_module: ShaderModule,
        p_identifier: *mut ShaderModuleIdentifierEXT,
    ) {
        let fp = self
            .commands()
            .get_shader_module_identifier_ext
            .expect("vkGetShaderModuleIdentifierEXT not loaded");
        unsafe { fp(self.handle(), shader_module, p_identifier) };
    }
    ///Wraps [`vkGetShaderModuleCreateInfoIdentifierEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetShaderModuleCreateInfoIdentifierEXT.html).
    /**
    Provided by **VK_EXT_shader_module_identifier**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Computes an identifier for a shader module from its create info
    ///(SPIR-V code) without creating the module. The identifier can be
    ///used in `PipelineShaderStageModuleIdentifierCreateInfoEXT` to
    ///create pipelines from cached shader data.
    ///
    ///Useful for pipeline caching workflows where the SPIR-V is
    ///available but you want to avoid full module creation.
    ///
    ///Requires `VK_EXT_shader_module_identifier`.
    pub unsafe fn get_shader_module_create_info_identifier_ext(
        &self,
        p_create_info: &ShaderModuleCreateInfo,
        p_identifier: *mut ShaderModuleIdentifierEXT,
    ) {
        let fp = self
            .commands()
            .get_shader_module_create_info_identifier_ext
            .expect("vkGetShaderModuleCreateInfoIdentifierEXT not loaded");
        unsafe { fp(self.handle(), p_create_info, p_identifier) };
    }
    ///Wraps [`vkGetImageSubresourceLayout2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetImageSubresourceLayout2.html).
    /**
    Provided by **VK_BASE_VERSION_1_4**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.4 version of `get_image_subresource_layout` that uses
    ///extensible structs via pNext.
    ///
    ///Returns the layout (offset, size, row pitch, array pitch, depth
    ///pitch) for a given subresource of an existing image. Chain
    ///`ImageCompressionPropertiesEXT` to query fixed-rate compression
    ///state.
    ///
    ///For linear-tiling images, this tells you how to access texels
    ///through a mapped pointer. For optimal-tiling images, the layout is
    ///opaque and implementation-defined.
    pub unsafe fn get_image_subresource_layout2(
        &self,
        image: Image,
        p_subresource: &ImageSubresource2,
        p_layout: *mut SubresourceLayout2,
    ) {
        let fp = self
            .commands()
            .get_image_subresource_layout2
            .expect("vkGetImageSubresourceLayout2 not loaded");
        unsafe { fp(self.handle(), image, p_subresource, p_layout) };
    }
    ///Wraps [`vkGetPipelinePropertiesEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPipelinePropertiesEXT.html).
    /**
    Provided by **VK_EXT_pipeline_properties**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries properties of a pipeline, such as its pipeline identifier.
    ///The identifier can be used to correlate pipelines across processes
    ///or with external tools.
    ///
    ///Requires `VK_EXT_pipeline_properties`.
    pub unsafe fn get_pipeline_properties_ext(
        &self,
        p_pipeline_info: &PipelineInfoEXT,
        p_pipeline_properties: *mut BaseOutStructure,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_pipeline_properties_ext
            .expect("vkGetPipelinePropertiesEXT not loaded");
        check(unsafe { fp(self.handle(), p_pipeline_info, p_pipeline_properties) })
    }
    ///Wraps [`vkExportMetalObjectsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkExportMetalObjectsEXT.html).
    /**
    Provided by **VK_EXT_metal_objects**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Exports the underlying Metal objects (device, command queue,
    ///textures, buffers, etc.) from Vulkan objects. Chain the
    ///appropriate export struct into the pNext of the info structure
    ///to select which Metal object to retrieve.
    ///
    ///Useful for Metal interop on Apple platforms where both APIs
    ///share the same GPU resources.
    ///
    ///Requires `VK_EXT_metal_objects`. macOS/iOS only.
    pub unsafe fn export_metal_objects_ext(
        &self,
        p_metal_objects_info: *mut ExportMetalObjectsInfoEXT,
    ) {
        let fp = self
            .commands()
            .export_metal_objects_ext
            .expect("vkExportMetalObjectsEXT not loaded");
        unsafe { fp(self.handle(), p_metal_objects_info) };
    }
    ///Wraps [`vkCmdBindTileMemoryQCOM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBindTileMemoryQCOM.html).
    /**
    Provided by **VK_QCOM_tile_memory_heap**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Binds tile memory for use within a per-tile execution region on
    ///Qualcomm tile-based GPUs. Pass `None` to unbind. Tile memory
    ///provides fast on-chip scratch storage scoped to each tile.
    ///
    ///Requires `VK_QCOM_tile_shading`.
    pub unsafe fn cmd_bind_tile_memory_qcom(
        &self,
        command_buffer: CommandBuffer,
        p_tile_memory_bind_info: Option<&TileMemoryBindInfoQCOM>,
    ) {
        let fp = self
            .commands()
            .cmd_bind_tile_memory_qcom
            .expect("vkCmdBindTileMemoryQCOM not loaded");
        let p_tile_memory_bind_info_ptr =
            p_tile_memory_bind_info.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(command_buffer, p_tile_memory_bind_info_ptr) };
    }
    ///Wraps [`vkGetFramebufferTilePropertiesQCOM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetFramebufferTilePropertiesQCOM.html).
    /**
    Provided by **VK_QCOM_tile_properties**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries tile properties (tile dimensions and extents) for a
    ///framebuffer on Qualcomm tile-based GPUs. Uses the two-call
    ///idiom. Useful for optimising rendering to match the hardware
    ///tile layout.
    ///
    ///Requires `VK_QCOM_tile_properties`.
    pub unsafe fn get_framebuffer_tile_properties_qcom(
        &self,
        framebuffer: Framebuffer,
    ) -> VkResult<Vec<TilePropertiesQCOM>> {
        let fp = self
            .commands()
            .get_framebuffer_tile_properties_qcom
            .expect("vkGetFramebufferTilePropertiesQCOM not loaded");
        enumerate_two_call(|count, data| unsafe { fp(self.handle(), framebuffer, count, data) })
    }
    ///Wraps [`vkGetDynamicRenderingTilePropertiesQCOM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDynamicRenderingTilePropertiesQCOM.html).
    /**
    Provided by **VK_QCOM_tile_properties**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries tile properties for a dynamic rendering pass (as opposed
    ///to a framebuffer-based render pass). Returns tile dimensions that
    ///would be used for the given rendering info on Qualcomm tile-based
    ///GPUs.
    ///
    ///Requires `VK_QCOM_tile_properties`.
    pub unsafe fn get_dynamic_rendering_tile_properties_qcom(
        &self,
        p_rendering_info: &RenderingInfo,
        p_properties: *mut TilePropertiesQCOM,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_dynamic_rendering_tile_properties_qcom
            .expect("vkGetDynamicRenderingTilePropertiesQCOM not loaded");
        check(unsafe { fp(self.handle(), p_rendering_info, p_properties) })
    }
    ///Wraps [`vkCreateOpticalFlowSessionNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateOpticalFlowSessionNV.html).
    /**
    Provided by **VK_NV_optical_flow**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates an optical flow session for GPU-accelerated motion
    ///estimation between image pairs. Configure the session with image
    ///format, resolution, and flow precision in the create info.
    ///
    ///Destroy with `destroy_optical_flow_session_nv`.
    ///
    ///Requires `VK_NV_optical_flow`.
    pub unsafe fn create_optical_flow_session_nv(
        &self,
        p_create_info: &OpticalFlowSessionCreateInfoNV,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<OpticalFlowSessionNV> {
        let fp = self
            .commands()
            .create_optical_flow_session_nv
            .expect("vkCreateOpticalFlowSessionNV not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyOpticalFlowSessionNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyOpticalFlowSessionNV.html).
    /**
    Provided by **VK_NV_optical_flow**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Destroys an optical flow session created with
    ///`create_optical_flow_session_nv`.
    ///
    ///Requires `VK_NV_optical_flow`.
    pub unsafe fn destroy_optical_flow_session_nv(
        &self,
        session: OpticalFlowSessionNV,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_optical_flow_session_nv
            .expect("vkDestroyOpticalFlowSessionNV not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), session, alloc_ptr) };
    }
    ///Wraps [`vkBindOpticalFlowSessionImageNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkBindOpticalFlowSessionImageNV.html).
    /**
    Provided by **VK_NV_optical_flow**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Binds an image view to an optical flow session at a specific
    ///binding point (reference, target, flow vector output, etc.).
    ///All required binding points must be bound before executing the
    ///optical flow.
    ///
    ///Requires `VK_NV_optical_flow`.
    pub unsafe fn bind_optical_flow_session_image_nv(
        &self,
        session: OpticalFlowSessionNV,
        binding_point: OpticalFlowSessionBindingPointNV,
        view: ImageView,
        layout: ImageLayout,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .bind_optical_flow_session_image_nv
            .expect("vkBindOpticalFlowSessionImageNV not loaded");
        check(unsafe { fp(self.handle(), session, binding_point, view, layout) })
    }
    ///Wraps [`vkCmdOpticalFlowExecuteNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdOpticalFlowExecuteNV.html).
    /**
    Provided by **VK_NV_optical_flow**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Executes optical flow estimation on the GPU using the bound
    ///reference and target images. Results are written to the bound
    ///flow vector output image.
    ///
    ///Requires `VK_NV_optical_flow`.
    pub unsafe fn cmd_optical_flow_execute_nv(
        &self,
        command_buffer: CommandBuffer,
        session: OpticalFlowSessionNV,
        p_execute_info: &OpticalFlowExecuteInfoNV,
    ) {
        let fp = self
            .commands()
            .cmd_optical_flow_execute_nv
            .expect("vkCmdOpticalFlowExecuteNV not loaded");
        unsafe { fp(command_buffer, session, p_execute_info) };
    }
    ///Wraps [`vkGetDeviceFaultInfoEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDeviceFaultInfoEXT.html).
    /**
    Provided by **VK_EXT_device_fault**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves diagnostic information after a device-lost error. Call
    ///once with a null info pointer to query the fault counts, then
    ///again with allocated structures to retrieve the fault details.
    ///
    ///The returned data is vendor-specific and intended for crash
    ///reporting and post-mortem debugging.
    ///
    ///Requires `VK_EXT_device_fault`.
    pub unsafe fn get_device_fault_info_ext(
        &self,
        p_fault_counts: *mut DeviceFaultCountsEXT,
        p_fault_info: *mut DeviceFaultInfoEXT,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_device_fault_info_ext
            .expect("vkGetDeviceFaultInfoEXT not loaded");
        check(unsafe { fp(self.handle(), p_fault_counts, p_fault_info) })
    }
    ///Wraps [`vkGetDeviceFaultReportsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDeviceFaultReportsKHR.html).
    /**
    Provided by **VK_KHR_device_fault**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves fault reports after a device loss (`ERROR_DEVICE_LOST`).
    ///Returns a list of `DeviceFaultInfoKHR` structs describing what
    ///went wrong, address faults, vendor-specific fault codes, etc.
    ///
    ///`timeout` specifies how long to wait (in nanoseconds) for the
    ///driver to collect fault data. Use `UINT64_MAX` to wait
    ///indefinitely.
    ///
    ///For raw debug data suitable for vendor tools, follow up with
    ///`get_device_fault_debug_info_khr`.
    ///
    ///Requires `VK_KHR_device_fault`.
    pub unsafe fn get_device_fault_reports_khr(
        &self,
        timeout: u64,
    ) -> VkResult<Vec<DeviceFaultInfoKHR>> {
        let fp = self
            .commands()
            .get_device_fault_reports_khr
            .expect("vkGetDeviceFaultReportsKHR not loaded");
        enumerate_two_call(|count, data| unsafe { fp(self.handle(), timeout, count, data) })
    }
    ///Wraps [`vkGetDeviceFaultDebugInfoKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDeviceFaultDebugInfoKHR.html).
    /**
    Provided by **VK_KHR_device_fault**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_NOT_ENOUGH_SPACE_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves debug information after a device fault (GPU crash or
    ///hang). The returned `DeviceFaultDebugInfoKHR` contains
    ///vendor-specific binary data that can be passed to GPU vendor
    ///diagnostic tools.
    ///
    ///Call `get_device_fault_reports_khr` first to get the high-level
    ///fault reports; this call provides the raw debug blob.
    ///
    ///Requires `VK_KHR_device_fault`.
    pub unsafe fn get_device_fault_debug_info_khr(
        &self,
        p_debug_info: *mut DeviceFaultDebugInfoKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_device_fault_debug_info_khr
            .expect("vkGetDeviceFaultDebugInfoKHR not loaded");
        check(unsafe { fp(self.handle(), p_debug_info) })
    }
    ///Wraps [`vkCmdSetDepthBias2EXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetDepthBias2EXT.html).
    /**
    Provided by **VK_EXT_depth_bias_control**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Extended version of `cmd_set_depth_bias` that takes a
    ///`DepthBiasInfoEXT` struct with pNext extensibility. This allows
    ///chaining `DepthBiasRepresentationInfoEXT` to control the depth
    ///bias representation (least-representable-value vs. float).
    ///
    ///Requires `VK_EXT_depth_bias_control`.
    pub unsafe fn cmd_set_depth_bias2_ext(
        &self,
        command_buffer: CommandBuffer,
        p_depth_bias_info: &DepthBiasInfoEXT,
    ) {
        let fp = self
            .commands()
            .cmd_set_depth_bias2_ext
            .expect("vkCmdSetDepthBias2EXT not loaded");
        unsafe { fp(command_buffer, p_depth_bias_info) };
    }
    ///Wraps [`vkReleaseSwapchainImagesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkReleaseSwapchainImagesKHR.html).
    /**
    Provided by **VK_KHR_swapchain_maintenance1**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Releases acquired swapchain images back to the swapchain without
    ///presenting them. Provided by `VK_KHR_swapchain_maintenance1`.
    ///
    ///Use this when you have acquired an image but decided not to render
    ///to it, for example, when aborting a frame due to a resize or
    ///error. Without this extension, the only way to return an acquired
    ///image is to present it.
    ///
    ///The released images return to the pool and can be acquired again.
    pub unsafe fn release_swapchain_images_khr(
        &self,
        p_release_info: &ReleaseSwapchainImagesInfoKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .release_swapchain_images_khr
            .expect("vkReleaseSwapchainImagesKHR not loaded");
        check(unsafe { fp(self.handle(), p_release_info) })
    }
    ///Wraps [`vkGetDeviceImageSubresourceLayout`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDeviceImageSubresourceLayout.html).
    /**
    Provided by **VK_BASE_VERSION_1_4**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.4 command that queries the subresource layout for an image
    ///**without creating it first**. Returns the offset, size, row pitch,
    ///array pitch, and depth pitch for a given subresource of a
    ///hypothetical image.
    ///
    ///Useful for pre-planning host-side memory layouts when using
    ///`HOST_TRANSFER` images, or for calculating buffer sizes for staging
    ///uploads.
    pub unsafe fn get_device_image_subresource_layout(
        &self,
        p_info: &DeviceImageSubresourceInfo,
        p_layout: *mut SubresourceLayout2,
    ) {
        let fp = self
            .commands()
            .get_device_image_subresource_layout
            .expect("vkGetDeviceImageSubresourceLayout not loaded");
        unsafe { fp(self.handle(), p_info, p_layout) };
    }
    ///Wraps [`vkMapMemory2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkMapMemory2.html).
    /**
    Provided by **VK_BASE_VERSION_1_4**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_MEMORY_MAP_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.4 version of `map_memory` that uses an extensible
    ///`MemoryMapInfo` struct. Supports additional flags via
    ///`MemoryMapFlags` that are not available in the original 1.0 version.
    ///
    ///`MEMORY_MAP_PLACED` (if supported) allows mapping at a
    ///caller-specified virtual address, useful for deterministic replay
    ///tools.
    ///
    ///For most applications, `map_memory` and `map_memory2` are
    ///equivalent. Prefer this when targeting Vulkan 1.4+.
    pub unsafe fn map_memory2(
        &self,
        p_memory_map_info: &MemoryMapInfo,
        pp_data: *mut *mut core::ffi::c_void,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .map_memory2
            .expect("vkMapMemory2 not loaded");
        check(unsafe { fp(self.handle(), p_memory_map_info, pp_data) })
    }
    ///Wraps [`vkUnmapMemory2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkUnmapMemory2.html).
    /**
    Provided by **VK_BASE_VERSION_1_4**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_MEMORY_MAP_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.4 version of `unmap_memory` that uses an extensible
    ///`MemoryUnmapInfo` struct. Supports `MEMORY_UNMAP_RESERVE` (if
    ///available) to keep the virtual address range reserved after
    ///unmapping, useful for placed mappings.
    ///
    ///For most applications, `unmap_memory` and `unmap_memory2` are
    ///equivalent. Prefer this when targeting Vulkan 1.4+.
    pub unsafe fn unmap_memory2(&self, p_memory_unmap_info: &MemoryUnmapInfo) -> VkResult<()> {
        let fp = self
            .commands()
            .unmap_memory2
            .expect("vkUnmapMemory2 not loaded");
        check(unsafe { fp(self.handle(), p_memory_unmap_info) })
    }
    ///Wraps [`vkCreateShadersEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateShadersEXT.html).
    /**
    Provided by **VK_EXT_shader_object**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates one or more shader objects from SPIR-V code. Shader
    ///objects are an alternative to pipelines, instead of baking
    ///shaders into monolithic pipeline objects, individual shader
    ///stages can be bound independently.
    ///
    ///Each `ShaderCreateInfoEXT` specifies the stage, code, entry
    ///point, and optional specialization constants.
    ///
    ///Bind with `cmd_bind_shaders_ext`. Destroy with
    ///`destroy_shader_ext`.
    ///
    ///Requires `VK_EXT_shader_object`.
    pub unsafe fn create_shaders_ext(
        &self,
        p_create_infos: &[ShaderCreateInfoEXT],
        allocator: Option<&AllocationCallbacks>,
        p_shaders: *mut ShaderEXT,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .create_shaders_ext
            .expect("vkCreateShadersEXT not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        check(unsafe {
            fp(
                self.handle(),
                p_create_infos.len() as u32,
                p_create_infos.as_ptr(),
                alloc_ptr,
                p_shaders,
            )
        })
    }
    ///Wraps [`vkDestroyShaderEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyShaderEXT.html).
    /**
    Provided by **VK_EXT_shader_object**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `shader` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a shader object created with `create_shaders_ext`.
    ///
    ///Requires `VK_EXT_shader_object`.
    pub unsafe fn destroy_shader_ext(
        &self,
        shader: ShaderEXT,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_shader_ext
            .expect("vkDestroyShaderEXT not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), shader, alloc_ptr) };
    }
    ///Wraps [`vkGetShaderBinaryDataEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetShaderBinaryDataEXT.html).
    /**
    Provided by **VK_EXT_shader_object**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves the binary representation of a compiled shader object.
    ///The binary data can be cached to disk and used to create shader
    ///objects without recompiling from SPIR-V on subsequent runs.
    ///
    ///Call once with a null buffer to query the size, then again with
    ///an appropriately sized buffer. Shader binaries are
    ///implementation-specific and not portable between devices or
    ///driver versions.
    ///
    ///Requires `VK_EXT_shader_object`.
    pub unsafe fn get_shader_binary_data_ext(
        &self,
        shader: ShaderEXT,
        p_data: *mut core::ffi::c_void,
    ) -> VkResult<usize> {
        let fp = self
            .commands()
            .get_shader_binary_data_ext
            .expect("vkGetShaderBinaryDataEXT not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), shader, &mut out, p_data) })?;
        Ok(out)
    }
    ///Wraps [`vkCmdBindShadersEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBindShadersEXT.html).
    /**
    Provided by **VK_EXT_shader_object**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Binds shader objects to specified shader stages for subsequent
    ///draw or dispatch commands. Pass arrays of stages and
    ///corresponding shader handles.
    ///
    ///To unbind a stage, pass a null handle for that stage. All
    ///required stages for the draw/dispatch type must be bound.
    ///
    ///When using shader objects, you must also set all relevant dynamic
    ///state, there is no pipeline to provide defaults.
    ///
    ///Requires `VK_EXT_shader_object`.
    pub unsafe fn cmd_bind_shaders_ext(
        &self,
        command_buffer: CommandBuffer,
        p_stages: &[ShaderStageFlagBits],
        p_shaders: Option<&ShaderEXT>,
    ) {
        let fp = self
            .commands()
            .cmd_bind_shaders_ext
            .expect("vkCmdBindShadersEXT not loaded");
        let p_shaders_ptr = p_shaders.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe {
            fp(
                command_buffer,
                p_stages.len() as u32,
                p_stages.as_ptr(),
                p_shaders_ptr,
            )
        };
    }
    ///Wraps [`vkSetSwapchainPresentTimingQueueSizeEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkSetSwapchainPresentTimingQueueSizeEXT.html).
    /**
    Provided by **VK_EXT_present_timing**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `swapchain` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Sets the maximum number of present timing results the driver
    ///will queue for later retrieval via `get_past_presentation_timing_ext`.
    ///A larger queue prevents timing data from being lost when the
    ///application cannot poll frequently.
    ///
    ///Requires `VK_EXT_present_timing`.
    pub unsafe fn set_swapchain_present_timing_queue_size_ext(
        &self,
        swapchain: SwapchainKHR,
        size: u32,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .set_swapchain_present_timing_queue_size_ext
            .expect("vkSetSwapchainPresentTimingQueueSizeEXT not loaded");
        check(unsafe { fp(self.handle(), swapchain, size) })
    }
    ///Wraps [`vkGetSwapchainTimingPropertiesEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetSwapchainTimingPropertiesEXT.html).
    /**
    Provided by **VK_EXT_present_timing**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `swapchain` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Queries the timing properties of a swapchain, such as the refresh
    ///duration and present margin. Use this to calibrate frame pacing
    ///and target the display's native refresh interval.
    ///
    ///Requires `VK_EXT_present_timing`.
    pub unsafe fn get_swapchain_timing_properties_ext(
        &self,
        swapchain: SwapchainKHR,
        p_swapchain_timing_properties: *mut SwapchainTimingPropertiesEXT,
    ) -> VkResult<u64> {
        let fp = self
            .commands()
            .get_swapchain_timing_properties_ext
            .expect("vkGetSwapchainTimingPropertiesEXT not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe {
            fp(
                self.handle(),
                swapchain,
                p_swapchain_timing_properties,
                &mut out,
            )
        })?;
        Ok(out)
    }
    ///Wraps [`vkGetSwapchainTimeDomainPropertiesEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetSwapchainTimeDomainPropertiesEXT.html).
    /**
    Provided by **VK_EXT_present_timing**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `swapchain` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Queries which time domains the swapchain supports for present
    ///timing. Use this to determine whether you can correlate present
    ///timestamps with the system clock or device-specific counters.
    ///
    ///Requires `VK_EXT_present_timing`.
    pub unsafe fn get_swapchain_time_domain_properties_ext(
        &self,
        swapchain: SwapchainKHR,
        p_swapchain_time_domain_properties: *mut SwapchainTimeDomainPropertiesEXT,
    ) -> VkResult<u64> {
        let fp = self
            .commands()
            .get_swapchain_time_domain_properties_ext
            .expect("vkGetSwapchainTimeDomainPropertiesEXT not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe {
            fp(
                self.handle(),
                swapchain,
                p_swapchain_time_domain_properties,
                &mut out,
            )
        })?;
        Ok(out)
    }
    ///Wraps [`vkGetPastPresentationTimingEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPastPresentationTimingEXT.html).
    /**
    Provided by **VK_EXT_present_timing**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_DEVICE_LOST`
    ///- `VK_ERROR_OUT_OF_DATE_KHR`
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves presentation timing data for previously presented images.
    ///Use this to measure actual vs. requested present times and detect
    ///missed frames or compositor latency.
    ///
    ///Requires `VK_EXT_present_timing`.
    pub unsafe fn get_past_presentation_timing_ext(
        &self,
        p_past_presentation_timing_info: &PastPresentationTimingInfoEXT,
        p_past_presentation_timing_properties: *mut PastPresentationTimingPropertiesEXT,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_past_presentation_timing_ext
            .expect("vkGetPastPresentationTimingEXT not loaded");
        check(unsafe {
            fp(
                self.handle(),
                p_past_presentation_timing_info,
                p_past_presentation_timing_properties,
            )
        })
    }
    ///Wraps [`vkGetScreenBufferPropertiesQNX`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetScreenBufferPropertiesQNX.html).
    /**
    Provided by **VK_QNX_external_memory_screen_buffer**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_INVALID_EXTERNAL_HANDLE_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries Vulkan memory properties for a QNX Screen buffer. Use
    ///before importing the buffer as Vulkan memory to determine
    ///compatible memory types and size. QNX only.
    ///
    ///Requires `VK_QNX_external_memory_screen_buffer`.
    pub unsafe fn get_screen_buffer_properties_qnx(
        &self,
        buffer: *const core::ffi::c_void,
        p_properties: *mut ScreenBufferPropertiesQNX,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_screen_buffer_properties_qnx
            .expect("vkGetScreenBufferPropertiesQNX not loaded");
        check(unsafe { fp(self.handle(), buffer, p_properties) })
    }
    ///Wraps [`vkGetExecutionGraphPipelineScratchSizeAMDX`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetExecutionGraphPipelineScratchSizeAMDX.html).
    /**
    Provided by **VK_AMDX_shader_enqueue**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the scratch memory size required to execute an execution
    ///graph pipeline. Allocate a buffer of at least this size and
    ///initialize it with `cmd_initialize_graph_scratch_memory_amdx`
    ///before dispatching the graph.
    ///
    ///Requires `VK_AMDX_shader_enqueue`.
    pub unsafe fn get_execution_graph_pipeline_scratch_size_amdx(
        &self,
        execution_graph: Pipeline,
        p_size_info: *mut ExecutionGraphPipelineScratchSizeAMDX,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_execution_graph_pipeline_scratch_size_amdx
            .expect("vkGetExecutionGraphPipelineScratchSizeAMDX not loaded");
        check(unsafe { fp(self.handle(), execution_graph, p_size_info) })
    }
    ///Wraps [`vkGetExecutionGraphPipelineNodeIndexAMDX`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetExecutionGraphPipelineNodeIndexAMDX.html).
    /**
    Provided by **VK_AMDX_shader_enqueue**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the node index for a named shader node in an execution
    ///graph pipeline. The index is used when dispatching or enqueuing
    ///work to a specific node in the graph.
    ///
    ///Requires `VK_AMDX_shader_enqueue`.
    pub unsafe fn get_execution_graph_pipeline_node_index_amdx(
        &self,
        execution_graph: Pipeline,
        p_node_info: &PipelineShaderStageNodeCreateInfoAMDX,
    ) -> VkResult<u32> {
        let fp = self
            .commands()
            .get_execution_graph_pipeline_node_index_amdx
            .expect("vkGetExecutionGraphPipelineNodeIndexAMDX not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), execution_graph, p_node_info, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkCreateExecutionGraphPipelinesAMDX`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateExecutionGraphPipelinesAMDX.html).
    /**
    Provided by **VK_AMDX_shader_enqueue**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `pipelineCache` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Creates one or more execution graph pipelines for GPU-driven
    ///shader dispatch. An execution graph is a DAG of shader nodes
    ///where each node can enqueue work to other nodes, enabling complex
    ///GPU-driven workflows without CPU round-trips.
    ///
    ///Requires `VK_AMDX_shader_enqueue`.
    pub unsafe fn create_execution_graph_pipelines_amdx(
        &self,
        pipeline_cache: PipelineCache,
        p_create_infos: &[ExecutionGraphPipelineCreateInfoAMDX],
        allocator: Option<&AllocationCallbacks>,
        p_pipelines: *mut Pipeline,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .create_execution_graph_pipelines_amdx
            .expect("vkCreateExecutionGraphPipelinesAMDX not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        check(unsafe {
            fp(
                self.handle(),
                pipeline_cache,
                p_create_infos.len() as u32,
                p_create_infos.as_ptr(),
                alloc_ptr,
                p_pipelines,
            )
        })
    }
    ///Wraps [`vkCmdInitializeGraphScratchMemoryAMDX`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdInitializeGraphScratchMemoryAMDX.html).
    /**
    Provided by **VK_AMDX_shader_enqueue**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Initializes the scratch memory buffer for an execution graph
    ///pipeline. Must be called before any `cmd_dispatch_graph_*_amdx`
    ///command that uses this scratch buffer.
    ///
    ///Requires `VK_AMDX_shader_enqueue`.
    pub unsafe fn cmd_initialize_graph_scratch_memory_amdx(
        &self,
        command_buffer: CommandBuffer,
        execution_graph: Pipeline,
        scratch: u64,
        scratch_size: u64,
    ) {
        let fp = self
            .commands()
            .cmd_initialize_graph_scratch_memory_amdx
            .expect("vkCmdInitializeGraphScratchMemoryAMDX not loaded");
        unsafe { fp(command_buffer, execution_graph, scratch, scratch_size) };
    }
    ///Wraps [`vkCmdDispatchGraphAMDX`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDispatchGraphAMDX.html).
    /**
    Provided by **VK_AMDX_shader_enqueue**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Dispatches an execution graph starting from the specified root
    ///nodes. The scratch buffer must have been initialized with
    ///`cmd_initialize_graph_scratch_memory_amdx`.
    ///
    ///Requires `VK_AMDX_shader_enqueue`.
    pub unsafe fn cmd_dispatch_graph_amdx(
        &self,
        command_buffer: CommandBuffer,
        scratch: u64,
        scratch_size: u64,
        p_count_info: &DispatchGraphCountInfoAMDX,
    ) {
        let fp = self
            .commands()
            .cmd_dispatch_graph_amdx
            .expect("vkCmdDispatchGraphAMDX not loaded");
        unsafe { fp(command_buffer, scratch, scratch_size, p_count_info) };
    }
    ///Wraps [`vkCmdDispatchGraphIndirectAMDX`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDispatchGraphIndirectAMDX.html).
    /**
    Provided by **VK_AMDX_shader_enqueue**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Indirect variant of `cmd_dispatch_graph_amdx`. Reads the graph
    ///dispatch payloads from a GPU buffer while the count info is
    ///provided on the CPU side.
    ///
    ///Requires `VK_AMDX_shader_enqueue`.
    pub unsafe fn cmd_dispatch_graph_indirect_amdx(
        &self,
        command_buffer: CommandBuffer,
        scratch: u64,
        scratch_size: u64,
        p_count_info: &DispatchGraphCountInfoAMDX,
    ) {
        let fp = self
            .commands()
            .cmd_dispatch_graph_indirect_amdx
            .expect("vkCmdDispatchGraphIndirectAMDX not loaded");
        unsafe { fp(command_buffer, scratch, scratch_size, p_count_info) };
    }
    ///Wraps [`vkCmdDispatchGraphIndirectCountAMDX`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDispatchGraphIndirectCountAMDX.html).
    /**
    Provided by **VK_AMDX_shader_enqueue**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Fully indirect variant of `cmd_dispatch_graph_amdx`. Both the
    ///dispatch payloads and the count are read from GPU buffers,
    ///enabling fully GPU-driven execution graph dispatch.
    ///
    ///Requires `VK_AMDX_shader_enqueue`.
    pub unsafe fn cmd_dispatch_graph_indirect_count_amdx(
        &self,
        command_buffer: CommandBuffer,
        scratch: u64,
        scratch_size: u64,
        count_info: u64,
    ) {
        let fp = self
            .commands()
            .cmd_dispatch_graph_indirect_count_amdx
            .expect("vkCmdDispatchGraphIndirectCountAMDX not loaded");
        unsafe { fp(command_buffer, scratch, scratch_size, count_info) };
    }
    ///Wraps [`vkCmdBindDescriptorSets2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBindDescriptorSets2.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_4**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.4 version of `cmd_bind_descriptor_sets` that uses an
    ///extensible `BindDescriptorSetsInfo` struct.
    ///
    ///Functionally equivalent to the 1.0 version. The extensible struct
    ///enables future extensions to modify binding behaviour via pNext.
    ///
    ///Prefer this when targeting Vulkan 1.4+.
    pub unsafe fn cmd_bind_descriptor_sets2(
        &self,
        command_buffer: CommandBuffer,
        p_bind_descriptor_sets_info: &BindDescriptorSetsInfo,
    ) {
        let fp = self
            .commands()
            .cmd_bind_descriptor_sets2
            .expect("vkCmdBindDescriptorSets2 not loaded");
        unsafe { fp(command_buffer, p_bind_descriptor_sets_info) };
    }
    ///Wraps [`vkCmdPushConstants2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdPushConstants2.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_4**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.4 version of `cmd_push_constants` that uses an extensible
    ///`PushConstantsInfo` struct.
    ///
    ///Functionally equivalent to the 1.0 version. Prefer this when
    ///targeting Vulkan 1.4+.
    pub unsafe fn cmd_push_constants2(
        &self,
        command_buffer: CommandBuffer,
        p_push_constants_info: &PushConstantsInfo,
    ) {
        let fp = self
            .commands()
            .cmd_push_constants2
            .expect("vkCmdPushConstants2 not loaded");
        unsafe { fp(command_buffer, p_push_constants_info) };
    }
    ///Wraps [`vkCmdPushDescriptorSet2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdPushDescriptorSet2.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_4**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.4 version of `cmd_push_descriptor_set` that uses an
    ///extensible `PushDescriptorSetInfo` struct.
    ///
    ///Functionally equivalent to the base version. Prefer this when
    ///targeting Vulkan 1.4+.
    pub unsafe fn cmd_push_descriptor_set2(
        &self,
        command_buffer: CommandBuffer,
        p_push_descriptor_set_info: &PushDescriptorSetInfo,
    ) {
        let fp = self
            .commands()
            .cmd_push_descriptor_set2
            .expect("vkCmdPushDescriptorSet2 not loaded");
        unsafe { fp(command_buffer, p_push_descriptor_set_info) };
    }
    ///Wraps [`vkCmdPushDescriptorSetWithTemplate2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdPushDescriptorSetWithTemplate2.html).
    /**
    Provided by **VK_COMPUTE_VERSION_1_4**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.4 version of `cmd_push_descriptor_set_with_template` that
    ///uses an extensible `PushDescriptorSetWithTemplateInfo` struct.
    ///
    ///Functionally equivalent to the base version. Prefer this when
    ///targeting Vulkan 1.4+.
    pub unsafe fn cmd_push_descriptor_set_with_template2(
        &self,
        command_buffer: CommandBuffer,
        p_push_descriptor_set_with_template_info: &PushDescriptorSetWithTemplateInfo,
    ) {
        let fp = self
            .commands()
            .cmd_push_descriptor_set_with_template2
            .expect("vkCmdPushDescriptorSetWithTemplate2 not loaded");
        unsafe { fp(command_buffer, p_push_descriptor_set_with_template_info) };
    }
    ///Wraps [`vkCmdSetDescriptorBufferOffsets2EXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetDescriptorBufferOffsets2EXT.html).
    /**
    Provided by **VK_KHR_maintenance6**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Extended version of `cmd_set_descriptor_buffer_offsets_ext` that
    ///takes a `SetDescriptorBufferOffsetsInfoEXT` struct with pNext
    ///extensibility.
    ///
    ///Sets the offsets into bound descriptor buffers for the specified
    ///pipeline layout. Each offset points to the start of a descriptor
    ///set's data within the bound descriptor buffer.
    ///
    ///Provided by `VK_KHR_maintenance6` (for the pNext-extensible
    ///variant of the `VK_EXT_descriptor_buffer` command).
    pub unsafe fn cmd_set_descriptor_buffer_offsets2_ext(
        &self,
        command_buffer: CommandBuffer,
        p_set_descriptor_buffer_offsets_info: &SetDescriptorBufferOffsetsInfoEXT,
    ) {
        let fp = self
            .commands()
            .cmd_set_descriptor_buffer_offsets2_ext
            .expect("vkCmdSetDescriptorBufferOffsets2EXT not loaded");
        unsafe { fp(command_buffer, p_set_descriptor_buffer_offsets_info) };
    }
    ///Wraps [`vkCmdBindDescriptorBufferEmbeddedSamplers2EXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBindDescriptorBufferEmbeddedSamplers2EXT.html).
    /**
    Provided by **VK_KHR_maintenance6**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Extended version of `cmd_bind_descriptor_buffer_embedded_samplers_ext`
    ///that takes a `BindDescriptorBufferEmbeddedSamplersInfoEXT` struct
    ///with pNext extensibility.
    ///
    ///Binds embedded immutable samplers from a descriptor set layout
    ///that was created with `CREATE_EMBEDDED_IMMUTABLE_SAMPLERS_BIT`.
    ///
    ///Provided by `VK_KHR_maintenance6` (for the pNext-extensible
    ///variant of the `VK_EXT_descriptor_buffer` command).
    pub unsafe fn cmd_bind_descriptor_buffer_embedded_samplers2_ext(
        &self,
        command_buffer: CommandBuffer,
        p_bind_descriptor_buffer_embedded_samplers_info: &BindDescriptorBufferEmbeddedSamplersInfoEXT,
    ) {
        let fp = self
            .commands()
            .cmd_bind_descriptor_buffer_embedded_samplers2_ext
            .expect("vkCmdBindDescriptorBufferEmbeddedSamplers2EXT not loaded");
        unsafe {
            fp(
                command_buffer,
                p_bind_descriptor_buffer_embedded_samplers_info,
            )
        };
    }
    ///Wraps [`vkSetLatencySleepModeNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkSetLatencySleepModeNV.html).
    /**
    Provided by **VK_NV_low_latency2**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Configures the NVIDIA Reflex low-latency sleep mode for a
    ///swapchain. Enables or disables latency sleep and sets the target
    ///sleep duration. Call before `latency_sleep_nv` to activate the
    ///system.
    ///
    ///Requires `VK_NV_low_latency2`.
    pub unsafe fn set_latency_sleep_mode_nv(
        &self,
        swapchain: SwapchainKHR,
        p_sleep_mode_info: &LatencySleepModeInfoNV,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .set_latency_sleep_mode_nv
            .expect("vkSetLatencySleepModeNV not loaded");
        check(unsafe { fp(self.handle(), swapchain, p_sleep_mode_info) })
    }
    ///Wraps [`vkLatencySleepNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkLatencySleepNV.html).
    /**
    Provided by **VK_NV_low_latency2**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Sleeps the calling thread until the optimal time to begin the
    ///next frame, as determined by the NVIDIA Reflex low-latency
    ///system. Reduces input-to-display latency by preventing the CPU
    ///from running too far ahead of the GPU.
    ///
    ///Requires `VK_NV_low_latency2`.
    pub unsafe fn latency_sleep_nv(
        &self,
        swapchain: SwapchainKHR,
        p_sleep_info: &LatencySleepInfoNV,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .latency_sleep_nv
            .expect("vkLatencySleepNV not loaded");
        check(unsafe { fp(self.handle(), swapchain, p_sleep_info) })
    }
    ///Wraps [`vkSetLatencyMarkerNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkSetLatencyMarkerNV.html).
    /**
    Provided by **VK_NV_low_latency2**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Sets a latency marker at a specific point in the frame lifecycle
    ///(simulation start, render start, present, etc.). The markers are
    ///later retrieved with `get_latency_timings_nv` to measure per-
    ///stage latency.
    ///
    ///Requires `VK_NV_low_latency2`.
    pub unsafe fn set_latency_marker_nv(
        &self,
        swapchain: SwapchainKHR,
        p_latency_marker_info: &SetLatencyMarkerInfoNV,
    ) {
        let fp = self
            .commands()
            .set_latency_marker_nv
            .expect("vkSetLatencyMarkerNV not loaded");
        unsafe { fp(self.handle(), swapchain, p_latency_marker_info) };
    }
    ///Wraps [`vkGetLatencyTimingsNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetLatencyTimingsNV.html).
    /**
    Provided by **VK_NV_low_latency2**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves per-frame latency timing data for a swapchain. Returns
    ///timestamps for each marker set with `set_latency_marker_nv`,
    ///enabling measurement of simulation, render, and present latency.
    ///
    ///Requires `VK_NV_low_latency2`.
    pub unsafe fn get_latency_timings_nv(
        &self,
        swapchain: SwapchainKHR,
        p_latency_marker_info: *mut GetLatencyMarkerInfoNV,
    ) {
        let fp = self
            .commands()
            .get_latency_timings_nv
            .expect("vkGetLatencyTimingsNV not loaded");
        unsafe { fp(self.handle(), swapchain, p_latency_marker_info) };
    }
    ///Wraps [`vkQueueNotifyOutOfBandNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkQueueNotifyOutOfBandNV.html).
    /**
    Provided by **VK_NV_low_latency2**.*/
    ///
    ///# Safety
    ///- `queue` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Notifies the low-latency system that a queue submission is
    ///out-of-band (e.g., a loading or async compute submission that
    ///should not be counted toward frame latency tracking).
    ///
    ///Requires `VK_NV_low_latency2`.
    pub unsafe fn queue_notify_out_of_band_nv(
        &self,
        queue: Queue,
        p_queue_type_info: &OutOfBandQueueTypeInfoNV,
    ) {
        let fp = self
            .commands()
            .queue_notify_out_of_band_nv
            .expect("vkQueueNotifyOutOfBandNV not loaded");
        unsafe { fp(queue, p_queue_type_info) };
    }
    ///Wraps [`vkCmdSetRenderingAttachmentLocations`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetRenderingAttachmentLocations.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_4**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically remaps colour attachment indices within a dynamic
    ///rendering instance. Allows fragment shader outputs to target
    ///different attachment slots without changing the pipeline.
    ///
    ///This is useful when the same shader outputs different data to
    ///different attachments depending on the rendering pass (e.g.
    ///G-buffer vs forward).
    ///
    ///Requires `dynamic_rendering_local_read` feature. Core in
    ///Vulkan 1.4.
    pub unsafe fn cmd_set_rendering_attachment_locations(
        &self,
        command_buffer: CommandBuffer,
        p_location_info: &RenderingAttachmentLocationInfo,
    ) {
        let fp = self
            .commands()
            .cmd_set_rendering_attachment_locations
            .expect("vkCmdSetRenderingAttachmentLocations not loaded");
        unsafe { fp(command_buffer, p_location_info) };
    }
    ///Wraps [`vkCmdSetRenderingInputAttachmentIndices`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetRenderingInputAttachmentIndices.html).
    /**
    Provided by **VK_GRAPHICS_VERSION_1_4**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically remaps input attachment indices within a dynamic
    ///rendering instance. Allows fragment shaders to read from different
    ///colour or depth/stencil attachments without changing the pipeline.
    ///
    ///Paired with `cmd_set_rendering_attachment_locations` to enable
    ///flexible attachment routing in multi-pass rendering with dynamic
    ///rendering.
    ///
    ///Requires `dynamic_rendering_local_read` feature. Core in
    ///Vulkan 1.4.
    pub unsafe fn cmd_set_rendering_input_attachment_indices(
        &self,
        command_buffer: CommandBuffer,
        p_input_attachment_index_info: &RenderingInputAttachmentIndexInfo,
    ) {
        let fp = self
            .commands()
            .cmd_set_rendering_input_attachment_indices
            .expect("vkCmdSetRenderingInputAttachmentIndices not loaded");
        unsafe { fp(command_buffer, p_input_attachment_index_info) };
    }
    ///Wraps [`vkCmdSetDepthClampRangeEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetDepthClampRangeEXT.html).
    /**
    Provided by **VK_EXT_shader_object**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Dynamically sets the depth clamp range. When depth clamping is
    ///enabled, fragments are clamped to the specified min/max depth
    ///values instead of the viewport near/far range.
    ///
    ///Pass null to use the default viewport depth range for clamping.
    ///
    ///Requires `VK_EXT_depth_clamp_control` and the
    ///`depthClampControl` feature.
    pub unsafe fn cmd_set_depth_clamp_range_ext(
        &self,
        command_buffer: CommandBuffer,
        depth_clamp_mode: DepthClampModeEXT,
        p_depth_clamp_range: Option<&DepthClampRangeEXT>,
    ) {
        let fp = self
            .commands()
            .cmd_set_depth_clamp_range_ext
            .expect("vkCmdSetDepthClampRangeEXT not loaded");
        let p_depth_clamp_range_ptr =
            p_depth_clamp_range.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(command_buffer, depth_clamp_mode, p_depth_clamp_range_ptr) };
    }
    ///Wraps [`vkGetMemoryMetalHandleEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetMemoryMetalHandleEXT.html).
    /**
    Provided by **VK_EXT_external_memory_metal**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_TOO_MANY_OBJECTS`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Exports a Vulkan device memory allocation as a Metal resource
    ///handle for cross-API interop on Apple platforms.
    ///
    ///Requires `VK_EXT_external_memory_metal`. macOS/iOS only.
    pub unsafe fn get_memory_metal_handle_ext(
        &self,
        p_get_metal_handle_info: &MemoryGetMetalHandleInfoEXT,
        p_handle: *mut *mut core::ffi::c_void,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_memory_metal_handle_ext
            .expect("vkGetMemoryMetalHandleEXT not loaded");
        check(unsafe { fp(self.handle(), p_get_metal_handle_info, p_handle) })
    }
    ///Wraps [`vkGetMemoryMetalHandlePropertiesEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetMemoryMetalHandlePropertiesEXT.html).
    /**
    Provided by **VK_EXT_external_memory_metal**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_INVALID_EXTERNAL_HANDLE`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries which Vulkan memory types are compatible with importing
    ///a Metal resource handle as external memory. Use before allocating
    ///device memory to determine valid memory type bits.
    ///
    ///Requires `VK_EXT_external_memory_metal`. macOS/iOS only.
    pub unsafe fn get_memory_metal_handle_properties_ext(
        &self,
        handle_type: ExternalMemoryHandleTypeFlagBits,
        p_handle: *const core::ffi::c_void,
        p_memory_metal_handle_properties: *mut MemoryMetalHandlePropertiesEXT,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_memory_metal_handle_properties_ext
            .expect("vkGetMemoryMetalHandlePropertiesEXT not loaded");
        check(unsafe {
            fp(
                self.handle(),
                handle_type,
                p_handle,
                p_memory_metal_handle_properties,
            )
        })
    }
    ///Wraps [`vkConvertCooperativeVectorMatrixNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkConvertCooperativeVectorMatrixNV.html).
    /**
    Provided by **VK_NV_cooperative_vector**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Host-side conversion of cooperative vector matrix data between
    ///formats or layouts. Use to prepare weight matrices on the CPU
    ///before uploading to the GPU for cooperative vector operations.
    ///
    ///Requires `VK_NV_cooperative_vector`.
    pub unsafe fn convert_cooperative_vector_matrix_nv(
        &self,
        p_info: &ConvertCooperativeVectorMatrixInfoNV,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .convert_cooperative_vector_matrix_nv
            .expect("vkConvertCooperativeVectorMatrixNV not loaded");
        check(unsafe { fp(self.handle(), p_info) })
    }
    ///Wraps [`vkCmdConvertCooperativeVectorMatrixNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdConvertCooperativeVectorMatrixNV.html).
    /**
    Provided by **VK_NV_cooperative_vector**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///GPU-side conversion of cooperative vector matrix data between
    ///formats or layouts. Converts one or more matrices in a command
    ///buffer. Use for repacking weight matrices for neural network
    ///inference on the GPU.
    ///
    ///Requires `VK_NV_cooperative_vector`.
    pub unsafe fn cmd_convert_cooperative_vector_matrix_nv(
        &self,
        command_buffer: CommandBuffer,
        p_infos: &[ConvertCooperativeVectorMatrixInfoNV],
    ) {
        let fp = self
            .commands()
            .cmd_convert_cooperative_vector_matrix_nv
            .expect("vkCmdConvertCooperativeVectorMatrixNV not loaded");
        unsafe { fp(command_buffer, p_infos.len() as u32, p_infos.as_ptr()) };
    }
    ///Wraps [`vkCmdDispatchTileQCOM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDispatchTileQCOM.html).
    /**
    Provided by **VK_QCOM_tile_shading**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Dispatches compute work within a per-tile execution region on
    ///Qualcomm tile-based GPUs. Must be called between
    ///`cmd_begin_per_tile_execution_qcom` and
    ///`cmd_end_per_tile_execution_qcom`.
    ///
    ///Requires `VK_QCOM_tile_shading`.
    pub unsafe fn cmd_dispatch_tile_qcom(
        &self,
        command_buffer: CommandBuffer,
        p_dispatch_tile_info: &DispatchTileInfoQCOM,
    ) {
        let fp = self
            .commands()
            .cmd_dispatch_tile_qcom
            .expect("vkCmdDispatchTileQCOM not loaded");
        unsafe { fp(command_buffer, p_dispatch_tile_info) };
    }
    ///Wraps [`vkCmdBeginPerTileExecutionQCOM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBeginPerTileExecutionQCOM.html).
    /**
    Provided by **VK_QCOM_tile_shading**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Begins a per-tile execution region within a render pass on
    ///Qualcomm tile-based GPUs. Commands recorded between this and
    ///`cmd_end_per_tile_execution_qcom` are executed once per tile,
    ///enabling tile-local compute and shading optimisations.
    ///
    ///Requires `VK_QCOM_tile_shading`.
    pub unsafe fn cmd_begin_per_tile_execution_qcom(
        &self,
        command_buffer: CommandBuffer,
        p_per_tile_begin_info: &PerTileBeginInfoQCOM,
    ) {
        let fp = self
            .commands()
            .cmd_begin_per_tile_execution_qcom
            .expect("vkCmdBeginPerTileExecutionQCOM not loaded");
        unsafe { fp(command_buffer, p_per_tile_begin_info) };
    }
    ///Wraps [`vkCmdEndPerTileExecutionQCOM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdEndPerTileExecutionQCOM.html).
    /**
    Provided by **VK_QCOM_tile_shading**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Ends a per-tile execution region started by
    ///`cmd_begin_per_tile_execution_qcom`. After this call, the
    ///command buffer returns to normal (non-tile-local) recording.
    ///
    ///Requires `VK_QCOM_tile_shading`.
    pub unsafe fn cmd_end_per_tile_execution_qcom(
        &self,
        command_buffer: CommandBuffer,
        p_per_tile_end_info: &PerTileEndInfoQCOM,
    ) {
        let fp = self
            .commands()
            .cmd_end_per_tile_execution_qcom
            .expect("vkCmdEndPerTileExecutionQCOM not loaded");
        unsafe { fp(command_buffer, p_per_tile_end_info) };
    }
    ///Wraps [`vkCreateExternalComputeQueueNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateExternalComputeQueueNV.html).
    /**
    Provided by **VK_NV_external_compute_queue**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_TOO_MANY_OBJECTS`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates an external compute queue that can be used to submit
    ///work from outside the Vulkan runtime (e.g., CUDA interop).
    ///Destroy with `destroy_external_compute_queue_nv`.
    ///
    ///Requires `VK_NV_external_compute_queue`.
    pub unsafe fn create_external_compute_queue_nv(
        &self,
        p_create_info: &ExternalComputeQueueCreateInfoNV,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<ExternalComputeQueueNV> {
        let fp = self
            .commands()
            .create_external_compute_queue_nv
            .expect("vkCreateExternalComputeQueueNV not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyExternalComputeQueueNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyExternalComputeQueueNV.html).
    /**
    Provided by **VK_NV_external_compute_queue**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Destroys an external compute queue created with
    ///`create_external_compute_queue_nv`.
    ///
    ///Requires `VK_NV_external_compute_queue`.
    pub unsafe fn destroy_external_compute_queue_nv(
        &self,
        external_queue: ExternalComputeQueueNV,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_external_compute_queue_nv
            .expect("vkDestroyExternalComputeQueueNV not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), external_queue, alloc_ptr) };
    }
    ///Wraps [`vkCreateShaderInstrumentationARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateShaderInstrumentationARM.html).
    /**
    Provided by **VK_ARM_shader_instrumentation**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a shader instrumentation object that configures which
    ///metrics to collect during shader execution. The instrumentation
    ///is later bound to command buffers with
    ///`cmd_begin_shader_instrumentation_arm`.
    ///
    ///Requires `VK_ARM_shader_instrumentation`.
    pub unsafe fn create_shader_instrumentation_arm(
        &self,
        p_create_info: &ShaderInstrumentationCreateInfoARM,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<ShaderInstrumentationARM> {
        let fp = self
            .commands()
            .create_shader_instrumentation_arm
            .expect("vkCreateShaderInstrumentationARM not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyShaderInstrumentationARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyShaderInstrumentationARM.html).
    /**
    Provided by **VK_ARM_shader_instrumentation**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `instrumentation` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a shader instrumentation object. The object must not be
    ///in use by any command buffer.
    ///
    ///Requires `VK_ARM_shader_instrumentation`.
    pub unsafe fn destroy_shader_instrumentation_arm(
        &self,
        instrumentation: ShaderInstrumentationARM,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_shader_instrumentation_arm
            .expect("vkDestroyShaderInstrumentationARM not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), instrumentation, alloc_ptr) };
    }
    ///Wraps [`vkCmdBeginShaderInstrumentationARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBeginShaderInstrumentationARM.html).
    /**
    Provided by **VK_ARM_shader_instrumentation**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///- `instrumentation` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Begins shader instrumentation collection in a command buffer.
    ///All subsequent draw and dispatch commands will collect the
    ///metrics configured in the instrumentation object until
    ///`cmd_end_shader_instrumentation_arm` is called.
    ///
    ///Requires `VK_ARM_shader_instrumentation`.
    pub unsafe fn cmd_begin_shader_instrumentation_arm(
        &self,
        command_buffer: CommandBuffer,
        instrumentation: ShaderInstrumentationARM,
    ) {
        let fp = self
            .commands()
            .cmd_begin_shader_instrumentation_arm
            .expect("vkCmdBeginShaderInstrumentationARM not loaded");
        unsafe { fp(command_buffer, instrumentation) };
    }
    ///Wraps [`vkCmdEndShaderInstrumentationARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdEndShaderInstrumentationARM.html).
    /**
    Provided by **VK_ARM_shader_instrumentation**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Ends shader instrumentation collection in a command buffer.
    ///Metrics collected since the matching
    ///`cmd_begin_shader_instrumentation_arm` can be retrieved with
    ///`get_shader_instrumentation_values_arm` after submission
    ///completes.
    ///
    ///Requires `VK_ARM_shader_instrumentation`.
    pub unsafe fn cmd_end_shader_instrumentation_arm(&self, command_buffer: CommandBuffer) {
        let fp = self
            .commands()
            .cmd_end_shader_instrumentation_arm
            .expect("vkCmdEndShaderInstrumentationARM not loaded");
        unsafe { fp(command_buffer) };
    }
    ///Wraps [`vkGetShaderInstrumentationValuesARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetShaderInstrumentationValuesARM.html).
    /**
    Provided by **VK_ARM_shader_instrumentation**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves collected shader instrumentation metric values after
    ///instrumented commands have completed execution. Returns metric
    ///blocks whose count is written to `p_metric_block_count`. Ensure
    ///the instrumented submission has finished before querying.
    ///
    ///Requires `VK_ARM_shader_instrumentation`.
    pub unsafe fn get_shader_instrumentation_values_arm(
        &self,
        instrumentation: ShaderInstrumentationARM,
        p_metric_block_count: *mut u32,
        flags: ShaderInstrumentationValuesFlagsARM,
    ) -> VkResult<core::ffi::c_void> {
        let fp = self
            .commands()
            .get_shader_instrumentation_values_arm
            .expect("vkGetShaderInstrumentationValuesARM not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe {
            fp(
                self.handle(),
                instrumentation,
                p_metric_block_count,
                &mut out,
                flags,
            )
        })?;
        Ok(out)
    }
    ///Wraps [`vkClearShaderInstrumentationMetricsARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkClearShaderInstrumentationMetricsARM.html).
    /**
    Provided by **VK_ARM_shader_instrumentation**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Resets accumulated metrics for a shader instrumentation object.
    ///Call between frames or profiling sessions to start collecting
    ///fresh data without destroying and recreating the object.
    ///
    ///Requires `VK_ARM_shader_instrumentation`.
    pub unsafe fn clear_shader_instrumentation_metrics_arm(
        &self,
        instrumentation: ShaderInstrumentationARM,
    ) {
        let fp = self
            .commands()
            .clear_shader_instrumentation_metrics_arm
            .expect("vkClearShaderInstrumentationMetricsARM not loaded");
        unsafe { fp(self.handle(), instrumentation) };
    }
    ///Wraps [`vkCreateTensorARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateTensorARM.html).
    /**
    Provided by **VK_ARM_tensors**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a tensor object for ARM's tensor extension. Tensors are
    ///multi-dimensional arrays with a defined format, dimensions, and
    ///usage flags. Must be bound to memory before use, similar to
    ///images and buffers.
    ///
    ///Requires `VK_ARM_tensors`.
    pub unsafe fn create_tensor_arm(
        &self,
        p_create_info: &TensorCreateInfoARM,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<TensorARM> {
        let fp = self
            .commands()
            .create_tensor_arm
            .expect("vkCreateTensorARM not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyTensorARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyTensorARM.html).
    /**
    Provided by **VK_ARM_tensors**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `tensor` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a tensor object. The tensor must not be in use by any
    ///command buffer or referenced by any tensor view.
    ///
    ///Requires `VK_ARM_tensors`.
    pub unsafe fn destroy_tensor_arm(
        &self,
        tensor: TensorARM,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_tensor_arm
            .expect("vkDestroyTensorARM not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), tensor, alloc_ptr) };
    }
    ///Wraps [`vkCreateTensorViewARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateTensorViewARM.html).
    /**
    Provided by **VK_ARM_tensors**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a view into a tensor, analogous to image views. A tensor
    ///view selects a subset of the tensor's dimensions or reinterprets
    ///its format for use in descriptors and shaders.
    ///
    ///Requires `VK_ARM_tensors`.
    pub unsafe fn create_tensor_view_arm(
        &self,
        p_create_info: &TensorViewCreateInfoARM,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<TensorViewARM> {
        let fp = self
            .commands()
            .create_tensor_view_arm
            .expect("vkCreateTensorViewARM not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyTensorViewARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyTensorViewARM.html).
    /**
    Provided by **VK_ARM_tensors**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `tensorView` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a tensor view. The view must not be in use by any
    ///command buffer or referenced by any descriptor set.
    ///
    ///Requires `VK_ARM_tensors`.
    pub unsafe fn destroy_tensor_view_arm(
        &self,
        tensor_view: TensorViewARM,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_tensor_view_arm
            .expect("vkDestroyTensorViewARM not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), tensor_view, alloc_ptr) };
    }
    ///Wraps [`vkGetTensorMemoryRequirementsARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetTensorMemoryRequirementsARM.html).
    /**
    Provided by **VK_ARM_tensors**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the memory requirements (size, alignment, memory type
    ///bits) for an existing tensor object. Call before
    ///`bind_tensor_memory_arm` to determine the allocation needed.
    ///
    ///Requires `VK_ARM_tensors`.
    pub unsafe fn get_tensor_memory_requirements_arm(
        &self,
        p_info: &TensorMemoryRequirementsInfoARM,
        p_memory_requirements: *mut MemoryRequirements2,
    ) {
        let fp = self
            .commands()
            .get_tensor_memory_requirements_arm
            .expect("vkGetTensorMemoryRequirementsARM not loaded");
        unsafe { fp(self.handle(), p_info, p_memory_requirements) };
    }
    ///Wraps [`vkBindTensorMemoryARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkBindTensorMemoryARM.html).
    /**
    Provided by **VK_ARM_tensors**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Binds device memory to one or more tensors. Each bind info
    ///specifies the tensor, memory, and offset. Must be called before
    ///the tensor is used in any command. Similar to
    ///`bind_buffer_memory2` / `bind_image_memory2`.
    ///
    ///Requires `VK_ARM_tensors`.
    pub unsafe fn bind_tensor_memory_arm(
        &self,
        p_bind_infos: &[BindTensorMemoryInfoARM],
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .bind_tensor_memory_arm
            .expect("vkBindTensorMemoryARM not loaded");
        check(unsafe {
            fp(
                self.handle(),
                p_bind_infos.len() as u32,
                p_bind_infos.as_ptr(),
            )
        })
    }
    ///Wraps [`vkGetDeviceTensorMemoryRequirementsARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDeviceTensorMemoryRequirementsARM.html).
    /**
    Provided by **VK_ARM_tensors**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries memory requirements for a tensor described by its create
    ///info, without creating the tensor first. Analogous to
    ///`get_device_buffer_memory_requirements` /
    ///`get_device_image_memory_requirements`.
    ///
    ///Requires `VK_ARM_tensors`.
    pub unsafe fn get_device_tensor_memory_requirements_arm(
        &self,
        p_info: &DeviceTensorMemoryRequirementsARM,
        p_memory_requirements: *mut MemoryRequirements2,
    ) {
        let fp = self
            .commands()
            .get_device_tensor_memory_requirements_arm
            .expect("vkGetDeviceTensorMemoryRequirementsARM not loaded");
        unsafe { fp(self.handle(), p_info, p_memory_requirements) };
    }
    ///Wraps [`vkCmdCopyTensorARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyTensorARM.html).
    /**
    Provided by **VK_ARM_tensors**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Copies data between tensors or between a tensor and a buffer.
    ///The copy info structure specifies the source and destination
    ///regions, similar to `cmd_copy_buffer` or `cmd_copy_image`.
    ///
    ///Requires `VK_ARM_tensors`.
    pub unsafe fn cmd_copy_tensor_arm(
        &self,
        command_buffer: CommandBuffer,
        p_copy_tensor_info: &CopyTensorInfoARM,
    ) {
        let fp = self
            .commands()
            .cmd_copy_tensor_arm
            .expect("vkCmdCopyTensorARM not loaded");
        unsafe { fp(command_buffer, p_copy_tensor_info) };
    }
    ///Wraps [`vkGetTensorOpaqueCaptureDescriptorDataARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetTensorOpaqueCaptureDescriptorDataARM.html).
    /**
    Provided by **VK_ARM_tensors**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves opaque capture data for a tensor's descriptor, used
    ///for descriptor buffer capture and replay. The returned data can
    ///be stored and replayed to recreate an equivalent descriptor.
    ///
    ///Requires `VK_ARM_tensors` + `VK_EXT_descriptor_buffer`.
    pub unsafe fn get_tensor_opaque_capture_descriptor_data_arm(
        &self,
        p_info: &TensorCaptureDescriptorDataInfoARM,
    ) -> VkResult<core::ffi::c_void> {
        let fp = self
            .commands()
            .get_tensor_opaque_capture_descriptor_data_arm
            .expect("vkGetTensorOpaqueCaptureDescriptorDataARM not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_info, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetTensorViewOpaqueCaptureDescriptorDataARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetTensorViewOpaqueCaptureDescriptorDataARM.html).
    /**
    Provided by **VK_ARM_tensors**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves opaque capture data for a tensor view's descriptor,
    ///used for descriptor buffer capture and replay. Similar to
    ///`get_tensor_opaque_capture_descriptor_data_arm` but for tensor
    ///views.
    ///
    ///Requires `VK_ARM_tensors` + `VK_EXT_descriptor_buffer`.
    pub unsafe fn get_tensor_view_opaque_capture_descriptor_data_arm(
        &self,
        p_info: &TensorViewCaptureDescriptorDataInfoARM,
    ) -> VkResult<core::ffi::c_void> {
        let fp = self
            .commands()
            .get_tensor_view_opaque_capture_descriptor_data_arm
            .expect("vkGetTensorViewOpaqueCaptureDescriptorDataARM not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_info, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkCreateDataGraphPipelinesARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateDataGraphPipelinesARM.html).
    /**
    Provided by **VK_ARM_data_graph**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates one or more data graph pipelines for ARM's data graph
    ///extension. Supports deferred compilation via a
    ///`DeferredOperationKHR` handle and pipeline caching. Data graph
    ///pipelines define GPU-side data processing graphs.
    ///
    ///Requires `VK_ARM_data_graph`.
    pub unsafe fn create_data_graph_pipelines_arm(
        &self,
        deferred_operation: DeferredOperationKHR,
        pipeline_cache: PipelineCache,
        p_create_infos: &[DataGraphPipelineCreateInfoARM],
        allocator: Option<&AllocationCallbacks>,
        p_pipelines: *mut Pipeline,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .create_data_graph_pipelines_arm
            .expect("vkCreateDataGraphPipelinesARM not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        check(unsafe {
            fp(
                self.handle(),
                deferred_operation,
                pipeline_cache,
                p_create_infos.len() as u32,
                p_create_infos.as_ptr(),
                alloc_ptr,
                p_pipelines,
            )
        })
    }
    ///Wraps [`vkCreateDataGraphPipelineSessionARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateDataGraphPipelineSessionARM.html).
    /**
    Provided by **VK_ARM_data_graph**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a session for executing a data graph pipeline. A session
    ///holds the runtime state and memory bindings needed to dispatch
    ///the graph. Must be bound to memory before dispatch.
    ///
    ///Requires `VK_ARM_data_graph`.
    pub unsafe fn create_data_graph_pipeline_session_arm(
        &self,
        p_create_info: &DataGraphPipelineSessionCreateInfoARM,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<DataGraphPipelineSessionARM> {
        let fp = self
            .commands()
            .create_data_graph_pipeline_session_arm
            .expect("vkCreateDataGraphPipelineSessionARM not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetDataGraphPipelineSessionBindPointRequirementsARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDataGraphPipelineSessionBindPointRequirementsARM.html).
    /**
    Provided by **VK_ARM_data_graph**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the memory bind point requirements for a data graph
    ///pipeline session. Uses the two-call idiom. Each returned
    ///requirement describes a bind point that must be satisfied with
    ///`bind_data_graph_pipeline_session_memory_arm` before dispatch.
    ///
    ///Requires `VK_ARM_data_graph`.
    pub unsafe fn get_data_graph_pipeline_session_bind_point_requirements_arm(
        &self,
        p_info: &DataGraphPipelineSessionBindPointRequirementsInfoARM,
    ) -> VkResult<Vec<DataGraphPipelineSessionBindPointRequirementARM>> {
        let fp = self
            .commands()
            .get_data_graph_pipeline_session_bind_point_requirements_arm
            .expect("vkGetDataGraphPipelineSessionBindPointRequirementsARM not loaded");
        enumerate_two_call(|count, data| unsafe { fp(self.handle(), p_info, count, data) })
    }
    ///Wraps [`vkGetDataGraphPipelineSessionMemoryRequirementsARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDataGraphPipelineSessionMemoryRequirementsARM.html).
    /**
    Provided by **VK_ARM_data_graph**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the memory requirements for a specific bind point within
    ///a data graph pipeline session. Returns a `MemoryRequirements2`
    ///describing the size, alignment, and compatible memory types.
    ///
    ///Requires `VK_ARM_data_graph`.
    pub unsafe fn get_data_graph_pipeline_session_memory_requirements_arm(
        &self,
        p_info: &DataGraphPipelineSessionMemoryRequirementsInfoARM,
        p_memory_requirements: *mut MemoryRequirements2,
    ) {
        let fp = self
            .commands()
            .get_data_graph_pipeline_session_memory_requirements_arm
            .expect("vkGetDataGraphPipelineSessionMemoryRequirementsARM not loaded");
        unsafe { fp(self.handle(), p_info, p_memory_requirements) };
    }
    ///Wraps [`vkBindDataGraphPipelineSessionMemoryARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkBindDataGraphPipelineSessionMemoryARM.html).
    /**
    Provided by **VK_ARM_data_graph**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Binds device memory to a data graph pipeline session at the bind
    ///points returned by
    ///`get_data_graph_pipeline_session_bind_point_requirements_arm`.
    ///Must be called before dispatching the session.
    ///
    ///Requires `VK_ARM_data_graph`.
    pub unsafe fn bind_data_graph_pipeline_session_memory_arm(
        &self,
        p_bind_infos: &[BindDataGraphPipelineSessionMemoryInfoARM],
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .bind_data_graph_pipeline_session_memory_arm
            .expect("vkBindDataGraphPipelineSessionMemoryARM not loaded");
        check(unsafe {
            fp(
                self.handle(),
                p_bind_infos.len() as u32,
                p_bind_infos.as_ptr(),
            )
        })
    }
    ///Wraps [`vkDestroyDataGraphPipelineSessionARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyDataGraphPipelineSessionARM.html).
    /**
    Provided by **VK_ARM_data_graph**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///- `session` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a data graph pipeline session and frees associated host
    ///resources. The session must not be in use by any command buffer.
    ///
    ///Requires `VK_ARM_data_graph`.
    pub unsafe fn destroy_data_graph_pipeline_session_arm(
        &self,
        session: DataGraphPipelineSessionARM,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_data_graph_pipeline_session_arm
            .expect("vkDestroyDataGraphPipelineSessionARM not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), session, alloc_ptr) };
    }
    ///Wraps [`vkCmdDispatchDataGraphARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDispatchDataGraphARM.html).
    /**
    Provided by **VK_ARM_data_graph**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Records a data graph pipeline dispatch into a command buffer.
    ///The session must have been created and bound to memory before
    ///dispatch. Optional dispatch info can configure execution
    ///parameters.
    ///
    ///Requires `VK_ARM_data_graph`.
    pub unsafe fn cmd_dispatch_data_graph_arm(
        &self,
        command_buffer: CommandBuffer,
        session: DataGraphPipelineSessionARM,
        p_info: Option<&DataGraphPipelineDispatchInfoARM>,
    ) {
        let fp = self
            .commands()
            .cmd_dispatch_data_graph_arm
            .expect("vkCmdDispatchDataGraphARM not loaded");
        let p_info_ptr = p_info.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(command_buffer, session, p_info_ptr) };
    }
    ///Wraps [`vkGetDataGraphPipelineAvailablePropertiesARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDataGraphPipelineAvailablePropertiesARM.html).
    /**
    Provided by **VK_ARM_data_graph**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Enumerates the queryable properties for a data graph pipeline.
    ///Uses the two-call idiom. The returned property descriptors can
    ///then be queried with `get_data_graph_pipeline_properties_arm`.
    ///
    ///Requires `VK_ARM_data_graph`.
    pub unsafe fn get_data_graph_pipeline_available_properties_arm(
        &self,
        p_pipeline_info: &DataGraphPipelineInfoARM,
    ) -> VkResult<Vec<DataGraphPipelinePropertyARM>> {
        let fp = self
            .commands()
            .get_data_graph_pipeline_available_properties_arm
            .expect("vkGetDataGraphPipelineAvailablePropertiesARM not loaded");
        enumerate_two_call(|count, data| unsafe { fp(self.handle(), p_pipeline_info, count, data) })
    }
    ///Wraps [`vkGetDataGraphPipelinePropertiesARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDataGraphPipelinePropertiesARM.html).
    /**
    Provided by **VK_ARM_data_graph**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries specific property values for a data graph pipeline.
    ///First enumerate available properties with
    ///`get_data_graph_pipeline_available_properties_arm`, then query
    ///the ones you need.
    ///
    ///Requires `VK_ARM_data_graph`.
    pub unsafe fn get_data_graph_pipeline_properties_arm(
        &self,
        p_pipeline_info: &DataGraphPipelineInfoARM,
        properties_count: u32,
        p_properties: *mut DataGraphPipelinePropertyQueryResultARM,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_data_graph_pipeline_properties_arm
            .expect("vkGetDataGraphPipelinePropertiesARM not loaded");
        check(unsafe {
            fp(
                self.handle(),
                p_pipeline_info,
                properties_count,
                p_properties,
            )
        })
    }
    ///Wraps [`vkGetNativeBufferPropertiesOHOS`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetNativeBufferPropertiesOHOS.html).
    /**
    Provided by **VK_OHOS_external_memory**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_INVALID_EXTERNAL_HANDLE_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries Vulkan memory properties (memory type bits, size) for
    ///an OHOS native buffer. Use before importing external memory to
    ///determine compatible memory types. OHOS only.
    ///
    ///Requires `VK_OHOS_external_memory`.
    pub unsafe fn get_native_buffer_properties_ohos(
        &self,
        buffer: *const core::ffi::c_void,
        p_properties: *mut NativeBufferPropertiesOHOS,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_native_buffer_properties_ohos
            .expect("vkGetNativeBufferPropertiesOHOS not loaded");
        check(unsafe { fp(self.handle(), buffer, p_properties) })
    }
    ///Wraps [`vkGetMemoryNativeBufferOHOS`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetMemoryNativeBufferOHOS.html).
    /**
    Provided by **VK_OHOS_external_memory**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Exports a Vulkan device memory allocation as an OHOS native
    ///buffer handle for sharing with other OpenHarmony services.
    ///OHOS only.
    ///
    ///Requires `VK_OHOS_external_memory`.
    pub unsafe fn get_memory_native_buffer_ohos(
        &self,
        p_info: &MemoryGetNativeBufferInfoOHOS,
        p_buffer: *mut *mut core::ffi::c_void,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_memory_native_buffer_ohos
            .expect("vkGetMemoryNativeBufferOHOS not loaded");
        check(unsafe { fp(self.handle(), p_info, p_buffer) })
    }
    ///Wraps [`vkGetSwapchainGrallocUsageOHOS`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetSwapchainGrallocUsageOHOS.html).
    ///
    ///# Errors
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the OHOS gralloc usage flags needed for swapchain images
    ///with the given format and Vulkan image usage. Used internally by
    ///the OHOS WSI implementation. OHOS only.
    ///
    ///Requires `VK_OHOS_native_buffer`.
    pub unsafe fn get_swapchain_gralloc_usage_ohos(
        &self,
        format: Format,
        image_usage: ImageUsageFlags,
    ) -> VkResult<u64> {
        let fp = self
            .commands()
            .get_swapchain_gralloc_usage_ohos
            .expect("vkGetSwapchainGrallocUsageOHOS not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), format, image_usage, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkAcquireImageOHOS`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkAcquireImageOHOS.html).
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Acquires ownership of a swapchain image on OpenHarmony OS.
    ///Takes a native fence file descriptor for synchronisation and
    ///can signal a Vulkan semaphore or fence on completion. OHOS only.
    ///
    ///Requires `VK_OHOS_native_buffer`.
    pub unsafe fn acquire_image_ohos(
        &self,
        image: Image,
        native_fence_fd: i32,
        semaphore: Semaphore,
        fence: Fence,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .acquire_image_ohos
            .expect("vkAcquireImageOHOS not loaded");
        check(unsafe { fp(self.handle(), image, native_fence_fd, semaphore, fence) })
    }
    ///Wraps [`vkQueueSignalReleaseImageOHOS`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkQueueSignalReleaseImageOHOS.html).
    ///
    ///# Errors
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `queue` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Releases a swapchain image back to the OHOS compositor after
    ///rendering. Waits on the given semaphores and returns a native
    ///fence FD for external synchronisation. OHOS only.
    ///
    ///Requires `VK_OHOS_native_buffer`.
    pub unsafe fn queue_signal_release_image_ohos(
        &self,
        queue: Queue,
        p_wait_semaphores: &[Semaphore],
        image: Image,
    ) -> VkResult<i32> {
        let fp = self
            .commands()
            .queue_signal_release_image_ohos
            .expect("vkQueueSignalReleaseImageOHOS not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe {
            fp(
                queue,
                p_wait_semaphores.len() as u32,
                p_wait_semaphores.as_ptr(),
                image,
                &mut out,
            )
        })?;
        Ok(out)
    }
    ///Wraps [`vkCmdSetComputeOccupancyPriorityNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetComputeOccupancyPriorityNV.html).
    /**
    Provided by **VK_NV_compute_occupancy_priority**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Sets the compute occupancy priority for subsequent dispatch
    ///commands. Higher priority may increase the number of warps
    ///resident on an SM, trading off per-warp resources for greater
    ///parallelism.
    ///
    ///Requires `VK_NV_compute_occupancy_priority`.
    pub unsafe fn cmd_set_compute_occupancy_priority_nv(
        &self,
        command_buffer: CommandBuffer,
        p_parameters: &ComputeOccupancyPriorityParametersNV,
    ) {
        let fp = self
            .commands()
            .cmd_set_compute_occupancy_priority_nv
            .expect("vkCmdSetComputeOccupancyPriorityNV not loaded");
        unsafe { fp(command_buffer, p_parameters) };
    }
    ///Wraps [`vkWriteSamplerDescriptorsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkWriteSamplerDescriptorsEXT.html).
    /**
    Provided by **VK_EXT_descriptor_heap**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Writes sampler descriptors to host-visible memory. This is the
    ///descriptor heap equivalent for sampler descriptors, instead of
    ///using descriptor sets, samplers are written directly to heap
    ///memory.
    ///
    ///Provided by `VK_EXT_descriptor_heap`.
    pub unsafe fn write_sampler_descriptors_ext(
        &self,
        p_samplers: &[SamplerCreateInfo],
        p_descriptors: &HostAddressRangeEXT,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .write_sampler_descriptors_ext
            .expect("vkWriteSamplerDescriptorsEXT not loaded");
        check(unsafe {
            fp(
                self.handle(),
                p_samplers.len() as u32,
                p_samplers.as_ptr(),
                p_descriptors,
            )
        })
    }
    ///Wraps [`vkWriteResourceDescriptorsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkWriteResourceDescriptorsEXT.html).
    /**
    Provided by **VK_EXT_descriptor_heap**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Writes resource descriptors (buffers, images, acceleration
    ///structures) to host-visible memory. This is the descriptor heap
    ///equivalent of writing descriptors, instead of using descriptor
    ///sets, descriptors are written directly to heap memory.
    ///
    ///Provided by `VK_EXT_descriptor_heap`.
    pub unsafe fn write_resource_descriptors_ext(
        &self,
        p_resources: &[ResourceDescriptorInfoEXT],
        p_descriptors: &HostAddressRangeEXT,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .write_resource_descriptors_ext
            .expect("vkWriteResourceDescriptorsEXT not loaded");
        check(unsafe {
            fp(
                self.handle(),
                p_resources.len() as u32,
                p_resources.as_ptr(),
                p_descriptors,
            )
        })
    }
    ///Wraps [`vkCmdBindSamplerHeapEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBindSamplerHeapEXT.html).
    /**
    Provided by **VK_EXT_descriptor_heap**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Binds a sampler descriptor heap for use in subsequent draw and
    ///dispatch commands. The `BindHeapInfoEXT` specifies the heap to
    ///bind.
    ///
    ///Sampler heaps hold sampler descriptors. Resource descriptors are
    ///bound separately with `cmd_bind_resource_heap_ext`.
    ///
    ///Provided by `VK_EXT_descriptor_heap`.
    pub unsafe fn cmd_bind_sampler_heap_ext(
        &self,
        command_buffer: CommandBuffer,
        p_bind_info: &BindHeapInfoEXT,
    ) {
        let fp = self
            .commands()
            .cmd_bind_sampler_heap_ext
            .expect("vkCmdBindSamplerHeapEXT not loaded");
        unsafe { fp(command_buffer, p_bind_info) };
    }
    ///Wraps [`vkCmdBindResourceHeapEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBindResourceHeapEXT.html).
    /**
    Provided by **VK_EXT_descriptor_heap**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Binds a resource descriptor heap for use in subsequent draw and
    ///dispatch commands. The `BindHeapInfoEXT` specifies the heap to
    ///bind.
    ///
    ///Resource heaps hold descriptors for buffers, images, and
    ///acceleration structures. Samplers are bound separately with
    ///`cmd_bind_sampler_heap_ext`.
    ///
    ///Provided by `VK_EXT_descriptor_heap`.
    pub unsafe fn cmd_bind_resource_heap_ext(
        &self,
        command_buffer: CommandBuffer,
        p_bind_info: &BindHeapInfoEXT,
    ) {
        let fp = self
            .commands()
            .cmd_bind_resource_heap_ext
            .expect("vkCmdBindResourceHeapEXT not loaded");
        unsafe { fp(command_buffer, p_bind_info) };
    }
    ///Wraps [`vkCmdPushDataEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdPushDataEXT.html).
    /**
    Provided by **VK_EXT_descriptor_heap**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Pushes inline data to the command buffer for use in shaders. The
    ///`PushDataInfoEXT` specifies the pipeline layout, stage flags,
    ///offset, and data bytes.
    ///
    ///Similar to push constants but used with the descriptor heap
    ///model. Data is accessible in shaders via the push data mechanism.
    ///
    ///Provided by `VK_EXT_descriptor_heap`.
    pub unsafe fn cmd_push_data_ext(
        &self,
        command_buffer: CommandBuffer,
        p_push_data_info: &PushDataInfoEXT,
    ) {
        let fp = self
            .commands()
            .cmd_push_data_ext
            .expect("vkCmdPushDataEXT not loaded");
        unsafe { fp(command_buffer, p_push_data_info) };
    }
    ///Wraps [`vkRegisterCustomBorderColorEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkRegisterCustomBorderColorEXT.html).
    /**
    Provided by **VK_EXT_descriptor_heap**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_TOO_MANY_OBJECTS`
    ///- `VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Registers a custom border color for use with samplers. Returns a
    ///handle that can be referenced when creating samplers with
    ///`BORDER_COLOR_CUSTOM` modes.
    ///
    ///The device has a limited number of custom border color slots
    ///(query `maxCustomBorderColors`).
    ///
    ///Unregister with `unregister_custom_border_color_ext` when no
    ///longer needed.
    ///
    ///Provided by `VK_EXT_descriptor_heap`.
    pub unsafe fn register_custom_border_color_ext(
        &self,
        p_border_color: &SamplerCustomBorderColorCreateInfoEXT,
        request_index: u32,
    ) -> VkResult<u32> {
        let fp = self
            .commands()
            .register_custom_border_color_ext
            .expect("vkRegisterCustomBorderColorEXT not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_border_color, request_index, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkUnregisterCustomBorderColorEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkUnregisterCustomBorderColorEXT.html).
    /**
    Provided by **VK_EXT_descriptor_heap**.*/
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Unregisters a custom border color previously registered with
    ///`register_custom_border_color_ext`, freeing the slot for reuse.
    ///
    ///Ensure no samplers referencing this border color are in use
    ///before unregistering.
    ///
    ///Provided by `VK_EXT_descriptor_heap`.
    pub unsafe fn unregister_custom_border_color_ext(&self, index: u32) {
        let fp = self
            .commands()
            .unregister_custom_border_color_ext
            .expect("vkUnregisterCustomBorderColorEXT not loaded");
        unsafe { fp(self.handle(), index) };
    }
    ///Wraps [`vkGetImageOpaqueCaptureDataEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetImageOpaqueCaptureDataEXT.html).
    /**
    Provided by **VK_EXT_descriptor_heap**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves opaque capture data for one or more images. The returned
    ///`HostAddressRangeEXT` data can be used to reconstruct image
    ///resource bindings during capture/replay.
    ///
    ///Provided by `VK_EXT_descriptor_heap`.
    pub unsafe fn get_image_opaque_capture_data_ext(
        &self,
        p_images: &[Image],
        p_datas: *mut HostAddressRangeEXT,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_image_opaque_capture_data_ext
            .expect("vkGetImageOpaqueCaptureDataEXT not loaded");
        check(unsafe {
            fp(
                self.handle(),
                p_images.len() as u32,
                p_images.as_ptr(),
                p_datas,
            )
        })
    }
    ///Wraps [`vkGetTensorOpaqueCaptureDataARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetTensorOpaqueCaptureDataARM.html).
    /**
    Provided by **VK_EXT_descriptor_heap**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Retrieves opaque capture data for one or more ARM tensors. The
    ///returned data can be used to reconstruct tensor resource bindings
    ///during capture/replay.
    ///
    ///Provided by `VK_ARM_tensors`.
    pub unsafe fn get_tensor_opaque_capture_data_arm(
        &self,
        p_tensors: &[TensorARM],
        p_datas: *mut HostAddressRangeEXT,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_tensor_opaque_capture_data_arm
            .expect("vkGetTensorOpaqueCaptureDataARM not loaded");
        check(unsafe {
            fp(
                self.handle(),
                p_tensors.len() as u32,
                p_tensors.as_ptr(),
                p_datas,
            )
        })
    }
    ///Wraps [`vkCmdCopyMemoryKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyMemoryKHR.html).
    /**
    Provided by **VK_KHR_device_address_commands**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Copies data between two device address ranges on the GPU. This is
    ///the device-address equivalent of `cmd_copy_buffer`, instead of
    ///buffer handles, source and destination are specified as device
    ///addresses in `CopyDeviceMemoryInfoKHR`.
    ///
    ///Useful for copying data between arbitrary device memory locations
    ///without needing buffer objects.
    ///
    ///Requires `VK_KHR_device_address_commands`.
    pub unsafe fn cmd_copy_memory_khr(
        &self,
        command_buffer: CommandBuffer,
        p_copy_memory_info: Option<&CopyDeviceMemoryInfoKHR>,
    ) {
        let fp = self
            .commands()
            .cmd_copy_memory_khr
            .expect("vkCmdCopyMemoryKHR not loaded");
        let p_copy_memory_info_ptr =
            p_copy_memory_info.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(command_buffer, p_copy_memory_info_ptr) };
    }
    ///Wraps [`vkCmdCopyMemoryToImageKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyMemoryToImageKHR.html).
    /**
    Provided by **VK_KHR_device_address_commands**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Device-address variant of `cmd_copy_buffer_to_image`. Copies data
    ///from a device address range into an image, instead of reading from
    ///a buffer handle.
    ///
    ///The `CopyDeviceMemoryImageInfoKHR` struct specifies the source
    ///device address, destination image, and region descriptions.
    ///
    ///This is the device-address counterpart, for the host-side
    ///equivalent, see `copy_memory_to_image` (core 1.4).
    ///
    ///Requires `VK_KHR_device_address_commands`.
    pub unsafe fn cmd_copy_memory_to_image_khr(
        &self,
        command_buffer: CommandBuffer,
        p_copy_memory_info: Option<&CopyDeviceMemoryImageInfoKHR>,
    ) {
        let fp = self
            .commands()
            .cmd_copy_memory_to_image_khr
            .expect("vkCmdCopyMemoryToImageKHR not loaded");
        let p_copy_memory_info_ptr =
            p_copy_memory_info.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(command_buffer, p_copy_memory_info_ptr) };
    }
    ///Wraps [`vkCmdCopyImageToMemoryKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyImageToMemoryKHR.html).
    /**
    Provided by **VK_KHR_device_address_commands**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Device-address variant of `cmd_copy_image_to_buffer`. Copies image
    ///texel data to a device address range instead of a buffer handle.
    ///
    ///The `CopyDeviceMemoryImageInfoKHR` struct specifies the source
    ///image, destination device address, and region descriptions.
    ///
    ///This is the device-address counterpart, for the host-side
    ///equivalent, see `copy_image_to_memory` (core 1.4).
    ///
    ///Requires `VK_KHR_device_address_commands`.
    pub unsafe fn cmd_copy_image_to_memory_khr(
        &self,
        command_buffer: CommandBuffer,
        p_copy_memory_info: Option<&CopyDeviceMemoryImageInfoKHR>,
    ) {
        let fp = self
            .commands()
            .cmd_copy_image_to_memory_khr
            .expect("vkCmdCopyImageToMemoryKHR not loaded");
        let p_copy_memory_info_ptr =
            p_copy_memory_info.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(command_buffer, p_copy_memory_info_ptr) };
    }
    ///Wraps [`vkCmdUpdateMemoryKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdUpdateMemoryKHR.html).
    /**
    Provided by **VK_KHR_device_address_commands**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Device-address variant of `cmd_update_buffer`. Writes a small
    ///amount of inline data to a device address range, instead of
    ///targeting a buffer handle.
    ///
    ///`data_size` must be ≤ 65536 bytes and a multiple of 4. For
    ///larger transfers, use `cmd_copy_memory_khr`.
    ///
    ///The `DeviceAddressRangeKHR` specifies the destination address.
    ///
    ///Requires `VK_KHR_device_address_commands`.
    pub unsafe fn cmd_update_memory_khr(
        &self,
        command_buffer: CommandBuffer,
        p_dst_range: &DeviceAddressRangeKHR,
        dst_flags: AddressCommandFlagsKHR,
        data_size: u64,
        p_data: *const core::ffi::c_void,
    ) {
        let fp = self
            .commands()
            .cmd_update_memory_khr
            .expect("vkCmdUpdateMemoryKHR not loaded");
        unsafe { fp(command_buffer, p_dst_range, dst_flags, data_size, p_data) };
    }
    ///Wraps [`vkCmdFillMemoryKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdFillMemoryKHR.html).
    /**
    Provided by **VK_KHR_device_address_commands**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Device-address variant of `cmd_fill_buffer`. Fills a device
    ///address range with a repeating 32-bit `data` value, instead of
    ///targeting a buffer handle.
    ///
    ///The `DeviceAddressRangeKHR` specifies the destination address
    ///and size. The fill size must be a multiple of 4 bytes.
    ///
    ///Requires `VK_KHR_device_address_commands`.
    pub unsafe fn cmd_fill_memory_khr(
        &self,
        command_buffer: CommandBuffer,
        p_dst_range: &DeviceAddressRangeKHR,
        dst_flags: AddressCommandFlagsKHR,
        data: u32,
    ) {
        let fp = self
            .commands()
            .cmd_fill_memory_khr
            .expect("vkCmdFillMemoryKHR not loaded");
        unsafe { fp(command_buffer, p_dst_range, dst_flags, data) };
    }
    ///Wraps [`vkCmdCopyQueryPoolResultsToMemoryKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyQueryPoolResultsToMemoryKHR.html).
    /**
    Provided by **VK_KHR_device_address_commands**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Device-address variant of `cmd_copy_query_pool_results`. Copies
    ///query results directly to a device address range instead of a
    ///buffer handle.
    ///
    ///`first_query` and `query_count` select which queries to copy.
    ///`query_result_flags` controls formatting (`_64_BIT`, `WAIT`,
    ///`WITH_AVAILABILITY`, `PARTIAL`).
    ///
    ///The `StridedDeviceAddressRangeKHR` specifies the destination
    ///address and stride between results.
    ///
    ///Requires `VK_KHR_device_address_commands`.
    pub unsafe fn cmd_copy_query_pool_results_to_memory_khr(
        &self,
        command_buffer: CommandBuffer,
        query_pool: QueryPool,
        first_query: u32,
        query_count: u32,
        p_dst_range: &StridedDeviceAddressRangeKHR,
        dst_flags: AddressCommandFlagsKHR,
        query_result_flags: QueryResultFlags,
    ) {
        let fp = self
            .commands()
            .cmd_copy_query_pool_results_to_memory_khr
            .expect("vkCmdCopyQueryPoolResultsToMemoryKHR not loaded");
        unsafe {
            fp(
                command_buffer,
                query_pool,
                first_query,
                query_count,
                p_dst_range,
                dst_flags,
                query_result_flags,
            )
        };
    }
    ///Wraps [`vkCmdBeginConditionalRendering2EXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBeginConditionalRendering2EXT.html).
    /**
    Provided by **VK_KHR_device_address_commands**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Device-address variant of `cmd_begin_conditional_rendering_ext`.
    ///Instead of a buffer handle + offset, the condition is read from a
    ///`DeviceAddress` specified in `ConditionalRenderingBeginInfo2EXT`.
    ///
    ///When the 32-bit value at the address is zero, subsequent rendering
    ///and dispatch commands are discarded (or the inverse, if
    ///`INVERTED` is set). End the conditional block with
    ///`cmd_end_conditional_rendering_ext`.
    ///
    ///Requires `VK_KHR_device_address_commands` and
    ///`VK_EXT_conditional_rendering`.
    pub unsafe fn cmd_begin_conditional_rendering2_ext(
        &self,
        command_buffer: CommandBuffer,
        p_conditional_rendering_begin: &ConditionalRenderingBeginInfo2EXT,
    ) {
        let fp = self
            .commands()
            .cmd_begin_conditional_rendering2_ext
            .expect("vkCmdBeginConditionalRendering2EXT not loaded");
        unsafe { fp(command_buffer, p_conditional_rendering_begin) };
    }
    ///Wraps [`vkCmdBindTransformFeedbackBuffers2EXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBindTransformFeedbackBuffers2EXT.html).
    /**
    Provided by **VK_KHR_device_address_commands**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Device-address variant of `cmd_bind_transform_feedback_buffers_ext`.
    ///Binds transform feedback output buffers using device addresses
    ///instead of buffer handles.
    ///
    ///Each `BindTransformFeedbackBuffer2InfoEXT` specifies a device
    ///address and size for one binding slot starting at `first_binding`.
    ///
    ///Requires `VK_KHR_device_address_commands` and
    ///`VK_EXT_transform_feedback`.
    pub unsafe fn cmd_bind_transform_feedback_buffers2_ext(
        &self,
        command_buffer: CommandBuffer,
        first_binding: u32,
        p_binding_infos: &[BindTransformFeedbackBuffer2InfoEXT],
    ) {
        let fp = self
            .commands()
            .cmd_bind_transform_feedback_buffers2_ext
            .expect("vkCmdBindTransformFeedbackBuffers2EXT not loaded");
        unsafe {
            fp(
                command_buffer,
                first_binding,
                p_binding_infos.len() as u32,
                p_binding_infos.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdBeginTransformFeedback2EXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBeginTransformFeedback2EXT.html).
    /**
    Provided by **VK_KHR_device_address_commands**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Device-address variant of `cmd_begin_transform_feedback_ext`.
    ///Activates transform feedback using counter buffers specified via
    ///device addresses in `BindTransformFeedbackBuffer2InfoEXT` rather
    ///than buffer handles.
    ///
    ///`first_counter_range` and the info array identify which transform
    ///feedback counter ranges to resume from. Pass empty counter infos
    ///to start from offset zero.
    ///
    ///End the transform feedback pass with
    ///`cmd_end_transform_feedback2_ext`.
    ///
    ///Requires `VK_KHR_device_address_commands` and
    ///`VK_EXT_transform_feedback`.
    pub unsafe fn cmd_begin_transform_feedback2_ext(
        &self,
        command_buffer: CommandBuffer,
        first_counter_range: u32,
        p_counter_infos: &[BindTransformFeedbackBuffer2InfoEXT],
    ) {
        let fp = self
            .commands()
            .cmd_begin_transform_feedback2_ext
            .expect("vkCmdBeginTransformFeedback2EXT not loaded");
        unsafe {
            fp(
                command_buffer,
                first_counter_range,
                p_counter_infos.len() as u32,
                p_counter_infos.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdEndTransformFeedback2EXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdEndTransformFeedback2EXT.html).
    /**
    Provided by **VK_KHR_device_address_commands**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Device-address variant of `cmd_end_transform_feedback_ext`.
    ///Stops transform feedback and writes counter values to device
    ///addresses specified in `BindTransformFeedbackBuffer2InfoEXT`.
    ///
    ///These saved counter values can be passed to
    ///`cmd_begin_transform_feedback2_ext` to resume feedback in a
    ///later render pass.
    ///
    ///Requires `VK_KHR_device_address_commands` and
    ///`VK_EXT_transform_feedback`.
    pub unsafe fn cmd_end_transform_feedback2_ext(
        &self,
        command_buffer: CommandBuffer,
        first_counter_range: u32,
        p_counter_infos: &[BindTransformFeedbackBuffer2InfoEXT],
    ) {
        let fp = self
            .commands()
            .cmd_end_transform_feedback2_ext
            .expect("vkCmdEndTransformFeedback2EXT not loaded");
        unsafe {
            fp(
                command_buffer,
                first_counter_range,
                p_counter_infos.len() as u32,
                p_counter_infos.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdDrawIndirectByteCount2EXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDrawIndirectByteCount2EXT.html).
    /**
    Provided by **VK_KHR_device_address_commands**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Device-address variant of `cmd_draw_indirect_byte_count_ext`.
    ///Draws vertices using a byte count from a transform feedback
    ///counter, with the counter buffer specified via device address
    ///instead of a buffer handle.
    ///
    ///`counter_offset` is the byte offset within the counter value
    ///to account for any header. `vertex_stride` determines how many
    ///bytes each vertex consumes.
    ///
    ///Requires `VK_KHR_device_address_commands` and
    ///`VK_EXT_transform_feedback`.
    pub unsafe fn cmd_draw_indirect_byte_count2_ext(
        &self,
        command_buffer: CommandBuffer,
        instance_count: u32,
        first_instance: u32,
        p_counter_info: &BindTransformFeedbackBuffer2InfoEXT,
        counter_offset: u32,
        vertex_stride: u32,
    ) {
        let fp = self
            .commands()
            .cmd_draw_indirect_byte_count2_ext
            .expect("vkCmdDrawIndirectByteCount2EXT not loaded");
        unsafe {
            fp(
                command_buffer,
                instance_count,
                first_instance,
                p_counter_info,
                counter_offset,
                vertex_stride,
            )
        };
    }
    ///Wraps [`vkCmdWriteMarkerToMemoryAMD`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdWriteMarkerToMemoryAMD.html).
    /**
    Provided by **VK_KHR_device_address_commands**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Device-address variant of the AMD buffer marker extension. Writes
    ///a 32-bit marker value to a device address at a specific pipeline
    ///stage.
    ///
    ///The `MemoryMarkerInfoAMD` specifies the pipeline stage, device
    ///address, and marker value. Useful for GPU crash debugging,
    ///markers that were written indicate how far the GPU progressed
    ///before a fault.
    ///
    ///Requires `VK_KHR_device_address_commands`. This is the
    ///device-address counterpart of `cmd_write_buffer_marker_amd`.
    pub unsafe fn cmd_write_marker_to_memory_amd(
        &self,
        command_buffer: CommandBuffer,
        p_info: &MemoryMarkerInfoAMD,
    ) {
        let fp = self
            .commands()
            .cmd_write_marker_to_memory_amd
            .expect("vkCmdWriteMarkerToMemoryAMD not loaded");
        unsafe { fp(command_buffer, p_info) };
    }
    ///Wraps [`vkCmdBindIndexBuffer3KHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBindIndexBuffer3KHR.html).
    /**
    Provided by **VK_KHR_device_address_commands**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Device-address variant of `cmd_bind_index_buffer`. Binds an index
    ///buffer for subsequent indexed draw commands using a device address
    ///instead of a buffer handle.
    ///
    ///The `BindIndexBuffer3InfoKHR` struct specifies the device address,
    ///size, and index type (`UINT16`, `UINT32`, or `UINT8` if enabled).
    ///
    ///Supersedes `cmd_bind_index_buffer` and `cmd_bind_index_buffer2`
    ///when using `VK_KHR_device_address_commands`.
    pub unsafe fn cmd_bind_index_buffer3_khr(
        &self,
        command_buffer: CommandBuffer,
        p_info: &BindIndexBuffer3InfoKHR,
    ) {
        let fp = self
            .commands()
            .cmd_bind_index_buffer3_khr
            .expect("vkCmdBindIndexBuffer3KHR not loaded");
        unsafe { fp(command_buffer, p_info) };
    }
    ///Wraps [`vkCmdBindVertexBuffers3KHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBindVertexBuffers3KHR.html).
    /**
    Provided by **VK_KHR_device_address_commands**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Device-address variant of `cmd_bind_vertex_buffers`. Binds vertex
    ///buffers for subsequent draw commands using device addresses instead
    ///of buffer handles.
    ///
    ///Each `BindVertexBuffer3InfoKHR` specifies a device address, size,
    ///and stride for one binding slot starting at `first_binding`.
    ///
    ///Supersedes `cmd_bind_vertex_buffers`, `cmd_bind_vertex_buffers2`,
    ///and the core 1.4 equivalent when using
    ///`VK_KHR_device_address_commands`.
    pub unsafe fn cmd_bind_vertex_buffers3_khr(
        &self,
        command_buffer: CommandBuffer,
        first_binding: u32,
        p_binding_infos: &[BindVertexBuffer3InfoKHR],
    ) {
        let fp = self
            .commands()
            .cmd_bind_vertex_buffers3_khr
            .expect("vkCmdBindVertexBuffers3KHR not loaded");
        unsafe {
            fp(
                command_buffer,
                first_binding,
                p_binding_infos.len() as u32,
                p_binding_infos.as_ptr(),
            )
        };
    }
    ///Wraps [`vkCmdDrawIndirect2KHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDrawIndirect2KHR.html).
    /**
    Provided by **VK_KHR_device_address_commands**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Device-address variant of `cmd_draw_indirect`. Reads non-indexed
    ///draw parameters from a device address instead of a buffer handle
    ///+ offset.
    ///
    ///The `DrawIndirect2InfoKHR` specifies the device address, draw
    ///count, and stride.
    ///
    ///Requires `VK_KHR_device_address_commands`.
    pub unsafe fn cmd_draw_indirect2_khr(
        &self,
        command_buffer: CommandBuffer,
        p_info: &DrawIndirect2InfoKHR,
    ) {
        let fp = self
            .commands()
            .cmd_draw_indirect2_khr
            .expect("vkCmdDrawIndirect2KHR not loaded");
        unsafe { fp(command_buffer, p_info) };
    }
    ///Wraps [`vkCmdDrawIndexedIndirect2KHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDrawIndexedIndirect2KHR.html).
    /**
    Provided by **VK_KHR_device_address_commands**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Device-address variant of `cmd_draw_indexed_indirect`. Reads
    ///indexed draw parameters from a device address instead of a buffer
    ///handle + offset.
    ///
    ///The `DrawIndirect2InfoKHR` specifies the device address, draw
    ///count, and stride.
    ///
    ///Requires `VK_KHR_device_address_commands`.
    pub unsafe fn cmd_draw_indexed_indirect2_khr(
        &self,
        command_buffer: CommandBuffer,
        p_info: &DrawIndirect2InfoKHR,
    ) {
        let fp = self
            .commands()
            .cmd_draw_indexed_indirect2_khr
            .expect("vkCmdDrawIndexedIndirect2KHR not loaded");
        unsafe { fp(command_buffer, p_info) };
    }
    ///Wraps [`vkCmdDrawIndirectCount2KHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDrawIndirectCount2KHR.html).
    /**
    Provided by **VK_KHR_device_address_commands**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Device-address variant of `cmd_draw_indirect_count`. Reads
    ///non-indexed draw parameters and the draw count from device
    ///addresses instead of buffer handles.
    ///
    ///The `DrawIndirectCount2InfoKHR` specifies both the draw parameter
    ///address and the count address, along with `max_draw_count` and
    ///stride.
    ///
    ///Requires `VK_KHR_device_address_commands`.
    pub unsafe fn cmd_draw_indirect_count2_khr(
        &self,
        command_buffer: CommandBuffer,
        p_info: &DrawIndirectCount2InfoKHR,
    ) {
        let fp = self
            .commands()
            .cmd_draw_indirect_count2_khr
            .expect("vkCmdDrawIndirectCount2KHR not loaded");
        unsafe { fp(command_buffer, p_info) };
    }
    ///Wraps [`vkCmdDrawIndexedIndirectCount2KHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDrawIndexedIndirectCount2KHR.html).
    /**
    Provided by **VK_KHR_device_address_commands**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Device-address variant of `cmd_draw_indexed_indirect_count`. Reads
    ///indexed draw parameters and the draw count from device addresses
    ///instead of buffer handles.
    ///
    ///The `DrawIndirectCount2InfoKHR` specifies both the draw parameter
    ///address and the count address, along with `max_draw_count` and
    ///stride.
    ///
    ///Requires `VK_KHR_device_address_commands`.
    pub unsafe fn cmd_draw_indexed_indirect_count2_khr(
        &self,
        command_buffer: CommandBuffer,
        p_info: &DrawIndirectCount2InfoKHR,
    ) {
        let fp = self
            .commands()
            .cmd_draw_indexed_indirect_count2_khr
            .expect("vkCmdDrawIndexedIndirectCount2KHR not loaded");
        unsafe { fp(command_buffer, p_info) };
    }
    ///Wraps [`vkCmdDrawMeshTasksIndirect2EXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDrawMeshTasksIndirect2EXT.html).
    /**
    Provided by **VK_KHR_device_address_commands**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Device-address variant of `cmd_draw_mesh_tasks_indirect_ext`.
    ///Reads mesh shader dispatch parameters from a device address
    ///instead of a buffer handle.
    ///
    ///The `DrawIndirect2InfoKHR` specifies the device address, draw
    ///count, and stride.
    ///
    ///Requires `VK_KHR_device_address_commands` and
    ///`VK_EXT_mesh_shader`.
    pub unsafe fn cmd_draw_mesh_tasks_indirect2_ext(
        &self,
        command_buffer: CommandBuffer,
        p_info: &DrawIndirect2InfoKHR,
    ) {
        let fp = self
            .commands()
            .cmd_draw_mesh_tasks_indirect2_ext
            .expect("vkCmdDrawMeshTasksIndirect2EXT not loaded");
        unsafe { fp(command_buffer, p_info) };
    }
    ///Wraps [`vkCmdDrawMeshTasksIndirectCount2EXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDrawMeshTasksIndirectCount2EXT.html).
    /**
    Provided by **VK_KHR_device_address_commands**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Device-address variant of `cmd_draw_mesh_tasks_indirect_count_ext`.
    ///Reads mesh shader dispatch parameters and the draw count from
    ///device addresses instead of buffer handles.
    ///
    ///The `DrawIndirectCount2InfoKHR` specifies both the draw parameter
    ///address and the count address, along with `max_draw_count` and
    ///stride.
    ///
    ///Requires `VK_KHR_device_address_commands` and
    ///`VK_EXT_mesh_shader`.
    pub unsafe fn cmd_draw_mesh_tasks_indirect_count2_ext(
        &self,
        command_buffer: CommandBuffer,
        p_info: &DrawIndirectCount2InfoKHR,
    ) {
        let fp = self
            .commands()
            .cmd_draw_mesh_tasks_indirect_count2_ext
            .expect("vkCmdDrawMeshTasksIndirectCount2EXT not loaded");
        unsafe { fp(command_buffer, p_info) };
    }
    ///Wraps [`vkCmdDispatchIndirect2KHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDispatchIndirect2KHR.html).
    /**
    Provided by **VK_KHR_device_address_commands**.*/
    ///
    ///# Safety
    ///- `commandBuffer` (self) must be valid and not destroyed.
    ///- `commandBuffer` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Device-address variant of `cmd_dispatch_indirect`. Reads the
    ///dispatch parameters (group counts) from a device address instead
    ///of a buffer handle + offset.
    ///
    ///The `DispatchIndirect2InfoKHR` specifies the device address where
    ///the `DispatchIndirectCommand` struct resides.
    ///
    ///Requires `VK_KHR_device_address_commands`.
    pub unsafe fn cmd_dispatch_indirect2_khr(
        &self,
        command_buffer: CommandBuffer,
        p_info: &DispatchIndirect2InfoKHR,
    ) {
        let fp = self
            .commands()
            .cmd_dispatch_indirect2_khr
            .expect("vkCmdDispatchIndirect2KHR not loaded");
        unsafe { fp(command_buffer, p_info) };
    }
    ///Wraps [`vkCreateAccelerationStructure2KHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateAccelerationStructure2KHR.html).
    /**
    Provided by **VK_KHR_device_address_commands**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS_KHR`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///
    ///# Safety
    ///- `device` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Device-address variant of `create_acceleration_structure_khr`.
    ///Creates a ray tracing acceleration structure backed by a device
    ///address range instead of a buffer handle.
    ///
    ///The `AccelerationStructureCreateInfo2KHR` specifies the device
    ///address, size, type (top-level or bottom-level), and optional
    ///capture/replay address.
    ///
    ///Destroy with `destroy_acceleration_structure_khr`.
    ///
    ///Requires `VK_KHR_device_address_commands` and
    ///`VK_KHR_acceleration_structure`.
    pub unsafe fn create_acceleration_structure2_khr(
        &self,
        p_create_info: &AccelerationStructureCreateInfo2KHR,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<AccelerationStructureKHR> {
        let fp = self
            .commands()
            .create_acceleration_structure2_khr
            .expect("vkCreateAccelerationStructure2KHR not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
}
