#![allow(unused_imports)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::missing_safety_doc)]
use crate::error::{VkResult, check, enumerate_two_call, fill_two_call};
use crate::vk::bitmasks::*;
use crate::vk::constants::*;
use crate::vk::enums::*;
use crate::vk::handles::*;
use crate::vk::structs::*;
impl crate::Device {
    pub unsafe fn get_device_proc_addr(&self, p_name: *const core::ffi::c_char) {
        let fp = self
            .commands()
            .get_device_proc_addr
            .expect("vkGetDeviceProcAddr not loaded");
        unsafe { fp(self.handle(), p_name) };
    }
    pub unsafe fn destroy_device(&self, allocator: Option<&AllocationCallbacks>) {
        let fp = self
            .commands()
            .destroy_device
            .expect("vkDestroyDevice not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), alloc_ptr) };
    }
    pub unsafe fn get_device_queue(&self, queue_family_index: u32, queue_index: u32) -> Queue {
        let fp = self
            .commands()
            .get_device_queue
            .expect("vkGetDeviceQueue not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(self.handle(), queue_family_index, queue_index, &mut out) };
        out
    }
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
    pub unsafe fn queue_wait_idle(&self, queue: Queue) -> VkResult<()> {
        let fp = self
            .commands()
            .queue_wait_idle
            .expect("vkQueueWaitIdle not loaded");
        check(unsafe { fp(queue) })
    }
    pub unsafe fn device_wait_idle(&self) -> VkResult<()> {
        let fp = self
            .commands()
            .device_wait_idle
            .expect("vkDeviceWaitIdle not loaded");
        check(unsafe { fp(self.handle()) })
    }
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
    pub unsafe fn unmap_memory(&self, memory: DeviceMemory) {
        let fp = self
            .commands()
            .unmap_memory
            .expect("vkUnmapMemory not loaded");
        unsafe { fp(self.handle(), memory) };
    }
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
    pub unsafe fn get_device_memory_commitment(&self, memory: DeviceMemory) -> u64 {
        let fp = self
            .commands()
            .get_device_memory_commitment
            .expect("vkGetDeviceMemoryCommitment not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(self.handle(), memory, &mut out) };
        out
    }
    pub unsafe fn get_buffer_memory_requirements(&self, buffer: Buffer) -> MemoryRequirements {
        let fp = self
            .commands()
            .get_buffer_memory_requirements
            .expect("vkGetBufferMemoryRequirements not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(self.handle(), buffer, &mut out) };
        out
    }
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
    pub unsafe fn get_image_memory_requirements(&self, image: Image) -> MemoryRequirements {
        let fp = self
            .commands()
            .get_image_memory_requirements
            .expect("vkGetImageMemoryRequirements not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(self.handle(), image, &mut out) };
        out
    }
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
    pub unsafe fn destroy_fence(&self, fence: Fence, allocator: Option<&AllocationCallbacks>) {
        let fp = self
            .commands()
            .destroy_fence
            .expect("vkDestroyFence not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), fence, alloc_ptr) };
    }
    pub unsafe fn reset_fences(&self, p_fences: &[Fence]) -> VkResult<()> {
        let fp = self
            .commands()
            .reset_fences
            .expect("vkResetFences not loaded");
        check(unsafe { fp(self.handle(), p_fences.len() as u32, p_fences.as_ptr()) })
    }
    pub unsafe fn get_fence_status(&self, fence: Fence) -> VkResult<()> {
        let fp = self
            .commands()
            .get_fence_status
            .expect("vkGetFenceStatus not loaded");
        check(unsafe { fp(self.handle(), fence) })
    }
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
    pub unsafe fn destroy_event(&self, event: Event, allocator: Option<&AllocationCallbacks>) {
        let fp = self
            .commands()
            .destroy_event
            .expect("vkDestroyEvent not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), event, alloc_ptr) };
    }
    pub unsafe fn get_event_status(&self, event: Event) -> VkResult<()> {
        let fp = self
            .commands()
            .get_event_status
            .expect("vkGetEventStatus not loaded");
        check(unsafe { fp(self.handle(), event) })
    }
    pub unsafe fn set_event(&self, event: Event) -> VkResult<()> {
        let fp = self.commands().set_event.expect("vkSetEvent not loaded");
        check(unsafe { fp(self.handle(), event) })
    }
    pub unsafe fn reset_event(&self, event: Event) -> VkResult<()> {
        let fp = self
            .commands()
            .reset_event
            .expect("vkResetEvent not loaded");
        check(unsafe { fp(self.handle(), event) })
    }
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
    pub unsafe fn destroy_buffer(&self, buffer: Buffer, allocator: Option<&AllocationCallbacks>) {
        let fp = self
            .commands()
            .destroy_buffer
            .expect("vkDestroyBuffer not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), buffer, alloc_ptr) };
    }
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
    pub unsafe fn destroy_image(&self, image: Image, allocator: Option<&AllocationCallbacks>) {
        let fp = self
            .commands()
            .destroy_image
            .expect("vkDestroyImage not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), image, alloc_ptr) };
    }
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
    pub unsafe fn get_render_area_granularity(&self, render_pass: RenderPass) -> Extent2D {
        let fp = self
            .commands()
            .get_render_area_granularity
            .expect("vkGetRenderAreaGranularity not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(self.handle(), render_pass, &mut out) };
        out
    }
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
    pub unsafe fn end_command_buffer(&self, command_buffer: CommandBuffer) -> VkResult<()> {
        let fp = self
            .commands()
            .end_command_buffer
            .expect("vkEndCommandBuffer not loaded");
        check(unsafe { fp(command_buffer) })
    }
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
    pub unsafe fn cmd_set_line_width(&self, command_buffer: CommandBuffer, line_width: f32) {
        let fp = self
            .commands()
            .cmd_set_line_width
            .expect("vkCmdSetLineWidth not loaded");
        unsafe { fp(command_buffer, line_width) };
    }
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
    pub unsafe fn cmd_subpass_shading_huawei(&self, command_buffer: CommandBuffer) {
        let fp = self
            .commands()
            .cmd_subpass_shading_huawei
            .expect("vkCmdSubpassShadingHUAWEI not loaded");
        unsafe { fp(command_buffer) };
    }
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
    pub unsafe fn cmd_end_conditional_rendering_ext(&self, command_buffer: CommandBuffer) {
        let fp = self
            .commands()
            .cmd_end_conditional_rendering_ext
            .expect("vkCmdEndConditionalRenderingEXT not loaded");
        unsafe { fp(command_buffer) };
    }
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
    pub unsafe fn cmd_end_render_pass(&self, command_buffer: CommandBuffer) {
        let fp = self
            .commands()
            .cmd_end_render_pass
            .expect("vkCmdEndRenderPass not loaded");
        unsafe { fp(command_buffer) };
    }
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
    pub unsafe fn get_swapchain_images_khr(&self, swapchain: SwapchainKHR) -> VkResult<Vec<Image>> {
        let fp = self
            .commands()
            .get_swapchain_images_khr
            .expect("vkGetSwapchainImagesKHR not loaded");
        enumerate_two_call(|count, data| unsafe { fp(self.handle(), swapchain, count, data) })
    }
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
    pub unsafe fn cmd_debug_marker_end_ext(&self, command_buffer: CommandBuffer) {
        let fp = self
            .commands()
            .cmd_debug_marker_end_ext
            .expect("vkCmdDebugMarkerEndEXT not loaded");
        unsafe { fp(command_buffer) };
    }
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
    pub unsafe fn trim_command_pool(&self, command_pool: CommandPool, flags: CommandPoolTrimFlags) {
        let fp = self
            .commands()
            .trim_command_pool
            .expect("vkTrimCommandPool not loaded");
        unsafe { fp(self.handle(), command_pool, flags) };
    }
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
    pub unsafe fn cmd_set_device_mask(&self, command_buffer: CommandBuffer, device_mask: u32) {
        let fp = self
            .commands()
            .cmd_set_device_mask
            .expect("vkCmdSetDeviceMask not loaded");
        unsafe { fp(command_buffer, device_mask) };
    }
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
    pub unsafe fn get_swapchain_status_khr(&self, swapchain: SwapchainKHR) -> VkResult<()> {
        let fp = self
            .commands()
            .get_swapchain_status_khr
            .expect("vkGetSwapchainStatusKHR not loaded");
        check(unsafe { fp(self.handle(), swapchain) })
    }
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
    pub unsafe fn get_device_queue2(&self, p_queue_info: &DeviceQueueInfo2) -> Queue {
        let fp = self
            .commands()
            .get_device_queue2
            .expect("vkGetDeviceQueue2 not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(self.handle(), p_queue_info, &mut out) };
        out
    }
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
    pub unsafe fn queue_end_debug_utils_label_ext(&self, queue: Queue) {
        let fp = self
            .commands()
            .queue_end_debug_utils_label_ext
            .expect("vkQueueEndDebugUtilsLabelEXT not loaded");
        unsafe { fp(queue) };
    }
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
    pub unsafe fn cmd_end_debug_utils_label_ext(&self, command_buffer: CommandBuffer) {
        let fp = self
            .commands()
            .cmd_end_debug_utils_label_ext
            .expect("vkCmdEndDebugUtilsLabelEXT not loaded");
        unsafe { fp(command_buffer) };
    }
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
    pub unsafe fn get_semaphore_counter_value(&self, semaphore: Semaphore) -> VkResult<u64> {
        let fp = self
            .commands()
            .get_semaphore_counter_value
            .expect("vkGetSemaphoreCounterValue not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), semaphore, &mut out) })?;
        Ok(out)
    }
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
    pub unsafe fn signal_semaphore(&self, p_signal_info: &SemaphoreSignalInfo) -> VkResult<()> {
        let fp = self
            .commands()
            .signal_semaphore
            .expect("vkSignalSemaphore not loaded");
        check(unsafe { fp(self.handle(), p_signal_info) })
    }
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
    pub unsafe fn get_queue_checkpoint_data_nv(&self, queue: Queue) -> Vec<CheckpointDataNV> {
        let fp = self
            .commands()
            .get_queue_checkpoint_data_nv
            .expect("vkGetQueueCheckpointDataNV not loaded");
        fill_two_call(|count, data| unsafe { fp(queue, count, data) })
    }
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
    pub unsafe fn compile_deferred_nv(&self, pipeline: Pipeline, shader: u32) -> VkResult<()> {
        let fp = self
            .commands()
            .compile_deferred_nv
            .expect("vkCompileDeferredNV not loaded");
        check(unsafe { fp(self.handle(), pipeline, shader) })
    }
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
    pub unsafe fn get_image_view_handle_nvx(&self, p_info: &ImageViewHandleInfoNVX) {
        let fp = self
            .commands()
            .get_image_view_handle_nvx
            .expect("vkGetImageViewHandleNVX not loaded");
        unsafe { fp(self.handle(), p_info) };
    }
    pub unsafe fn get_image_view_handle64_nvx(&self, p_info: &ImageViewHandleInfoNVX) {
        let fp = self
            .commands()
            .get_image_view_handle64_nvx
            .expect("vkGetImageViewHandle64NVX not loaded");
        unsafe { fp(self.handle(), p_info) };
    }
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
    pub unsafe fn release_profiling_lock_khr(&self) {
        let fp = self
            .commands()
            .release_profiling_lock_khr
            .expect("vkReleaseProfilingLockKHR not loaded");
        unsafe { fp(self.handle()) };
    }
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
    pub unsafe fn get_buffer_opaque_capture_address(&self, p_info: &BufferDeviceAddressInfo) {
        let fp = self
            .commands()
            .get_buffer_opaque_capture_address
            .expect("vkGetBufferOpaqueCaptureAddress not loaded");
        unsafe { fp(self.handle(), p_info) };
    }
    pub unsafe fn get_buffer_device_address(&self, p_info: &BufferDeviceAddressInfo) {
        let fp = self
            .commands()
            .get_buffer_device_address
            .expect("vkGetBufferDeviceAddress not loaded");
        unsafe { fp(self.handle(), p_info) };
    }
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
    pub unsafe fn uninitialize_performance_api_intel(&self) {
        let fp = self
            .commands()
            .uninitialize_performance_api_intel
            .expect("vkUninitializePerformanceApiINTEL not loaded");
        unsafe { fp(self.handle()) };
    }
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
    pub unsafe fn anti_lag_update_amd(&self, p_data: &AntiLagDataAMD) {
        let fp = self
            .commands()
            .anti_lag_update_amd
            .expect("vkAntiLagUpdateAMD not loaded");
        unsafe { fp(self.handle(), p_data) };
    }
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
    pub unsafe fn cmd_set_front_face(&self, command_buffer: CommandBuffer, front_face: FrontFace) {
        let fp = self
            .commands()
            .cmd_set_front_face
            .expect("vkCmdSetFrontFace not loaded");
        unsafe { fp(command_buffer, front_face) };
    }
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
    pub unsafe fn cmd_set_logic_op_ext(&self, command_buffer: CommandBuffer, logic_op: LogicOp) {
        let fp = self
            .commands()
            .cmd_set_logic_op_ext
            .expect("vkCmdSetLogicOpEXT not loaded");
        unsafe { fp(command_buffer, logic_op) };
    }
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
    pub unsafe fn get_queue_checkpoint_data2_nv(&self, queue: Queue) -> Vec<CheckpointData2NV> {
        let fp = self
            .commands()
            .get_queue_checkpoint_data2_nv
            .expect("vkGetQueueCheckpointData2NV not loaded");
        fill_two_call(|count, data| unsafe { fp(queue, count, data) })
    }
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
    pub unsafe fn get_descriptor_set_layout_size_ext(&self, layout: DescriptorSetLayout) -> u64 {
        let fp = self
            .commands()
            .get_descriptor_set_layout_size_ext
            .expect("vkGetDescriptorSetLayoutSizeEXT not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(self.handle(), layout, &mut out) };
        out
    }
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
    pub unsafe fn set_device_memory_priority_ext(&self, memory: DeviceMemory, priority: f32) {
        let fp = self
            .commands()
            .set_device_memory_priority_ext
            .expect("vkSetDeviceMemoryPriorityEXT not loaded");
        unsafe { fp(self.handle(), memory, priority) };
    }
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
    pub unsafe fn cmd_end_rendering(&self, command_buffer: CommandBuffer) {
        let fp = self
            .commands()
            .cmd_end_rendering
            .expect("vkCmdEndRendering not loaded");
        unsafe { fp(command_buffer) };
    }
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
    pub unsafe fn unmap_memory2(&self, p_memory_unmap_info: &MemoryUnmapInfo) -> VkResult<()> {
        let fp = self
            .commands()
            .unmap_memory2
            .expect("vkUnmapMemory2 not loaded");
        check(unsafe { fp(self.handle(), p_memory_unmap_info) })
    }
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
    pub unsafe fn cmd_end_shader_instrumentation_arm(&self, command_buffer: CommandBuffer) {
        let fp = self
            .commands()
            .cmd_end_shader_instrumentation_arm
            .expect("vkCmdEndShaderInstrumentationARM not loaded");
        unsafe { fp(command_buffer) };
    }
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
    pub unsafe fn unregister_custom_border_color_ext(&self, index: u32) {
        let fp = self
            .commands()
            .unregister_custom_border_color_ext
            .expect("vkUnregisterCustomBorderColorEXT not loaded");
        unsafe { fp(self.handle(), index) };
    }
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
