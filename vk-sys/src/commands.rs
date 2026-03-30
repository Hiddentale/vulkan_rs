#![allow(clippy::manual_c_str_literals)]
#![allow(clippy::missing_transmute_annotations)]
#![allow(clippy::field_reassign_with_default)]
use super::structs::*;
use super::enums::*;
use super::handles::*;
use super::bitmasks::*;
pub type PFN_vkCreateInstance = Option<
    unsafe extern "system" fn(
        p_create_info: *const InstanceCreateInfo,
        p_allocator: *const AllocationCallbacks,
        p_instance: *mut Instance,
    ) -> Result,
>;
pub type PFN_vkDestroyInstance = Option<
    unsafe extern "system" fn(
        instance: Instance,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkEnumeratePhysicalDevices = Option<
    unsafe extern "system" fn(
        instance: Instance,
        p_physical_device_count: *mut u32,
        p_physical_devices: *mut PhysicalDevice,
    ) -> Result,
>;
pub type PFN_vkGetDeviceProcAddr = Option<
    unsafe extern "system" fn(
        device: Device,
        p_name: *const std::ffi::c_char,
    ) -> PFN_vkVoidFunction,
>;
pub type PFN_vkGetInstanceProcAddr = Option<
    unsafe extern "system" fn(
        instance: Instance,
        p_name: *const std::ffi::c_char,
    ) -> PFN_vkVoidFunction,
>;
pub type PFN_vkGetPhysicalDeviceProperties = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_properties: *mut PhysicalDeviceProperties,
    ),
>;
pub type PFN_vkGetPhysicalDeviceQueueFamilyProperties = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_queue_family_property_count: *mut u32,
        p_queue_family_properties: *mut QueueFamilyProperties,
    ),
>;
pub type PFN_vkGetPhysicalDeviceMemoryProperties = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_memory_properties: *mut PhysicalDeviceMemoryProperties,
    ),
>;
pub type PFN_vkGetPhysicalDeviceFeatures = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_features: *mut PhysicalDeviceFeatures,
    ),
>;
pub type PFN_vkGetPhysicalDeviceFormatProperties = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        format: Format,
        p_format_properties: *mut FormatProperties,
    ),
>;
pub type PFN_vkGetPhysicalDeviceImageFormatProperties = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        format: Format,
        r#type: ImageType,
        tiling: ImageTiling,
        usage: ImageUsageFlags,
        flags: ImageCreateFlags,
        p_image_format_properties: *mut ImageFormatProperties,
    ) -> Result,
>;
pub type PFN_vkCreateDevice = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_create_info: *const DeviceCreateInfo,
        p_allocator: *const AllocationCallbacks,
        p_device: *mut Device,
    ) -> Result,
>;
pub type PFN_vkDestroyDevice = Option<
    unsafe extern "system" fn(device: Device, p_allocator: *const AllocationCallbacks),
>;
pub type PFN_vkEnumerateInstanceVersion = Option<
    unsafe extern "system" fn(p_api_version: *mut u32) -> Result,
>;
pub type PFN_vkEnumerateInstanceLayerProperties = Option<
    unsafe extern "system" fn(
        p_property_count: *mut u32,
        p_properties: *mut LayerProperties,
    ) -> Result,
>;
pub type PFN_vkEnumerateInstanceExtensionProperties = Option<
    unsafe extern "system" fn(
        p_layer_name: *const std::ffi::c_char,
        p_property_count: *mut u32,
        p_properties: *mut ExtensionProperties,
    ) -> Result,
>;
pub type PFN_vkEnumerateDeviceLayerProperties = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_property_count: *mut u32,
        p_properties: *mut LayerProperties,
    ) -> Result,
>;
pub type PFN_vkEnumerateDeviceExtensionProperties = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_layer_name: *const std::ffi::c_char,
        p_property_count: *mut u32,
        p_properties: *mut ExtensionProperties,
    ) -> Result,
>;
pub type PFN_vkGetDeviceQueue = Option<
    unsafe extern "system" fn(
        device: Device,
        queue_family_index: u32,
        queue_index: u32,
        p_queue: *mut Queue,
    ),
>;
pub type PFN_vkQueueSubmit = Option<
    unsafe extern "system" fn(
        queue: Queue,
        submit_count: u32,
        p_submits: *const SubmitInfo,
        fence: Fence,
    ) -> Result,
>;
pub type PFN_vkQueueWaitIdle = Option<unsafe extern "system" fn(queue: Queue) -> Result>;
pub type PFN_vkDeviceWaitIdle = Option<
    unsafe extern "system" fn(device: Device) -> Result,
>;
pub type PFN_vkAllocateMemory = Option<
    unsafe extern "system" fn(
        device: Device,
        p_allocate_info: *const MemoryAllocateInfo,
        p_allocator: *const AllocationCallbacks,
        p_memory: *mut DeviceMemory,
    ) -> Result,
>;
pub type PFN_vkFreeMemory = Option<
    unsafe extern "system" fn(
        device: Device,
        memory: DeviceMemory,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkMapMemory = Option<
    unsafe extern "system" fn(
        device: Device,
        memory: DeviceMemory,
        offset: u64,
        size: u64,
        flags: MemoryMapFlags,
        pp_data: *mut *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkUnmapMemory = Option<
    unsafe extern "system" fn(device: Device, memory: DeviceMemory),
>;
pub type PFN_vkFlushMappedMemoryRanges = Option<
    unsafe extern "system" fn(
        device: Device,
        memory_range_count: u32,
        p_memory_ranges: *const MappedMemoryRange,
    ) -> Result,
>;
pub type PFN_vkInvalidateMappedMemoryRanges = Option<
    unsafe extern "system" fn(
        device: Device,
        memory_range_count: u32,
        p_memory_ranges: *const MappedMemoryRange,
    ) -> Result,
>;
pub type PFN_vkGetDeviceMemoryCommitment = Option<
    unsafe extern "system" fn(
        device: Device,
        memory: DeviceMemory,
        p_committed_memory_in_bytes: *mut u64,
    ),
>;
pub type PFN_vkGetBufferMemoryRequirements = Option<
    unsafe extern "system" fn(
        device: Device,
        buffer: Buffer,
        p_memory_requirements: *mut MemoryRequirements,
    ),
>;
pub type PFN_vkBindBufferMemory = Option<
    unsafe extern "system" fn(
        device: Device,
        buffer: Buffer,
        memory: DeviceMemory,
        memory_offset: u64,
    ) -> Result,
>;
pub type PFN_vkGetImageMemoryRequirements = Option<
    unsafe extern "system" fn(
        device: Device,
        image: Image,
        p_memory_requirements: *mut MemoryRequirements,
    ),
>;
pub type PFN_vkBindImageMemory = Option<
    unsafe extern "system" fn(
        device: Device,
        image: Image,
        memory: DeviceMemory,
        memory_offset: u64,
    ) -> Result,
>;
pub type PFN_vkGetImageSparseMemoryRequirements = Option<
    unsafe extern "system" fn(
        device: Device,
        image: Image,
        p_sparse_memory_requirement_count: *mut u32,
        p_sparse_memory_requirements: *mut SparseImageMemoryRequirements,
    ),
>;
pub type PFN_vkGetPhysicalDeviceSparseImageFormatProperties = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        format: Format,
        r#type: ImageType,
        samples: SampleCountFlagBits,
        usage: ImageUsageFlags,
        tiling: ImageTiling,
        p_property_count: *mut u32,
        p_properties: *mut SparseImageFormatProperties,
    ),
>;
pub type PFN_vkQueueBindSparse = Option<
    unsafe extern "system" fn(
        queue: Queue,
        bind_info_count: u32,
        p_bind_info: *const BindSparseInfo,
        fence: Fence,
    ) -> Result,
>;
pub type PFN_vkCreateFence = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const FenceCreateInfo,
        p_allocator: *const AllocationCallbacks,
        p_fence: *mut Fence,
    ) -> Result,
>;
pub type PFN_vkDestroyFence = Option<
    unsafe extern "system" fn(
        device: Device,
        fence: Fence,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkResetFences = Option<
    unsafe extern "system" fn(
        device: Device,
        fence_count: u32,
        p_fences: *const Fence,
    ) -> Result,
>;
pub type PFN_vkGetFenceStatus = Option<
    unsafe extern "system" fn(device: Device, fence: Fence) -> Result,
>;
pub type PFN_vkWaitForFences = Option<
    unsafe extern "system" fn(
        device: Device,
        fence_count: u32,
        p_fences: *const Fence,
        wait_all: u32,
        timeout: u64,
    ) -> Result,
>;
pub type PFN_vkCreateSemaphore = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const SemaphoreCreateInfo,
        p_allocator: *const AllocationCallbacks,
        p_semaphore: *mut Semaphore,
    ) -> Result,
>;
pub type PFN_vkDestroySemaphore = Option<
    unsafe extern "system" fn(
        device: Device,
        semaphore: Semaphore,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkCreateEvent = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const EventCreateInfo,
        p_allocator: *const AllocationCallbacks,
        p_event: *mut Event,
    ) -> Result,
>;
pub type PFN_vkDestroyEvent = Option<
    unsafe extern "system" fn(
        device: Device,
        event: Event,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkGetEventStatus = Option<
    unsafe extern "system" fn(device: Device, event: Event) -> Result,
>;
pub type PFN_vkSetEvent = Option<
    unsafe extern "system" fn(device: Device, event: Event) -> Result,
>;
pub type PFN_vkResetEvent = Option<
    unsafe extern "system" fn(device: Device, event: Event) -> Result,
>;
pub type PFN_vkCreateQueryPool = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const QueryPoolCreateInfo,
        p_allocator: *const AllocationCallbacks,
        p_query_pool: *mut QueryPool,
    ) -> Result,
>;
pub type PFN_vkDestroyQueryPool = Option<
    unsafe extern "system" fn(
        device: Device,
        query_pool: QueryPool,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkGetQueryPoolResults = Option<
    unsafe extern "system" fn(
        device: Device,
        query_pool: QueryPool,
        first_query: u32,
        query_count: u32,
        data_size: usize,
        p_data: *mut std::ffi::c_void,
        stride: u64,
        flags: QueryResultFlags,
    ) -> Result,
>;
pub type PFN_vkResetQueryPool = Option<
    unsafe extern "system" fn(
        device: Device,
        query_pool: QueryPool,
        first_query: u32,
        query_count: u32,
    ),
>;
pub type PFN_vkCreateBuffer = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const BufferCreateInfo,
        p_allocator: *const AllocationCallbacks,
        p_buffer: *mut Buffer,
    ) -> Result,
>;
pub type PFN_vkDestroyBuffer = Option<
    unsafe extern "system" fn(
        device: Device,
        buffer: Buffer,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkCreateBufferView = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const BufferViewCreateInfo,
        p_allocator: *const AllocationCallbacks,
        p_view: *mut BufferView,
    ) -> Result,
>;
pub type PFN_vkDestroyBufferView = Option<
    unsafe extern "system" fn(
        device: Device,
        buffer_view: BufferView,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkCreateImage = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const ImageCreateInfo,
        p_allocator: *const AllocationCallbacks,
        p_image: *mut Image,
    ) -> Result,
>;
pub type PFN_vkDestroyImage = Option<
    unsafe extern "system" fn(
        device: Device,
        image: Image,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkGetImageSubresourceLayout = Option<
    unsafe extern "system" fn(
        device: Device,
        image: Image,
        p_subresource: *const ImageSubresource,
        p_layout: *mut SubresourceLayout,
    ),
>;
pub type PFN_vkCreateImageView = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const ImageViewCreateInfo,
        p_allocator: *const AllocationCallbacks,
        p_view: *mut ImageView,
    ) -> Result,
>;
pub type PFN_vkDestroyImageView = Option<
    unsafe extern "system" fn(
        device: Device,
        image_view: ImageView,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkCreateShaderModule = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const ShaderModuleCreateInfo,
        p_allocator: *const AllocationCallbacks,
        p_shader_module: *mut ShaderModule,
    ) -> Result,
>;
pub type PFN_vkDestroyShaderModule = Option<
    unsafe extern "system" fn(
        device: Device,
        shader_module: ShaderModule,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkCreatePipelineCache = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const PipelineCacheCreateInfo,
        p_allocator: *const AllocationCallbacks,
        p_pipeline_cache: *mut PipelineCache,
    ) -> Result,
>;
pub type PFN_vkDestroyPipelineCache = Option<
    unsafe extern "system" fn(
        device: Device,
        pipeline_cache: PipelineCache,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkGetPipelineCacheData = Option<
    unsafe extern "system" fn(
        device: Device,
        pipeline_cache: PipelineCache,
        p_data_size: *mut usize,
        p_data: *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkMergePipelineCaches = Option<
    unsafe extern "system" fn(
        device: Device,
        dst_cache: PipelineCache,
        src_cache_count: u32,
        p_src_caches: *const PipelineCache,
    ) -> Result,
>;
pub type PFN_vkCreatePipelineBinariesKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const PipelineBinaryCreateInfoKHR,
        p_allocator: *const AllocationCallbacks,
        p_binaries: *mut PipelineBinaryHandlesInfoKHR,
    ) -> Result,
>;
pub type PFN_vkDestroyPipelineBinaryKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        pipeline_binary: PipelineBinaryKHR,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkGetPipelineKeyKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_pipeline_create_info: *const PipelineCreateInfoKHR,
        p_pipeline_key: *mut PipelineBinaryKeyKHR,
    ) -> Result,
>;
pub type PFN_vkGetPipelineBinaryDataKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const PipelineBinaryDataInfoKHR,
        p_pipeline_binary_key: *mut PipelineBinaryKeyKHR,
        p_pipeline_binary_data_size: *mut usize,
        p_pipeline_binary_data: *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkReleaseCapturedPipelineDataKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const ReleaseCapturedPipelineDataInfoKHR,
        p_allocator: *const AllocationCallbacks,
    ) -> Result,
>;
pub type PFN_vkCreateGraphicsPipelines = Option<
    unsafe extern "system" fn(
        device: Device,
        pipeline_cache: PipelineCache,
        create_info_count: u32,
        p_create_infos: *const GraphicsPipelineCreateInfo,
        p_allocator: *const AllocationCallbacks,
        p_pipelines: *mut Pipeline,
    ) -> Result,
>;
pub type PFN_vkCreateComputePipelines = Option<
    unsafe extern "system" fn(
        device: Device,
        pipeline_cache: PipelineCache,
        create_info_count: u32,
        p_create_infos: *const ComputePipelineCreateInfo,
        p_allocator: *const AllocationCallbacks,
        p_pipelines: *mut Pipeline,
    ) -> Result,
>;
pub type PFN_vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI = Option<
    unsafe extern "system" fn(
        device: Device,
        renderpass: RenderPass,
        p_max_workgroup_size: *mut Extent2D,
    ) -> Result,
>;
pub type PFN_vkDestroyPipeline = Option<
    unsafe extern "system" fn(
        device: Device,
        pipeline: Pipeline,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkCreatePipelineLayout = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const PipelineLayoutCreateInfo,
        p_allocator: *const AllocationCallbacks,
        p_pipeline_layout: *mut PipelineLayout,
    ) -> Result,
>;
pub type PFN_vkDestroyPipelineLayout = Option<
    unsafe extern "system" fn(
        device: Device,
        pipeline_layout: PipelineLayout,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkCreateSampler = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const SamplerCreateInfo,
        p_allocator: *const AllocationCallbacks,
        p_sampler: *mut Sampler,
    ) -> Result,
>;
pub type PFN_vkDestroySampler = Option<
    unsafe extern "system" fn(
        device: Device,
        sampler: Sampler,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkCreateDescriptorSetLayout = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const DescriptorSetLayoutCreateInfo,
        p_allocator: *const AllocationCallbacks,
        p_set_layout: *mut DescriptorSetLayout,
    ) -> Result,
>;
pub type PFN_vkDestroyDescriptorSetLayout = Option<
    unsafe extern "system" fn(
        device: Device,
        descriptor_set_layout: DescriptorSetLayout,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkCreateDescriptorPool = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const DescriptorPoolCreateInfo,
        p_allocator: *const AllocationCallbacks,
        p_descriptor_pool: *mut DescriptorPool,
    ) -> Result,
>;
pub type PFN_vkDestroyDescriptorPool = Option<
    unsafe extern "system" fn(
        device: Device,
        descriptor_pool: DescriptorPool,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkResetDescriptorPool = Option<
    unsafe extern "system" fn(
        device: Device,
        descriptor_pool: DescriptorPool,
        flags: DescriptorPoolResetFlags,
    ) -> Result,
>;
pub type PFN_vkAllocateDescriptorSets = Option<
    unsafe extern "system" fn(
        device: Device,
        p_allocate_info: *const DescriptorSetAllocateInfo,
        p_descriptor_sets: *mut DescriptorSet,
    ) -> Result,
>;
pub type PFN_vkFreeDescriptorSets = Option<
    unsafe extern "system" fn(
        device: Device,
        descriptor_pool: DescriptorPool,
        descriptor_set_count: u32,
        p_descriptor_sets: *const DescriptorSet,
    ) -> Result,
>;
pub type PFN_vkUpdateDescriptorSets = Option<
    unsafe extern "system" fn(
        device: Device,
        descriptor_write_count: u32,
        p_descriptor_writes: *const WriteDescriptorSet,
        descriptor_copy_count: u32,
        p_descriptor_copies: *const CopyDescriptorSet,
    ),
>;
pub type PFN_vkCreateFramebuffer = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const FramebufferCreateInfo,
        p_allocator: *const AllocationCallbacks,
        p_framebuffer: *mut Framebuffer,
    ) -> Result,
>;
pub type PFN_vkDestroyFramebuffer = Option<
    unsafe extern "system" fn(
        device: Device,
        framebuffer: Framebuffer,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkCreateRenderPass = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const RenderPassCreateInfo,
        p_allocator: *const AllocationCallbacks,
        p_render_pass: *mut RenderPass,
    ) -> Result,
>;
pub type PFN_vkDestroyRenderPass = Option<
    unsafe extern "system" fn(
        device: Device,
        render_pass: RenderPass,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkGetRenderAreaGranularity = Option<
    unsafe extern "system" fn(
        device: Device,
        render_pass: RenderPass,
        p_granularity: *mut Extent2D,
    ),
>;
pub type PFN_vkGetRenderingAreaGranularity = Option<
    unsafe extern "system" fn(
        device: Device,
        p_rendering_area_info: *const RenderingAreaInfo,
        p_granularity: *mut Extent2D,
    ),
>;
pub type PFN_vkCreateCommandPool = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const CommandPoolCreateInfo,
        p_allocator: *const AllocationCallbacks,
        p_command_pool: *mut CommandPool,
    ) -> Result,
>;
pub type PFN_vkDestroyCommandPool = Option<
    unsafe extern "system" fn(
        device: Device,
        command_pool: CommandPool,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkResetCommandPool = Option<
    unsafe extern "system" fn(
        device: Device,
        command_pool: CommandPool,
        flags: CommandPoolResetFlags,
    ) -> Result,
>;
pub type PFN_vkAllocateCommandBuffers = Option<
    unsafe extern "system" fn(
        device: Device,
        p_allocate_info: *const CommandBufferAllocateInfo,
        p_command_buffers: *mut CommandBuffer,
    ) -> Result,
>;
pub type PFN_vkFreeCommandBuffers = Option<
    unsafe extern "system" fn(
        device: Device,
        command_pool: CommandPool,
        command_buffer_count: u32,
        p_command_buffers: *const CommandBuffer,
    ),
>;
pub type PFN_vkBeginCommandBuffer = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_begin_info: *const CommandBufferBeginInfo,
    ) -> Result,
>;
pub type PFN_vkEndCommandBuffer = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer) -> Result,
>;
pub type PFN_vkResetCommandBuffer = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        flags: CommandBufferResetFlags,
    ) -> Result,
>;
pub type PFN_vkCmdBindPipeline = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        pipeline_bind_point: PipelineBindPoint,
        pipeline: Pipeline,
    ),
>;
pub type PFN_vkCmdSetAttachmentFeedbackLoopEnableEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        aspect_mask: ImageAspectFlags,
    ),
>;
pub type PFN_vkCmdSetViewport = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        first_viewport: u32,
        viewport_count: u32,
        p_viewports: *const Viewport,
    ),
>;
pub type PFN_vkCmdSetScissor = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        first_scissor: u32,
        scissor_count: u32,
        p_scissors: *const Rect2D,
    ),
>;
pub type PFN_vkCmdSetLineWidth = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer, line_width: f32),
>;
pub type PFN_vkCmdSetDepthBias = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        depth_bias_constant_factor: f32,
        depth_bias_clamp: f32,
        depth_bias_slope_factor: f32,
    ),
>;
pub type PFN_vkCmdSetBlendConstants = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer, blend_constants: f32),
>;
pub type PFN_vkCmdSetDepthBounds = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        min_depth_bounds: f32,
        max_depth_bounds: f32,
    ),
>;
pub type PFN_vkCmdSetStencilCompareMask = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        face_mask: StencilFaceFlags,
        compare_mask: u32,
    ),
>;
pub type PFN_vkCmdSetStencilWriteMask = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        face_mask: StencilFaceFlags,
        write_mask: u32,
    ),
>;
pub type PFN_vkCmdSetStencilReference = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        face_mask: StencilFaceFlags,
        reference: u32,
    ),
>;
pub type PFN_vkCmdBindDescriptorSets = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        pipeline_bind_point: PipelineBindPoint,
        layout: PipelineLayout,
        first_set: u32,
        descriptor_set_count: u32,
        p_descriptor_sets: *const DescriptorSet,
        dynamic_offset_count: u32,
        p_dynamic_offsets: *const u32,
    ),
>;
pub type PFN_vkCmdBindIndexBuffer = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        buffer: Buffer,
        offset: u64,
        index_type: IndexType,
    ),
>;
pub type PFN_vkCmdBindVertexBuffers = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        first_binding: u32,
        binding_count: u32,
        p_buffers: *const Buffer,
        p_offsets: *const u64,
    ),
>;
pub type PFN_vkCmdDraw = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ),
>;
pub type PFN_vkCmdDrawIndexed = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ),
>;
pub type PFN_vkCmdDrawMultiEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        draw_count: u32,
        p_vertex_info: *const MultiDrawInfoEXT,
        instance_count: u32,
        first_instance: u32,
        stride: u32,
    ),
>;
pub type PFN_vkCmdDrawMultiIndexedEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        draw_count: u32,
        p_index_info: *const MultiDrawIndexedInfoEXT,
        instance_count: u32,
        first_instance: u32,
        stride: u32,
        p_vertex_offset: *const i32,
    ),
>;
pub type PFN_vkCmdDrawIndirect = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        buffer: Buffer,
        offset: u64,
        draw_count: u32,
        stride: u32,
    ),
>;
pub type PFN_vkCmdDrawIndexedIndirect = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        buffer: Buffer,
        offset: u64,
        draw_count: u32,
        stride: u32,
    ),
>;
pub type PFN_vkCmdDispatch = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ),
>;
pub type PFN_vkCmdDispatchIndirect = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer, buffer: Buffer, offset: u64),
>;
pub type PFN_vkCmdSubpassShadingHUAWEI = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer),
>;
pub type PFN_vkCmdDrawClusterHUAWEI = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ),
>;
pub type PFN_vkCmdDrawClusterIndirectHUAWEI = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer, buffer: Buffer, offset: u64),
>;
pub type PFN_vkCmdUpdatePipelineIndirectBufferNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        pipeline_bind_point: PipelineBindPoint,
        pipeline: Pipeline,
    ),
>;
pub type PFN_vkCmdCopyBuffer = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        src_buffer: Buffer,
        dst_buffer: Buffer,
        region_count: u32,
        p_regions: *const BufferCopy,
    ),
>;
pub type PFN_vkCmdCopyImage = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        src_image: Image,
        src_image_layout: ImageLayout,
        dst_image: Image,
        dst_image_layout: ImageLayout,
        region_count: u32,
        p_regions: *const ImageCopy,
    ),
>;
pub type PFN_vkCmdBlitImage = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        src_image: Image,
        src_image_layout: ImageLayout,
        dst_image: Image,
        dst_image_layout: ImageLayout,
        region_count: u32,
        p_regions: *const ImageBlit,
        filter: Filter,
    ),
>;
pub type PFN_vkCmdCopyBufferToImage = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        src_buffer: Buffer,
        dst_image: Image,
        dst_image_layout: ImageLayout,
        region_count: u32,
        p_regions: *const BufferImageCopy,
    ),
>;
pub type PFN_vkCmdCopyImageToBuffer = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        src_image: Image,
        src_image_layout: ImageLayout,
        dst_buffer: Buffer,
        region_count: u32,
        p_regions: *const BufferImageCopy,
    ),
>;
pub type PFN_vkCmdCopyMemoryIndirectNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        copy_buffer_address: u64,
        copy_count: u32,
        stride: u32,
    ),
>;
pub type PFN_vkCmdCopyMemoryIndirectKHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_copy_memory_indirect_info: *const CopyMemoryIndirectInfoKHR,
    ),
>;
pub type PFN_vkCmdCopyMemoryToImageIndirectNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        copy_buffer_address: u64,
        copy_count: u32,
        stride: u32,
        dst_image: Image,
        dst_image_layout: ImageLayout,
        p_image_subresources: *const ImageSubresourceLayers,
    ),
>;
pub type PFN_vkCmdCopyMemoryToImageIndirectKHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_copy_memory_to_image_indirect_info: *const CopyMemoryToImageIndirectInfoKHR,
    ),
>;
pub type PFN_vkCmdUpdateBuffer = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        dst_buffer: Buffer,
        dst_offset: u64,
        data_size: u64,
        p_data: *const std::ffi::c_void,
    ),
>;
pub type PFN_vkCmdFillBuffer = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        dst_buffer: Buffer,
        dst_offset: u64,
        size: u64,
        data: u32,
    ),
>;
pub type PFN_vkCmdClearColorImage = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        image: Image,
        image_layout: ImageLayout,
        p_color: *const ClearColorValue,
        range_count: u32,
        p_ranges: *const ImageSubresourceRange,
    ),
>;
pub type PFN_vkCmdClearDepthStencilImage = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        image: Image,
        image_layout: ImageLayout,
        p_depth_stencil: *const ClearDepthStencilValue,
        range_count: u32,
        p_ranges: *const ImageSubresourceRange,
    ),
>;
pub type PFN_vkCmdClearAttachments = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        attachment_count: u32,
        p_attachments: *const ClearAttachment,
        rect_count: u32,
        p_rects: *const ClearRect,
    ),
>;
pub type PFN_vkCmdResolveImage = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        src_image: Image,
        src_image_layout: ImageLayout,
        dst_image: Image,
        dst_image_layout: ImageLayout,
        region_count: u32,
        p_regions: *const ImageResolve,
    ),
>;
pub type PFN_vkCmdSetEvent = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        event: Event,
        stage_mask: PipelineStageFlags,
    ),
>;
pub type PFN_vkCmdResetEvent = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        event: Event,
        stage_mask: PipelineStageFlags,
    ),
>;
pub type PFN_vkCmdWaitEvents = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        event_count: u32,
        p_events: *const Event,
        src_stage_mask: PipelineStageFlags,
        dst_stage_mask: PipelineStageFlags,
        memory_barrier_count: u32,
        p_memory_barriers: *const MemoryBarrier,
        buffer_memory_barrier_count: u32,
        p_buffer_memory_barriers: *const BufferMemoryBarrier,
        image_memory_barrier_count: u32,
        p_image_memory_barriers: *const ImageMemoryBarrier,
    ),
>;
pub type PFN_vkCmdPipelineBarrier = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        src_stage_mask: PipelineStageFlags,
        dst_stage_mask: PipelineStageFlags,
        dependency_flags: DependencyFlags,
        memory_barrier_count: u32,
        p_memory_barriers: *const MemoryBarrier,
        buffer_memory_barrier_count: u32,
        p_buffer_memory_barriers: *const BufferMemoryBarrier,
        image_memory_barrier_count: u32,
        p_image_memory_barriers: *const ImageMemoryBarrier,
    ),
>;
pub type PFN_vkCmdBeginQuery = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        query_pool: QueryPool,
        query: u32,
        flags: QueryControlFlags,
    ),
>;
pub type PFN_vkCmdEndQuery = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        query_pool: QueryPool,
        query: u32,
    ),
>;
pub type PFN_vkCmdBeginConditionalRenderingEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_conditional_rendering_begin: *const ConditionalRenderingBeginInfoEXT,
    ),
>;
pub type PFN_vkCmdEndConditionalRenderingEXT = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer),
>;
pub type PFN_vkCmdBeginCustomResolveEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_begin_custom_resolve_info: *const BeginCustomResolveInfoEXT,
    ),
>;
pub type PFN_vkCmdResetQueryPool = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        query_pool: QueryPool,
        first_query: u32,
        query_count: u32,
    ),
>;
pub type PFN_vkCmdWriteTimestamp = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        pipeline_stage: PipelineStageFlagBits,
        query_pool: QueryPool,
        query: u32,
    ),
>;
pub type PFN_vkCmdCopyQueryPoolResults = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        query_pool: QueryPool,
        first_query: u32,
        query_count: u32,
        dst_buffer: Buffer,
        dst_offset: u64,
        stride: u64,
        flags: QueryResultFlags,
    ),
>;
pub type PFN_vkCmdPushConstants = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        layout: PipelineLayout,
        stage_flags: ShaderStageFlags,
        offset: u32,
        size: u32,
        p_values: *const std::ffi::c_void,
    ),
>;
pub type PFN_vkCmdBeginRenderPass = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_render_pass_begin: *const RenderPassBeginInfo,
        contents: SubpassContents,
    ),
>;
pub type PFN_vkCmdNextSubpass = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer, contents: SubpassContents),
>;
pub type PFN_vkCmdEndRenderPass = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer),
>;
pub type PFN_vkCmdExecuteCommands = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        command_buffer_count: u32,
        p_command_buffers: *const CommandBuffer,
    ),
>;
pub type PFN_vkCreateAndroidSurfaceKHR = Option<
    unsafe extern "system" fn(
        instance: Instance,
        p_create_info: *const AndroidSurfaceCreateInfoKHR,
        p_allocator: *const AllocationCallbacks,
        p_surface: *mut SurfaceKHR,
    ) -> Result,
>;
pub type PFN_vkCreateSurfaceOHOS = Option<
    unsafe extern "system" fn(
        instance: Instance,
        p_create_info: *const SurfaceCreateInfoOHOS,
        p_allocator: *const AllocationCallbacks,
        p_surface: *mut SurfaceKHR,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceDisplayPropertiesKHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_property_count: *mut u32,
        p_properties: *mut DisplayPropertiesKHR,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceDisplayPlanePropertiesKHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_property_count: *mut u32,
        p_properties: *mut DisplayPlanePropertiesKHR,
    ) -> Result,
>;
pub type PFN_vkGetDisplayPlaneSupportedDisplaysKHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        plane_index: u32,
        p_display_count: *mut u32,
        p_displays: *mut DisplayKHR,
    ) -> Result,
>;
pub type PFN_vkGetDisplayModePropertiesKHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        display: DisplayKHR,
        p_property_count: *mut u32,
        p_properties: *mut DisplayModePropertiesKHR,
    ) -> Result,
>;
pub type PFN_vkCreateDisplayModeKHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        display: DisplayKHR,
        p_create_info: *const DisplayModeCreateInfoKHR,
        p_allocator: *const AllocationCallbacks,
        p_mode: *mut DisplayModeKHR,
    ) -> Result,
>;
pub type PFN_vkGetDisplayPlaneCapabilitiesKHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        mode: DisplayModeKHR,
        plane_index: u32,
        p_capabilities: *mut DisplayPlaneCapabilitiesKHR,
    ) -> Result,
>;
pub type PFN_vkCreateDisplayPlaneSurfaceKHR = Option<
    unsafe extern "system" fn(
        instance: Instance,
        p_create_info: *const DisplaySurfaceCreateInfoKHR,
        p_allocator: *const AllocationCallbacks,
        p_surface: *mut SurfaceKHR,
    ) -> Result,
>;
pub type PFN_vkCreateSharedSwapchainsKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        swapchain_count: u32,
        p_create_infos: *const SwapchainCreateInfoKHR,
        p_create_infos: *const SwapchainCreateInfoKHR,
        p_allocator: *const AllocationCallbacks,
        p_swapchains: *mut SwapchainKHR,
    ) -> Result,
>;
pub type PFN_vkDestroySurfaceKHR = Option<
    unsafe extern "system" fn(
        instance: Instance,
        surface: SurfaceKHR,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkGetPhysicalDeviceSurfaceSupportKHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        queue_family_index: u32,
        surface: SurfaceKHR,
        p_supported: *mut u32,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        surface: SurfaceKHR,
        p_surface_capabilities: *mut SurfaceCapabilitiesKHR,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceSurfaceFormatsKHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        surface: SurfaceKHR,
        p_surface_format_count: *mut u32,
        p_surface_formats: *mut SurfaceFormatKHR,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceSurfacePresentModesKHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        surface: SurfaceKHR,
        p_present_mode_count: *mut u32,
        p_present_modes: *mut PresentModeKHR,
    ) -> Result,
>;
pub type PFN_vkCreateSwapchainKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const SwapchainCreateInfoKHR,
        p_create_info: *const SwapchainCreateInfoKHR,
        p_allocator: *const AllocationCallbacks,
        p_swapchain: *mut SwapchainKHR,
    ) -> Result,
>;
pub type PFN_vkDestroySwapchainKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        swapchain: SwapchainKHR,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkGetSwapchainImagesKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        swapchain: SwapchainKHR,
        p_swapchain_image_count: *mut u32,
        p_swapchain_images: *mut Image,
    ) -> Result,
>;
pub type PFN_vkAcquireNextImageKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        swapchain: SwapchainKHR,
        timeout: u64,
        semaphore: Semaphore,
        fence: Fence,
        p_image_index: *mut u32,
    ) -> Result,
>;
pub type PFN_vkQueuePresentKHR = Option<
    unsafe extern "system" fn(
        queue: Queue,
        p_present_info: *const PresentInfoKHR,
    ) -> Result,
>;
pub type PFN_vkCreateViSurfaceNN = Option<
    unsafe extern "system" fn(
        instance: Instance,
        p_create_info: *const ViSurfaceCreateInfoNN,
        p_allocator: *const AllocationCallbacks,
        p_surface: *mut SurfaceKHR,
    ) -> Result,
>;
pub type PFN_vkCreateWaylandSurfaceKHR = Option<
    unsafe extern "system" fn(
        instance: Instance,
        p_create_info: *const WaylandSurfaceCreateInfoKHR,
        p_allocator: *const AllocationCallbacks,
        p_surface: *mut SurfaceKHR,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceWaylandPresentationSupportKHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        queue_family_index: u32,
        display: *mut std::ffi::c_void,
    ) -> u32,
>;
pub type PFN_vkCreateUbmSurfaceSEC = Option<
    unsafe extern "system" fn(
        instance: Instance,
        p_create_info: *const UbmSurfaceCreateInfoSEC,
        p_allocator: *const AllocationCallbacks,
        p_surface: *mut SurfaceKHR,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceUbmPresentationSupportSEC = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        queue_family_index: u32,
        device: *mut std::ffi::c_void,
    ) -> u32,
>;
pub type PFN_vkCreateWin32SurfaceKHR = Option<
    unsafe extern "system" fn(
        instance: Instance,
        p_create_info: *const Win32SurfaceCreateInfoKHR,
        p_allocator: *const AllocationCallbacks,
        p_surface: *mut SurfaceKHR,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceWin32PresentationSupportKHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        queue_family_index: u32,
    ) -> u32,
>;
pub type PFN_vkCreateXlibSurfaceKHR = Option<
    unsafe extern "system" fn(
        instance: Instance,
        p_create_info: *const XlibSurfaceCreateInfoKHR,
        p_allocator: *const AllocationCallbacks,
        p_surface: *mut SurfaceKHR,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceXlibPresentationSupportKHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        queue_family_index: u32,
        dpy: *mut std::ffi::c_void,
        visual_id: std::ffi::c_ulong,
    ) -> u32,
>;
pub type PFN_vkCreateXcbSurfaceKHR = Option<
    unsafe extern "system" fn(
        instance: Instance,
        p_create_info: *const XcbSurfaceCreateInfoKHR,
        p_allocator: *const AllocationCallbacks,
        p_surface: *mut SurfaceKHR,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceXcbPresentationSupportKHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        queue_family_index: u32,
        connection: *mut std::ffi::c_void,
        visual_id: u32,
    ) -> u32,
>;
pub type PFN_vkCreateDirectFBSurfaceEXT = Option<
    unsafe extern "system" fn(
        instance: Instance,
        p_create_info: *const DirectFBSurfaceCreateInfoEXT,
        p_allocator: *const AllocationCallbacks,
        p_surface: *mut SurfaceKHR,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceDirectFBPresentationSupportEXT = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        queue_family_index: u32,
        dfb: *mut std::ffi::c_void,
    ) -> u32,
>;
pub type PFN_vkCreateImagePipeSurfaceFUCHSIA = Option<
    unsafe extern "system" fn(
        instance: Instance,
        p_create_info: *const ImagePipeSurfaceCreateInfoFUCHSIA,
        p_allocator: *const AllocationCallbacks,
        p_surface: *mut SurfaceKHR,
    ) -> Result,
>;
pub type PFN_vkCreateStreamDescriptorSurfaceGGP = Option<
    unsafe extern "system" fn(
        instance: Instance,
        p_create_info: *const StreamDescriptorSurfaceCreateInfoGGP,
        p_allocator: *const AllocationCallbacks,
        p_surface: *mut SurfaceKHR,
    ) -> Result,
>;
pub type PFN_vkCreateScreenSurfaceQNX = Option<
    unsafe extern "system" fn(
        instance: Instance,
        p_create_info: *const ScreenSurfaceCreateInfoQNX,
        p_allocator: *const AllocationCallbacks,
        p_surface: *mut SurfaceKHR,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceScreenPresentationSupportQNX = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        queue_family_index: u32,
        window: *mut std::ffi::c_void,
    ) -> u32,
>;
pub type PFN_vkCreateDebugReportCallbackEXT = Option<
    unsafe extern "system" fn(
        instance: Instance,
        p_create_info: *const DebugReportCallbackCreateInfoEXT,
        p_allocator: *const AllocationCallbacks,
        p_callback: *mut DebugReportCallbackEXT,
    ) -> Result,
>;
pub type PFN_vkDestroyDebugReportCallbackEXT = Option<
    unsafe extern "system" fn(
        instance: Instance,
        callback: DebugReportCallbackEXT,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkDebugReportMessageEXT = Option<
    unsafe extern "system" fn(
        instance: Instance,
        flags: DebugReportFlagsEXT,
        object_type: DebugReportObjectTypeEXT,
        object: u64,
        location: usize,
        message_code: i32,
        p_layer_prefix: *const std::ffi::c_char,
        p_message: *const std::ffi::c_char,
    ),
>;
pub type PFN_vkDebugMarkerSetObjectNameEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        p_name_info: *const DebugMarkerObjectNameInfoEXT,
    ) -> Result,
>;
pub type PFN_vkDebugMarkerSetObjectTagEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        p_tag_info: *const DebugMarkerObjectTagInfoEXT,
    ) -> Result,
>;
pub type PFN_vkCmdDebugMarkerBeginEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_marker_info: *const DebugMarkerMarkerInfoEXT,
    ),
>;
pub type PFN_vkCmdDebugMarkerEndEXT = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer),
>;
pub type PFN_vkCmdDebugMarkerInsertEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_marker_info: *const DebugMarkerMarkerInfoEXT,
    ),
>;
pub type PFN_vkGetPhysicalDeviceExternalImageFormatPropertiesNV = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        format: Format,
        r#type: ImageType,
        tiling: ImageTiling,
        usage: ImageUsageFlags,
        flags: ImageCreateFlags,
        external_handle_type: ExternalMemoryHandleTypeFlagsNV,
        p_external_image_format_properties: *mut ExternalImageFormatPropertiesNV,
    ) -> Result,
>;
pub type PFN_vkGetMemoryWin32HandleNV = Option<
    unsafe extern "system" fn(
        device: Device,
        memory: DeviceMemory,
        handle_type: ExternalMemoryHandleTypeFlagsNV,
        p_handle: *mut isize,
    ) -> Result,
>;
pub type PFN_vkCmdExecuteGeneratedCommandsNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        is_preprocessed: u32,
        p_generated_commands_info: *const GeneratedCommandsInfoNV,
    ),
>;
pub type PFN_vkCmdPreprocessGeneratedCommandsNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_generated_commands_info: *const GeneratedCommandsInfoNV,
    ),
>;
pub type PFN_vkCmdBindPipelineShaderGroupNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        pipeline_bind_point: PipelineBindPoint,
        pipeline: Pipeline,
        group_index: u32,
    ),
>;
pub type PFN_vkGetGeneratedCommandsMemoryRequirementsNV = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const GeneratedCommandsMemoryRequirementsInfoNV,
        p_memory_requirements: *mut MemoryRequirements2,
    ),
>;
pub type PFN_vkCreateIndirectCommandsLayoutNV = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const IndirectCommandsLayoutCreateInfoNV,
        p_allocator: *const AllocationCallbacks,
        p_indirect_commands_layout: *mut IndirectCommandsLayoutNV,
    ) -> Result,
>;
pub type PFN_vkDestroyIndirectCommandsLayoutNV = Option<
    unsafe extern "system" fn(
        device: Device,
        indirect_commands_layout: IndirectCommandsLayoutNV,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkCmdExecuteGeneratedCommandsEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        is_preprocessed: u32,
        p_generated_commands_info: *const GeneratedCommandsInfoEXT,
    ),
>;
pub type PFN_vkCmdPreprocessGeneratedCommandsEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_generated_commands_info: *const GeneratedCommandsInfoEXT,
        state_command_buffer: CommandBuffer,
    ),
>;
pub type PFN_vkGetGeneratedCommandsMemoryRequirementsEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const GeneratedCommandsMemoryRequirementsInfoEXT,
        p_memory_requirements: *mut MemoryRequirements2,
    ),
>;
pub type PFN_vkCreateIndirectCommandsLayoutEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const IndirectCommandsLayoutCreateInfoEXT,
        p_allocator: *const AllocationCallbacks,
        p_indirect_commands_layout: *mut IndirectCommandsLayoutEXT,
    ) -> Result,
>;
pub type PFN_vkDestroyIndirectCommandsLayoutEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        indirect_commands_layout: IndirectCommandsLayoutEXT,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkCreateIndirectExecutionSetEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const IndirectExecutionSetCreateInfoEXT,
        p_allocator: *const AllocationCallbacks,
        p_indirect_execution_set: *mut IndirectExecutionSetEXT,
    ) -> Result,
>;
pub type PFN_vkDestroyIndirectExecutionSetEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        indirect_execution_set: IndirectExecutionSetEXT,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkUpdateIndirectExecutionSetPipelineEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        indirect_execution_set: IndirectExecutionSetEXT,
        execution_set_write_count: u32,
        p_execution_set_writes: *const WriteIndirectExecutionSetPipelineEXT,
    ),
>;
pub type PFN_vkUpdateIndirectExecutionSetShaderEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        indirect_execution_set: IndirectExecutionSetEXT,
        execution_set_write_count: u32,
        p_execution_set_writes: *const WriteIndirectExecutionSetShaderEXT,
    ),
>;
pub type PFN_vkGetPhysicalDeviceFeatures2 = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_features: *mut PhysicalDeviceFeatures2,
    ),
>;
pub type PFN_vkGetPhysicalDeviceProperties2 = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_properties: *mut PhysicalDeviceProperties2,
    ),
>;
pub type PFN_vkGetPhysicalDeviceFormatProperties2 = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        format: Format,
        p_format_properties: *mut FormatProperties2,
    ),
>;
pub type PFN_vkGetPhysicalDeviceImageFormatProperties2 = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_image_format_info: *const PhysicalDeviceImageFormatInfo2,
        p_image_format_properties: *mut ImageFormatProperties2,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceQueueFamilyProperties2 = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_queue_family_property_count: *mut u32,
        p_queue_family_properties: *mut QueueFamilyProperties2,
    ),
>;
pub type PFN_vkGetPhysicalDeviceMemoryProperties2 = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_memory_properties: *mut PhysicalDeviceMemoryProperties2,
    ),
>;
pub type PFN_vkGetPhysicalDeviceSparseImageFormatProperties2 = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_format_info: *const PhysicalDeviceSparseImageFormatInfo2,
        p_property_count: *mut u32,
        p_properties: *mut SparseImageFormatProperties2,
    ),
>;
pub type PFN_vkCmdPushDescriptorSet = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        pipeline_bind_point: PipelineBindPoint,
        layout: PipelineLayout,
        set: u32,
        descriptor_write_count: u32,
        p_descriptor_writes: *const WriteDescriptorSet,
    ),
>;
pub type PFN_vkTrimCommandPool = Option<
    unsafe extern "system" fn(
        device: Device,
        command_pool: CommandPool,
        flags: CommandPoolTrimFlags,
    ),
>;
pub type PFN_vkGetPhysicalDeviceExternalBufferProperties = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_external_buffer_info: *const PhysicalDeviceExternalBufferInfo,
        p_external_buffer_properties: *mut ExternalBufferProperties,
    ),
>;
pub type PFN_vkGetMemoryWin32HandleKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_get_win32_handle_info: *const MemoryGetWin32HandleInfoKHR,
        p_handle: *mut isize,
    ) -> Result,
>;
pub type PFN_vkGetMemoryWin32HandlePropertiesKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        handle_type: ExternalMemoryHandleTypeFlagBits,
        handle: isize,
        p_memory_win32_handle_properties: *mut MemoryWin32HandlePropertiesKHR,
    ) -> Result,
>;
pub type PFN_vkGetMemoryFdKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_get_fd_info: *const MemoryGetFdInfoKHR,
        p_fd: *mut std::ffi::c_int,
    ) -> Result,
>;
pub type PFN_vkGetMemoryFdPropertiesKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        handle_type: ExternalMemoryHandleTypeFlagBits,
        fd: std::ffi::c_int,
        p_memory_fd_properties: *mut MemoryFdPropertiesKHR,
    ) -> Result,
>;
pub type PFN_vkGetMemoryZirconHandleFUCHSIA = Option<
    unsafe extern "system" fn(
        device: Device,
        p_get_zircon_handle_info: *const MemoryGetZirconHandleInfoFUCHSIA,
        p_zircon_handle: *mut u32,
    ) -> Result,
>;
pub type PFN_vkGetMemoryZirconHandlePropertiesFUCHSIA = Option<
    unsafe extern "system" fn(
        device: Device,
        handle_type: ExternalMemoryHandleTypeFlagBits,
        zircon_handle: u32,
        p_memory_zircon_handle_properties: *mut MemoryZirconHandlePropertiesFUCHSIA,
    ) -> Result,
>;
pub type PFN_vkGetMemoryRemoteAddressNV = Option<
    unsafe extern "system" fn(
        device: Device,
        p_memory_get_remote_address_info: *const MemoryGetRemoteAddressInfoNV,
        p_address: *mut *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkGetMemorySciBufNV = Option<
    unsafe extern "system" fn(
        device: Device,
        p_get_sci_buf_info: *const MemoryGetSciBufInfoNV,
        p_handle: *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceExternalMemorySciBufPropertiesNV = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        handle_type: ExternalMemoryHandleTypeFlagBits,
        handle: *const std::ffi::c_void,
        p_memory_sci_buf_properties: *mut MemorySciBufPropertiesNV,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceSciBufAttributesNV = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_attributes: *const std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceExternalSemaphoreProperties = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_external_semaphore_info: *const PhysicalDeviceExternalSemaphoreInfo,
        p_external_semaphore_properties: *mut ExternalSemaphoreProperties,
    ),
>;
pub type PFN_vkGetSemaphoreWin32HandleKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_get_win32_handle_info: *const SemaphoreGetWin32HandleInfoKHR,
        p_handle: *mut isize,
    ) -> Result,
>;
pub type PFN_vkImportSemaphoreWin32HandleKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_import_semaphore_win32_handle_info: *const ImportSemaphoreWin32HandleInfoKHR,
    ) -> Result,
>;
pub type PFN_vkGetSemaphoreFdKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_get_fd_info: *const SemaphoreGetFdInfoKHR,
        p_fd: *mut std::ffi::c_int,
    ) -> Result,
>;
pub type PFN_vkImportSemaphoreFdKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_import_semaphore_fd_info: *const ImportSemaphoreFdInfoKHR,
    ) -> Result,
>;
pub type PFN_vkGetSemaphoreZirconHandleFUCHSIA = Option<
    unsafe extern "system" fn(
        device: Device,
        p_get_zircon_handle_info: *const SemaphoreGetZirconHandleInfoFUCHSIA,
        p_zircon_handle: *mut u32,
    ) -> Result,
>;
pub type PFN_vkImportSemaphoreZirconHandleFUCHSIA = Option<
    unsafe extern "system" fn(
        device: Device,
        p_import_semaphore_zircon_handle_info: *const ImportSemaphoreZirconHandleInfoFUCHSIA,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceExternalFenceProperties = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_external_fence_info: *const PhysicalDeviceExternalFenceInfo,
        p_external_fence_properties: *mut ExternalFenceProperties,
    ),
>;
pub type PFN_vkGetFenceWin32HandleKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_get_win32_handle_info: *const FenceGetWin32HandleInfoKHR,
        p_handle: *mut isize,
    ) -> Result,
>;
pub type PFN_vkImportFenceWin32HandleKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_import_fence_win32_handle_info: *const ImportFenceWin32HandleInfoKHR,
    ) -> Result,
>;
pub type PFN_vkGetFenceFdKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_get_fd_info: *const FenceGetFdInfoKHR,
        p_fd: *mut std::ffi::c_int,
    ) -> Result,
>;
pub type PFN_vkImportFenceFdKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_import_fence_fd_info: *const ImportFenceFdInfoKHR,
    ) -> Result,
>;
pub type PFN_vkGetFenceSciSyncFenceNV = Option<
    unsafe extern "system" fn(
        device: Device,
        p_get_sci_sync_handle_info: *const FenceGetSciSyncInfoNV,
        p_handle: *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkGetFenceSciSyncObjNV = Option<
    unsafe extern "system" fn(
        device: Device,
        p_get_sci_sync_handle_info: *const FenceGetSciSyncInfoNV,
        p_handle: *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkImportFenceSciSyncFenceNV = Option<
    unsafe extern "system" fn(
        device: Device,
        p_import_fence_sci_sync_info: *const ImportFenceSciSyncInfoNV,
    ) -> Result,
>;
pub type PFN_vkImportFenceSciSyncObjNV = Option<
    unsafe extern "system" fn(
        device: Device,
        p_import_fence_sci_sync_info: *const ImportFenceSciSyncInfoNV,
    ) -> Result,
>;
pub type PFN_vkGetSemaphoreSciSyncObjNV = Option<
    unsafe extern "system" fn(
        device: Device,
        p_get_sci_sync_info: *const SemaphoreGetSciSyncInfoNV,
        p_handle: *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkImportSemaphoreSciSyncObjNV = Option<
    unsafe extern "system" fn(
        device: Device,
        p_import_semaphore_sci_sync_info: *const ImportSemaphoreSciSyncInfoNV,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceSciSyncAttributesNV = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_sci_sync_attributes_info: *const SciSyncAttributesInfoNV,
        p_attributes: *const std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkCreateSemaphoreSciSyncPoolNV = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const SemaphoreSciSyncPoolCreateInfoNV,
        p_allocator: *const AllocationCallbacks,
        p_semaphore_pool: *mut SemaphoreSciSyncPoolNV,
    ) -> Result,
>;
pub type PFN_vkDestroySemaphoreSciSyncPoolNV = Option<
    unsafe extern "system" fn(
        device: Device,
        semaphore_pool: SemaphoreSciSyncPoolNV,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkReleaseDisplayEXT = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        display: DisplayKHR,
    ) -> Result,
>;
pub type PFN_vkAcquireXlibDisplayEXT = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        dpy: *mut std::ffi::c_void,
        display: DisplayKHR,
    ) -> Result,
>;
pub type PFN_vkGetRandROutputDisplayEXT = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        dpy: *mut std::ffi::c_void,
        rr_output: std::ffi::c_ulong,
        p_display: *mut DisplayKHR,
    ) -> Result,
>;
pub type PFN_vkAcquireWinrtDisplayNV = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        display: DisplayKHR,
    ) -> Result,
>;
pub type PFN_vkGetWinrtDisplayNV = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        device_relative_id: u32,
        p_display: *mut DisplayKHR,
    ) -> Result,
>;
pub type PFN_vkDisplayPowerControlEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        display: DisplayKHR,
        p_display_power_info: *const DisplayPowerInfoEXT,
    ) -> Result,
>;
pub type PFN_vkRegisterDeviceEventEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        p_device_event_info: *const DeviceEventInfoEXT,
        p_allocator: *const AllocationCallbacks,
        p_fence: *mut Fence,
    ) -> Result,
>;
pub type PFN_vkRegisterDisplayEventEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        display: DisplayKHR,
        p_display_event_info: *const DisplayEventInfoEXT,
        p_allocator: *const AllocationCallbacks,
        p_fence: *mut Fence,
    ) -> Result,
>;
pub type PFN_vkGetSwapchainCounterEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        swapchain: SwapchainKHR,
        counter: SurfaceCounterFlagBitsEXT,
        p_counter_value: *mut u64,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceSurfaceCapabilities2EXT = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        surface: SurfaceKHR,
        p_surface_capabilities: *mut SurfaceCapabilities2EXT,
    ) -> Result,
>;
pub type PFN_vkEnumeratePhysicalDeviceGroups = Option<
    unsafe extern "system" fn(
        instance: Instance,
        p_physical_device_group_count: *mut u32,
        p_physical_device_group_properties: *mut PhysicalDeviceGroupProperties,
    ) -> Result,
>;
pub type PFN_vkGetDeviceGroupPeerMemoryFeatures = Option<
    unsafe extern "system" fn(
        device: Device,
        heap_index: u32,
        local_device_index: u32,
        remote_device_index: u32,
        p_peer_memory_features: *mut PeerMemoryFeatureFlags,
    ),
>;
pub type PFN_vkBindBufferMemory2 = Option<
    unsafe extern "system" fn(
        device: Device,
        bind_info_count: u32,
        p_bind_infos: *const BindBufferMemoryInfo,
    ) -> Result,
>;
pub type PFN_vkBindImageMemory2 = Option<
    unsafe extern "system" fn(
        device: Device,
        bind_info_count: u32,
        p_bind_infos: *const BindImageMemoryInfo,
    ) -> Result,
>;
pub type PFN_vkCmdSetDeviceMask = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer, device_mask: u32),
>;
pub type PFN_vkGetDeviceGroupPresentCapabilitiesKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_device_group_present_capabilities: *mut DeviceGroupPresentCapabilitiesKHR,
    ) -> Result,
>;
pub type PFN_vkGetDeviceGroupSurfacePresentModesKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        surface: SurfaceKHR,
        p_modes: *mut DeviceGroupPresentModeFlagsKHR,
    ) -> Result,
>;
pub type PFN_vkAcquireNextImage2KHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_acquire_info: *const AcquireNextImageInfoKHR,
        p_image_index: *mut u32,
    ) -> Result,
>;
pub type PFN_vkCmdDispatchBase = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        base_group_x: u32,
        base_group_y: u32,
        base_group_z: u32,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ),
>;
pub type PFN_vkGetPhysicalDevicePresentRectanglesKHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        surface: SurfaceKHR,
        p_rect_count: *mut u32,
        p_rects: *mut Rect2D,
    ) -> Result,
>;
pub type PFN_vkCreateDescriptorUpdateTemplate = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const DescriptorUpdateTemplateCreateInfo,
        p_allocator: *const AllocationCallbacks,
        p_descriptor_update_template: *mut DescriptorUpdateTemplate,
    ) -> Result,
>;
pub type PFN_vkDestroyDescriptorUpdateTemplate = Option<
    unsafe extern "system" fn(
        device: Device,
        descriptor_update_template: DescriptorUpdateTemplate,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkUpdateDescriptorSetWithTemplate = Option<
    unsafe extern "system" fn(
        device: Device,
        descriptor_set: DescriptorSet,
        descriptor_update_template: DescriptorUpdateTemplate,
        p_data: *const std::ffi::c_void,
    ),
>;
pub type PFN_vkCmdPushDescriptorSetWithTemplate = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        descriptor_update_template: DescriptorUpdateTemplate,
        layout: PipelineLayout,
        set: u32,
        p_data: *const std::ffi::c_void,
    ),
>;
pub type PFN_vkSetHdrMetadataEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        swapchain_count: u32,
        p_swapchains: *const SwapchainKHR,
        p_metadata: *const HdrMetadataEXT,
    ),
>;
pub type PFN_vkGetSwapchainStatusKHR = Option<
    unsafe extern "system" fn(device: Device, swapchain: SwapchainKHR) -> Result,
>;
pub type PFN_vkGetRefreshCycleDurationGOOGLE = Option<
    unsafe extern "system" fn(
        device: Device,
        swapchain: SwapchainKHR,
        p_display_timing_properties: *mut RefreshCycleDurationGOOGLE,
    ) -> Result,
>;
pub type PFN_vkGetPastPresentationTimingGOOGLE = Option<
    unsafe extern "system" fn(
        device: Device,
        swapchain: SwapchainKHR,
        p_presentation_timing_count: *mut u32,
        p_presentation_timings: *mut PastPresentationTimingGOOGLE,
    ) -> Result,
>;
pub type PFN_vkCreateIOSSurfaceMVK = Option<
    unsafe extern "system" fn(
        instance: Instance,
        p_create_info: *const IOSSurfaceCreateInfoMVK,
        p_allocator: *const AllocationCallbacks,
        p_surface: *mut SurfaceKHR,
    ) -> Result,
>;
pub type PFN_vkCreateMacOSSurfaceMVK = Option<
    unsafe extern "system" fn(
        instance: Instance,
        p_create_info: *const MacOSSurfaceCreateInfoMVK,
        p_allocator: *const AllocationCallbacks,
        p_surface: *mut SurfaceKHR,
    ) -> Result,
>;
pub type PFN_vkCreateMetalSurfaceEXT = Option<
    unsafe extern "system" fn(
        instance: Instance,
        p_create_info: *const MetalSurfaceCreateInfoEXT,
        p_allocator: *const AllocationCallbacks,
        p_surface: *mut SurfaceKHR,
    ) -> Result,
>;
pub type PFN_vkCmdSetViewportWScalingNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        first_viewport: u32,
        viewport_count: u32,
        p_viewport_w_scalings: *const ViewportWScalingNV,
    ),
>;
pub type PFN_vkCmdSetDiscardRectangleEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        first_discard_rectangle: u32,
        discard_rectangle_count: u32,
        p_discard_rectangles: *const Rect2D,
    ),
>;
pub type PFN_vkCmdSetDiscardRectangleEnableEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        discard_rectangle_enable: u32,
    ),
>;
pub type PFN_vkCmdSetDiscardRectangleModeEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        discard_rectangle_mode: DiscardRectangleModeEXT,
    ),
>;
pub type PFN_vkCmdSetSampleLocationsEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_sample_locations_info: *const SampleLocationsInfoEXT,
    ),
>;
pub type PFN_vkGetPhysicalDeviceMultisamplePropertiesEXT = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        samples: SampleCountFlagBits,
        p_multisample_properties: *mut MultisamplePropertiesEXT,
    ),
>;
pub type PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_surface_info: *const PhysicalDeviceSurfaceInfo2KHR,
        p_surface_capabilities: *mut SurfaceCapabilities2KHR,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceSurfaceFormats2KHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_surface_info: *const PhysicalDeviceSurfaceInfo2KHR,
        p_surface_format_count: *mut u32,
        p_surface_formats: *mut SurfaceFormat2KHR,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceDisplayProperties2KHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_property_count: *mut u32,
        p_properties: *mut DisplayProperties2KHR,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceDisplayPlaneProperties2KHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_property_count: *mut u32,
        p_properties: *mut DisplayPlaneProperties2KHR,
    ) -> Result,
>;
pub type PFN_vkGetDisplayModeProperties2KHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        display: DisplayKHR,
        p_property_count: *mut u32,
        p_properties: *mut DisplayModeProperties2KHR,
    ) -> Result,
>;
pub type PFN_vkGetDisplayPlaneCapabilities2KHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_display_plane_info: *const DisplayPlaneInfo2KHR,
        p_capabilities: *mut DisplayPlaneCapabilities2KHR,
    ) -> Result,
>;
pub type PFN_vkGetBufferMemoryRequirements2 = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const BufferMemoryRequirementsInfo2,
        p_memory_requirements: *mut MemoryRequirements2,
    ),
>;
pub type PFN_vkGetImageMemoryRequirements2 = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const ImageMemoryRequirementsInfo2,
        p_memory_requirements: *mut MemoryRequirements2,
    ),
>;
pub type PFN_vkGetImageSparseMemoryRequirements2 = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const ImageSparseMemoryRequirementsInfo2,
        p_sparse_memory_requirement_count: *mut u32,
        p_sparse_memory_requirements: *mut SparseImageMemoryRequirements2,
    ),
>;
pub type PFN_vkGetDeviceBufferMemoryRequirements = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const DeviceBufferMemoryRequirements,
        p_memory_requirements: *mut MemoryRequirements2,
    ),
>;
pub type PFN_vkGetDeviceImageMemoryRequirements = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const DeviceImageMemoryRequirements,
        p_memory_requirements: *mut MemoryRequirements2,
    ),
>;
pub type PFN_vkGetDeviceImageSparseMemoryRequirements = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const DeviceImageMemoryRequirements,
        p_sparse_memory_requirement_count: *mut u32,
        p_sparse_memory_requirements: *mut SparseImageMemoryRequirements2,
    ),
>;
pub type PFN_vkCreateSamplerYcbcrConversion = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const SamplerYcbcrConversionCreateInfo,
        p_allocator: *const AllocationCallbacks,
        p_ycbcr_conversion: *mut SamplerYcbcrConversion,
    ) -> Result,
>;
pub type PFN_vkDestroySamplerYcbcrConversion = Option<
    unsafe extern "system" fn(
        device: Device,
        ycbcr_conversion: SamplerYcbcrConversion,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkGetDeviceQueue2 = Option<
    unsafe extern "system" fn(
        device: Device,
        p_queue_info: *const DeviceQueueInfo2,
        p_queue: *mut Queue,
    ),
>;
pub type PFN_vkCreateValidationCacheEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const ValidationCacheCreateInfoEXT,
        p_allocator: *const AllocationCallbacks,
        p_validation_cache: *mut ValidationCacheEXT,
    ) -> Result,
>;
pub type PFN_vkDestroyValidationCacheEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        validation_cache: ValidationCacheEXT,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkGetValidationCacheDataEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        validation_cache: ValidationCacheEXT,
        p_data_size: *mut usize,
        p_data: *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkMergeValidationCachesEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        dst_cache: ValidationCacheEXT,
        src_cache_count: u32,
        p_src_caches: *const ValidationCacheEXT,
    ) -> Result,
>;
pub type PFN_vkGetDescriptorSetLayoutSupport = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const DescriptorSetLayoutCreateInfo,
        p_support: *mut DescriptorSetLayoutSupport,
    ),
>;
pub type PFN_vkGetSwapchainGrallocUsageANDROID = Option<
    unsafe extern "system" fn(
        device: Device,
        format: Format,
        image_usage: ImageUsageFlags,
        gralloc_usage: *mut std::ffi::c_int,
    ) -> Result,
>;
pub type PFN_vkGetSwapchainGrallocUsage2ANDROID = Option<
    unsafe extern "system" fn(
        device: Device,
        format: Format,
        image_usage: ImageUsageFlags,
        swapchain_image_usage: SwapchainImageUsageFlagsANDROID,
        gralloc_consumer_usage: *mut u64,
        gralloc_producer_usage: *mut u64,
    ) -> Result,
>;
pub type PFN_vkAcquireImageANDROID = Option<
    unsafe extern "system" fn(
        device: Device,
        image: Image,
        native_fence_fd: std::ffi::c_int,
        semaphore: Semaphore,
        fence: Fence,
    ) -> Result,
>;
pub type PFN_vkQueueSignalReleaseImageANDROID = Option<
    unsafe extern "system" fn(
        queue: Queue,
        wait_semaphore_count: u32,
        p_wait_semaphores: *const Semaphore,
        image: Image,
        p_native_fence_fd: *mut std::ffi::c_int,
    ) -> Result,
>;
pub type PFN_vkGetShaderInfoAMD = Option<
    unsafe extern "system" fn(
        device: Device,
        pipeline: Pipeline,
        shader_stage: ShaderStageFlagBits,
        info_type: ShaderInfoTypeAMD,
        p_info_size: *mut usize,
        p_info: *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkSetLocalDimmingAMD = Option<
    unsafe extern "system" fn(
        device: Device,
        swap_chain: SwapchainKHR,
        local_dimming_enable: u32,
    ),
>;
pub type PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_time_domain_count: *mut u32,
        p_time_domains: *mut TimeDomainKHR,
    ) -> Result,
>;
pub type PFN_vkGetCalibratedTimestampsKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        timestamp_count: u32,
        p_timestamp_infos: *const CalibratedTimestampInfoKHR,
        p_timestamps: *mut u64,
        p_max_deviation: *mut u64,
    ) -> Result,
>;
pub type PFN_vkSetDebugUtilsObjectNameEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        p_name_info: *const DebugUtilsObjectNameInfoEXT,
    ) -> Result,
>;
pub type PFN_vkSetDebugUtilsObjectTagEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        p_tag_info: *const DebugUtilsObjectTagInfoEXT,
    ) -> Result,
>;
pub type PFN_vkQueueBeginDebugUtilsLabelEXT = Option<
    unsafe extern "system" fn(queue: Queue, p_label_info: *const DebugUtilsLabelEXT),
>;
pub type PFN_vkQueueEndDebugUtilsLabelEXT = Option<
    unsafe extern "system" fn(queue: Queue),
>;
pub type PFN_vkQueueInsertDebugUtilsLabelEXT = Option<
    unsafe extern "system" fn(queue: Queue, p_label_info: *const DebugUtilsLabelEXT),
>;
pub type PFN_vkCmdBeginDebugUtilsLabelEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_label_info: *const DebugUtilsLabelEXT,
    ),
>;
pub type PFN_vkCmdEndDebugUtilsLabelEXT = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer),
>;
pub type PFN_vkCmdInsertDebugUtilsLabelEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_label_info: *const DebugUtilsLabelEXT,
    ),
>;
pub type PFN_vkCreateDebugUtilsMessengerEXT = Option<
    unsafe extern "system" fn(
        instance: Instance,
        p_create_info: *const DebugUtilsMessengerCreateInfoEXT,
        p_allocator: *const AllocationCallbacks,
        p_messenger: *mut DebugUtilsMessengerEXT,
    ) -> Result,
>;
pub type PFN_vkDestroyDebugUtilsMessengerEXT = Option<
    unsafe extern "system" fn(
        instance: Instance,
        messenger: DebugUtilsMessengerEXT,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkSubmitDebugUtilsMessageEXT = Option<
    unsafe extern "system" fn(
        instance: Instance,
        message_severity: DebugUtilsMessageSeverityFlagBitsEXT,
        message_types: DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const DebugUtilsMessengerCallbackDataEXT,
    ),
>;
pub type PFN_vkGetMemoryHostPointerPropertiesEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        handle_type: ExternalMemoryHandleTypeFlagBits,
        p_host_pointer: *const std::ffi::c_void,
        p_memory_host_pointer_properties: *mut MemoryHostPointerPropertiesEXT,
    ) -> Result,
>;
pub type PFN_vkCmdWriteBufferMarkerAMD = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        pipeline_stage: PipelineStageFlagBits,
        dst_buffer: Buffer,
        dst_offset: u64,
        marker: u32,
    ),
>;
pub type PFN_vkCreateRenderPass2 = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const RenderPassCreateInfo2,
        p_allocator: *const AllocationCallbacks,
        p_render_pass: *mut RenderPass,
    ) -> Result,
>;
pub type PFN_vkCmdBeginRenderPass2 = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_render_pass_begin: *const RenderPassBeginInfo,
        p_subpass_begin_info: *const SubpassBeginInfo,
    ),
>;
pub type PFN_vkCmdNextSubpass2 = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_subpass_begin_info: *const SubpassBeginInfo,
        p_subpass_end_info: *const SubpassEndInfo,
    ),
>;
pub type PFN_vkCmdEndRenderPass2 = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_subpass_end_info: *const SubpassEndInfo,
    ),
>;
pub type PFN_vkGetSemaphoreCounterValue = Option<
    unsafe extern "system" fn(
        device: Device,
        semaphore: Semaphore,
        p_value: *mut u64,
    ) -> Result,
>;
pub type PFN_vkWaitSemaphores = Option<
    unsafe extern "system" fn(
        device: Device,
        p_wait_info: *const SemaphoreWaitInfo,
        timeout: u64,
    ) -> Result,
>;
pub type PFN_vkSignalSemaphore = Option<
    unsafe extern "system" fn(
        device: Device,
        p_signal_info: *const SemaphoreSignalInfo,
    ) -> Result,
>;
pub type PFN_vkGetAndroidHardwareBufferPropertiesANDROID = Option<
    unsafe extern "system" fn(
        device: Device,
        buffer: *const std::ffi::c_void,
        p_properties: *mut AndroidHardwareBufferPropertiesANDROID,
    ) -> Result,
>;
pub type PFN_vkGetMemoryAndroidHardwareBufferANDROID = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const MemoryGetAndroidHardwareBufferInfoANDROID,
        p_buffer: *mut *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkCmdDrawIndirectCount = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        buffer: Buffer,
        offset: u64,
        count_buffer: Buffer,
        count_buffer_offset: u64,
        max_draw_count: u32,
        stride: u32,
    ),
>;
pub type PFN_vkCmdDrawIndexedIndirectCount = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        buffer: Buffer,
        offset: u64,
        count_buffer: Buffer,
        count_buffer_offset: u64,
        max_draw_count: u32,
        stride: u32,
    ),
>;
pub type PFN_vkCmdSetCheckpointNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_checkpoint_marker: *const std::ffi::c_void,
    ),
>;
pub type PFN_vkGetQueueCheckpointDataNV = Option<
    unsafe extern "system" fn(
        queue: Queue,
        p_checkpoint_data_count: *mut u32,
        p_checkpoint_data: *mut CheckpointDataNV,
    ),
>;
pub type PFN_vkCmdBindTransformFeedbackBuffersEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        first_binding: u32,
        binding_count: u32,
        p_buffers: *const Buffer,
        p_offsets: *const u64,
        p_sizes: *const u64,
    ),
>;
pub type PFN_vkCmdBeginTransformFeedbackEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        first_counter_buffer: u32,
        counter_buffer_count: u32,
        p_counter_buffers: *const Buffer,
        p_counter_buffer_offsets: *const u64,
    ),
>;
pub type PFN_vkCmdEndTransformFeedbackEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        first_counter_buffer: u32,
        counter_buffer_count: u32,
        p_counter_buffers: *const Buffer,
        p_counter_buffer_offsets: *const u64,
    ),
>;
pub type PFN_vkCmdBeginQueryIndexedEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        query_pool: QueryPool,
        query: u32,
        flags: QueryControlFlags,
        index: u32,
    ),
>;
pub type PFN_vkCmdEndQueryIndexedEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        query_pool: QueryPool,
        query: u32,
        index: u32,
    ),
>;
pub type PFN_vkCmdDrawIndirectByteCountEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        instance_count: u32,
        first_instance: u32,
        counter_buffer: Buffer,
        counter_buffer_offset: u64,
        counter_offset: u32,
        vertex_stride: u32,
    ),
>;
pub type PFN_vkCmdSetExclusiveScissorNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        first_exclusive_scissor: u32,
        exclusive_scissor_count: u32,
        p_exclusive_scissors: *const Rect2D,
    ),
>;
pub type PFN_vkCmdSetExclusiveScissorEnableNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        first_exclusive_scissor: u32,
        exclusive_scissor_count: u32,
        p_exclusive_scissor_enables: *const u32,
    ),
>;
pub type PFN_vkCmdBindShadingRateImageNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        image_view: ImageView,
        image_layout: ImageLayout,
    ),
>;
pub type PFN_vkCmdSetViewportShadingRatePaletteNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        first_viewport: u32,
        viewport_count: u32,
        p_shading_rate_palettes: *const ShadingRatePaletteNV,
    ),
>;
pub type PFN_vkCmdSetCoarseSampleOrderNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        sample_order_type: CoarseSampleOrderTypeNV,
        custom_sample_order_count: u32,
        p_custom_sample_orders: *const CoarseSampleOrderCustomNV,
    ),
>;
pub type PFN_vkCmdDrawMeshTasksNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        task_count: u32,
        first_task: u32,
    ),
>;
pub type PFN_vkCmdDrawMeshTasksIndirectNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        buffer: Buffer,
        offset: u64,
        draw_count: u32,
        stride: u32,
    ),
>;
pub type PFN_vkCmdDrawMeshTasksIndirectCountNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        buffer: Buffer,
        offset: u64,
        count_buffer: Buffer,
        count_buffer_offset: u64,
        max_draw_count: u32,
        stride: u32,
    ),
>;
pub type PFN_vkCmdDrawMeshTasksEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ),
>;
pub type PFN_vkCmdDrawMeshTasksIndirectEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        buffer: Buffer,
        offset: u64,
        draw_count: u32,
        stride: u32,
    ),
>;
pub type PFN_vkCmdDrawMeshTasksIndirectCountEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        buffer: Buffer,
        offset: u64,
        count_buffer: Buffer,
        count_buffer_offset: u64,
        max_draw_count: u32,
        stride: u32,
    ),
>;
pub type PFN_vkCompileDeferredNV = Option<
    unsafe extern "system" fn(device: Device, pipeline: Pipeline, shader: u32) -> Result,
>;
pub type PFN_vkCreateAccelerationStructureNV = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const AccelerationStructureCreateInfoNV,
        p_allocator: *const AllocationCallbacks,
        p_acceleration_structure: *mut AccelerationStructureNV,
    ) -> Result,
>;
pub type PFN_vkCmdBindInvocationMaskHUAWEI = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        image_view: ImageView,
        image_layout: ImageLayout,
    ),
>;
pub type PFN_vkDestroyAccelerationStructureKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        acceleration_structure: AccelerationStructureKHR,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkDestroyAccelerationStructureNV = Option<
    unsafe extern "system" fn(
        device: Device,
        acceleration_structure: AccelerationStructureNV,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkGetAccelerationStructureMemoryRequirementsNV = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const AccelerationStructureMemoryRequirementsInfoNV,
        p_memory_requirements: *mut MemoryRequirements2KHR,
    ),
>;
pub type PFN_vkBindAccelerationStructureMemoryNV = Option<
    unsafe extern "system" fn(
        device: Device,
        bind_info_count: u32,
        p_bind_infos: *const BindAccelerationStructureMemoryInfoNV,
    ) -> Result,
>;
pub type PFN_vkCmdCopyAccelerationStructureNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        dst: AccelerationStructureNV,
        src: AccelerationStructureNV,
        mode: CopyAccelerationStructureModeKHR,
    ),
>;
pub type PFN_vkCmdCopyAccelerationStructureKHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_info: *const CopyAccelerationStructureInfoKHR,
    ),
>;
pub type PFN_vkCopyAccelerationStructureKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        deferred_operation: DeferredOperationKHR,
        p_info: *const CopyAccelerationStructureInfoKHR,
    ) -> Result,
>;
pub type PFN_vkCmdCopyAccelerationStructureToMemoryKHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_info: *const CopyAccelerationStructureToMemoryInfoKHR,
    ),
>;
pub type PFN_vkCopyAccelerationStructureToMemoryKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        deferred_operation: DeferredOperationKHR,
        p_info: *const CopyAccelerationStructureToMemoryInfoKHR,
    ) -> Result,
>;
pub type PFN_vkCmdCopyMemoryToAccelerationStructureKHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_info: *const CopyMemoryToAccelerationStructureInfoKHR,
    ),
>;
pub type PFN_vkCopyMemoryToAccelerationStructureKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        deferred_operation: DeferredOperationKHR,
        p_info: *const CopyMemoryToAccelerationStructureInfoKHR,
    ) -> Result,
>;
pub type PFN_vkCmdWriteAccelerationStructuresPropertiesKHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        acceleration_structure_count: u32,
        p_acceleration_structures: *const AccelerationStructureKHR,
        query_type: QueryType,
        query_pool: QueryPool,
        first_query: u32,
    ),
>;
pub type PFN_vkCmdWriteAccelerationStructuresPropertiesNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        acceleration_structure_count: u32,
        p_acceleration_structures: *const AccelerationStructureNV,
        query_type: QueryType,
        query_pool: QueryPool,
        first_query: u32,
    ),
>;
pub type PFN_vkCmdBuildAccelerationStructureNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_info: *const AccelerationStructureInfoNV,
        instance_data: Buffer,
        instance_offset: u64,
        update: u32,
        dst: AccelerationStructureNV,
        src: AccelerationStructureNV,
        scratch: Buffer,
        scratch_offset: u64,
    ),
>;
pub type PFN_vkWriteAccelerationStructuresPropertiesKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        acceleration_structure_count: u32,
        p_acceleration_structures: *const AccelerationStructureKHR,
        query_type: QueryType,
        data_size: usize,
        p_data: *mut std::ffi::c_void,
        stride: usize,
    ) -> Result,
>;
pub type PFN_vkCmdTraceRaysKHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_raygen_shader_binding_table: *const StridedDeviceAddressRegionKHR,
        p_miss_shader_binding_table: *const StridedDeviceAddressRegionKHR,
        p_hit_shader_binding_table: *const StridedDeviceAddressRegionKHR,
        p_callable_shader_binding_table: *const StridedDeviceAddressRegionKHR,
        width: u32,
        height: u32,
        depth: u32,
    ),
>;
pub type PFN_vkCmdTraceRaysNV = Option<
    unsafe extern "system" fn(
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
    ),
>;
pub type PFN_vkGetRayTracingShaderGroupHandlesKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        pipeline: Pipeline,
        first_group: u32,
        group_count: u32,
        data_size: usize,
        p_data: *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        pipeline: Pipeline,
        first_group: u32,
        group_count: u32,
        data_size: usize,
        p_data: *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkGetAccelerationStructureHandleNV = Option<
    unsafe extern "system" fn(
        device: Device,
        acceleration_structure: AccelerationStructureNV,
        data_size: usize,
        p_data: *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkCreateRayTracingPipelinesNV = Option<
    unsafe extern "system" fn(
        device: Device,
        pipeline_cache: PipelineCache,
        create_info_count: u32,
        p_create_infos: *const RayTracingPipelineCreateInfoNV,
        p_allocator: *const AllocationCallbacks,
        p_pipelines: *mut Pipeline,
    ) -> Result,
>;
pub type PFN_vkCreateRayTracingPipelinesKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        deferred_operation: DeferredOperationKHR,
        pipeline_cache: PipelineCache,
        create_info_count: u32,
        p_create_infos: *const RayTracingPipelineCreateInfoKHR,
        p_allocator: *const AllocationCallbacks,
        p_pipelines: *mut Pipeline,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesNV = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_property_count: *mut u32,
        p_properties: *mut CooperativeMatrixPropertiesNV,
    ) -> Result,
>;
pub type PFN_vkCmdTraceRaysIndirectKHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_raygen_shader_binding_table: *const StridedDeviceAddressRegionKHR,
        p_miss_shader_binding_table: *const StridedDeviceAddressRegionKHR,
        p_hit_shader_binding_table: *const StridedDeviceAddressRegionKHR,
        p_callable_shader_binding_table: *const StridedDeviceAddressRegionKHR,
        indirect_device_address: u64,
    ),
>;
pub type PFN_vkCmdTraceRaysIndirect2KHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        indirect_device_address: u64,
    ),
>;
pub type PFN_vkGetClusterAccelerationStructureBuildSizesNV = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const ClusterAccelerationStructureInputInfoNV,
        p_size_info: *mut AccelerationStructureBuildSizesInfoKHR,
    ),
>;
pub type PFN_vkCmdBuildClusterAccelerationStructureIndirectNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_command_infos: *const ClusterAccelerationStructureCommandsInfoNV,
    ),
>;
pub type PFN_vkGetDeviceAccelerationStructureCompatibilityKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_version_info: *const AccelerationStructureVersionInfoKHR,
        p_compatibility: *mut AccelerationStructureCompatibilityKHR,
    ),
>;
pub type PFN_vkGetRayTracingShaderGroupStackSizeKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        pipeline: Pipeline,
        group: u32,
        group_shader: ShaderGroupShaderKHR,
    ) -> u64,
>;
pub type PFN_vkCmdSetRayTracingPipelineStackSizeKHR = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer, pipeline_stack_size: u32),
>;
pub type PFN_vkGetImageViewHandleNVX = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const ImageViewHandleInfoNVX,
    ) -> u32,
>;
pub type PFN_vkGetImageViewHandle64NVX = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const ImageViewHandleInfoNVX,
    ) -> u64,
>;
pub type PFN_vkGetImageViewAddressNVX = Option<
    unsafe extern "system" fn(
        device: Device,
        image_view: ImageView,
        p_properties: *mut ImageViewAddressPropertiesNVX,
    ) -> Result,
>;
pub type PFN_vkGetDeviceCombinedImageSamplerIndexNVX = Option<
    unsafe extern "system" fn(
        device: Device,
        image_view_index: u64,
        sampler_index: u64,
    ) -> u64,
>;
pub type PFN_vkGetPhysicalDeviceSurfacePresentModes2EXT = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_surface_info: *const PhysicalDeviceSurfaceInfo2KHR,
        p_present_mode_count: *mut u32,
        p_present_modes: *mut PresentModeKHR,
    ) -> Result,
>;
pub type PFN_vkGetDeviceGroupSurfacePresentModes2EXT = Option<
    unsafe extern "system" fn(
        device: Device,
        p_surface_info: *const PhysicalDeviceSurfaceInfo2KHR,
        p_modes: *mut DeviceGroupPresentModeFlagsKHR,
    ) -> Result,
>;
pub type PFN_vkAcquireFullScreenExclusiveModeEXT = Option<
    unsafe extern "system" fn(device: Device, swapchain: SwapchainKHR) -> Result,
>;
pub type PFN_vkReleaseFullScreenExclusiveModeEXT = Option<
    unsafe extern "system" fn(device: Device, swapchain: SwapchainKHR) -> Result,
>;
pub type PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        queue_family_index: u32,
        p_counter_count: *mut u32,
        p_counters: *mut PerformanceCounterKHR,
        p_counter_descriptions: *mut PerformanceCounterDescriptionKHR,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_performance_query_create_info: *const QueryPoolPerformanceCreateInfoKHR,
        p_num_passes: *mut u32,
    ),
>;
pub type PFN_vkAcquireProfilingLockKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const AcquireProfilingLockInfoKHR,
    ) -> Result,
>;
pub type PFN_vkReleaseProfilingLockKHR = Option<
    unsafe extern "system" fn(device: Device),
>;
pub type PFN_vkGetImageDrmFormatModifierPropertiesEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        image: Image,
        p_properties: *mut ImageDrmFormatModifierPropertiesEXT,
    ) -> Result,
>;
pub type PFN_vkGetBufferOpaqueCaptureAddress = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const BufferDeviceAddressInfo,
    ) -> u64,
>;
pub type PFN_vkGetBufferDeviceAddress = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const BufferDeviceAddressInfo,
    ) -> u64,
>;
pub type PFN_vkCreateHeadlessSurfaceEXT = Option<
    unsafe extern "system" fn(
        instance: Instance,
        p_create_info: *const HeadlessSurfaceCreateInfoEXT,
        p_allocator: *const AllocationCallbacks,
        p_surface: *mut SurfaceKHR,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_combination_count: *mut u32,
        p_combinations: *mut FramebufferMixedSamplesCombinationNV,
    ) -> Result,
>;
pub type PFN_vkInitializePerformanceApiINTEL = Option<
    unsafe extern "system" fn(
        device: Device,
        p_initialize_info: *const InitializePerformanceApiInfoINTEL,
    ) -> Result,
>;
pub type PFN_vkUninitializePerformanceApiINTEL = Option<
    unsafe extern "system" fn(device: Device),
>;
pub type PFN_vkCmdSetPerformanceMarkerINTEL = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_marker_info: *const PerformanceMarkerInfoINTEL,
    ) -> Result,
>;
pub type PFN_vkCmdSetPerformanceStreamMarkerINTEL = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_marker_info: *const PerformanceStreamMarkerInfoINTEL,
    ) -> Result,
>;
pub type PFN_vkCmdSetPerformanceOverrideINTEL = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_override_info: *const PerformanceOverrideInfoINTEL,
    ) -> Result,
>;
pub type PFN_vkAcquirePerformanceConfigurationINTEL = Option<
    unsafe extern "system" fn(
        device: Device,
        p_acquire_info: *const PerformanceConfigurationAcquireInfoINTEL,
        p_configuration: *mut PerformanceConfigurationINTEL,
    ) -> Result,
>;
pub type PFN_vkReleasePerformanceConfigurationINTEL = Option<
    unsafe extern "system" fn(
        device: Device,
        configuration: PerformanceConfigurationINTEL,
    ) -> Result,
>;
pub type PFN_vkQueueSetPerformanceConfigurationINTEL = Option<
    unsafe extern "system" fn(
        queue: Queue,
        configuration: PerformanceConfigurationINTEL,
    ) -> Result,
>;
pub type PFN_vkGetPerformanceParameterINTEL = Option<
    unsafe extern "system" fn(
        device: Device,
        parameter: PerformanceParameterTypeINTEL,
        p_value: *mut PerformanceValueINTEL,
    ) -> Result,
>;
pub type PFN_vkGetDeviceMemoryOpaqueCaptureAddress = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const DeviceMemoryOpaqueCaptureAddressInfo,
    ) -> u64,
>;
pub type PFN_vkGetPipelineExecutablePropertiesKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_pipeline_info: *const PipelineInfoKHR,
        p_executable_count: *mut u32,
        p_properties: *mut PipelineExecutablePropertiesKHR,
    ) -> Result,
>;
pub type PFN_vkGetPipelineExecutableStatisticsKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_executable_info: *const PipelineExecutableInfoKHR,
        p_statistic_count: *mut u32,
        p_statistics: *mut PipelineExecutableStatisticKHR,
    ) -> Result,
>;
pub type PFN_vkGetPipelineExecutableInternalRepresentationsKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_executable_info: *const PipelineExecutableInfoKHR,
        p_internal_representation_count: *mut u32,
        p_internal_representations: *mut PipelineExecutableInternalRepresentationKHR,
    ) -> Result,
>;
pub type PFN_vkCmdSetLineStipple = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        line_stipple_factor: u32,
        line_stipple_pattern: u16,
    ),
>;
pub type PFN_vkGetFaultData = Option<
    unsafe extern "system" fn(
        device: Device,
        fault_query_behavior: FaultQueryBehavior,
        p_unrecorded_faults: *mut u32,
        p_fault_count: *mut u32,
        p_faults: *mut FaultData,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceToolProperties = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_tool_count: *mut u32,
        p_tool_properties: *mut PhysicalDeviceToolProperties,
    ) -> Result,
>;
pub type PFN_vkCreateAccelerationStructureKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const AccelerationStructureCreateInfoKHR,
        p_allocator: *const AllocationCallbacks,
        p_acceleration_structure: *mut AccelerationStructureKHR,
    ) -> Result,
>;
pub type PFN_vkCmdBuildAccelerationStructuresKHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        info_count: u32,
        p_infos: *const AccelerationStructureBuildGeometryInfoKHR,
        pp_build_range_infos: *const *const AccelerationStructureBuildRangeInfoKHR,
    ),
>;
pub type PFN_vkCmdBuildAccelerationStructuresIndirectKHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        info_count: u32,
        p_infos: *const AccelerationStructureBuildGeometryInfoKHR,
        p_indirect_device_addresses: *const u64,
        p_indirect_strides: *const u32,
        pp_max_primitive_counts: *const *const u32,
    ),
>;
pub type PFN_vkBuildAccelerationStructuresKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        deferred_operation: DeferredOperationKHR,
        info_count: u32,
        p_infos: *const AccelerationStructureBuildGeometryInfoKHR,
        pp_build_range_infos: *const *const AccelerationStructureBuildRangeInfoKHR,
    ) -> Result,
>;
pub type PFN_vkGetAccelerationStructureDeviceAddressKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const AccelerationStructureDeviceAddressInfoKHR,
    ) -> u64,
>;
pub type PFN_vkCreateDeferredOperationKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_allocator: *const AllocationCallbacks,
        p_deferred_operation: *mut DeferredOperationKHR,
    ) -> Result,
>;
pub type PFN_vkDestroyDeferredOperationKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        operation: DeferredOperationKHR,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkGetDeferredOperationMaxConcurrencyKHR = Option<
    unsafe extern "system" fn(device: Device, operation: DeferredOperationKHR) -> u32,
>;
pub type PFN_vkGetDeferredOperationResultKHR = Option<
    unsafe extern "system" fn(device: Device, operation: DeferredOperationKHR) -> Result,
>;
pub type PFN_vkDeferredOperationJoinKHR = Option<
    unsafe extern "system" fn(device: Device, operation: DeferredOperationKHR) -> Result,
>;
pub type PFN_vkGetPipelineIndirectMemoryRequirementsNV = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const ComputePipelineCreateInfo,
        p_memory_requirements: *mut MemoryRequirements2,
    ),
>;
pub type PFN_vkGetPipelineIndirectDeviceAddressNV = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const PipelineIndirectDeviceAddressInfoNV,
    ) -> u64,
>;
pub type PFN_vkAntiLagUpdateAMD = Option<
    unsafe extern "system" fn(device: Device, p_data: *const AntiLagDataAMD),
>;
pub type PFN_vkCmdSetCullMode = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer, cull_mode: CullModeFlags),
>;
pub type PFN_vkCmdSetFrontFace = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer, front_face: FrontFace),
>;
pub type PFN_vkCmdSetPrimitiveTopology = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        primitive_topology: PrimitiveTopology,
    ),
>;
pub type PFN_vkCmdSetViewportWithCount = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        viewport_count: u32,
        p_viewports: *const Viewport,
    ),
>;
pub type PFN_vkCmdSetScissorWithCount = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        scissor_count: u32,
        p_scissors: *const Rect2D,
    ),
>;
pub type PFN_vkCmdBindIndexBuffer2 = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        buffer: Buffer,
        offset: u64,
        size: u64,
        index_type: IndexType,
    ),
>;
pub type PFN_vkCmdBindVertexBuffers2 = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        first_binding: u32,
        binding_count: u32,
        p_buffers: *const Buffer,
        p_offsets: *const u64,
        p_sizes: *const u64,
        p_strides: *const u64,
    ),
>;
pub type PFN_vkCmdSetDepthTestEnable = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer, depth_test_enable: u32),
>;
pub type PFN_vkCmdSetDepthWriteEnable = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer, depth_write_enable: u32),
>;
pub type PFN_vkCmdSetDepthCompareOp = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer, depth_compare_op: CompareOp),
>;
pub type PFN_vkCmdSetDepthBoundsTestEnable = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        depth_bounds_test_enable: u32,
    ),
>;
pub type PFN_vkCmdSetStencilTestEnable = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer, stencil_test_enable: u32),
>;
pub type PFN_vkCmdSetStencilOp = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        face_mask: StencilFaceFlags,
        fail_op: StencilOp,
        pass_op: StencilOp,
        depth_fail_op: StencilOp,
        compare_op: CompareOp,
    ),
>;
pub type PFN_vkCmdSetPatchControlPointsEXT = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer, patch_control_points: u32),
>;
pub type PFN_vkCmdSetRasterizerDiscardEnable = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        rasterizer_discard_enable: u32,
    ),
>;
pub type PFN_vkCmdSetDepthBiasEnable = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer, depth_bias_enable: u32),
>;
pub type PFN_vkCmdSetLogicOpEXT = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer, logic_op: LogicOp),
>;
pub type PFN_vkCmdSetPrimitiveRestartEnable = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        primitive_restart_enable: u32,
    ),
>;
pub type PFN_vkCmdSetTessellationDomainOriginEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        domain_origin: TessellationDomainOrigin,
    ),
>;
pub type PFN_vkCmdSetDepthClampEnableEXT = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer, depth_clamp_enable: u32),
>;
pub type PFN_vkCmdSetPolygonModeEXT = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer, polygon_mode: PolygonMode),
>;
pub type PFN_vkCmdSetRasterizationSamplesEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        rasterization_samples: SampleCountFlagBits,
    ),
>;
pub type PFN_vkCmdSetSampleMaskEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        samples: SampleCountFlagBits,
        p_sample_mask: *const u32,
    ),
>;
pub type PFN_vkCmdSetAlphaToCoverageEnableEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        alpha_to_coverage_enable: u32,
    ),
>;
pub type PFN_vkCmdSetAlphaToOneEnableEXT = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer, alpha_to_one_enable: u32),
>;
pub type PFN_vkCmdSetLogicOpEnableEXT = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer, logic_op_enable: u32),
>;
pub type PFN_vkCmdSetColorBlendEnableEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        first_attachment: u32,
        attachment_count: u32,
        p_color_blend_enables: *const u32,
    ),
>;
pub type PFN_vkCmdSetColorBlendEquationEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        first_attachment: u32,
        attachment_count: u32,
        p_color_blend_equations: *const ColorBlendEquationEXT,
    ),
>;
pub type PFN_vkCmdSetColorWriteMaskEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        first_attachment: u32,
        attachment_count: u32,
        p_color_write_masks: *const ColorComponentFlags,
    ),
>;
pub type PFN_vkCmdSetRasterizationStreamEXT = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer, rasterization_stream: u32),
>;
pub type PFN_vkCmdSetConservativeRasterizationModeEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        conservative_rasterization_mode: ConservativeRasterizationModeEXT,
    ),
>;
pub type PFN_vkCmdSetExtraPrimitiveOverestimationSizeEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        extra_primitive_overestimation_size: f32,
    ),
>;
pub type PFN_vkCmdSetDepthClipEnableEXT = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer, depth_clip_enable: u32),
>;
pub type PFN_vkCmdSetSampleLocationsEnableEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        sample_locations_enable: u32,
    ),
>;
pub type PFN_vkCmdSetColorBlendAdvancedEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        first_attachment: u32,
        attachment_count: u32,
        p_color_blend_advanced: *const ColorBlendAdvancedEXT,
    ),
>;
pub type PFN_vkCmdSetProvokingVertexModeEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        provoking_vertex_mode: ProvokingVertexModeEXT,
    ),
>;
pub type PFN_vkCmdSetLineRasterizationModeEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        line_rasterization_mode: LineRasterizationModeEXT,
    ),
>;
pub type PFN_vkCmdSetLineStippleEnableEXT = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer, stippled_line_enable: u32),
>;
pub type PFN_vkCmdSetDepthClipNegativeOneToOneEXT = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer, negative_one_to_one: u32),
>;
pub type PFN_vkCmdSetViewportWScalingEnableNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        viewport_w_scaling_enable: u32,
    ),
>;
pub type PFN_vkCmdSetViewportSwizzleNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        first_viewport: u32,
        viewport_count: u32,
        p_viewport_swizzles: *const ViewportSwizzleNV,
    ),
>;
pub type PFN_vkCmdSetCoverageToColorEnableNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        coverage_to_color_enable: u32,
    ),
>;
pub type PFN_vkCmdSetCoverageToColorLocationNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        coverage_to_color_location: u32,
    ),
>;
pub type PFN_vkCmdSetCoverageModulationModeNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        coverage_modulation_mode: CoverageModulationModeNV,
    ),
>;
pub type PFN_vkCmdSetCoverageModulationTableEnableNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        coverage_modulation_table_enable: u32,
    ),
>;
pub type PFN_vkCmdSetCoverageModulationTableNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        coverage_modulation_table_count: u32,
        p_coverage_modulation_table: *const f32,
    ),
>;
pub type PFN_vkCmdSetShadingRateImageEnableNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        shading_rate_image_enable: u32,
    ),
>;
pub type PFN_vkCmdSetCoverageReductionModeNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        coverage_reduction_mode: CoverageReductionModeNV,
    ),
>;
pub type PFN_vkCmdSetRepresentativeFragmentTestEnableNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        representative_fragment_test_enable: u32,
    ),
>;
pub type PFN_vkCreatePrivateDataSlot = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const PrivateDataSlotCreateInfo,
        p_allocator: *const AllocationCallbacks,
        p_private_data_slot: *mut PrivateDataSlot,
    ) -> Result,
>;
pub type PFN_vkDestroyPrivateDataSlot = Option<
    unsafe extern "system" fn(
        device: Device,
        private_data_slot: PrivateDataSlot,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkSetPrivateData = Option<
    unsafe extern "system" fn(
        device: Device,
        object_type: ObjectType,
        object_handle: u64,
        private_data_slot: PrivateDataSlot,
        data: u64,
    ) -> Result,
>;
pub type PFN_vkGetPrivateData = Option<
    unsafe extern "system" fn(
        device: Device,
        object_type: ObjectType,
        object_handle: u64,
        private_data_slot: PrivateDataSlot,
        p_data: *mut u64,
    ),
>;
pub type PFN_vkCmdCopyBuffer2 = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_copy_buffer_info: *const CopyBufferInfo2,
    ),
>;
pub type PFN_vkCmdCopyImage2 = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_copy_image_info: *const CopyImageInfo2,
    ),
>;
pub type PFN_vkCmdBlitImage2 = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_blit_image_info: *const BlitImageInfo2,
    ),
>;
pub type PFN_vkCmdCopyBufferToImage2 = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_copy_buffer_to_image_info: *const CopyBufferToImageInfo2,
    ),
>;
pub type PFN_vkCmdCopyImageToBuffer2 = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_copy_image_to_buffer_info: *const CopyImageToBufferInfo2,
    ),
>;
pub type PFN_vkCmdResolveImage2 = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_resolve_image_info: *const ResolveImageInfo2,
    ),
>;
pub type PFN_vkCmdRefreshObjectsKHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_refresh_objects: *const RefreshObjectListKHR,
    ),
>;
pub type PFN_vkGetPhysicalDeviceRefreshableObjectTypesKHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_refreshable_object_type_count: *mut u32,
        p_refreshable_object_types: *mut ObjectType,
    ) -> Result,
>;
pub type PFN_vkCmdSetFragmentShadingRateKHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_fragment_size: *const Extent2D,
        combiner_ops: FragmentShadingRateCombinerOpKHR,
    ),
>;
pub type PFN_vkGetPhysicalDeviceFragmentShadingRatesKHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_fragment_shading_rate_count: *mut u32,
        p_fragment_shading_rates: *mut PhysicalDeviceFragmentShadingRateKHR,
    ) -> Result,
>;
pub type PFN_vkCmdSetFragmentShadingRateEnumNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        shading_rate: FragmentShadingRateNV,
        combiner_ops: FragmentShadingRateCombinerOpKHR,
    ),
>;
pub type PFN_vkGetAccelerationStructureBuildSizesKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        build_type: AccelerationStructureBuildTypeKHR,
        p_build_info: *const AccelerationStructureBuildGeometryInfoKHR,
        p_max_primitive_counts: *const u32,
        p_size_info: *mut AccelerationStructureBuildSizesInfoKHR,
    ),
>;
pub type PFN_vkCmdSetVertexInputEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        vertex_binding_description_count: u32,
        p_vertex_binding_descriptions: *const VertexInputBindingDescription2EXT,
        vertex_attribute_description_count: u32,
        p_vertex_attribute_descriptions: *const VertexInputAttributeDescription2EXT,
    ),
>;
pub type PFN_vkCmdSetColorWriteEnableEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        attachment_count: u32,
        p_color_write_enables: *const u32,
    ),
>;
pub type PFN_vkCmdSetEvent2 = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        event: Event,
        p_dependency_info: *const DependencyInfo,
    ),
>;
pub type PFN_vkCmdResetEvent2 = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        event: Event,
        stage_mask: PipelineStageFlags2,
    ),
>;
pub type PFN_vkCmdWaitEvents2 = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        event_count: u32,
        p_events: *const Event,
        p_dependency_infos: *const DependencyInfo,
    ),
>;
pub type PFN_vkCmdPipelineBarrier2 = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_dependency_info: *const DependencyInfo,
    ),
>;
pub type PFN_vkQueueSubmit2 = Option<
    unsafe extern "system" fn(
        queue: Queue,
        submit_count: u32,
        p_submits: *const SubmitInfo2,
        fence: Fence,
    ) -> Result,
>;
pub type PFN_vkCmdWriteTimestamp2 = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        stage: PipelineStageFlags2,
        query_pool: QueryPool,
        query: u32,
    ),
>;
pub type PFN_vkCmdWriteBufferMarker2AMD = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        stage: PipelineStageFlags2,
        dst_buffer: Buffer,
        dst_offset: u64,
        marker: u32,
    ),
>;
pub type PFN_vkGetQueueCheckpointData2NV = Option<
    unsafe extern "system" fn(
        queue: Queue,
        p_checkpoint_data_count: *mut u32,
        p_checkpoint_data: *mut CheckpointData2NV,
    ),
>;
pub type PFN_vkCopyMemoryToImage = Option<
    unsafe extern "system" fn(
        device: Device,
        p_copy_memory_to_image_info: *const CopyMemoryToImageInfo,
    ) -> Result,
>;
pub type PFN_vkCopyImageToMemory = Option<
    unsafe extern "system" fn(
        device: Device,
        p_copy_image_to_memory_info: *const CopyImageToMemoryInfo,
    ) -> Result,
>;
pub type PFN_vkCopyImageToImage = Option<
    unsafe extern "system" fn(
        device: Device,
        p_copy_image_to_image_info: *const CopyImageToImageInfo,
    ) -> Result,
>;
pub type PFN_vkTransitionImageLayout = Option<
    unsafe extern "system" fn(
        device: Device,
        transition_count: u32,
        p_transitions: *const HostImageLayoutTransitionInfo,
    ) -> Result,
>;
pub type PFN_vkGetCommandPoolMemoryConsumption = Option<
    unsafe extern "system" fn(
        device: Device,
        command_pool: CommandPool,
        command_buffer: CommandBuffer,
        p_consumption: *mut CommandPoolMemoryConsumption,
    ),
>;
pub type PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_video_profile: *const VideoProfileInfoKHR,
        p_capabilities: *mut VideoCapabilitiesKHR,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_video_format_info: *const PhysicalDeviceVideoFormatInfoKHR,
        p_video_format_property_count: *mut u32,
        p_video_format_properties: *mut VideoFormatPropertiesKHR,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_quality_level_info: *const PhysicalDeviceVideoEncodeQualityLevelInfoKHR,
        p_quality_level_properties: *mut VideoEncodeQualityLevelPropertiesKHR,
    ) -> Result,
>;
pub type PFN_vkCreateVideoSessionKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const VideoSessionCreateInfoKHR,
        p_allocator: *const AllocationCallbacks,
        p_video_session: *mut VideoSessionKHR,
    ) -> Result,
>;
pub type PFN_vkDestroyVideoSessionKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        video_session: VideoSessionKHR,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkCreateVideoSessionParametersKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const VideoSessionParametersCreateInfoKHR,
        p_allocator: *const AllocationCallbacks,
        p_video_session_parameters: *mut VideoSessionParametersKHR,
    ) -> Result,
>;
pub type PFN_vkUpdateVideoSessionParametersKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        video_session_parameters: VideoSessionParametersKHR,
        p_update_info: *const VideoSessionParametersUpdateInfoKHR,
    ) -> Result,
>;
pub type PFN_vkGetEncodedVideoSessionParametersKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_video_session_parameters_info: *const VideoEncodeSessionParametersGetInfoKHR,
        p_feedback_info: *mut VideoEncodeSessionParametersFeedbackInfoKHR,
        p_data_size: *mut usize,
        p_data: *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkDestroyVideoSessionParametersKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        video_session_parameters: VideoSessionParametersKHR,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkGetVideoSessionMemoryRequirementsKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        video_session: VideoSessionKHR,
        p_memory_requirements_count: *mut u32,
        p_memory_requirements: *mut VideoSessionMemoryRequirementsKHR,
    ) -> Result,
>;
pub type PFN_vkBindVideoSessionMemoryKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        video_session: VideoSessionKHR,
        bind_session_memory_info_count: u32,
        p_bind_session_memory_infos: *const BindVideoSessionMemoryInfoKHR,
    ) -> Result,
>;
pub type PFN_vkCmdDecodeVideoKHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_decode_info: *const VideoDecodeInfoKHR,
    ),
>;
pub type PFN_vkCmdBeginVideoCodingKHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_begin_info: *const VideoBeginCodingInfoKHR,
    ),
>;
pub type PFN_vkCmdControlVideoCodingKHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_coding_control_info: *const VideoCodingControlInfoKHR,
    ),
>;
pub type PFN_vkCmdEndVideoCodingKHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_end_coding_info: *const VideoEndCodingInfoKHR,
    ),
>;
pub type PFN_vkCmdEncodeVideoKHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_encode_info: *const VideoEncodeInfoKHR,
    ),
>;
pub type PFN_vkCmdDecompressMemoryNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        decompress_region_count: u32,
        p_decompress_memory_regions: *const DecompressMemoryRegionNV,
    ),
>;
pub type PFN_vkCmdDecompressMemoryIndirectCountNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        indirect_commands_address: u64,
        indirect_commands_count_address: u64,
        stride: u32,
    ),
>;
pub type PFN_vkGetPartitionedAccelerationStructuresBuildSizesNV = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const PartitionedAccelerationStructureInstancesInputNV,
        p_size_info: *mut AccelerationStructureBuildSizesInfoKHR,
    ),
>;
pub type PFN_vkCmdBuildPartitionedAccelerationStructuresNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_build_info: *const BuildPartitionedAccelerationStructureInfoNV,
    ),
>;
pub type PFN_vkCmdDecompressMemoryEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_decompress_memory_info_ext: *const DecompressMemoryInfoEXT,
    ),
>;
pub type PFN_vkCmdDecompressMemoryIndirectCountEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        decompression_method: MemoryDecompressionMethodFlagsEXT,
        indirect_commands_address: u64,
        indirect_commands_count_address: u64,
        max_decompression_count: u32,
        stride: u32,
    ),
>;
pub type PFN_vkCreateCuModuleNVX = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const CuModuleCreateInfoNVX,
        p_allocator: *const AllocationCallbacks,
        p_module: *mut CuModuleNVX,
    ) -> Result,
>;
pub type PFN_vkCreateCuFunctionNVX = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const CuFunctionCreateInfoNVX,
        p_allocator: *const AllocationCallbacks,
        p_function: *mut CuFunctionNVX,
    ) -> Result,
>;
pub type PFN_vkDestroyCuModuleNVX = Option<
    unsafe extern "system" fn(
        device: Device,
        module: CuModuleNVX,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkDestroyCuFunctionNVX = Option<
    unsafe extern "system" fn(
        device: Device,
        function: CuFunctionNVX,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkCmdCuLaunchKernelNVX = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_launch_info: *const CuLaunchInfoNVX,
    ),
>;
pub type PFN_vkGetDescriptorSetLayoutSizeEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        layout: DescriptorSetLayout,
        p_layout_size_in_bytes: *mut u64,
    ),
>;
pub type PFN_vkGetDescriptorSetLayoutBindingOffsetEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        layout: DescriptorSetLayout,
        binding: u32,
        p_offset: *mut u64,
    ),
>;
pub type PFN_vkGetDescriptorEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        p_descriptor_info: *const DescriptorGetInfoEXT,
        data_size: usize,
        p_descriptor: *mut std::ffi::c_void,
    ),
>;
pub type PFN_vkCmdBindDescriptorBuffersEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        buffer_count: u32,
        p_binding_infos: *const DescriptorBufferBindingInfoEXT,
    ),
>;
pub type PFN_vkCmdSetDescriptorBufferOffsetsEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        pipeline_bind_point: PipelineBindPoint,
        layout: PipelineLayout,
        first_set: u32,
        set_count: u32,
        p_buffer_indices: *const u32,
        p_offsets: *const u64,
    ),
>;
pub type PFN_vkCmdBindDescriptorBufferEmbeddedSamplersEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        pipeline_bind_point: PipelineBindPoint,
        layout: PipelineLayout,
        set: u32,
    ),
>;
pub type PFN_vkGetBufferOpaqueCaptureDescriptorDataEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const BufferCaptureDescriptorDataInfoEXT,
        p_data: *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkGetImageOpaqueCaptureDescriptorDataEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const ImageCaptureDescriptorDataInfoEXT,
        p_data: *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkGetImageViewOpaqueCaptureDescriptorDataEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const ImageViewCaptureDescriptorDataInfoEXT,
        p_data: *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkGetSamplerOpaqueCaptureDescriptorDataEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const SamplerCaptureDescriptorDataInfoEXT,
        p_data: *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const AccelerationStructureCaptureDescriptorDataInfoEXT,
        p_data: *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkSetDeviceMemoryPriorityEXT = Option<
    unsafe extern "system" fn(device: Device, memory: DeviceMemory, priority: f32),
>;
pub type PFN_vkAcquireDrmDisplayEXT = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        drm_fd: i32,
        display: DisplayKHR,
    ) -> Result,
>;
pub type PFN_vkGetDrmDisplayEXT = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        drm_fd: i32,
        connector_id: u32,
        display: *mut DisplayKHR,
    ) -> Result,
>;
pub type PFN_vkWaitForPresent2KHR = Option<
    unsafe extern "system" fn(
        device: Device,
        swapchain: SwapchainKHR,
        p_present_wait2_info: *const PresentWait2InfoKHR,
    ) -> Result,
>;
pub type PFN_vkWaitForPresentKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        swapchain: SwapchainKHR,
        present_id: u64,
        timeout: u64,
    ) -> Result,
>;
pub type PFN_vkCreateBufferCollectionFUCHSIA = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const BufferCollectionCreateInfoFUCHSIA,
        p_allocator: *const AllocationCallbacks,
        p_collection: *mut BufferCollectionFUCHSIA,
    ) -> Result,
>;
pub type PFN_vkSetBufferCollectionBufferConstraintsFUCHSIA = Option<
    unsafe extern "system" fn(
        device: Device,
        collection: BufferCollectionFUCHSIA,
        p_buffer_constraints_info: *const BufferConstraintsInfoFUCHSIA,
    ) -> Result,
>;
pub type PFN_vkSetBufferCollectionImageConstraintsFUCHSIA = Option<
    unsafe extern "system" fn(
        device: Device,
        collection: BufferCollectionFUCHSIA,
        p_image_constraints_info: *const ImageConstraintsInfoFUCHSIA,
    ) -> Result,
>;
pub type PFN_vkDestroyBufferCollectionFUCHSIA = Option<
    unsafe extern "system" fn(
        device: Device,
        collection: BufferCollectionFUCHSIA,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkGetBufferCollectionPropertiesFUCHSIA = Option<
    unsafe extern "system" fn(
        device: Device,
        collection: BufferCollectionFUCHSIA,
        p_properties: *mut BufferCollectionPropertiesFUCHSIA,
    ) -> Result,
>;
pub type PFN_vkCreateCudaModuleNV = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const CudaModuleCreateInfoNV,
        p_allocator: *const AllocationCallbacks,
        p_module: *mut CudaModuleNV,
    ) -> Result,
>;
pub type PFN_vkGetCudaModuleCacheNV = Option<
    unsafe extern "system" fn(
        device: Device,
        module: CudaModuleNV,
        p_cache_size: *mut usize,
        p_cache_data: *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkCreateCudaFunctionNV = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const CudaFunctionCreateInfoNV,
        p_allocator: *const AllocationCallbacks,
        p_function: *mut CudaFunctionNV,
    ) -> Result,
>;
pub type PFN_vkDestroyCudaModuleNV = Option<
    unsafe extern "system" fn(
        device: Device,
        module: CudaModuleNV,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkDestroyCudaFunctionNV = Option<
    unsafe extern "system" fn(
        device: Device,
        function: CudaFunctionNV,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkCmdCudaLaunchKernelNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_launch_info: *const CudaLaunchInfoNV,
    ),
>;
pub type PFN_vkCmdBeginRendering = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_rendering_info: *const RenderingInfo,
    ),
>;
pub type PFN_vkCmdEndRendering = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer),
>;
pub type PFN_vkCmdEndRendering2KHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_rendering_end_info: *const RenderingEndInfoKHR,
    ),
>;
pub type PFN_vkGetDescriptorSetLayoutHostMappingInfoVALVE = Option<
    unsafe extern "system" fn(
        device: Device,
        p_binding_reference: *const DescriptorSetBindingReferenceVALVE,
        p_host_mapping: *mut DescriptorSetLayoutHostMappingInfoVALVE,
    ),
>;
pub type PFN_vkGetDescriptorSetHostMappingVALVE = Option<
    unsafe extern "system" fn(
        device: Device,
        descriptor_set: DescriptorSet,
        pp_data: *mut *mut std::ffi::c_void,
    ),
>;
pub type PFN_vkCreateMicromapEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const MicromapCreateInfoEXT,
        p_allocator: *const AllocationCallbacks,
        p_micromap: *mut MicromapEXT,
    ) -> Result,
>;
pub type PFN_vkCmdBuildMicromapsEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        info_count: u32,
        p_infos: *const MicromapBuildInfoEXT,
    ),
>;
pub type PFN_vkBuildMicromapsEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        deferred_operation: DeferredOperationKHR,
        info_count: u32,
        p_infos: *const MicromapBuildInfoEXT,
    ) -> Result,
>;
pub type PFN_vkDestroyMicromapEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        micromap: MicromapEXT,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkCmdCopyMicromapEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_info: *const CopyMicromapInfoEXT,
    ),
>;
pub type PFN_vkCopyMicromapEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        deferred_operation: DeferredOperationKHR,
        p_info: *const CopyMicromapInfoEXT,
    ) -> Result,
>;
pub type PFN_vkCmdCopyMicromapToMemoryEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_info: *const CopyMicromapToMemoryInfoEXT,
    ),
>;
pub type PFN_vkCopyMicromapToMemoryEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        deferred_operation: DeferredOperationKHR,
        p_info: *const CopyMicromapToMemoryInfoEXT,
    ) -> Result,
>;
pub type PFN_vkCmdCopyMemoryToMicromapEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_info: *const CopyMemoryToMicromapInfoEXT,
    ),
>;
pub type PFN_vkCopyMemoryToMicromapEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        deferred_operation: DeferredOperationKHR,
        p_info: *const CopyMemoryToMicromapInfoEXT,
    ) -> Result,
>;
pub type PFN_vkCmdWriteMicromapsPropertiesEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        micromap_count: u32,
        p_micromaps: *const MicromapEXT,
        query_type: QueryType,
        query_pool: QueryPool,
        first_query: u32,
    ),
>;
pub type PFN_vkWriteMicromapsPropertiesEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        micromap_count: u32,
        p_micromaps: *const MicromapEXT,
        query_type: QueryType,
        data_size: usize,
        p_data: *mut std::ffi::c_void,
        stride: usize,
    ) -> Result,
>;
pub type PFN_vkGetDeviceMicromapCompatibilityEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        p_version_info: *const MicromapVersionInfoEXT,
        p_compatibility: *mut AccelerationStructureCompatibilityKHR,
    ),
>;
pub type PFN_vkGetMicromapBuildSizesEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        build_type: AccelerationStructureBuildTypeKHR,
        p_build_info: *const MicromapBuildInfoEXT,
        p_size_info: *mut MicromapBuildSizesInfoEXT,
    ),
>;
pub type PFN_vkGetShaderModuleIdentifierEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        shader_module: ShaderModule,
        p_identifier: *mut ShaderModuleIdentifierEXT,
    ),
>;
pub type PFN_vkGetShaderModuleCreateInfoIdentifierEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const ShaderModuleCreateInfo,
        p_identifier: *mut ShaderModuleIdentifierEXT,
    ),
>;
pub type PFN_vkGetImageSubresourceLayout2 = Option<
    unsafe extern "system" fn(
        device: Device,
        image: Image,
        p_subresource: *const ImageSubresource2,
        p_layout: *mut SubresourceLayout2,
    ),
>;
pub type PFN_vkGetPipelinePropertiesEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        p_pipeline_info: *const PipelineInfoEXT,
        p_pipeline_properties: *mut BaseOutStructure,
    ) -> Result,
>;
pub type PFN_vkExportMetalObjectsEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        p_metal_objects_info: *mut ExportMetalObjectsInfoEXT,
    ),
>;
pub type PFN_vkCmdBindTileMemoryQCOM = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_tile_memory_bind_info: *const TileMemoryBindInfoQCOM,
    ),
>;
pub type PFN_vkGetFramebufferTilePropertiesQCOM = Option<
    unsafe extern "system" fn(
        device: Device,
        framebuffer: Framebuffer,
        p_properties_count: *mut u32,
        p_properties: *mut TilePropertiesQCOM,
    ) -> Result,
>;
pub type PFN_vkGetDynamicRenderingTilePropertiesQCOM = Option<
    unsafe extern "system" fn(
        device: Device,
        p_rendering_info: *const RenderingInfo,
        p_properties: *mut TilePropertiesQCOM,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceOpticalFlowImageFormatsNV = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_optical_flow_image_format_info: *const OpticalFlowImageFormatInfoNV,
        p_format_count: *mut u32,
        p_image_format_properties: *mut OpticalFlowImageFormatPropertiesNV,
    ) -> Result,
>;
pub type PFN_vkCreateOpticalFlowSessionNV = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const OpticalFlowSessionCreateInfoNV,
        p_allocator: *const AllocationCallbacks,
        p_session: *mut OpticalFlowSessionNV,
    ) -> Result,
>;
pub type PFN_vkDestroyOpticalFlowSessionNV = Option<
    unsafe extern "system" fn(
        device: Device,
        session: OpticalFlowSessionNV,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkBindOpticalFlowSessionImageNV = Option<
    unsafe extern "system" fn(
        device: Device,
        session: OpticalFlowSessionNV,
        binding_point: OpticalFlowSessionBindingPointNV,
        view: ImageView,
        layout: ImageLayout,
    ) -> Result,
>;
pub type PFN_vkCmdOpticalFlowExecuteNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        session: OpticalFlowSessionNV,
        p_execute_info: *const OpticalFlowExecuteInfoNV,
    ),
>;
pub type PFN_vkGetDeviceFaultInfoEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        p_fault_counts: *mut DeviceFaultCountsEXT,
        p_fault_info: *mut DeviceFaultInfoEXT,
    ) -> Result,
>;
pub type PFN_vkGetDeviceFaultReportsKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        timeout: u64,
        p_fault_counts: *mut u32,
        p_fault_info: *mut DeviceFaultInfoKHR,
    ) -> Result,
>;
pub type PFN_vkGetDeviceFaultDebugInfoKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_debug_info: *mut DeviceFaultDebugInfoKHR,
    ) -> Result,
>;
pub type PFN_vkCmdSetDepthBias2EXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_depth_bias_info: *const DepthBiasInfoEXT,
    ),
>;
pub type PFN_vkReleaseSwapchainImagesKHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_release_info: *const ReleaseSwapchainImagesInfoKHR,
    ) -> Result,
>;
pub type PFN_vkGetDeviceImageSubresourceLayout = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const DeviceImageSubresourceInfo,
        p_layout: *mut SubresourceLayout2,
    ),
>;
pub type PFN_vkMapMemory2 = Option<
    unsafe extern "system" fn(
        device: Device,
        p_memory_map_info: *const MemoryMapInfo,
        pp_data: *mut *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkUnmapMemory2 = Option<
    unsafe extern "system" fn(
        device: Device,
        p_memory_unmap_info: *const MemoryUnmapInfo,
    ) -> Result,
>;
pub type PFN_vkCreateShadersEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        create_info_count: u32,
        p_create_infos: *const ShaderCreateInfoEXT,
        p_allocator: *const AllocationCallbacks,
        p_shaders: *mut ShaderEXT,
    ) -> Result,
>;
pub type PFN_vkDestroyShaderEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        shader: ShaderEXT,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkGetShaderBinaryDataEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        shader: ShaderEXT,
        p_data_size: *mut usize,
        p_data: *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkCmdBindShadersEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        stage_count: u32,
        p_stages: *const ShaderStageFlagBits,
        p_shaders: *const ShaderEXT,
    ),
>;
pub type PFN_vkSetSwapchainPresentTimingQueueSizeEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        swapchain: SwapchainKHR,
        size: u32,
    ) -> Result,
>;
pub type PFN_vkGetSwapchainTimingPropertiesEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        swapchain: SwapchainKHR,
        p_swapchain_timing_properties: *mut SwapchainTimingPropertiesEXT,
        p_swapchain_timing_properties_counter: *mut u64,
    ) -> Result,
>;
pub type PFN_vkGetSwapchainTimeDomainPropertiesEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        swapchain: SwapchainKHR,
        p_swapchain_time_domain_properties: *mut SwapchainTimeDomainPropertiesEXT,
        p_time_domains_counter: *mut u64,
    ) -> Result,
>;
pub type PFN_vkGetPastPresentationTimingEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        p_past_presentation_timing_info: *const PastPresentationTimingInfoEXT,
        p_past_presentation_timing_properties: *mut PastPresentationTimingPropertiesEXT,
    ) -> Result,
>;
pub type PFN_vkGetScreenBufferPropertiesQNX = Option<
    unsafe extern "system" fn(
        device: Device,
        buffer: *const std::ffi::c_void,
        p_properties: *mut ScreenBufferPropertiesQNX,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_property_count: *mut u32,
        p_properties: *mut CooperativeMatrixPropertiesKHR,
    ) -> Result,
>;
pub type PFN_vkGetExecutionGraphPipelineScratchSizeAMDX = Option<
    unsafe extern "system" fn(
        device: Device,
        execution_graph: Pipeline,
        p_size_info: *mut ExecutionGraphPipelineScratchSizeAMDX,
    ) -> Result,
>;
pub type PFN_vkGetExecutionGraphPipelineNodeIndexAMDX = Option<
    unsafe extern "system" fn(
        device: Device,
        execution_graph: Pipeline,
        p_node_info: *const PipelineShaderStageNodeCreateInfoAMDX,
        p_node_index: *mut u32,
    ) -> Result,
>;
pub type PFN_vkCreateExecutionGraphPipelinesAMDX = Option<
    unsafe extern "system" fn(
        device: Device,
        pipeline_cache: PipelineCache,
        create_info_count: u32,
        p_create_infos: *const ExecutionGraphPipelineCreateInfoAMDX,
        p_allocator: *const AllocationCallbacks,
        p_pipelines: *mut Pipeline,
    ) -> Result,
>;
pub type PFN_vkCmdInitializeGraphScratchMemoryAMDX = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        execution_graph: Pipeline,
        scratch: u64,
        scratch_size: u64,
    ),
>;
pub type PFN_vkCmdDispatchGraphAMDX = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        scratch: u64,
        scratch_size: u64,
        p_count_info: *const DispatchGraphCountInfoAMDX,
    ),
>;
pub type PFN_vkCmdDispatchGraphIndirectAMDX = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        scratch: u64,
        scratch_size: u64,
        p_count_info: *const DispatchGraphCountInfoAMDX,
    ),
>;
pub type PFN_vkCmdDispatchGraphIndirectCountAMDX = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        scratch: u64,
        scratch_size: u64,
        count_info: u64,
    ),
>;
pub type PFN_vkCmdBindDescriptorSets2 = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_bind_descriptor_sets_info: *const BindDescriptorSetsInfo,
    ),
>;
pub type PFN_vkCmdPushConstants2 = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_push_constants_info: *const PushConstantsInfo,
    ),
>;
pub type PFN_vkCmdPushDescriptorSet2 = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_push_descriptor_set_info: *const PushDescriptorSetInfo,
    ),
>;
pub type PFN_vkCmdPushDescriptorSetWithTemplate2 = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_push_descriptor_set_with_template_info: *const PushDescriptorSetWithTemplateInfo,
    ),
>;
pub type PFN_vkCmdSetDescriptorBufferOffsets2EXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_set_descriptor_buffer_offsets_info: *const SetDescriptorBufferOffsetsInfoEXT,
    ),
>;
pub type PFN_vkCmdBindDescriptorBufferEmbeddedSamplers2EXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_bind_descriptor_buffer_embedded_samplers_info: *const BindDescriptorBufferEmbeddedSamplersInfoEXT,
    ),
>;
pub type PFN_vkSetLatencySleepModeNV = Option<
    unsafe extern "system" fn(
        device: Device,
        swapchain: SwapchainKHR,
        p_sleep_mode_info: *const LatencySleepModeInfoNV,
    ) -> Result,
>;
pub type PFN_vkLatencySleepNV = Option<
    unsafe extern "system" fn(
        device: Device,
        swapchain: SwapchainKHR,
        p_sleep_info: *const LatencySleepInfoNV,
    ) -> Result,
>;
pub type PFN_vkSetLatencyMarkerNV = Option<
    unsafe extern "system" fn(
        device: Device,
        swapchain: SwapchainKHR,
        p_latency_marker_info: *const SetLatencyMarkerInfoNV,
    ),
>;
pub type PFN_vkGetLatencyTimingsNV = Option<
    unsafe extern "system" fn(
        device: Device,
        swapchain: SwapchainKHR,
        p_latency_marker_info: *mut GetLatencyMarkerInfoNV,
    ),
>;
pub type PFN_vkQueueNotifyOutOfBandNV = Option<
    unsafe extern "system" fn(
        queue: Queue,
        p_queue_type_info: *const OutOfBandQueueTypeInfoNV,
    ),
>;
pub type PFN_vkCmdSetRenderingAttachmentLocations = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_location_info: *const RenderingAttachmentLocationInfo,
    ),
>;
pub type PFN_vkCmdSetRenderingInputAttachmentIndices = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_input_attachment_index_info: *const RenderingInputAttachmentIndexInfo,
    ),
>;
pub type PFN_vkCmdSetDepthClampRangeEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        depth_clamp_mode: DepthClampModeEXT,
        p_depth_clamp_range: *const DepthClampRangeEXT,
    ),
>;
pub type PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_property_count: *mut u32,
        p_properties: *mut CooperativeMatrixFlexibleDimensionsPropertiesNV,
    ) -> Result,
>;
pub type PFN_vkGetMemoryMetalHandleEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        p_get_metal_handle_info: *const MemoryGetMetalHandleInfoEXT,
        p_handle: *mut *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkGetMemoryMetalHandlePropertiesEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        handle_type: ExternalMemoryHandleTypeFlagBits,
        p_handle: *const std::ffi::c_void,
        p_memory_metal_handle_properties: *mut MemoryMetalHandlePropertiesEXT,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceCooperativeVectorPropertiesNV = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_property_count: *mut u32,
        p_properties: *mut CooperativeVectorPropertiesNV,
    ) -> Result,
>;
pub type PFN_vkConvertCooperativeVectorMatrixNV = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const ConvertCooperativeVectorMatrixInfoNV,
    ) -> Result,
>;
pub type PFN_vkCmdConvertCooperativeVectorMatrixNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        info_count: u32,
        p_infos: *const ConvertCooperativeVectorMatrixInfoNV,
    ),
>;
pub type PFN_vkCmdDispatchTileQCOM = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_dispatch_tile_info: *const DispatchTileInfoQCOM,
    ),
>;
pub type PFN_vkCmdBeginPerTileExecutionQCOM = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_per_tile_begin_info: *const PerTileBeginInfoQCOM,
    ),
>;
pub type PFN_vkCmdEndPerTileExecutionQCOM = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_per_tile_end_info: *const PerTileEndInfoQCOM,
    ),
>;
pub type PFN_vkCreateExternalComputeQueueNV = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const ExternalComputeQueueCreateInfoNV,
        p_allocator: *const AllocationCallbacks,
        p_external_queue: *mut ExternalComputeQueueNV,
    ) -> Result,
>;
pub type PFN_vkDestroyExternalComputeQueueNV = Option<
    unsafe extern "system" fn(
        device: Device,
        external_queue: ExternalComputeQueueNV,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkGetExternalComputeQueueDataNV = Option<
    unsafe extern "system" fn(
        external_queue: ExternalComputeQueueNV,
        params: *mut ExternalComputeQueueDataParamsNV,
        p_data: *mut std::ffi::c_void,
    ),
>;
pub type PFN_vkEnumeratePhysicalDeviceShaderInstrumentationMetricsARM = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_description_count: *mut u32,
        p_descriptions: *mut ShaderInstrumentationMetricDescriptionARM,
    ) -> Result,
>;
pub type PFN_vkCreateShaderInstrumentationARM = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const ShaderInstrumentationCreateInfoARM,
        p_allocator: *const AllocationCallbacks,
        p_instrumentation: *mut ShaderInstrumentationARM,
    ) -> Result,
>;
pub type PFN_vkDestroyShaderInstrumentationARM = Option<
    unsafe extern "system" fn(
        device: Device,
        instrumentation: ShaderInstrumentationARM,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkCmdBeginShaderInstrumentationARM = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        instrumentation: ShaderInstrumentationARM,
    ),
>;
pub type PFN_vkCmdEndShaderInstrumentationARM = Option<
    unsafe extern "system" fn(command_buffer: CommandBuffer),
>;
pub type PFN_vkGetShaderInstrumentationValuesARM = Option<
    unsafe extern "system" fn(
        device: Device,
        instrumentation: ShaderInstrumentationARM,
        p_metric_block_count: *mut u32,
        p_metric_values: *mut std::ffi::c_void,
        flags: ShaderInstrumentationValuesFlagsARM,
    ) -> Result,
>;
pub type PFN_vkClearShaderInstrumentationMetricsARM = Option<
    unsafe extern "system" fn(device: Device, instrumentation: ShaderInstrumentationARM),
>;
pub type PFN_vkCreateTensorARM = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const TensorCreateInfoARM,
        p_allocator: *const AllocationCallbacks,
        p_tensor: *mut TensorARM,
    ) -> Result,
>;
pub type PFN_vkDestroyTensorARM = Option<
    unsafe extern "system" fn(
        device: Device,
        tensor: TensorARM,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkCreateTensorViewARM = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const TensorViewCreateInfoARM,
        p_allocator: *const AllocationCallbacks,
        p_view: *mut TensorViewARM,
    ) -> Result,
>;
pub type PFN_vkDestroyTensorViewARM = Option<
    unsafe extern "system" fn(
        device: Device,
        tensor_view: TensorViewARM,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkGetTensorMemoryRequirementsARM = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const TensorMemoryRequirementsInfoARM,
        p_memory_requirements: *mut MemoryRequirements2,
    ),
>;
pub type PFN_vkBindTensorMemoryARM = Option<
    unsafe extern "system" fn(
        device: Device,
        bind_info_count: u32,
        p_bind_infos: *const BindTensorMemoryInfoARM,
    ) -> Result,
>;
pub type PFN_vkGetDeviceTensorMemoryRequirementsARM = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const DeviceTensorMemoryRequirementsARM,
        p_memory_requirements: *mut MemoryRequirements2,
    ),
>;
pub type PFN_vkCmdCopyTensorARM = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_copy_tensor_info: *const CopyTensorInfoARM,
    ),
>;
pub type PFN_vkGetTensorOpaqueCaptureDescriptorDataARM = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const TensorCaptureDescriptorDataInfoARM,
        p_data: *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkGetTensorViewOpaqueCaptureDescriptorDataARM = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const TensorViewCaptureDescriptorDataInfoARM,
        p_data: *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceExternalTensorPropertiesARM = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_external_tensor_info: *const PhysicalDeviceExternalTensorInfoARM,
        p_external_tensor_properties: *mut ExternalTensorPropertiesARM,
    ),
>;
pub type PFN_vkCreateDataGraphPipelinesARM = Option<
    unsafe extern "system" fn(
        device: Device,
        deferred_operation: DeferredOperationKHR,
        pipeline_cache: PipelineCache,
        create_info_count: u32,
        p_create_infos: *const DataGraphPipelineCreateInfoARM,
        p_allocator: *const AllocationCallbacks,
        p_pipelines: *mut Pipeline,
    ) -> Result,
>;
pub type PFN_vkCreateDataGraphPipelineSessionARM = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const DataGraphPipelineSessionCreateInfoARM,
        p_allocator: *const AllocationCallbacks,
        p_session: *mut DataGraphPipelineSessionARM,
    ) -> Result,
>;
pub type PFN_vkGetDataGraphPipelineSessionBindPointRequirementsARM = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const DataGraphPipelineSessionBindPointRequirementsInfoARM,
        p_bind_point_requirement_count: *mut u32,
        p_bind_point_requirements: *mut DataGraphPipelineSessionBindPointRequirementARM,
    ) -> Result,
>;
pub type PFN_vkGetDataGraphPipelineSessionMemoryRequirementsARM = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const DataGraphPipelineSessionMemoryRequirementsInfoARM,
        p_memory_requirements: *mut MemoryRequirements2,
    ),
>;
pub type PFN_vkBindDataGraphPipelineSessionMemoryARM = Option<
    unsafe extern "system" fn(
        device: Device,
        bind_info_count: u32,
        p_bind_infos: *const BindDataGraphPipelineSessionMemoryInfoARM,
    ) -> Result,
>;
pub type PFN_vkDestroyDataGraphPipelineSessionARM = Option<
    unsafe extern "system" fn(
        device: Device,
        session: DataGraphPipelineSessionARM,
        p_allocator: *const AllocationCallbacks,
    ),
>;
pub type PFN_vkCmdDispatchDataGraphARM = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        session: DataGraphPipelineSessionARM,
        p_info: *const DataGraphPipelineDispatchInfoARM,
    ),
>;
pub type PFN_vkGetDataGraphPipelineAvailablePropertiesARM = Option<
    unsafe extern "system" fn(
        device: Device,
        p_pipeline_info: *const DataGraphPipelineInfoARM,
        p_properties_count: *mut u32,
        p_properties: *mut DataGraphPipelinePropertyARM,
    ) -> Result,
>;
pub type PFN_vkGetDataGraphPipelinePropertiesARM = Option<
    unsafe extern "system" fn(
        device: Device,
        p_pipeline_info: *const DataGraphPipelineInfoARM,
        properties_count: u32,
        p_properties: *mut DataGraphPipelinePropertyQueryResultARM,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        queue_family_index: u32,
        p_queue_family_data_graph_property_count: *mut u32,
        p_queue_family_data_graph_properties: *mut QueueFamilyDataGraphPropertiesARM,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceQueueFamilyDataGraphProcessingEnginePropertiesARM = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_queue_family_data_graph_processing_engine_info: *const PhysicalDeviceQueueFamilyDataGraphProcessingEngineInfoARM,
        p_queue_family_data_graph_processing_engine_properties: *mut QueueFamilyDataGraphProcessingEnginePropertiesARM,
    ),
>;
pub type PFN_vkGetNativeBufferPropertiesOHOS = Option<
    unsafe extern "system" fn(
        device: Device,
        buffer: *const std::ffi::c_void,
        p_properties: *mut NativeBufferPropertiesOHOS,
    ) -> Result,
>;
pub type PFN_vkGetMemoryNativeBufferOHOS = Option<
    unsafe extern "system" fn(
        device: Device,
        p_info: *const MemoryGetNativeBufferInfoOHOS,
        p_buffer: *mut *mut std::ffi::c_void,
    ) -> Result,
>;
pub type PFN_vkGetSwapchainGrallocUsageOHOS = Option<
    unsafe extern "system" fn(
        device: Device,
        format: Format,
        image_usage: ImageUsageFlags,
        gralloc_usage: *mut u64,
    ) -> Result,
>;
pub type PFN_vkAcquireImageOHOS = Option<
    unsafe extern "system" fn(
        device: Device,
        image: Image,
        native_fence_fd: i32,
        semaphore: Semaphore,
        fence: Fence,
    ) -> Result,
>;
pub type PFN_vkQueueSignalReleaseImageOHOS = Option<
    unsafe extern "system" fn(
        queue: Queue,
        wait_semaphore_count: u32,
        p_wait_semaphores: *const Semaphore,
        image: Image,
        p_native_fence_fd: *mut i32,
    ) -> Result,
>;
pub type PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceCountersByRegionARM = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        queue_family_index: u32,
        p_counter_count: *mut u32,
        p_counters: *mut PerformanceCounterARM,
        p_counter_descriptions: *mut PerformanceCounterDescriptionARM,
    ) -> Result,
>;
pub type PFN_vkCmdSetComputeOccupancyPriorityNV = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_parameters: *const ComputeOccupancyPriorityParametersNV,
    ),
>;
pub type PFN_vkWriteSamplerDescriptorsEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        sampler_count: u32,
        p_samplers: *const SamplerCreateInfo,
        p_descriptors: *const HostAddressRangeEXT,
    ) -> Result,
>;
pub type PFN_vkWriteResourceDescriptorsEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        resource_count: u32,
        p_resources: *const ResourceDescriptorInfoEXT,
        p_descriptors: *const HostAddressRangeEXT,
    ) -> Result,
>;
pub type PFN_vkCmdBindSamplerHeapEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_bind_info: *const BindHeapInfoEXT,
    ),
>;
pub type PFN_vkCmdBindResourceHeapEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_bind_info: *const BindHeapInfoEXT,
    ),
>;
pub type PFN_vkCmdPushDataEXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_push_data_info: *const PushDataInfoEXT,
    ),
>;
pub type PFN_vkRegisterCustomBorderColorEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        p_border_color: *const SamplerCustomBorderColorCreateInfoEXT,
        request_index: u32,
        p_index: *mut u32,
    ) -> Result,
>;
pub type PFN_vkUnregisterCustomBorderColorEXT = Option<
    unsafe extern "system" fn(device: Device, index: u32),
>;
pub type PFN_vkGetImageOpaqueCaptureDataEXT = Option<
    unsafe extern "system" fn(
        device: Device,
        image_count: u32,
        p_images: *const Image,
        p_datas: *mut HostAddressRangeEXT,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceDescriptorSizeEXT = Option<
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        descriptor_type: DescriptorType,
    ) -> u64,
>;
pub type PFN_vkGetTensorOpaqueCaptureDataARM = Option<
    unsafe extern "system" fn(
        device: Device,
        tensor_count: u32,
        p_tensors: *const TensorARM,
        p_datas: *mut HostAddressRangeEXT,
    ) -> Result,
>;
pub type PFN_vkCmdCopyMemoryKHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_copy_memory_info: *const CopyDeviceMemoryInfoKHR,
    ),
>;
pub type PFN_vkCmdCopyMemoryToImageKHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_copy_memory_info: *const CopyDeviceMemoryImageInfoKHR,
    ),
>;
pub type PFN_vkCmdCopyImageToMemoryKHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_copy_memory_info: *const CopyDeviceMemoryImageInfoKHR,
    ),
>;
pub type PFN_vkCmdUpdateMemoryKHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_dst_range: *const DeviceAddressRangeKHR,
        dst_flags: AddressCommandFlagsKHR,
        data_size: u64,
        p_data: *const std::ffi::c_void,
    ),
>;
pub type PFN_vkCmdFillMemoryKHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_dst_range: *const DeviceAddressRangeKHR,
        dst_flags: AddressCommandFlagsKHR,
        data: u32,
    ),
>;
pub type PFN_vkCmdCopyQueryPoolResultsToMemoryKHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        query_pool: QueryPool,
        first_query: u32,
        query_count: u32,
        p_dst_range: *const StridedDeviceAddressRangeKHR,
        dst_flags: AddressCommandFlagsKHR,
        query_result_flags: QueryResultFlags,
    ),
>;
pub type PFN_vkCmdBeginConditionalRendering2EXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_conditional_rendering_begin: *const ConditionalRenderingBeginInfo2EXT,
    ),
>;
pub type PFN_vkCmdBindTransformFeedbackBuffers2EXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        first_binding: u32,
        binding_count: u32,
        p_binding_infos: *const BindTransformFeedbackBuffer2InfoEXT,
    ),
>;
pub type PFN_vkCmdBeginTransformFeedback2EXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        first_counter_range: u32,
        counter_range_count: u32,
        p_counter_infos: *const BindTransformFeedbackBuffer2InfoEXT,
    ),
>;
pub type PFN_vkCmdEndTransformFeedback2EXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        first_counter_range: u32,
        counter_range_count: u32,
        p_counter_infos: *const BindTransformFeedbackBuffer2InfoEXT,
    ),
>;
pub type PFN_vkCmdDrawIndirectByteCount2EXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        instance_count: u32,
        first_instance: u32,
        p_counter_info: *const BindTransformFeedbackBuffer2InfoEXT,
        counter_offset: u32,
        vertex_stride: u32,
    ),
>;
pub type PFN_vkCmdWriteMarkerToMemoryAMD = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_info: *const MemoryMarkerInfoAMD,
    ),
>;
pub type PFN_vkCmdBindIndexBuffer3KHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_info: *const BindIndexBuffer3InfoKHR,
    ),
>;
pub type PFN_vkCmdBindVertexBuffers3KHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        first_binding: u32,
        binding_count: u32,
        p_binding_infos: *const BindVertexBuffer3InfoKHR,
    ),
>;
pub type PFN_vkCmdDrawIndirect2KHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_info: *const DrawIndirect2InfoKHR,
    ),
>;
pub type PFN_vkCmdDrawIndexedIndirect2KHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_info: *const DrawIndirect2InfoKHR,
    ),
>;
pub type PFN_vkCmdDrawIndirectCount2KHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_info: *const DrawIndirectCount2InfoKHR,
    ),
>;
pub type PFN_vkCmdDrawIndexedIndirectCount2KHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_info: *const DrawIndirectCount2InfoKHR,
    ),
>;
pub type PFN_vkCmdDrawMeshTasksIndirect2EXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_info: *const DrawIndirect2InfoKHR,
    ),
>;
pub type PFN_vkCmdDrawMeshTasksIndirectCount2EXT = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_info: *const DrawIndirectCount2InfoKHR,
    ),
>;
pub type PFN_vkCmdDispatchIndirect2KHR = Option<
    unsafe extern "system" fn(
        command_buffer: CommandBuffer,
        p_info: *const DispatchIndirect2InfoKHR,
    ),
>;
pub type PFN_vkCreateAccelerationStructure2KHR = Option<
    unsafe extern "system" fn(
        device: Device,
        p_create_info: *const AccelerationStructureCreateInfo2KHR,
        p_allocator: *const AllocationCallbacks,
        p_acceleration_structure: *mut AccelerationStructureKHR,
    ) -> Result,
>;
pub type PFN_vkGetPhysicalDeviceFeatures2KHR = PFN_vkGetPhysicalDeviceFeatures2;
pub type PFN_vkCmdSetFrontFaceEXT = PFN_vkCmdSetFrontFace;
pub type PFN_vkDestroySamplerYcbcrConversionKHR = PFN_vkDestroySamplerYcbcrConversion;
pub type PFN_vkCmdSetDepthBiasEnableEXT = PFN_vkCmdSetDepthBiasEnable;
pub type PFN_vkResetQueryPoolEXT = PFN_vkResetQueryPool;
pub type PFN_vkCmdCopyImageToBuffer2KHR = PFN_vkCmdCopyImageToBuffer2;
pub type PFN_vkCmdSetScissorWithCountEXT = PFN_vkCmdSetScissorWithCount;
pub type PFN_vkCmdCopyBufferToImage2KHR = PFN_vkCmdCopyBufferToImage2;
pub type PFN_vkGetImageSubresourceLayout2KHR = PFN_vkGetImageSubresourceLayout2;
pub type PFN_vkCmdSetCullModeEXT = PFN_vkCmdSetCullMode;
pub type PFN_vkEnumeratePhysicalDeviceGroupsKHR = PFN_vkEnumeratePhysicalDeviceGroups;
pub type PFN_vkGetBufferDeviceAddressKHR = PFN_vkGetBufferDeviceAddress;
pub type PFN_vkBindImageMemory2KHR = PFN_vkBindImageMemory2;
pub type PFN_vkGetSemaphoreCounterValueKHR = PFN_vkGetSemaphoreCounterValue;
pub type PFN_vkGetImageMemoryRequirements2KHR = PFN_vkGetImageMemoryRequirements2;
pub type PFN_vkCmdSetViewportWithCountEXT = PFN_vkCmdSetViewportWithCount;
pub type PFN_vkCmdResetEvent2KHR = PFN_vkCmdResetEvent2;
pub type PFN_vkCopyImageToImageEXT = PFN_vkCopyImageToImage;
pub type PFN_vkGetPhysicalDeviceMemoryProperties2KHR = PFN_vkGetPhysicalDeviceMemoryProperties2;
pub type PFN_vkCmdSetStencilTestEnableEXT = PFN_vkCmdSetStencilTestEnable;
pub type PFN_vkCmdSetPrimitiveRestartEnableEXT = PFN_vkCmdSetPrimitiveRestartEnable;
pub type PFN_vkGetPhysicalDeviceExternalSemaphorePropertiesKHR = PFN_vkGetPhysicalDeviceExternalSemaphoreProperties;
pub type PFN_vkGetDeviceBufferMemoryRequirementsKHR = PFN_vkGetDeviceBufferMemoryRequirements;
pub type PFN_vkReleaseSwapchainImagesEXT = PFN_vkReleaseSwapchainImagesKHR;
pub type PFN_vkCmdPushDescriptorSetWithTemplateKHR = PFN_vkCmdPushDescriptorSetWithTemplate;
pub type PFN_vkMapMemory2KHR = PFN_vkMapMemory2;
pub type PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR = PFN_vkGetPhysicalDeviceExternalBufferProperties;
pub type PFN_vkCreateRenderPass2KHR = PFN_vkCreateRenderPass2;
pub type PFN_vkDestroyPrivateDataSlotEXT = PFN_vkDestroyPrivateDataSlot;
pub type PFN_vkCmdSetRenderingInputAttachmentIndicesKHR = PFN_vkCmdSetRenderingInputAttachmentIndices;
pub type PFN_vkCmdSetDepthCompareOpEXT = PFN_vkCmdSetDepthCompareOp;
pub type PFN_vkCmdCopyBuffer2KHR = PFN_vkCmdCopyBuffer2;
pub type PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT = PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR;
pub type PFN_vkCmdSetEvent2KHR = PFN_vkCmdSetEvent2;
pub type PFN_vkCmdEndRenderPass2KHR = PFN_vkCmdEndRenderPass2;
pub type PFN_vkCreatePrivateDataSlotEXT = PFN_vkCreatePrivateDataSlot;
pub type PFN_vkTransitionImageLayoutEXT = PFN_vkTransitionImageLayout;
pub type PFN_vkUnmapMemory2KHR = PFN_vkUnmapMemory2;
pub type PFN_vkQueueSubmit2KHR = PFN_vkQueueSubmit2;
pub type PFN_vkGetDeviceImageSubresourceLayoutKHR = PFN_vkGetDeviceImageSubresourceLayout;
pub type PFN_vkCmdSetLineStippleEXT = PFN_vkCmdSetLineStipple;
pub type PFN_vkCmdSetDepthWriteEnableEXT = PFN_vkCmdSetDepthWriteEnable;
pub type PFN_vkGetPhysicalDeviceSparseImageFormatProperties2KHR = PFN_vkGetPhysicalDeviceSparseImageFormatProperties2;
pub type PFN_vkCmdBeginRenderPass2KHR = PFN_vkCmdBeginRenderPass2;
pub type PFN_vkCmdNextSubpass2KHR = PFN_vkCmdNextSubpass2;
pub type PFN_vkCmdSetDepthBoundsTestEnableEXT = PFN_vkCmdSetDepthBoundsTestEnable;
pub type PFN_vkCmdResolveImage2KHR = PFN_vkCmdResolveImage2;
pub type PFN_vkGetPhysicalDeviceFormatProperties2KHR = PFN_vkGetPhysicalDeviceFormatProperties2;
pub type PFN_vkGetDescriptorSetLayoutSupportKHR = PFN_vkGetDescriptorSetLayoutSupport;
pub type PFN_vkBindBufferMemory2KHR = PFN_vkBindBufferMemory2;
pub type PFN_vkCmdDrawIndexedIndirectCountAMD = PFN_vkCmdDrawIndexedIndirectCount;
pub type PFN_vkCmdBindDescriptorSets2KHR = PFN_vkCmdBindDescriptorSets2;
pub type PFN_vkGetDeviceMemoryOpaqueCaptureAddressKHR = PFN_vkGetDeviceMemoryOpaqueCaptureAddress;
pub type PFN_vkGetPhysicalDeviceToolPropertiesEXT = PFN_vkGetPhysicalDeviceToolProperties;
pub type PFN_vkCmdDrawIndirectCountAMD = PFN_vkCmdDrawIndirectCount;
pub type PFN_vkWaitSemaphoresKHR = PFN_vkWaitSemaphores;
pub type PFN_vkCreateSamplerYcbcrConversionKHR = PFN_vkCreateSamplerYcbcrConversion;
pub type PFN_vkCmdBindVertexBuffers2EXT = PFN_vkCmdBindVertexBuffers2;
pub type PFN_vkCmdWriteTimestamp2KHR = PFN_vkCmdWriteTimestamp2;
pub type PFN_vkCmdPushDescriptorSetKHR = PFN_vkCmdPushDescriptorSet;
pub type PFN_vkCmdBlitImage2KHR = PFN_vkCmdBlitImage2;
pub type PFN_vkCmdSetPrimitiveTopologyEXT = PFN_vkCmdSetPrimitiveTopology;
pub type PFN_vkCmdSetRasterizerDiscardEnableEXT = PFN_vkCmdSetRasterizerDiscardEnable;
pub type PFN_vkCmdEndRendering2EXT = PFN_vkCmdEndRendering2KHR;
pub type PFN_vkCmdPushDescriptorSetWithTemplate2KHR = PFN_vkCmdPushDescriptorSetWithTemplate2;
pub type PFN_vkCmdDispatchBaseKHR = PFN_vkCmdDispatchBase;
pub type PFN_vkGetPrivateDataEXT = PFN_vkGetPrivateData;
pub type PFN_vkCmdEndRenderingKHR = PFN_vkCmdEndRendering;
pub type PFN_vkCopyImageToMemoryEXT = PFN_vkCopyImageToMemory;
pub type PFN_vkGetPhysicalDeviceProperties2KHR = PFN_vkGetPhysicalDeviceProperties2;
pub type PFN_vkGetImageSubresourceLayout2EXT = PFN_vkGetImageSubresourceLayout2;
pub type PFN_vkCmdWaitEvents2KHR = PFN_vkCmdWaitEvents2;
pub type PFN_vkCreateDescriptorUpdateTemplateKHR = PFN_vkCreateDescriptorUpdateTemplate;
pub type PFN_vkGetBufferDeviceAddressEXT = PFN_vkGetBufferDeviceAddress;
pub type PFN_vkGetRayTracingShaderGroupHandlesNV = PFN_vkGetRayTracingShaderGroupHandlesKHR;
pub type PFN_vkGetRenderingAreaGranularityKHR = PFN_vkGetRenderingAreaGranularity;
pub type PFN_vkDestroyDescriptorUpdateTemplateKHR = PFN_vkDestroyDescriptorUpdateTemplate;
pub type PFN_vkGetDeviceGroupPeerMemoryFeaturesKHR = PFN_vkGetDeviceGroupPeerMemoryFeatures;
pub type PFN_vkCmdSetStencilOpEXT = PFN_vkCmdSetStencilOp;
pub type PFN_vkGetDeviceImageSparseMemoryRequirementsKHR = PFN_vkGetDeviceImageSparseMemoryRequirements;
pub type PFN_vkCmdBeginRenderingKHR = PFN_vkCmdBeginRendering;
pub type PFN_vkCmdPushConstants2KHR = PFN_vkCmdPushConstants2;
pub type PFN_vkCmdPushDescriptorSet2KHR = PFN_vkCmdPushDescriptorSet2;
pub type PFN_vkGetDeviceImageMemoryRequirementsKHR = PFN_vkGetDeviceImageMemoryRequirements;
pub type PFN_vkGetPhysicalDeviceImageFormatProperties2KHR = PFN_vkGetPhysicalDeviceImageFormatProperties2;
pub type PFN_vkCmdPipelineBarrier2KHR = PFN_vkCmdPipelineBarrier2;
pub type PFN_vkGetBufferOpaqueCaptureAddressKHR = PFN_vkGetBufferOpaqueCaptureAddress;
pub type PFN_vkGetBufferMemoryRequirements2KHR = PFN_vkGetBufferMemoryRequirements2;
pub type PFN_vkCmdSetDeviceMaskKHR = PFN_vkCmdSetDeviceMask;
pub type PFN_vkGetCalibratedTimestampsEXT = PFN_vkGetCalibratedTimestampsKHR;
pub type PFN_vkCopyMemoryToImageEXT = PFN_vkCopyMemoryToImage;
pub type PFN_vkGetImageSparseMemoryRequirements2KHR = PFN_vkGetImageSparseMemoryRequirements2;
pub type PFN_vkGetPhysicalDeviceExternalFencePropertiesKHR = PFN_vkGetPhysicalDeviceExternalFenceProperties;
pub type PFN_vkCmdSetRenderingAttachmentLocationsKHR = PFN_vkCmdSetRenderingAttachmentLocations;
pub type PFN_vkCmdSetLineStippleKHR = PFN_vkCmdSetLineStipple;
pub type PFN_vkSignalSemaphoreKHR = PFN_vkSignalSemaphore;
pub type PFN_vkUpdateDescriptorSetWithTemplateKHR = PFN_vkUpdateDescriptorSetWithTemplate;
pub type PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR = PFN_vkGetPhysicalDeviceQueueFamilyProperties2;
pub type PFN_vkCmdSetDepthTestEnableEXT = PFN_vkCmdSetDepthTestEnable;
pub type PFN_vkTrimCommandPoolKHR = PFN_vkTrimCommandPool;
pub type PFN_vkCmdDrawIndirectCountKHR = PFN_vkCmdDrawIndirectCount;
pub type PFN_vkCmdDrawIndexedIndirectCountKHR = PFN_vkCmdDrawIndexedIndirectCount;
pub type PFN_vkCmdBindIndexBuffer2KHR = PFN_vkCmdBindIndexBuffer2;
pub type PFN_vkSetPrivateDataEXT = PFN_vkSetPrivateData;
pub type PFN_vkCmdCopyImage2KHR = PFN_vkCmdCopyImage2;
pub struct EntryCommands {
    pub create_instance: PFN_vkCreateInstance,
    pub enumerate_instance_version: PFN_vkEnumerateInstanceVersion,
    pub enumerate_instance_layer_properties: PFN_vkEnumerateInstanceLayerProperties,
    pub enumerate_instance_extension_properties: PFN_vkEnumerateInstanceExtensionProperties,
    pub get_external_compute_queue_data_nv: PFN_vkGetExternalComputeQueueDataNV,
}
impl Default for EntryCommands {
    #[inline]
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}
impl EntryCommands {
    /// Load all function pointers from the given loader callback.
    ///
    /// Load all function pointers from the given loader callback.
    ///
    /// # Safety
    ///
    /// The loader must return valid function pointers compatible with
    /// each command's signature, or null for unavailable commands.
    pub unsafe fn load(
        mut f: impl FnMut(&std::ffi::CStr) -> *const std::ffi::c_void,
    ) -> Self {
        unsafe {
            let mut cmd = Self::default();
            cmd.create_instance = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCreateInstance\0")),
            );
            cmd.enumerate_instance_version = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkEnumerateInstanceVersion\0",
                    ),
                ),
            );
            cmd.enumerate_instance_layer_properties = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkEnumerateInstanceLayerProperties\0",
                    ),
                ),
            );
            cmd.enumerate_instance_extension_properties = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkEnumerateInstanceExtensionProperties\0",
                    ),
                ),
            );
            cmd.get_external_compute_queue_data_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetExternalComputeQueueDataNV\0",
                    ),
                ),
            );
            cmd
        }
    }
}
pub struct InstanceCommands {
    pub destroy_instance: PFN_vkDestroyInstance,
    pub enumerate_physical_devices: PFN_vkEnumeratePhysicalDevices,
    pub get_instance_proc_addr: PFN_vkGetInstanceProcAddr,
    pub get_physical_device_properties: PFN_vkGetPhysicalDeviceProperties,
    pub get_physical_device_queue_family_properties: PFN_vkGetPhysicalDeviceQueueFamilyProperties,
    pub get_physical_device_memory_properties: PFN_vkGetPhysicalDeviceMemoryProperties,
    pub get_physical_device_features: PFN_vkGetPhysicalDeviceFeatures,
    pub get_physical_device_format_properties: PFN_vkGetPhysicalDeviceFormatProperties,
    pub get_physical_device_image_format_properties: PFN_vkGetPhysicalDeviceImageFormatProperties,
    pub create_device: PFN_vkCreateDevice,
    pub enumerate_device_layer_properties: PFN_vkEnumerateDeviceLayerProperties,
    pub enumerate_device_extension_properties: PFN_vkEnumerateDeviceExtensionProperties,
    pub get_physical_device_sparse_image_format_properties: PFN_vkGetPhysicalDeviceSparseImageFormatProperties,
    pub create_android_surface_khr: PFN_vkCreateAndroidSurfaceKHR,
    pub create_surface_ohos: PFN_vkCreateSurfaceOHOS,
    pub get_physical_device_display_properties_khr: PFN_vkGetPhysicalDeviceDisplayPropertiesKHR,
    pub get_physical_device_display_plane_properties_khr: PFN_vkGetPhysicalDeviceDisplayPlanePropertiesKHR,
    pub get_display_plane_supported_displays_khr: PFN_vkGetDisplayPlaneSupportedDisplaysKHR,
    pub get_display_mode_properties_khr: PFN_vkGetDisplayModePropertiesKHR,
    pub create_display_mode_khr: PFN_vkCreateDisplayModeKHR,
    pub get_display_plane_capabilities_khr: PFN_vkGetDisplayPlaneCapabilitiesKHR,
    pub create_display_plane_surface_khr: PFN_vkCreateDisplayPlaneSurfaceKHR,
    pub destroy_surface_khr: PFN_vkDestroySurfaceKHR,
    pub get_physical_device_surface_support_khr: PFN_vkGetPhysicalDeviceSurfaceSupportKHR,
    pub get_physical_device_surface_capabilities_khr: PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR,
    pub get_physical_device_surface_formats_khr: PFN_vkGetPhysicalDeviceSurfaceFormatsKHR,
    pub get_physical_device_surface_present_modes_khr: PFN_vkGetPhysicalDeviceSurfacePresentModesKHR,
    pub create_vi_surface_nn: PFN_vkCreateViSurfaceNN,
    pub create_wayland_surface_khr: PFN_vkCreateWaylandSurfaceKHR,
    pub get_physical_device_wayland_presentation_support_khr: PFN_vkGetPhysicalDeviceWaylandPresentationSupportKHR,
    pub create_ubm_surface_sec: PFN_vkCreateUbmSurfaceSEC,
    pub get_physical_device_ubm_presentation_support_sec: PFN_vkGetPhysicalDeviceUbmPresentationSupportSEC,
    pub create_win32_surface_khr: PFN_vkCreateWin32SurfaceKHR,
    pub get_physical_device_win32_presentation_support_khr: PFN_vkGetPhysicalDeviceWin32PresentationSupportKHR,
    pub create_xlib_surface_khr: PFN_vkCreateXlibSurfaceKHR,
    pub get_physical_device_xlib_presentation_support_khr: PFN_vkGetPhysicalDeviceXlibPresentationSupportKHR,
    pub create_xcb_surface_khr: PFN_vkCreateXcbSurfaceKHR,
    pub get_physical_device_xcb_presentation_support_khr: PFN_vkGetPhysicalDeviceXcbPresentationSupportKHR,
    pub create_direct_fb_surface_ext: PFN_vkCreateDirectFBSurfaceEXT,
    pub get_physical_device_direct_fb_presentation_support_ext: PFN_vkGetPhysicalDeviceDirectFBPresentationSupportEXT,
    pub create_image_pipe_surface_fuchsia: PFN_vkCreateImagePipeSurfaceFUCHSIA,
    pub create_stream_descriptor_surface_ggp: PFN_vkCreateStreamDescriptorSurfaceGGP,
    pub create_screen_surface_qnx: PFN_vkCreateScreenSurfaceQNX,
    pub get_physical_device_screen_presentation_support_qnx: PFN_vkGetPhysicalDeviceScreenPresentationSupportQNX,
    pub create_debug_report_callback_ext: PFN_vkCreateDebugReportCallbackEXT,
    pub destroy_debug_report_callback_ext: PFN_vkDestroyDebugReportCallbackEXT,
    pub debug_report_message_ext: PFN_vkDebugReportMessageEXT,
    pub get_physical_device_external_image_format_properties_nv: PFN_vkGetPhysicalDeviceExternalImageFormatPropertiesNV,
    pub get_physical_device_features2: PFN_vkGetPhysicalDeviceFeatures2,
    pub get_physical_device_properties2: PFN_vkGetPhysicalDeviceProperties2,
    pub get_physical_device_format_properties2: PFN_vkGetPhysicalDeviceFormatProperties2,
    pub get_physical_device_image_format_properties2: PFN_vkGetPhysicalDeviceImageFormatProperties2,
    pub get_physical_device_queue_family_properties2: PFN_vkGetPhysicalDeviceQueueFamilyProperties2,
    pub get_physical_device_memory_properties2: PFN_vkGetPhysicalDeviceMemoryProperties2,
    pub get_physical_device_sparse_image_format_properties2: PFN_vkGetPhysicalDeviceSparseImageFormatProperties2,
    pub get_physical_device_external_buffer_properties: PFN_vkGetPhysicalDeviceExternalBufferProperties,
    pub get_physical_device_external_memory_sci_buf_properties_nv: PFN_vkGetPhysicalDeviceExternalMemorySciBufPropertiesNV,
    pub get_physical_device_sci_buf_attributes_nv: PFN_vkGetPhysicalDeviceSciBufAttributesNV,
    pub get_physical_device_external_semaphore_properties: PFN_vkGetPhysicalDeviceExternalSemaphoreProperties,
    pub get_physical_device_external_fence_properties: PFN_vkGetPhysicalDeviceExternalFenceProperties,
    pub get_physical_device_sci_sync_attributes_nv: PFN_vkGetPhysicalDeviceSciSyncAttributesNV,
    pub release_display_ext: PFN_vkReleaseDisplayEXT,
    pub acquire_xlib_display_ext: PFN_vkAcquireXlibDisplayEXT,
    pub get_rand_r_output_display_ext: PFN_vkGetRandROutputDisplayEXT,
    pub acquire_winrt_display_nv: PFN_vkAcquireWinrtDisplayNV,
    pub get_winrt_display_nv: PFN_vkGetWinrtDisplayNV,
    pub get_physical_device_surface_capabilities2_ext: PFN_vkGetPhysicalDeviceSurfaceCapabilities2EXT,
    pub enumerate_physical_device_groups: PFN_vkEnumeratePhysicalDeviceGroups,
    pub get_physical_device_present_rectangles_khr: PFN_vkGetPhysicalDevicePresentRectanglesKHR,
    pub create_ios_surface_mvk: PFN_vkCreateIOSSurfaceMVK,
    pub create_mac_os_surface_mvk: PFN_vkCreateMacOSSurfaceMVK,
    pub create_metal_surface_ext: PFN_vkCreateMetalSurfaceEXT,
    pub get_physical_device_multisample_properties_ext: PFN_vkGetPhysicalDeviceMultisamplePropertiesEXT,
    pub get_physical_device_surface_capabilities2_khr: PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR,
    pub get_physical_device_surface_formats2_khr: PFN_vkGetPhysicalDeviceSurfaceFormats2KHR,
    pub get_physical_device_display_properties2_khr: PFN_vkGetPhysicalDeviceDisplayProperties2KHR,
    pub get_physical_device_display_plane_properties2_khr: PFN_vkGetPhysicalDeviceDisplayPlaneProperties2KHR,
    pub get_display_mode_properties2_khr: PFN_vkGetDisplayModeProperties2KHR,
    pub get_display_plane_capabilities2_khr: PFN_vkGetDisplayPlaneCapabilities2KHR,
    pub get_physical_device_calibrateable_time_domains_khr: PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR,
    pub create_debug_utils_messenger_ext: PFN_vkCreateDebugUtilsMessengerEXT,
    pub destroy_debug_utils_messenger_ext: PFN_vkDestroyDebugUtilsMessengerEXT,
    pub submit_debug_utils_message_ext: PFN_vkSubmitDebugUtilsMessageEXT,
    pub get_physical_device_cooperative_matrix_properties_nv: PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesNV,
    pub get_physical_device_surface_present_modes2_ext: PFN_vkGetPhysicalDeviceSurfacePresentModes2EXT,
    pub enumerate_physical_device_queue_family_performance_query_counters_khr: PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR,
    pub get_physical_device_queue_family_performance_query_passes_khr: PFN_vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR,
    pub create_headless_surface_ext: PFN_vkCreateHeadlessSurfaceEXT,
    pub get_physical_device_supported_framebuffer_mixed_samples_combinations_nv: PFN_vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV,
    pub get_physical_device_tool_properties: PFN_vkGetPhysicalDeviceToolProperties,
    pub get_physical_device_refreshable_object_types_khr: PFN_vkGetPhysicalDeviceRefreshableObjectTypesKHR,
    pub get_physical_device_fragment_shading_rates_khr: PFN_vkGetPhysicalDeviceFragmentShadingRatesKHR,
    pub get_physical_device_video_capabilities_khr: PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR,
    pub get_physical_device_video_format_properties_khr: PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR,
    pub get_physical_device_video_encode_quality_level_properties_khr: PFN_vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR,
    pub acquire_drm_display_ext: PFN_vkAcquireDrmDisplayEXT,
    pub get_drm_display_ext: PFN_vkGetDrmDisplayEXT,
    pub get_physical_device_optical_flow_image_formats_nv: PFN_vkGetPhysicalDeviceOpticalFlowImageFormatsNV,
    pub get_physical_device_cooperative_matrix_properties_khr: PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR,
    pub get_physical_device_cooperative_matrix_flexible_dimensions_properties_nv: PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV,
    pub get_physical_device_cooperative_vector_properties_nv: PFN_vkGetPhysicalDeviceCooperativeVectorPropertiesNV,
    pub enumerate_physical_device_shader_instrumentation_metrics_arm: PFN_vkEnumeratePhysicalDeviceShaderInstrumentationMetricsARM,
    pub get_physical_device_external_tensor_properties_arm: PFN_vkGetPhysicalDeviceExternalTensorPropertiesARM,
    pub get_physical_device_queue_family_data_graph_properties_arm: PFN_vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM,
    pub get_physical_device_queue_family_data_graph_processing_engine_properties_arm: PFN_vkGetPhysicalDeviceQueueFamilyDataGraphProcessingEnginePropertiesARM,
    pub enumerate_physical_device_queue_family_performance_counters_by_region_arm: PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceCountersByRegionARM,
    pub get_physical_device_descriptor_size_ext: PFN_vkGetPhysicalDeviceDescriptorSizeEXT,
}
impl Default for InstanceCommands {
    #[inline]
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}
impl InstanceCommands {
    /// Load all function pointers from the given loader callback.
    ///
    /// Load all function pointers from the given loader callback.
    ///
    /// # Safety
    ///
    /// The loader must return valid function pointers compatible with
    /// each command's signature, or null for unavailable commands.
    pub unsafe fn load(
        mut f: impl FnMut(&std::ffi::CStr) -> *const std::ffi::c_void,
    ) -> Self {
        unsafe {
            let mut cmd = Self::default();
            cmd.destroy_instance = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkDestroyInstance\0")),
            );
            cmd.enumerate_physical_devices = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkEnumeratePhysicalDevices\0",
                    ),
                ),
            );
            cmd.get_instance_proc_addr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetInstanceProcAddr\0",
                    ),
                ),
            );
            cmd.get_physical_device_properties = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceProperties\0",
                    ),
                ),
            );
            cmd.get_physical_device_queue_family_properties = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceQueueFamilyProperties\0",
                    ),
                ),
            );
            cmd.get_physical_device_memory_properties = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceMemoryProperties\0",
                    ),
                ),
            );
            cmd.get_physical_device_features = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceFeatures\0",
                    ),
                ),
            );
            cmd.get_physical_device_format_properties = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceFormatProperties\0",
                    ),
                ),
            );
            cmd.get_physical_device_image_format_properties = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceImageFormatProperties\0",
                    ),
                ),
            );
            cmd.create_device = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCreateDevice\0")),
            );
            cmd.enumerate_device_layer_properties = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkEnumerateDeviceLayerProperties\0",
                    ),
                ),
            );
            cmd.enumerate_device_extension_properties = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkEnumerateDeviceExtensionProperties\0",
                    ),
                ),
            );
            cmd.get_physical_device_sparse_image_format_properties = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceSparseImageFormatProperties\0",
                    ),
                ),
            );
            cmd.create_android_surface_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateAndroidSurfaceKHR\0",
                    ),
                ),
            );
            cmd.create_surface_ohos = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateSurfaceOHOS\0",
                    ),
                ),
            );
            cmd.get_physical_device_display_properties_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceDisplayPropertiesKHR\0",
                    ),
                ),
            );
            cmd.get_physical_device_display_plane_properties_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceDisplayPlanePropertiesKHR\0",
                    ),
                ),
            );
            cmd.get_display_plane_supported_displays_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDisplayPlaneSupportedDisplaysKHR\0",
                    ),
                ),
            );
            cmd.get_display_mode_properties_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDisplayModePropertiesKHR\0",
                    ),
                ),
            );
            cmd.create_display_mode_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateDisplayModeKHR\0",
                    ),
                ),
            );
            cmd.get_display_plane_capabilities_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDisplayPlaneCapabilitiesKHR\0",
                    ),
                ),
            );
            cmd.create_display_plane_surface_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateDisplayPlaneSurfaceKHR\0",
                    ),
                ),
            );
            cmd.destroy_surface_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroySurfaceKHR\0",
                    ),
                ),
            );
            cmd.get_physical_device_surface_support_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceSurfaceSupportKHR\0",
                    ),
                ),
            );
            cmd.get_physical_device_surface_capabilities_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceSurfaceCapabilitiesKHR\0",
                    ),
                ),
            );
            cmd.get_physical_device_surface_formats_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceSurfaceFormatsKHR\0",
                    ),
                ),
            );
            cmd.get_physical_device_surface_present_modes_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceSurfacePresentModesKHR\0",
                    ),
                ),
            );
            cmd.create_vi_surface_nn = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateViSurfaceNN\0",
                    ),
                ),
            );
            cmd.create_wayland_surface_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateWaylandSurfaceKHR\0",
                    ),
                ),
            );
            cmd.get_physical_device_wayland_presentation_support_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceWaylandPresentationSupportKHR\0",
                    ),
                ),
            );
            cmd.create_ubm_surface_sec = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateUbmSurfaceSEC\0",
                    ),
                ),
            );
            cmd.get_physical_device_ubm_presentation_support_sec = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceUbmPresentationSupportSEC\0",
                    ),
                ),
            );
            cmd.create_win32_surface_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateWin32SurfaceKHR\0",
                    ),
                ),
            );
            cmd.get_physical_device_win32_presentation_support_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceWin32PresentationSupportKHR\0",
                    ),
                ),
            );
            cmd.create_xlib_surface_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateXlibSurfaceKHR\0",
                    ),
                ),
            );
            cmd.get_physical_device_xlib_presentation_support_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceXlibPresentationSupportKHR\0",
                    ),
                ),
            );
            cmd.create_xcb_surface_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateXcbSurfaceKHR\0",
                    ),
                ),
            );
            cmd.get_physical_device_xcb_presentation_support_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceXcbPresentationSupportKHR\0",
                    ),
                ),
            );
            cmd.create_direct_fb_surface_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateDirectFBSurfaceEXT\0",
                    ),
                ),
            );
            cmd.get_physical_device_direct_fb_presentation_support_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceDirectFBPresentationSupportEXT\0",
                    ),
                ),
            );
            cmd.create_image_pipe_surface_fuchsia = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateImagePipeSurfaceFUCHSIA\0",
                    ),
                ),
            );
            cmd.create_stream_descriptor_surface_ggp = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateStreamDescriptorSurfaceGGP\0",
                    ),
                ),
            );
            cmd.create_screen_surface_qnx = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateScreenSurfaceQNX\0",
                    ),
                ),
            );
            cmd.get_physical_device_screen_presentation_support_qnx = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceScreenPresentationSupportQNX\0",
                    ),
                ),
            );
            cmd.create_debug_report_callback_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateDebugReportCallbackEXT\0",
                    ),
                ),
            );
            cmd.destroy_debug_report_callback_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyDebugReportCallbackEXT\0",
                    ),
                ),
            );
            cmd.debug_report_message_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDebugReportMessageEXT\0",
                    ),
                ),
            );
            cmd.get_physical_device_external_image_format_properties_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceExternalImageFormatPropertiesNV\0",
                    ),
                ),
            );
            cmd.get_physical_device_features2 = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceFeatures2\0",
                    ),
                ),
            );
            if cmd.get_physical_device_features2.is_none() {
                cmd.get_physical_device_features2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetPhysicalDeviceFeatures2KHR\0",
                        ),
                    ),
                );
            }
            cmd.get_physical_device_properties2 = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceProperties2\0",
                    ),
                ),
            );
            if cmd.get_physical_device_properties2.is_none() {
                cmd.get_physical_device_properties2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetPhysicalDeviceProperties2KHR\0",
                        ),
                    ),
                );
            }
            cmd.get_physical_device_format_properties2 = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceFormatProperties2\0",
                    ),
                ),
            );
            if cmd.get_physical_device_format_properties2.is_none() {
                cmd.get_physical_device_format_properties2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetPhysicalDeviceFormatProperties2KHR\0",
                        ),
                    ),
                );
            }
            cmd.get_physical_device_image_format_properties2 = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceImageFormatProperties2\0",
                    ),
                ),
            );
            if cmd.get_physical_device_image_format_properties2.is_none() {
                cmd.get_physical_device_image_format_properties2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetPhysicalDeviceImageFormatProperties2KHR\0",
                        ),
                    ),
                );
            }
            cmd.get_physical_device_queue_family_properties2 = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceQueueFamilyProperties2\0",
                    ),
                ),
            );
            if cmd.get_physical_device_queue_family_properties2.is_none() {
                cmd.get_physical_device_queue_family_properties2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetPhysicalDeviceQueueFamilyProperties2KHR\0",
                        ),
                    ),
                );
            }
            cmd.get_physical_device_memory_properties2 = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceMemoryProperties2\0",
                    ),
                ),
            );
            if cmd.get_physical_device_memory_properties2.is_none() {
                cmd.get_physical_device_memory_properties2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetPhysicalDeviceMemoryProperties2KHR\0",
                        ),
                    ),
                );
            }
            cmd.get_physical_device_sparse_image_format_properties2 = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceSparseImageFormatProperties2\0",
                    ),
                ),
            );
            if cmd.get_physical_device_sparse_image_format_properties2.is_none() {
                cmd.get_physical_device_sparse_image_format_properties2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetPhysicalDeviceSparseImageFormatProperties2KHR\0",
                        ),
                    ),
                );
            }
            cmd.get_physical_device_external_buffer_properties = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceExternalBufferProperties\0",
                    ),
                ),
            );
            if cmd.get_physical_device_external_buffer_properties.is_none() {
                cmd.get_physical_device_external_buffer_properties = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetPhysicalDeviceExternalBufferPropertiesKHR\0",
                        ),
                    ),
                );
            }
            cmd.get_physical_device_external_memory_sci_buf_properties_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceExternalMemorySciBufPropertiesNV\0",
                    ),
                ),
            );
            cmd.get_physical_device_sci_buf_attributes_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceSciBufAttributesNV\0",
                    ),
                ),
            );
            cmd.get_physical_device_external_semaphore_properties = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceExternalSemaphoreProperties\0",
                    ),
                ),
            );
            if cmd.get_physical_device_external_semaphore_properties.is_none() {
                cmd.get_physical_device_external_semaphore_properties = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetPhysicalDeviceExternalSemaphorePropertiesKHR\0",
                        ),
                    ),
                );
            }
            cmd.get_physical_device_external_fence_properties = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceExternalFenceProperties\0",
                    ),
                ),
            );
            if cmd.get_physical_device_external_fence_properties.is_none() {
                cmd.get_physical_device_external_fence_properties = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetPhysicalDeviceExternalFencePropertiesKHR\0",
                        ),
                    ),
                );
            }
            cmd.get_physical_device_sci_sync_attributes_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceSciSyncAttributesNV\0",
                    ),
                ),
            );
            cmd.release_display_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkReleaseDisplayEXT\0",
                    ),
                ),
            );
            cmd.acquire_xlib_display_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkAcquireXlibDisplayEXT\0",
                    ),
                ),
            );
            cmd.get_rand_r_output_display_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetRandROutputDisplayEXT\0",
                    ),
                ),
            );
            cmd.acquire_winrt_display_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkAcquireWinrtDisplayNV\0",
                    ),
                ),
            );
            cmd.get_winrt_display_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetWinrtDisplayNV\0",
                    ),
                ),
            );
            cmd.get_physical_device_surface_capabilities2_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceSurfaceCapabilities2EXT\0",
                    ),
                ),
            );
            cmd.enumerate_physical_device_groups = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkEnumeratePhysicalDeviceGroups\0",
                    ),
                ),
            );
            if cmd.enumerate_physical_device_groups.is_none() {
                cmd.enumerate_physical_device_groups = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkEnumeratePhysicalDeviceGroupsKHR\0",
                        ),
                    ),
                );
            }
            cmd.get_physical_device_present_rectangles_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDevicePresentRectanglesKHR\0",
                    ),
                ),
            );
            cmd.create_ios_surface_mvk = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateIOSSurfaceMVK\0",
                    ),
                ),
            );
            cmd.create_mac_os_surface_mvk = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateMacOSSurfaceMVK\0",
                    ),
                ),
            );
            cmd.create_metal_surface_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateMetalSurfaceEXT\0",
                    ),
                ),
            );
            cmd.get_physical_device_multisample_properties_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceMultisamplePropertiesEXT\0",
                    ),
                ),
            );
            cmd.get_physical_device_surface_capabilities2_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceSurfaceCapabilities2KHR\0",
                    ),
                ),
            );
            cmd.get_physical_device_surface_formats2_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceSurfaceFormats2KHR\0",
                    ),
                ),
            );
            cmd.get_physical_device_display_properties2_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceDisplayProperties2KHR\0",
                    ),
                ),
            );
            cmd.get_physical_device_display_plane_properties2_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceDisplayPlaneProperties2KHR\0",
                    ),
                ),
            );
            cmd.get_display_mode_properties2_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDisplayModeProperties2KHR\0",
                    ),
                ),
            );
            cmd.get_display_plane_capabilities2_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDisplayPlaneCapabilities2KHR\0",
                    ),
                ),
            );
            cmd.get_physical_device_calibrateable_time_domains_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceCalibrateableTimeDomainsKHR\0",
                    ),
                ),
            );
            if cmd.get_physical_device_calibrateable_time_domains_khr.is_none() {
                cmd.get_physical_device_calibrateable_time_domains_khr = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetPhysicalDeviceCalibrateableTimeDomainsEXT\0",
                        ),
                    ),
                );
            }
            cmd.create_debug_utils_messenger_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateDebugUtilsMessengerEXT\0",
                    ),
                ),
            );
            cmd.destroy_debug_utils_messenger_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyDebugUtilsMessengerEXT\0",
                    ),
                ),
            );
            cmd.submit_debug_utils_message_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkSubmitDebugUtilsMessageEXT\0",
                    ),
                ),
            );
            cmd.get_physical_device_cooperative_matrix_properties_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceCooperativeMatrixPropertiesNV\0",
                    ),
                ),
            );
            cmd.get_physical_device_surface_present_modes2_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceSurfacePresentModes2EXT\0",
                    ),
                ),
            );
            cmd.enumerate_physical_device_queue_family_performance_query_counters_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR\0",
                    ),
                ),
            );
            cmd.get_physical_device_queue_family_performance_query_passes_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR\0",
                    ),
                ),
            );
            cmd.create_headless_surface_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateHeadlessSurfaceEXT\0",
                    ),
                ),
            );
            cmd
                .get_physical_device_supported_framebuffer_mixed_samples_combinations_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV\0",
                    ),
                ),
            );
            cmd.get_physical_device_tool_properties = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceToolProperties\0",
                    ),
                ),
            );
            if cmd.get_physical_device_tool_properties.is_none() {
                cmd.get_physical_device_tool_properties = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetPhysicalDeviceToolPropertiesEXT\0",
                        ),
                    ),
                );
            }
            cmd.get_physical_device_refreshable_object_types_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceRefreshableObjectTypesKHR\0",
                    ),
                ),
            );
            cmd.get_physical_device_fragment_shading_rates_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceFragmentShadingRatesKHR\0",
                    ),
                ),
            );
            cmd.get_physical_device_video_capabilities_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceVideoCapabilitiesKHR\0",
                    ),
                ),
            );
            cmd.get_physical_device_video_format_properties_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceVideoFormatPropertiesKHR\0",
                    ),
                ),
            );
            cmd.get_physical_device_video_encode_quality_level_properties_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR\0",
                    ),
                ),
            );
            cmd.acquire_drm_display_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkAcquireDrmDisplayEXT\0",
                    ),
                ),
            );
            cmd.get_drm_display_ext = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkGetDrmDisplayEXT\0")),
            );
            cmd.get_physical_device_optical_flow_image_formats_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceOpticalFlowImageFormatsNV\0",
                    ),
                ),
            );
            cmd.get_physical_device_cooperative_matrix_properties_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR\0",
                    ),
                ),
            );
            cmd
                .get_physical_device_cooperative_matrix_flexible_dimensions_properties_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV\0",
                    ),
                ),
            );
            cmd.get_physical_device_cooperative_vector_properties_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceCooperativeVectorPropertiesNV\0",
                    ),
                ),
            );
            cmd.enumerate_physical_device_shader_instrumentation_metrics_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkEnumeratePhysicalDeviceShaderInstrumentationMetricsARM\0",
                    ),
                ),
            );
            cmd.get_physical_device_external_tensor_properties_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceExternalTensorPropertiesARM\0",
                    ),
                ),
            );
            cmd.get_physical_device_queue_family_data_graph_properties_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM\0",
                    ),
                ),
            );
            cmd
                .get_physical_device_queue_family_data_graph_processing_engine_properties_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceQueueFamilyDataGraphProcessingEnginePropertiesARM\0",
                    ),
                ),
            );
            cmd
                .enumerate_physical_device_queue_family_performance_counters_by_region_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkEnumeratePhysicalDeviceQueueFamilyPerformanceCountersByRegionARM\0",
                    ),
                ),
            );
            cmd.get_physical_device_descriptor_size_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPhysicalDeviceDescriptorSizeEXT\0",
                    ),
                ),
            );
            cmd
        }
    }
}
pub struct DeviceCommands {
    pub get_device_proc_addr: PFN_vkGetDeviceProcAddr,
    pub destroy_device: PFN_vkDestroyDevice,
    pub get_device_queue: PFN_vkGetDeviceQueue,
    pub queue_submit: PFN_vkQueueSubmit,
    pub queue_wait_idle: PFN_vkQueueWaitIdle,
    pub device_wait_idle: PFN_vkDeviceWaitIdle,
    pub allocate_memory: PFN_vkAllocateMemory,
    pub free_memory: PFN_vkFreeMemory,
    pub map_memory: PFN_vkMapMemory,
    pub unmap_memory: PFN_vkUnmapMemory,
    pub flush_mapped_memory_ranges: PFN_vkFlushMappedMemoryRanges,
    pub invalidate_mapped_memory_ranges: PFN_vkInvalidateMappedMemoryRanges,
    pub get_device_memory_commitment: PFN_vkGetDeviceMemoryCommitment,
    pub get_buffer_memory_requirements: PFN_vkGetBufferMemoryRequirements,
    pub bind_buffer_memory: PFN_vkBindBufferMemory,
    pub get_image_memory_requirements: PFN_vkGetImageMemoryRequirements,
    pub bind_image_memory: PFN_vkBindImageMemory,
    pub get_image_sparse_memory_requirements: PFN_vkGetImageSparseMemoryRequirements,
    pub queue_bind_sparse: PFN_vkQueueBindSparse,
    pub create_fence: PFN_vkCreateFence,
    pub destroy_fence: PFN_vkDestroyFence,
    pub reset_fences: PFN_vkResetFences,
    pub get_fence_status: PFN_vkGetFenceStatus,
    pub wait_for_fences: PFN_vkWaitForFences,
    pub create_semaphore: PFN_vkCreateSemaphore,
    pub destroy_semaphore: PFN_vkDestroySemaphore,
    pub create_event: PFN_vkCreateEvent,
    pub destroy_event: PFN_vkDestroyEvent,
    pub get_event_status: PFN_vkGetEventStatus,
    pub set_event: PFN_vkSetEvent,
    pub reset_event: PFN_vkResetEvent,
    pub create_query_pool: PFN_vkCreateQueryPool,
    pub destroy_query_pool: PFN_vkDestroyQueryPool,
    pub get_query_pool_results: PFN_vkGetQueryPoolResults,
    pub reset_query_pool: PFN_vkResetQueryPool,
    pub create_buffer: PFN_vkCreateBuffer,
    pub destroy_buffer: PFN_vkDestroyBuffer,
    pub create_buffer_view: PFN_vkCreateBufferView,
    pub destroy_buffer_view: PFN_vkDestroyBufferView,
    pub create_image: PFN_vkCreateImage,
    pub destroy_image: PFN_vkDestroyImage,
    pub get_image_subresource_layout: PFN_vkGetImageSubresourceLayout,
    pub create_image_view: PFN_vkCreateImageView,
    pub destroy_image_view: PFN_vkDestroyImageView,
    pub create_shader_module: PFN_vkCreateShaderModule,
    pub destroy_shader_module: PFN_vkDestroyShaderModule,
    pub create_pipeline_cache: PFN_vkCreatePipelineCache,
    pub destroy_pipeline_cache: PFN_vkDestroyPipelineCache,
    pub get_pipeline_cache_data: PFN_vkGetPipelineCacheData,
    pub merge_pipeline_caches: PFN_vkMergePipelineCaches,
    pub create_pipeline_binaries_khr: PFN_vkCreatePipelineBinariesKHR,
    pub destroy_pipeline_binary_khr: PFN_vkDestroyPipelineBinaryKHR,
    pub get_pipeline_key_khr: PFN_vkGetPipelineKeyKHR,
    pub get_pipeline_binary_data_khr: PFN_vkGetPipelineBinaryDataKHR,
    pub release_captured_pipeline_data_khr: PFN_vkReleaseCapturedPipelineDataKHR,
    pub create_graphics_pipelines: PFN_vkCreateGraphicsPipelines,
    pub create_compute_pipelines: PFN_vkCreateComputePipelines,
    pub get_device_subpass_shading_max_workgroup_size_huawei: PFN_vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI,
    pub destroy_pipeline: PFN_vkDestroyPipeline,
    pub create_pipeline_layout: PFN_vkCreatePipelineLayout,
    pub destroy_pipeline_layout: PFN_vkDestroyPipelineLayout,
    pub create_sampler: PFN_vkCreateSampler,
    pub destroy_sampler: PFN_vkDestroySampler,
    pub create_descriptor_set_layout: PFN_vkCreateDescriptorSetLayout,
    pub destroy_descriptor_set_layout: PFN_vkDestroyDescriptorSetLayout,
    pub create_descriptor_pool: PFN_vkCreateDescriptorPool,
    pub destroy_descriptor_pool: PFN_vkDestroyDescriptorPool,
    pub reset_descriptor_pool: PFN_vkResetDescriptorPool,
    pub allocate_descriptor_sets: PFN_vkAllocateDescriptorSets,
    pub free_descriptor_sets: PFN_vkFreeDescriptorSets,
    pub update_descriptor_sets: PFN_vkUpdateDescriptorSets,
    pub create_framebuffer: PFN_vkCreateFramebuffer,
    pub destroy_framebuffer: PFN_vkDestroyFramebuffer,
    pub create_render_pass: PFN_vkCreateRenderPass,
    pub destroy_render_pass: PFN_vkDestroyRenderPass,
    pub get_render_area_granularity: PFN_vkGetRenderAreaGranularity,
    pub get_rendering_area_granularity: PFN_vkGetRenderingAreaGranularity,
    pub create_command_pool: PFN_vkCreateCommandPool,
    pub destroy_command_pool: PFN_vkDestroyCommandPool,
    pub reset_command_pool: PFN_vkResetCommandPool,
    pub allocate_command_buffers: PFN_vkAllocateCommandBuffers,
    pub free_command_buffers: PFN_vkFreeCommandBuffers,
    pub begin_command_buffer: PFN_vkBeginCommandBuffer,
    pub end_command_buffer: PFN_vkEndCommandBuffer,
    pub reset_command_buffer: PFN_vkResetCommandBuffer,
    pub cmd_bind_pipeline: PFN_vkCmdBindPipeline,
    pub cmd_set_attachment_feedback_loop_enable_ext: PFN_vkCmdSetAttachmentFeedbackLoopEnableEXT,
    pub cmd_set_viewport: PFN_vkCmdSetViewport,
    pub cmd_set_scissor: PFN_vkCmdSetScissor,
    pub cmd_set_line_width: PFN_vkCmdSetLineWidth,
    pub cmd_set_depth_bias: PFN_vkCmdSetDepthBias,
    pub cmd_set_blend_constants: PFN_vkCmdSetBlendConstants,
    pub cmd_set_depth_bounds: PFN_vkCmdSetDepthBounds,
    pub cmd_set_stencil_compare_mask: PFN_vkCmdSetStencilCompareMask,
    pub cmd_set_stencil_write_mask: PFN_vkCmdSetStencilWriteMask,
    pub cmd_set_stencil_reference: PFN_vkCmdSetStencilReference,
    pub cmd_bind_descriptor_sets: PFN_vkCmdBindDescriptorSets,
    pub cmd_bind_index_buffer: PFN_vkCmdBindIndexBuffer,
    pub cmd_bind_vertex_buffers: PFN_vkCmdBindVertexBuffers,
    pub cmd_draw: PFN_vkCmdDraw,
    pub cmd_draw_indexed: PFN_vkCmdDrawIndexed,
    pub cmd_draw_multi_ext: PFN_vkCmdDrawMultiEXT,
    pub cmd_draw_multi_indexed_ext: PFN_vkCmdDrawMultiIndexedEXT,
    pub cmd_draw_indirect: PFN_vkCmdDrawIndirect,
    pub cmd_draw_indexed_indirect: PFN_vkCmdDrawIndexedIndirect,
    pub cmd_dispatch: PFN_vkCmdDispatch,
    pub cmd_dispatch_indirect: PFN_vkCmdDispatchIndirect,
    pub cmd_subpass_shading_huawei: PFN_vkCmdSubpassShadingHUAWEI,
    pub cmd_draw_cluster_huawei: PFN_vkCmdDrawClusterHUAWEI,
    pub cmd_draw_cluster_indirect_huawei: PFN_vkCmdDrawClusterIndirectHUAWEI,
    pub cmd_update_pipeline_indirect_buffer_nv: PFN_vkCmdUpdatePipelineIndirectBufferNV,
    pub cmd_copy_buffer: PFN_vkCmdCopyBuffer,
    pub cmd_copy_image: PFN_vkCmdCopyImage,
    pub cmd_blit_image: PFN_vkCmdBlitImage,
    pub cmd_copy_buffer_to_image: PFN_vkCmdCopyBufferToImage,
    pub cmd_copy_image_to_buffer: PFN_vkCmdCopyImageToBuffer,
    pub cmd_copy_memory_indirect_nv: PFN_vkCmdCopyMemoryIndirectNV,
    pub cmd_copy_memory_indirect_khr: PFN_vkCmdCopyMemoryIndirectKHR,
    pub cmd_copy_memory_to_image_indirect_nv: PFN_vkCmdCopyMemoryToImageIndirectNV,
    pub cmd_copy_memory_to_image_indirect_khr: PFN_vkCmdCopyMemoryToImageIndirectKHR,
    pub cmd_update_buffer: PFN_vkCmdUpdateBuffer,
    pub cmd_fill_buffer: PFN_vkCmdFillBuffer,
    pub cmd_clear_color_image: PFN_vkCmdClearColorImage,
    pub cmd_clear_depth_stencil_image: PFN_vkCmdClearDepthStencilImage,
    pub cmd_clear_attachments: PFN_vkCmdClearAttachments,
    pub cmd_resolve_image: PFN_vkCmdResolveImage,
    pub cmd_set_event: PFN_vkCmdSetEvent,
    pub cmd_reset_event: PFN_vkCmdResetEvent,
    pub cmd_wait_events: PFN_vkCmdWaitEvents,
    pub cmd_pipeline_barrier: PFN_vkCmdPipelineBarrier,
    pub cmd_begin_query: PFN_vkCmdBeginQuery,
    pub cmd_end_query: PFN_vkCmdEndQuery,
    pub cmd_begin_conditional_rendering_ext: PFN_vkCmdBeginConditionalRenderingEXT,
    pub cmd_end_conditional_rendering_ext: PFN_vkCmdEndConditionalRenderingEXT,
    pub cmd_begin_custom_resolve_ext: PFN_vkCmdBeginCustomResolveEXT,
    pub cmd_reset_query_pool: PFN_vkCmdResetQueryPool,
    pub cmd_write_timestamp: PFN_vkCmdWriteTimestamp,
    pub cmd_copy_query_pool_results: PFN_vkCmdCopyQueryPoolResults,
    pub cmd_push_constants: PFN_vkCmdPushConstants,
    pub cmd_begin_render_pass: PFN_vkCmdBeginRenderPass,
    pub cmd_next_subpass: PFN_vkCmdNextSubpass,
    pub cmd_end_render_pass: PFN_vkCmdEndRenderPass,
    pub cmd_execute_commands: PFN_vkCmdExecuteCommands,
    pub create_shared_swapchains_khr: PFN_vkCreateSharedSwapchainsKHR,
    pub create_swapchain_khr: PFN_vkCreateSwapchainKHR,
    pub destroy_swapchain_khr: PFN_vkDestroySwapchainKHR,
    pub get_swapchain_images_khr: PFN_vkGetSwapchainImagesKHR,
    pub acquire_next_image_khr: PFN_vkAcquireNextImageKHR,
    pub queue_present_khr: PFN_vkQueuePresentKHR,
    pub debug_marker_set_object_name_ext: PFN_vkDebugMarkerSetObjectNameEXT,
    pub debug_marker_set_object_tag_ext: PFN_vkDebugMarkerSetObjectTagEXT,
    pub cmd_debug_marker_begin_ext: PFN_vkCmdDebugMarkerBeginEXT,
    pub cmd_debug_marker_end_ext: PFN_vkCmdDebugMarkerEndEXT,
    pub cmd_debug_marker_insert_ext: PFN_vkCmdDebugMarkerInsertEXT,
    pub get_memory_win32_handle_nv: PFN_vkGetMemoryWin32HandleNV,
    pub cmd_execute_generated_commands_nv: PFN_vkCmdExecuteGeneratedCommandsNV,
    pub cmd_preprocess_generated_commands_nv: PFN_vkCmdPreprocessGeneratedCommandsNV,
    pub cmd_bind_pipeline_shader_group_nv: PFN_vkCmdBindPipelineShaderGroupNV,
    pub get_generated_commands_memory_requirements_nv: PFN_vkGetGeneratedCommandsMemoryRequirementsNV,
    pub create_indirect_commands_layout_nv: PFN_vkCreateIndirectCommandsLayoutNV,
    pub destroy_indirect_commands_layout_nv: PFN_vkDestroyIndirectCommandsLayoutNV,
    pub cmd_execute_generated_commands_ext: PFN_vkCmdExecuteGeneratedCommandsEXT,
    pub cmd_preprocess_generated_commands_ext: PFN_vkCmdPreprocessGeneratedCommandsEXT,
    pub get_generated_commands_memory_requirements_ext: PFN_vkGetGeneratedCommandsMemoryRequirementsEXT,
    pub create_indirect_commands_layout_ext: PFN_vkCreateIndirectCommandsLayoutEXT,
    pub destroy_indirect_commands_layout_ext: PFN_vkDestroyIndirectCommandsLayoutEXT,
    pub create_indirect_execution_set_ext: PFN_vkCreateIndirectExecutionSetEXT,
    pub destroy_indirect_execution_set_ext: PFN_vkDestroyIndirectExecutionSetEXT,
    pub update_indirect_execution_set_pipeline_ext: PFN_vkUpdateIndirectExecutionSetPipelineEXT,
    pub update_indirect_execution_set_shader_ext: PFN_vkUpdateIndirectExecutionSetShaderEXT,
    pub cmd_push_descriptor_set: PFN_vkCmdPushDescriptorSet,
    pub trim_command_pool: PFN_vkTrimCommandPool,
    pub get_memory_win32_handle_khr: PFN_vkGetMemoryWin32HandleKHR,
    pub get_memory_win32_handle_properties_khr: PFN_vkGetMemoryWin32HandlePropertiesKHR,
    pub get_memory_fd_khr: PFN_vkGetMemoryFdKHR,
    pub get_memory_fd_properties_khr: PFN_vkGetMemoryFdPropertiesKHR,
    pub get_memory_zircon_handle_fuchsia: PFN_vkGetMemoryZirconHandleFUCHSIA,
    pub get_memory_zircon_handle_properties_fuchsia: PFN_vkGetMemoryZirconHandlePropertiesFUCHSIA,
    pub get_memory_remote_address_nv: PFN_vkGetMemoryRemoteAddressNV,
    pub get_memory_sci_buf_nv: PFN_vkGetMemorySciBufNV,
    pub get_semaphore_win32_handle_khr: PFN_vkGetSemaphoreWin32HandleKHR,
    pub import_semaphore_win32_handle_khr: PFN_vkImportSemaphoreWin32HandleKHR,
    pub get_semaphore_fd_khr: PFN_vkGetSemaphoreFdKHR,
    pub import_semaphore_fd_khr: PFN_vkImportSemaphoreFdKHR,
    pub get_semaphore_zircon_handle_fuchsia: PFN_vkGetSemaphoreZirconHandleFUCHSIA,
    pub import_semaphore_zircon_handle_fuchsia: PFN_vkImportSemaphoreZirconHandleFUCHSIA,
    pub get_fence_win32_handle_khr: PFN_vkGetFenceWin32HandleKHR,
    pub import_fence_win32_handle_khr: PFN_vkImportFenceWin32HandleKHR,
    pub get_fence_fd_khr: PFN_vkGetFenceFdKHR,
    pub import_fence_fd_khr: PFN_vkImportFenceFdKHR,
    pub get_fence_sci_sync_fence_nv: PFN_vkGetFenceSciSyncFenceNV,
    pub get_fence_sci_sync_obj_nv: PFN_vkGetFenceSciSyncObjNV,
    pub import_fence_sci_sync_fence_nv: PFN_vkImportFenceSciSyncFenceNV,
    pub import_fence_sci_sync_obj_nv: PFN_vkImportFenceSciSyncObjNV,
    pub get_semaphore_sci_sync_obj_nv: PFN_vkGetSemaphoreSciSyncObjNV,
    pub import_semaphore_sci_sync_obj_nv: PFN_vkImportSemaphoreSciSyncObjNV,
    pub create_semaphore_sci_sync_pool_nv: PFN_vkCreateSemaphoreSciSyncPoolNV,
    pub destroy_semaphore_sci_sync_pool_nv: PFN_vkDestroySemaphoreSciSyncPoolNV,
    pub display_power_control_ext: PFN_vkDisplayPowerControlEXT,
    pub register_device_event_ext: PFN_vkRegisterDeviceEventEXT,
    pub register_display_event_ext: PFN_vkRegisterDisplayEventEXT,
    pub get_swapchain_counter_ext: PFN_vkGetSwapchainCounterEXT,
    pub get_device_group_peer_memory_features: PFN_vkGetDeviceGroupPeerMemoryFeatures,
    pub bind_buffer_memory2: PFN_vkBindBufferMemory2,
    pub bind_image_memory2: PFN_vkBindImageMemory2,
    pub cmd_set_device_mask: PFN_vkCmdSetDeviceMask,
    pub get_device_group_present_capabilities_khr: PFN_vkGetDeviceGroupPresentCapabilitiesKHR,
    pub get_device_group_surface_present_modes_khr: PFN_vkGetDeviceGroupSurfacePresentModesKHR,
    pub acquire_next_image2_khr: PFN_vkAcquireNextImage2KHR,
    pub cmd_dispatch_base: PFN_vkCmdDispatchBase,
    pub create_descriptor_update_template: PFN_vkCreateDescriptorUpdateTemplate,
    pub destroy_descriptor_update_template: PFN_vkDestroyDescriptorUpdateTemplate,
    pub update_descriptor_set_with_template: PFN_vkUpdateDescriptorSetWithTemplate,
    pub cmd_push_descriptor_set_with_template: PFN_vkCmdPushDescriptorSetWithTemplate,
    pub set_hdr_metadata_ext: PFN_vkSetHdrMetadataEXT,
    pub get_swapchain_status_khr: PFN_vkGetSwapchainStatusKHR,
    pub get_refresh_cycle_duration_google: PFN_vkGetRefreshCycleDurationGOOGLE,
    pub get_past_presentation_timing_google: PFN_vkGetPastPresentationTimingGOOGLE,
    pub cmd_set_viewport_w_scaling_nv: PFN_vkCmdSetViewportWScalingNV,
    pub cmd_set_discard_rectangle_ext: PFN_vkCmdSetDiscardRectangleEXT,
    pub cmd_set_discard_rectangle_enable_ext: PFN_vkCmdSetDiscardRectangleEnableEXT,
    pub cmd_set_discard_rectangle_mode_ext: PFN_vkCmdSetDiscardRectangleModeEXT,
    pub cmd_set_sample_locations_ext: PFN_vkCmdSetSampleLocationsEXT,
    pub get_buffer_memory_requirements2: PFN_vkGetBufferMemoryRequirements2,
    pub get_image_memory_requirements2: PFN_vkGetImageMemoryRequirements2,
    pub get_image_sparse_memory_requirements2: PFN_vkGetImageSparseMemoryRequirements2,
    pub get_device_buffer_memory_requirements: PFN_vkGetDeviceBufferMemoryRequirements,
    pub get_device_image_memory_requirements: PFN_vkGetDeviceImageMemoryRequirements,
    pub get_device_image_sparse_memory_requirements: PFN_vkGetDeviceImageSparseMemoryRequirements,
    pub create_sampler_ycbcr_conversion: PFN_vkCreateSamplerYcbcrConversion,
    pub destroy_sampler_ycbcr_conversion: PFN_vkDestroySamplerYcbcrConversion,
    pub get_device_queue2: PFN_vkGetDeviceQueue2,
    pub create_validation_cache_ext: PFN_vkCreateValidationCacheEXT,
    pub destroy_validation_cache_ext: PFN_vkDestroyValidationCacheEXT,
    pub get_validation_cache_data_ext: PFN_vkGetValidationCacheDataEXT,
    pub merge_validation_caches_ext: PFN_vkMergeValidationCachesEXT,
    pub get_descriptor_set_layout_support: PFN_vkGetDescriptorSetLayoutSupport,
    pub get_swapchain_gralloc_usage_android: PFN_vkGetSwapchainGrallocUsageANDROID,
    pub get_swapchain_gralloc_usage2_android: PFN_vkGetSwapchainGrallocUsage2ANDROID,
    pub acquire_image_android: PFN_vkAcquireImageANDROID,
    pub queue_signal_release_image_android: PFN_vkQueueSignalReleaseImageANDROID,
    pub get_shader_info_amd: PFN_vkGetShaderInfoAMD,
    pub set_local_dimming_amd: PFN_vkSetLocalDimmingAMD,
    pub get_calibrated_timestamps_khr: PFN_vkGetCalibratedTimestampsKHR,
    pub set_debug_utils_object_name_ext: PFN_vkSetDebugUtilsObjectNameEXT,
    pub set_debug_utils_object_tag_ext: PFN_vkSetDebugUtilsObjectTagEXT,
    pub queue_begin_debug_utils_label_ext: PFN_vkQueueBeginDebugUtilsLabelEXT,
    pub queue_end_debug_utils_label_ext: PFN_vkQueueEndDebugUtilsLabelEXT,
    pub queue_insert_debug_utils_label_ext: PFN_vkQueueInsertDebugUtilsLabelEXT,
    pub cmd_begin_debug_utils_label_ext: PFN_vkCmdBeginDebugUtilsLabelEXT,
    pub cmd_end_debug_utils_label_ext: PFN_vkCmdEndDebugUtilsLabelEXT,
    pub cmd_insert_debug_utils_label_ext: PFN_vkCmdInsertDebugUtilsLabelEXT,
    pub get_memory_host_pointer_properties_ext: PFN_vkGetMemoryHostPointerPropertiesEXT,
    pub cmd_write_buffer_marker_amd: PFN_vkCmdWriteBufferMarkerAMD,
    pub create_render_pass2: PFN_vkCreateRenderPass2,
    pub cmd_begin_render_pass2: PFN_vkCmdBeginRenderPass2,
    pub cmd_next_subpass2: PFN_vkCmdNextSubpass2,
    pub cmd_end_render_pass2: PFN_vkCmdEndRenderPass2,
    pub get_semaphore_counter_value: PFN_vkGetSemaphoreCounterValue,
    pub wait_semaphores: PFN_vkWaitSemaphores,
    pub signal_semaphore: PFN_vkSignalSemaphore,
    pub get_android_hardware_buffer_properties_android: PFN_vkGetAndroidHardwareBufferPropertiesANDROID,
    pub get_memory_android_hardware_buffer_android: PFN_vkGetMemoryAndroidHardwareBufferANDROID,
    pub cmd_draw_indirect_count: PFN_vkCmdDrawIndirectCount,
    pub cmd_draw_indexed_indirect_count: PFN_vkCmdDrawIndexedIndirectCount,
    pub cmd_set_checkpoint_nv: PFN_vkCmdSetCheckpointNV,
    pub get_queue_checkpoint_data_nv: PFN_vkGetQueueCheckpointDataNV,
    pub cmd_bind_transform_feedback_buffers_ext: PFN_vkCmdBindTransformFeedbackBuffersEXT,
    pub cmd_begin_transform_feedback_ext: PFN_vkCmdBeginTransformFeedbackEXT,
    pub cmd_end_transform_feedback_ext: PFN_vkCmdEndTransformFeedbackEXT,
    pub cmd_begin_query_indexed_ext: PFN_vkCmdBeginQueryIndexedEXT,
    pub cmd_end_query_indexed_ext: PFN_vkCmdEndQueryIndexedEXT,
    pub cmd_draw_indirect_byte_count_ext: PFN_vkCmdDrawIndirectByteCountEXT,
    pub cmd_set_exclusive_scissor_nv: PFN_vkCmdSetExclusiveScissorNV,
    pub cmd_set_exclusive_scissor_enable_nv: PFN_vkCmdSetExclusiveScissorEnableNV,
    pub cmd_bind_shading_rate_image_nv: PFN_vkCmdBindShadingRateImageNV,
    pub cmd_set_viewport_shading_rate_palette_nv: PFN_vkCmdSetViewportShadingRatePaletteNV,
    pub cmd_set_coarse_sample_order_nv: PFN_vkCmdSetCoarseSampleOrderNV,
    pub cmd_draw_mesh_tasks_nv: PFN_vkCmdDrawMeshTasksNV,
    pub cmd_draw_mesh_tasks_indirect_nv: PFN_vkCmdDrawMeshTasksIndirectNV,
    pub cmd_draw_mesh_tasks_indirect_count_nv: PFN_vkCmdDrawMeshTasksIndirectCountNV,
    pub cmd_draw_mesh_tasks_ext: PFN_vkCmdDrawMeshTasksEXT,
    pub cmd_draw_mesh_tasks_indirect_ext: PFN_vkCmdDrawMeshTasksIndirectEXT,
    pub cmd_draw_mesh_tasks_indirect_count_ext: PFN_vkCmdDrawMeshTasksIndirectCountEXT,
    pub compile_deferred_nv: PFN_vkCompileDeferredNV,
    pub create_acceleration_structure_nv: PFN_vkCreateAccelerationStructureNV,
    pub cmd_bind_invocation_mask_huawei: PFN_vkCmdBindInvocationMaskHUAWEI,
    pub destroy_acceleration_structure_khr: PFN_vkDestroyAccelerationStructureKHR,
    pub destroy_acceleration_structure_nv: PFN_vkDestroyAccelerationStructureNV,
    pub get_acceleration_structure_memory_requirements_nv: PFN_vkGetAccelerationStructureMemoryRequirementsNV,
    pub bind_acceleration_structure_memory_nv: PFN_vkBindAccelerationStructureMemoryNV,
    pub cmd_copy_acceleration_structure_nv: PFN_vkCmdCopyAccelerationStructureNV,
    pub cmd_copy_acceleration_structure_khr: PFN_vkCmdCopyAccelerationStructureKHR,
    pub copy_acceleration_structure_khr: PFN_vkCopyAccelerationStructureKHR,
    pub cmd_copy_acceleration_structure_to_memory_khr: PFN_vkCmdCopyAccelerationStructureToMemoryKHR,
    pub copy_acceleration_structure_to_memory_khr: PFN_vkCopyAccelerationStructureToMemoryKHR,
    pub cmd_copy_memory_to_acceleration_structure_khr: PFN_vkCmdCopyMemoryToAccelerationStructureKHR,
    pub copy_memory_to_acceleration_structure_khr: PFN_vkCopyMemoryToAccelerationStructureKHR,
    pub cmd_write_acceleration_structures_properties_khr: PFN_vkCmdWriteAccelerationStructuresPropertiesKHR,
    pub cmd_write_acceleration_structures_properties_nv: PFN_vkCmdWriteAccelerationStructuresPropertiesNV,
    pub cmd_build_acceleration_structure_nv: PFN_vkCmdBuildAccelerationStructureNV,
    pub write_acceleration_structures_properties_khr: PFN_vkWriteAccelerationStructuresPropertiesKHR,
    pub cmd_trace_rays_khr: PFN_vkCmdTraceRaysKHR,
    pub cmd_trace_rays_nv: PFN_vkCmdTraceRaysNV,
    pub get_ray_tracing_shader_group_handles_khr: PFN_vkGetRayTracingShaderGroupHandlesKHR,
    pub get_ray_tracing_capture_replay_shader_group_handles_khr: PFN_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR,
    pub get_acceleration_structure_handle_nv: PFN_vkGetAccelerationStructureHandleNV,
    pub create_ray_tracing_pipelines_nv: PFN_vkCreateRayTracingPipelinesNV,
    pub create_ray_tracing_pipelines_khr: PFN_vkCreateRayTracingPipelinesKHR,
    pub cmd_trace_rays_indirect_khr: PFN_vkCmdTraceRaysIndirectKHR,
    pub cmd_trace_rays_indirect2_khr: PFN_vkCmdTraceRaysIndirect2KHR,
    pub get_cluster_acceleration_structure_build_sizes_nv: PFN_vkGetClusterAccelerationStructureBuildSizesNV,
    pub cmd_build_cluster_acceleration_structure_indirect_nv: PFN_vkCmdBuildClusterAccelerationStructureIndirectNV,
    pub get_device_acceleration_structure_compatibility_khr: PFN_vkGetDeviceAccelerationStructureCompatibilityKHR,
    pub get_ray_tracing_shader_group_stack_size_khr: PFN_vkGetRayTracingShaderGroupStackSizeKHR,
    pub cmd_set_ray_tracing_pipeline_stack_size_khr: PFN_vkCmdSetRayTracingPipelineStackSizeKHR,
    pub get_image_view_handle_nvx: PFN_vkGetImageViewHandleNVX,
    pub get_image_view_handle64_nvx: PFN_vkGetImageViewHandle64NVX,
    pub get_image_view_address_nvx: PFN_vkGetImageViewAddressNVX,
    pub get_device_combined_image_sampler_index_nvx: PFN_vkGetDeviceCombinedImageSamplerIndexNVX,
    pub get_device_group_surface_present_modes2_ext: PFN_vkGetDeviceGroupSurfacePresentModes2EXT,
    pub acquire_full_screen_exclusive_mode_ext: PFN_vkAcquireFullScreenExclusiveModeEXT,
    pub release_full_screen_exclusive_mode_ext: PFN_vkReleaseFullScreenExclusiveModeEXT,
    pub acquire_profiling_lock_khr: PFN_vkAcquireProfilingLockKHR,
    pub release_profiling_lock_khr: PFN_vkReleaseProfilingLockKHR,
    pub get_image_drm_format_modifier_properties_ext: PFN_vkGetImageDrmFormatModifierPropertiesEXT,
    pub get_buffer_opaque_capture_address: PFN_vkGetBufferOpaqueCaptureAddress,
    pub get_buffer_device_address: PFN_vkGetBufferDeviceAddress,
    pub initialize_performance_api_intel: PFN_vkInitializePerformanceApiINTEL,
    pub uninitialize_performance_api_intel: PFN_vkUninitializePerformanceApiINTEL,
    pub cmd_set_performance_marker_intel: PFN_vkCmdSetPerformanceMarkerINTEL,
    pub cmd_set_performance_stream_marker_intel: PFN_vkCmdSetPerformanceStreamMarkerINTEL,
    pub cmd_set_performance_override_intel: PFN_vkCmdSetPerformanceOverrideINTEL,
    pub acquire_performance_configuration_intel: PFN_vkAcquirePerformanceConfigurationINTEL,
    pub release_performance_configuration_intel: PFN_vkReleasePerformanceConfigurationINTEL,
    pub queue_set_performance_configuration_intel: PFN_vkQueueSetPerformanceConfigurationINTEL,
    pub get_performance_parameter_intel: PFN_vkGetPerformanceParameterINTEL,
    pub get_device_memory_opaque_capture_address: PFN_vkGetDeviceMemoryOpaqueCaptureAddress,
    pub get_pipeline_executable_properties_khr: PFN_vkGetPipelineExecutablePropertiesKHR,
    pub get_pipeline_executable_statistics_khr: PFN_vkGetPipelineExecutableStatisticsKHR,
    pub get_pipeline_executable_internal_representations_khr: PFN_vkGetPipelineExecutableInternalRepresentationsKHR,
    pub cmd_set_line_stipple: PFN_vkCmdSetLineStipple,
    pub get_fault_data: PFN_vkGetFaultData,
    pub create_acceleration_structure_khr: PFN_vkCreateAccelerationStructureKHR,
    pub cmd_build_acceleration_structures_khr: PFN_vkCmdBuildAccelerationStructuresKHR,
    pub cmd_build_acceleration_structures_indirect_khr: PFN_vkCmdBuildAccelerationStructuresIndirectKHR,
    pub build_acceleration_structures_khr: PFN_vkBuildAccelerationStructuresKHR,
    pub get_acceleration_structure_device_address_khr: PFN_vkGetAccelerationStructureDeviceAddressKHR,
    pub create_deferred_operation_khr: PFN_vkCreateDeferredOperationKHR,
    pub destroy_deferred_operation_khr: PFN_vkDestroyDeferredOperationKHR,
    pub get_deferred_operation_max_concurrency_khr: PFN_vkGetDeferredOperationMaxConcurrencyKHR,
    pub get_deferred_operation_result_khr: PFN_vkGetDeferredOperationResultKHR,
    pub deferred_operation_join_khr: PFN_vkDeferredOperationJoinKHR,
    pub get_pipeline_indirect_memory_requirements_nv: PFN_vkGetPipelineIndirectMemoryRequirementsNV,
    pub get_pipeline_indirect_device_address_nv: PFN_vkGetPipelineIndirectDeviceAddressNV,
    pub anti_lag_update_amd: PFN_vkAntiLagUpdateAMD,
    pub cmd_set_cull_mode: PFN_vkCmdSetCullMode,
    pub cmd_set_front_face: PFN_vkCmdSetFrontFace,
    pub cmd_set_primitive_topology: PFN_vkCmdSetPrimitiveTopology,
    pub cmd_set_viewport_with_count: PFN_vkCmdSetViewportWithCount,
    pub cmd_set_scissor_with_count: PFN_vkCmdSetScissorWithCount,
    pub cmd_bind_index_buffer2: PFN_vkCmdBindIndexBuffer2,
    pub cmd_bind_vertex_buffers2: PFN_vkCmdBindVertexBuffers2,
    pub cmd_set_depth_test_enable: PFN_vkCmdSetDepthTestEnable,
    pub cmd_set_depth_write_enable: PFN_vkCmdSetDepthWriteEnable,
    pub cmd_set_depth_compare_op: PFN_vkCmdSetDepthCompareOp,
    pub cmd_set_depth_bounds_test_enable: PFN_vkCmdSetDepthBoundsTestEnable,
    pub cmd_set_stencil_test_enable: PFN_vkCmdSetStencilTestEnable,
    pub cmd_set_stencil_op: PFN_vkCmdSetStencilOp,
    pub cmd_set_patch_control_points_ext: PFN_vkCmdSetPatchControlPointsEXT,
    pub cmd_set_rasterizer_discard_enable: PFN_vkCmdSetRasterizerDiscardEnable,
    pub cmd_set_depth_bias_enable: PFN_vkCmdSetDepthBiasEnable,
    pub cmd_set_logic_op_ext: PFN_vkCmdSetLogicOpEXT,
    pub cmd_set_primitive_restart_enable: PFN_vkCmdSetPrimitiveRestartEnable,
    pub cmd_set_tessellation_domain_origin_ext: PFN_vkCmdSetTessellationDomainOriginEXT,
    pub cmd_set_depth_clamp_enable_ext: PFN_vkCmdSetDepthClampEnableEXT,
    pub cmd_set_polygon_mode_ext: PFN_vkCmdSetPolygonModeEXT,
    pub cmd_set_rasterization_samples_ext: PFN_vkCmdSetRasterizationSamplesEXT,
    pub cmd_set_sample_mask_ext: PFN_vkCmdSetSampleMaskEXT,
    pub cmd_set_alpha_to_coverage_enable_ext: PFN_vkCmdSetAlphaToCoverageEnableEXT,
    pub cmd_set_alpha_to_one_enable_ext: PFN_vkCmdSetAlphaToOneEnableEXT,
    pub cmd_set_logic_op_enable_ext: PFN_vkCmdSetLogicOpEnableEXT,
    pub cmd_set_color_blend_enable_ext: PFN_vkCmdSetColorBlendEnableEXT,
    pub cmd_set_color_blend_equation_ext: PFN_vkCmdSetColorBlendEquationEXT,
    pub cmd_set_color_write_mask_ext: PFN_vkCmdSetColorWriteMaskEXT,
    pub cmd_set_rasterization_stream_ext: PFN_vkCmdSetRasterizationStreamEXT,
    pub cmd_set_conservative_rasterization_mode_ext: PFN_vkCmdSetConservativeRasterizationModeEXT,
    pub cmd_set_extra_primitive_overestimation_size_ext: PFN_vkCmdSetExtraPrimitiveOverestimationSizeEXT,
    pub cmd_set_depth_clip_enable_ext: PFN_vkCmdSetDepthClipEnableEXT,
    pub cmd_set_sample_locations_enable_ext: PFN_vkCmdSetSampleLocationsEnableEXT,
    pub cmd_set_color_blend_advanced_ext: PFN_vkCmdSetColorBlendAdvancedEXT,
    pub cmd_set_provoking_vertex_mode_ext: PFN_vkCmdSetProvokingVertexModeEXT,
    pub cmd_set_line_rasterization_mode_ext: PFN_vkCmdSetLineRasterizationModeEXT,
    pub cmd_set_line_stipple_enable_ext: PFN_vkCmdSetLineStippleEnableEXT,
    pub cmd_set_depth_clip_negative_one_to_one_ext: PFN_vkCmdSetDepthClipNegativeOneToOneEXT,
    pub cmd_set_viewport_w_scaling_enable_nv: PFN_vkCmdSetViewportWScalingEnableNV,
    pub cmd_set_viewport_swizzle_nv: PFN_vkCmdSetViewportSwizzleNV,
    pub cmd_set_coverage_to_color_enable_nv: PFN_vkCmdSetCoverageToColorEnableNV,
    pub cmd_set_coverage_to_color_location_nv: PFN_vkCmdSetCoverageToColorLocationNV,
    pub cmd_set_coverage_modulation_mode_nv: PFN_vkCmdSetCoverageModulationModeNV,
    pub cmd_set_coverage_modulation_table_enable_nv: PFN_vkCmdSetCoverageModulationTableEnableNV,
    pub cmd_set_coverage_modulation_table_nv: PFN_vkCmdSetCoverageModulationTableNV,
    pub cmd_set_shading_rate_image_enable_nv: PFN_vkCmdSetShadingRateImageEnableNV,
    pub cmd_set_coverage_reduction_mode_nv: PFN_vkCmdSetCoverageReductionModeNV,
    pub cmd_set_representative_fragment_test_enable_nv: PFN_vkCmdSetRepresentativeFragmentTestEnableNV,
    pub create_private_data_slot: PFN_vkCreatePrivateDataSlot,
    pub destroy_private_data_slot: PFN_vkDestroyPrivateDataSlot,
    pub set_private_data: PFN_vkSetPrivateData,
    pub get_private_data: PFN_vkGetPrivateData,
    pub cmd_copy_buffer2: PFN_vkCmdCopyBuffer2,
    pub cmd_copy_image2: PFN_vkCmdCopyImage2,
    pub cmd_blit_image2: PFN_vkCmdBlitImage2,
    pub cmd_copy_buffer_to_image2: PFN_vkCmdCopyBufferToImage2,
    pub cmd_copy_image_to_buffer2: PFN_vkCmdCopyImageToBuffer2,
    pub cmd_resolve_image2: PFN_vkCmdResolveImage2,
    pub cmd_refresh_objects_khr: PFN_vkCmdRefreshObjectsKHR,
    pub cmd_set_fragment_shading_rate_khr: PFN_vkCmdSetFragmentShadingRateKHR,
    pub cmd_set_fragment_shading_rate_enum_nv: PFN_vkCmdSetFragmentShadingRateEnumNV,
    pub get_acceleration_structure_build_sizes_khr: PFN_vkGetAccelerationStructureBuildSizesKHR,
    pub cmd_set_vertex_input_ext: PFN_vkCmdSetVertexInputEXT,
    pub cmd_set_color_write_enable_ext: PFN_vkCmdSetColorWriteEnableEXT,
    pub cmd_set_event2: PFN_vkCmdSetEvent2,
    pub cmd_reset_event2: PFN_vkCmdResetEvent2,
    pub cmd_wait_events2: PFN_vkCmdWaitEvents2,
    pub cmd_pipeline_barrier2: PFN_vkCmdPipelineBarrier2,
    pub queue_submit2: PFN_vkQueueSubmit2,
    pub cmd_write_timestamp2: PFN_vkCmdWriteTimestamp2,
    pub cmd_write_buffer_marker2_amd: PFN_vkCmdWriteBufferMarker2AMD,
    pub get_queue_checkpoint_data2_nv: PFN_vkGetQueueCheckpointData2NV,
    pub copy_memory_to_image: PFN_vkCopyMemoryToImage,
    pub copy_image_to_memory: PFN_vkCopyImageToMemory,
    pub copy_image_to_image: PFN_vkCopyImageToImage,
    pub transition_image_layout: PFN_vkTransitionImageLayout,
    pub get_command_pool_memory_consumption: PFN_vkGetCommandPoolMemoryConsumption,
    pub create_video_session_khr: PFN_vkCreateVideoSessionKHR,
    pub destroy_video_session_khr: PFN_vkDestroyVideoSessionKHR,
    pub create_video_session_parameters_khr: PFN_vkCreateVideoSessionParametersKHR,
    pub update_video_session_parameters_khr: PFN_vkUpdateVideoSessionParametersKHR,
    pub get_encoded_video_session_parameters_khr: PFN_vkGetEncodedVideoSessionParametersKHR,
    pub destroy_video_session_parameters_khr: PFN_vkDestroyVideoSessionParametersKHR,
    pub get_video_session_memory_requirements_khr: PFN_vkGetVideoSessionMemoryRequirementsKHR,
    pub bind_video_session_memory_khr: PFN_vkBindVideoSessionMemoryKHR,
    pub cmd_decode_video_khr: PFN_vkCmdDecodeVideoKHR,
    pub cmd_begin_video_coding_khr: PFN_vkCmdBeginVideoCodingKHR,
    pub cmd_control_video_coding_khr: PFN_vkCmdControlVideoCodingKHR,
    pub cmd_end_video_coding_khr: PFN_vkCmdEndVideoCodingKHR,
    pub cmd_encode_video_khr: PFN_vkCmdEncodeVideoKHR,
    pub cmd_decompress_memory_nv: PFN_vkCmdDecompressMemoryNV,
    pub cmd_decompress_memory_indirect_count_nv: PFN_vkCmdDecompressMemoryIndirectCountNV,
    pub get_partitioned_acceleration_structures_build_sizes_nv: PFN_vkGetPartitionedAccelerationStructuresBuildSizesNV,
    pub cmd_build_partitioned_acceleration_structures_nv: PFN_vkCmdBuildPartitionedAccelerationStructuresNV,
    pub cmd_decompress_memory_ext: PFN_vkCmdDecompressMemoryEXT,
    pub cmd_decompress_memory_indirect_count_ext: PFN_vkCmdDecompressMemoryIndirectCountEXT,
    pub create_cu_module_nvx: PFN_vkCreateCuModuleNVX,
    pub create_cu_function_nvx: PFN_vkCreateCuFunctionNVX,
    pub destroy_cu_module_nvx: PFN_vkDestroyCuModuleNVX,
    pub destroy_cu_function_nvx: PFN_vkDestroyCuFunctionNVX,
    pub cmd_cu_launch_kernel_nvx: PFN_vkCmdCuLaunchKernelNVX,
    pub get_descriptor_set_layout_size_ext: PFN_vkGetDescriptorSetLayoutSizeEXT,
    pub get_descriptor_set_layout_binding_offset_ext: PFN_vkGetDescriptorSetLayoutBindingOffsetEXT,
    pub get_descriptor_ext: PFN_vkGetDescriptorEXT,
    pub cmd_bind_descriptor_buffers_ext: PFN_vkCmdBindDescriptorBuffersEXT,
    pub cmd_set_descriptor_buffer_offsets_ext: PFN_vkCmdSetDescriptorBufferOffsetsEXT,
    pub cmd_bind_descriptor_buffer_embedded_samplers_ext: PFN_vkCmdBindDescriptorBufferEmbeddedSamplersEXT,
    pub get_buffer_opaque_capture_descriptor_data_ext: PFN_vkGetBufferOpaqueCaptureDescriptorDataEXT,
    pub get_image_opaque_capture_descriptor_data_ext: PFN_vkGetImageOpaqueCaptureDescriptorDataEXT,
    pub get_image_view_opaque_capture_descriptor_data_ext: PFN_vkGetImageViewOpaqueCaptureDescriptorDataEXT,
    pub get_sampler_opaque_capture_descriptor_data_ext: PFN_vkGetSamplerOpaqueCaptureDescriptorDataEXT,
    pub get_acceleration_structure_opaque_capture_descriptor_data_ext: PFN_vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT,
    pub set_device_memory_priority_ext: PFN_vkSetDeviceMemoryPriorityEXT,
    pub wait_for_present2_khr: PFN_vkWaitForPresent2KHR,
    pub wait_for_present_khr: PFN_vkWaitForPresentKHR,
    pub create_buffer_collection_fuchsia: PFN_vkCreateBufferCollectionFUCHSIA,
    pub set_buffer_collection_buffer_constraints_fuchsia: PFN_vkSetBufferCollectionBufferConstraintsFUCHSIA,
    pub set_buffer_collection_image_constraints_fuchsia: PFN_vkSetBufferCollectionImageConstraintsFUCHSIA,
    pub destroy_buffer_collection_fuchsia: PFN_vkDestroyBufferCollectionFUCHSIA,
    pub get_buffer_collection_properties_fuchsia: PFN_vkGetBufferCollectionPropertiesFUCHSIA,
    pub create_cuda_module_nv: PFN_vkCreateCudaModuleNV,
    pub get_cuda_module_cache_nv: PFN_vkGetCudaModuleCacheNV,
    pub create_cuda_function_nv: PFN_vkCreateCudaFunctionNV,
    pub destroy_cuda_module_nv: PFN_vkDestroyCudaModuleNV,
    pub destroy_cuda_function_nv: PFN_vkDestroyCudaFunctionNV,
    pub cmd_cuda_launch_kernel_nv: PFN_vkCmdCudaLaunchKernelNV,
    pub cmd_begin_rendering: PFN_vkCmdBeginRendering,
    pub cmd_end_rendering: PFN_vkCmdEndRendering,
    pub cmd_end_rendering2_khr: PFN_vkCmdEndRendering2KHR,
    pub get_descriptor_set_layout_host_mapping_info_valve: PFN_vkGetDescriptorSetLayoutHostMappingInfoVALVE,
    pub get_descriptor_set_host_mapping_valve: PFN_vkGetDescriptorSetHostMappingVALVE,
    pub create_micromap_ext: PFN_vkCreateMicromapEXT,
    pub cmd_build_micromaps_ext: PFN_vkCmdBuildMicromapsEXT,
    pub build_micromaps_ext: PFN_vkBuildMicromapsEXT,
    pub destroy_micromap_ext: PFN_vkDestroyMicromapEXT,
    pub cmd_copy_micromap_ext: PFN_vkCmdCopyMicromapEXT,
    pub copy_micromap_ext: PFN_vkCopyMicromapEXT,
    pub cmd_copy_micromap_to_memory_ext: PFN_vkCmdCopyMicromapToMemoryEXT,
    pub copy_micromap_to_memory_ext: PFN_vkCopyMicromapToMemoryEXT,
    pub cmd_copy_memory_to_micromap_ext: PFN_vkCmdCopyMemoryToMicromapEXT,
    pub copy_memory_to_micromap_ext: PFN_vkCopyMemoryToMicromapEXT,
    pub cmd_write_micromaps_properties_ext: PFN_vkCmdWriteMicromapsPropertiesEXT,
    pub write_micromaps_properties_ext: PFN_vkWriteMicromapsPropertiesEXT,
    pub get_device_micromap_compatibility_ext: PFN_vkGetDeviceMicromapCompatibilityEXT,
    pub get_micromap_build_sizes_ext: PFN_vkGetMicromapBuildSizesEXT,
    pub get_shader_module_identifier_ext: PFN_vkGetShaderModuleIdentifierEXT,
    pub get_shader_module_create_info_identifier_ext: PFN_vkGetShaderModuleCreateInfoIdentifierEXT,
    pub get_image_subresource_layout2: PFN_vkGetImageSubresourceLayout2,
    pub get_pipeline_properties_ext: PFN_vkGetPipelinePropertiesEXT,
    pub export_metal_objects_ext: PFN_vkExportMetalObjectsEXT,
    pub cmd_bind_tile_memory_qcom: PFN_vkCmdBindTileMemoryQCOM,
    pub get_framebuffer_tile_properties_qcom: PFN_vkGetFramebufferTilePropertiesQCOM,
    pub get_dynamic_rendering_tile_properties_qcom: PFN_vkGetDynamicRenderingTilePropertiesQCOM,
    pub create_optical_flow_session_nv: PFN_vkCreateOpticalFlowSessionNV,
    pub destroy_optical_flow_session_nv: PFN_vkDestroyOpticalFlowSessionNV,
    pub bind_optical_flow_session_image_nv: PFN_vkBindOpticalFlowSessionImageNV,
    pub cmd_optical_flow_execute_nv: PFN_vkCmdOpticalFlowExecuteNV,
    pub get_device_fault_info_ext: PFN_vkGetDeviceFaultInfoEXT,
    pub get_device_fault_reports_khr: PFN_vkGetDeviceFaultReportsKHR,
    pub get_device_fault_debug_info_khr: PFN_vkGetDeviceFaultDebugInfoKHR,
    pub cmd_set_depth_bias2_ext: PFN_vkCmdSetDepthBias2EXT,
    pub release_swapchain_images_khr: PFN_vkReleaseSwapchainImagesKHR,
    pub get_device_image_subresource_layout: PFN_vkGetDeviceImageSubresourceLayout,
    pub map_memory2: PFN_vkMapMemory2,
    pub unmap_memory2: PFN_vkUnmapMemory2,
    pub create_shaders_ext: PFN_vkCreateShadersEXT,
    pub destroy_shader_ext: PFN_vkDestroyShaderEXT,
    pub get_shader_binary_data_ext: PFN_vkGetShaderBinaryDataEXT,
    pub cmd_bind_shaders_ext: PFN_vkCmdBindShadersEXT,
    pub set_swapchain_present_timing_queue_size_ext: PFN_vkSetSwapchainPresentTimingQueueSizeEXT,
    pub get_swapchain_timing_properties_ext: PFN_vkGetSwapchainTimingPropertiesEXT,
    pub get_swapchain_time_domain_properties_ext: PFN_vkGetSwapchainTimeDomainPropertiesEXT,
    pub get_past_presentation_timing_ext: PFN_vkGetPastPresentationTimingEXT,
    pub get_screen_buffer_properties_qnx: PFN_vkGetScreenBufferPropertiesQNX,
    pub get_execution_graph_pipeline_scratch_size_amdx: PFN_vkGetExecutionGraphPipelineScratchSizeAMDX,
    pub get_execution_graph_pipeline_node_index_amdx: PFN_vkGetExecutionGraphPipelineNodeIndexAMDX,
    pub create_execution_graph_pipelines_amdx: PFN_vkCreateExecutionGraphPipelinesAMDX,
    pub cmd_initialize_graph_scratch_memory_amdx: PFN_vkCmdInitializeGraphScratchMemoryAMDX,
    pub cmd_dispatch_graph_amdx: PFN_vkCmdDispatchGraphAMDX,
    pub cmd_dispatch_graph_indirect_amdx: PFN_vkCmdDispatchGraphIndirectAMDX,
    pub cmd_dispatch_graph_indirect_count_amdx: PFN_vkCmdDispatchGraphIndirectCountAMDX,
    pub cmd_bind_descriptor_sets2: PFN_vkCmdBindDescriptorSets2,
    pub cmd_push_constants2: PFN_vkCmdPushConstants2,
    pub cmd_push_descriptor_set2: PFN_vkCmdPushDescriptorSet2,
    pub cmd_push_descriptor_set_with_template2: PFN_vkCmdPushDescriptorSetWithTemplate2,
    pub cmd_set_descriptor_buffer_offsets2_ext: PFN_vkCmdSetDescriptorBufferOffsets2EXT,
    pub cmd_bind_descriptor_buffer_embedded_samplers2_ext: PFN_vkCmdBindDescriptorBufferEmbeddedSamplers2EXT,
    pub set_latency_sleep_mode_nv: PFN_vkSetLatencySleepModeNV,
    pub latency_sleep_nv: PFN_vkLatencySleepNV,
    pub set_latency_marker_nv: PFN_vkSetLatencyMarkerNV,
    pub get_latency_timings_nv: PFN_vkGetLatencyTimingsNV,
    pub queue_notify_out_of_band_nv: PFN_vkQueueNotifyOutOfBandNV,
    pub cmd_set_rendering_attachment_locations: PFN_vkCmdSetRenderingAttachmentLocations,
    pub cmd_set_rendering_input_attachment_indices: PFN_vkCmdSetRenderingInputAttachmentIndices,
    pub cmd_set_depth_clamp_range_ext: PFN_vkCmdSetDepthClampRangeEXT,
    pub get_memory_metal_handle_ext: PFN_vkGetMemoryMetalHandleEXT,
    pub get_memory_metal_handle_properties_ext: PFN_vkGetMemoryMetalHandlePropertiesEXT,
    pub convert_cooperative_vector_matrix_nv: PFN_vkConvertCooperativeVectorMatrixNV,
    pub cmd_convert_cooperative_vector_matrix_nv: PFN_vkCmdConvertCooperativeVectorMatrixNV,
    pub cmd_dispatch_tile_qcom: PFN_vkCmdDispatchTileQCOM,
    pub cmd_begin_per_tile_execution_qcom: PFN_vkCmdBeginPerTileExecutionQCOM,
    pub cmd_end_per_tile_execution_qcom: PFN_vkCmdEndPerTileExecutionQCOM,
    pub create_external_compute_queue_nv: PFN_vkCreateExternalComputeQueueNV,
    pub destroy_external_compute_queue_nv: PFN_vkDestroyExternalComputeQueueNV,
    pub create_shader_instrumentation_arm: PFN_vkCreateShaderInstrumentationARM,
    pub destroy_shader_instrumentation_arm: PFN_vkDestroyShaderInstrumentationARM,
    pub cmd_begin_shader_instrumentation_arm: PFN_vkCmdBeginShaderInstrumentationARM,
    pub cmd_end_shader_instrumentation_arm: PFN_vkCmdEndShaderInstrumentationARM,
    pub get_shader_instrumentation_values_arm: PFN_vkGetShaderInstrumentationValuesARM,
    pub clear_shader_instrumentation_metrics_arm: PFN_vkClearShaderInstrumentationMetricsARM,
    pub create_tensor_arm: PFN_vkCreateTensorARM,
    pub destroy_tensor_arm: PFN_vkDestroyTensorARM,
    pub create_tensor_view_arm: PFN_vkCreateTensorViewARM,
    pub destroy_tensor_view_arm: PFN_vkDestroyTensorViewARM,
    pub get_tensor_memory_requirements_arm: PFN_vkGetTensorMemoryRequirementsARM,
    pub bind_tensor_memory_arm: PFN_vkBindTensorMemoryARM,
    pub get_device_tensor_memory_requirements_arm: PFN_vkGetDeviceTensorMemoryRequirementsARM,
    pub cmd_copy_tensor_arm: PFN_vkCmdCopyTensorARM,
    pub get_tensor_opaque_capture_descriptor_data_arm: PFN_vkGetTensorOpaqueCaptureDescriptorDataARM,
    pub get_tensor_view_opaque_capture_descriptor_data_arm: PFN_vkGetTensorViewOpaqueCaptureDescriptorDataARM,
    pub create_data_graph_pipelines_arm: PFN_vkCreateDataGraphPipelinesARM,
    pub create_data_graph_pipeline_session_arm: PFN_vkCreateDataGraphPipelineSessionARM,
    pub get_data_graph_pipeline_session_bind_point_requirements_arm: PFN_vkGetDataGraphPipelineSessionBindPointRequirementsARM,
    pub get_data_graph_pipeline_session_memory_requirements_arm: PFN_vkGetDataGraphPipelineSessionMemoryRequirementsARM,
    pub bind_data_graph_pipeline_session_memory_arm: PFN_vkBindDataGraphPipelineSessionMemoryARM,
    pub destroy_data_graph_pipeline_session_arm: PFN_vkDestroyDataGraphPipelineSessionARM,
    pub cmd_dispatch_data_graph_arm: PFN_vkCmdDispatchDataGraphARM,
    pub get_data_graph_pipeline_available_properties_arm: PFN_vkGetDataGraphPipelineAvailablePropertiesARM,
    pub get_data_graph_pipeline_properties_arm: PFN_vkGetDataGraphPipelinePropertiesARM,
    pub get_native_buffer_properties_ohos: PFN_vkGetNativeBufferPropertiesOHOS,
    pub get_memory_native_buffer_ohos: PFN_vkGetMemoryNativeBufferOHOS,
    pub get_swapchain_gralloc_usage_ohos: PFN_vkGetSwapchainGrallocUsageOHOS,
    pub acquire_image_ohos: PFN_vkAcquireImageOHOS,
    pub queue_signal_release_image_ohos: PFN_vkQueueSignalReleaseImageOHOS,
    pub cmd_set_compute_occupancy_priority_nv: PFN_vkCmdSetComputeOccupancyPriorityNV,
    pub write_sampler_descriptors_ext: PFN_vkWriteSamplerDescriptorsEXT,
    pub write_resource_descriptors_ext: PFN_vkWriteResourceDescriptorsEXT,
    pub cmd_bind_sampler_heap_ext: PFN_vkCmdBindSamplerHeapEXT,
    pub cmd_bind_resource_heap_ext: PFN_vkCmdBindResourceHeapEXT,
    pub cmd_push_data_ext: PFN_vkCmdPushDataEXT,
    pub register_custom_border_color_ext: PFN_vkRegisterCustomBorderColorEXT,
    pub unregister_custom_border_color_ext: PFN_vkUnregisterCustomBorderColorEXT,
    pub get_image_opaque_capture_data_ext: PFN_vkGetImageOpaqueCaptureDataEXT,
    pub get_tensor_opaque_capture_data_arm: PFN_vkGetTensorOpaqueCaptureDataARM,
    pub cmd_copy_memory_khr: PFN_vkCmdCopyMemoryKHR,
    pub cmd_copy_memory_to_image_khr: PFN_vkCmdCopyMemoryToImageKHR,
    pub cmd_copy_image_to_memory_khr: PFN_vkCmdCopyImageToMemoryKHR,
    pub cmd_update_memory_khr: PFN_vkCmdUpdateMemoryKHR,
    pub cmd_fill_memory_khr: PFN_vkCmdFillMemoryKHR,
    pub cmd_copy_query_pool_results_to_memory_khr: PFN_vkCmdCopyQueryPoolResultsToMemoryKHR,
    pub cmd_begin_conditional_rendering2_ext: PFN_vkCmdBeginConditionalRendering2EXT,
    pub cmd_bind_transform_feedback_buffers2_ext: PFN_vkCmdBindTransformFeedbackBuffers2EXT,
    pub cmd_begin_transform_feedback2_ext: PFN_vkCmdBeginTransformFeedback2EXT,
    pub cmd_end_transform_feedback2_ext: PFN_vkCmdEndTransformFeedback2EXT,
    pub cmd_draw_indirect_byte_count2_ext: PFN_vkCmdDrawIndirectByteCount2EXT,
    pub cmd_write_marker_to_memory_amd: PFN_vkCmdWriteMarkerToMemoryAMD,
    pub cmd_bind_index_buffer3_khr: PFN_vkCmdBindIndexBuffer3KHR,
    pub cmd_bind_vertex_buffers3_khr: PFN_vkCmdBindVertexBuffers3KHR,
    pub cmd_draw_indirect2_khr: PFN_vkCmdDrawIndirect2KHR,
    pub cmd_draw_indexed_indirect2_khr: PFN_vkCmdDrawIndexedIndirect2KHR,
    pub cmd_draw_indirect_count2_khr: PFN_vkCmdDrawIndirectCount2KHR,
    pub cmd_draw_indexed_indirect_count2_khr: PFN_vkCmdDrawIndexedIndirectCount2KHR,
    pub cmd_draw_mesh_tasks_indirect2_ext: PFN_vkCmdDrawMeshTasksIndirect2EXT,
    pub cmd_draw_mesh_tasks_indirect_count2_ext: PFN_vkCmdDrawMeshTasksIndirectCount2EXT,
    pub cmd_dispatch_indirect2_khr: PFN_vkCmdDispatchIndirect2KHR,
    pub create_acceleration_structure2_khr: PFN_vkCreateAccelerationStructure2KHR,
}
impl Default for DeviceCommands {
    #[inline]
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}
impl DeviceCommands {
    /// Load all function pointers from the given loader callback.
    ///
    /// Load all function pointers from the given loader callback.
    ///
    /// # Safety
    ///
    /// The loader must return valid function pointers compatible with
    /// each command's signature, or null for unavailable commands.
    pub unsafe fn load(
        mut f: impl FnMut(&std::ffi::CStr) -> *const std::ffi::c_void,
    ) -> Self {
        unsafe {
            let mut cmd = Self::default();
            cmd.get_device_proc_addr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDeviceProcAddr\0",
                    ),
                ),
            );
            cmd.destroy_device = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkDestroyDevice\0")),
            );
            cmd.get_device_queue = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkGetDeviceQueue\0")),
            );
            cmd.queue_submit = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkQueueSubmit\0")),
            );
            cmd.queue_wait_idle = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkQueueWaitIdle\0")),
            );
            cmd.device_wait_idle = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkDeviceWaitIdle\0")),
            );
            cmd.allocate_memory = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkAllocateMemory\0")),
            );
            cmd.free_memory = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkFreeMemory\0")),
            );
            cmd.map_memory = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkMapMemory\0")),
            );
            cmd.unmap_memory = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkUnmapMemory\0")),
            );
            cmd.flush_mapped_memory_ranges = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkFlushMappedMemoryRanges\0",
                    ),
                ),
            );
            cmd.invalidate_mapped_memory_ranges = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkInvalidateMappedMemoryRanges\0",
                    ),
                ),
            );
            cmd.get_device_memory_commitment = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDeviceMemoryCommitment\0",
                    ),
                ),
            );
            cmd.get_buffer_memory_requirements = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetBufferMemoryRequirements\0",
                    ),
                ),
            );
            cmd.bind_buffer_memory = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkBindBufferMemory\0")),
            );
            cmd.get_image_memory_requirements = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetImageMemoryRequirements\0",
                    ),
                ),
            );
            cmd.bind_image_memory = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkBindImageMemory\0")),
            );
            cmd.get_image_sparse_memory_requirements = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetImageSparseMemoryRequirements\0",
                    ),
                ),
            );
            cmd.queue_bind_sparse = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkQueueBindSparse\0")),
            );
            cmd.create_fence = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCreateFence\0")),
            );
            cmd.destroy_fence = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkDestroyFence\0")),
            );
            cmd.reset_fences = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkResetFences\0")),
            );
            cmd.get_fence_status = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkGetFenceStatus\0")),
            );
            cmd.wait_for_fences = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkWaitForFences\0")),
            );
            cmd.create_semaphore = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCreateSemaphore\0")),
            );
            cmd.destroy_semaphore = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkDestroySemaphore\0")),
            );
            cmd.create_event = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCreateEvent\0")),
            );
            cmd.destroy_event = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkDestroyEvent\0")),
            );
            cmd.get_event_status = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkGetEventStatus\0")),
            );
            cmd.set_event = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkSetEvent\0")),
            );
            cmd.reset_event = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkResetEvent\0")),
            );
            cmd.create_query_pool = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCreateQueryPool\0")),
            );
            cmd.destroy_query_pool = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkDestroyQueryPool\0")),
            );
            cmd.get_query_pool_results = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetQueryPoolResults\0",
                    ),
                ),
            );
            cmd.reset_query_pool = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkResetQueryPool\0")),
            );
            if cmd.reset_query_pool.is_none() {
                cmd.reset_query_pool = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkResetQueryPoolEXT\0",
                        ),
                    ),
                );
            }
            cmd.create_buffer = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCreateBuffer\0")),
            );
            cmd.destroy_buffer = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkDestroyBuffer\0")),
            );
            cmd.create_buffer_view = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCreateBufferView\0")),
            );
            cmd.destroy_buffer_view = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyBufferView\0",
                    ),
                ),
            );
            cmd.create_image = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCreateImage\0")),
            );
            cmd.destroy_image = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkDestroyImage\0")),
            );
            cmd.get_image_subresource_layout = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetImageSubresourceLayout\0",
                    ),
                ),
            );
            cmd.create_image_view = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCreateImageView\0")),
            );
            cmd.destroy_image_view = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkDestroyImageView\0")),
            );
            cmd.create_shader_module = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateShaderModule\0",
                    ),
                ),
            );
            cmd.destroy_shader_module = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyShaderModule\0",
                    ),
                ),
            );
            cmd.create_pipeline_cache = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreatePipelineCache\0",
                    ),
                ),
            );
            cmd.destroy_pipeline_cache = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyPipelineCache\0",
                    ),
                ),
            );
            cmd.get_pipeline_cache_data = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPipelineCacheData\0",
                    ),
                ),
            );
            cmd.merge_pipeline_caches = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkMergePipelineCaches\0",
                    ),
                ),
            );
            cmd.create_pipeline_binaries_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreatePipelineBinariesKHR\0",
                    ),
                ),
            );
            cmd.destroy_pipeline_binary_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyPipelineBinaryKHR\0",
                    ),
                ),
            );
            cmd.get_pipeline_key_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPipelineKeyKHR\0",
                    ),
                ),
            );
            cmd.get_pipeline_binary_data_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPipelineBinaryDataKHR\0",
                    ),
                ),
            );
            cmd.release_captured_pipeline_data_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkReleaseCapturedPipelineDataKHR\0",
                    ),
                ),
            );
            cmd.create_graphics_pipelines = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateGraphicsPipelines\0",
                    ),
                ),
            );
            cmd.create_compute_pipelines = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateComputePipelines\0",
                    ),
                ),
            );
            cmd.get_device_subpass_shading_max_workgroup_size_huawei = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI\0",
                    ),
                ),
            );
            cmd.destroy_pipeline = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkDestroyPipeline\0")),
            );
            cmd.create_pipeline_layout = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreatePipelineLayout\0",
                    ),
                ),
            );
            cmd.destroy_pipeline_layout = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyPipelineLayout\0",
                    ),
                ),
            );
            cmd.create_sampler = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCreateSampler\0")),
            );
            cmd.destroy_sampler = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkDestroySampler\0")),
            );
            cmd.create_descriptor_set_layout = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateDescriptorSetLayout\0",
                    ),
                ),
            );
            cmd.destroy_descriptor_set_layout = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyDescriptorSetLayout\0",
                    ),
                ),
            );
            cmd.create_descriptor_pool = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateDescriptorPool\0",
                    ),
                ),
            );
            cmd.destroy_descriptor_pool = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyDescriptorPool\0",
                    ),
                ),
            );
            cmd.reset_descriptor_pool = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkResetDescriptorPool\0",
                    ),
                ),
            );
            cmd.allocate_descriptor_sets = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkAllocateDescriptorSets\0",
                    ),
                ),
            );
            cmd.free_descriptor_sets = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkFreeDescriptorSets\0",
                    ),
                ),
            );
            cmd.update_descriptor_sets = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkUpdateDescriptorSets\0",
                    ),
                ),
            );
            cmd.create_framebuffer = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateFramebuffer\0",
                    ),
                ),
            );
            cmd.destroy_framebuffer = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyFramebuffer\0",
                    ),
                ),
            );
            cmd.create_render_pass = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCreateRenderPass\0")),
            );
            cmd.destroy_render_pass = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyRenderPass\0",
                    ),
                ),
            );
            cmd.get_render_area_granularity = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetRenderAreaGranularity\0",
                    ),
                ),
            );
            cmd.get_rendering_area_granularity = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetRenderingAreaGranularity\0",
                    ),
                ),
            );
            if cmd.get_rendering_area_granularity.is_none() {
                cmd.get_rendering_area_granularity = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetRenderingAreaGranularityKHR\0",
                        ),
                    ),
                );
            }
            cmd.create_command_pool = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateCommandPool\0",
                    ),
                ),
            );
            cmd.destroy_command_pool = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyCommandPool\0",
                    ),
                ),
            );
            cmd.reset_command_pool = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkResetCommandPool\0")),
            );
            cmd.allocate_command_buffers = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkAllocateCommandBuffers\0",
                    ),
                ),
            );
            cmd.free_command_buffers = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkFreeCommandBuffers\0",
                    ),
                ),
            );
            cmd.begin_command_buffer = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkBeginCommandBuffer\0",
                    ),
                ),
            );
            cmd.end_command_buffer = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkEndCommandBuffer\0")),
            );
            cmd.reset_command_buffer = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkResetCommandBuffer\0",
                    ),
                ),
            );
            cmd.cmd_bind_pipeline = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdBindPipeline\0")),
            );
            cmd.cmd_set_attachment_feedback_loop_enable_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetAttachmentFeedbackLoopEnableEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_viewport = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdSetViewport\0")),
            );
            cmd.cmd_set_scissor = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdSetScissor\0")),
            );
            cmd.cmd_set_line_width = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdSetLineWidth\0")),
            );
            cmd.cmd_set_depth_bias = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdSetDepthBias\0")),
            );
            cmd.cmd_set_blend_constants = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetBlendConstants\0",
                    ),
                ),
            );
            cmd.cmd_set_depth_bounds = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetDepthBounds\0",
                    ),
                ),
            );
            cmd.cmd_set_stencil_compare_mask = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetStencilCompareMask\0",
                    ),
                ),
            );
            cmd.cmd_set_stencil_write_mask = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetStencilWriteMask\0",
                    ),
                ),
            );
            cmd.cmd_set_stencil_reference = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetStencilReference\0",
                    ),
                ),
            );
            cmd.cmd_bind_descriptor_sets = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBindDescriptorSets\0",
                    ),
                ),
            );
            cmd.cmd_bind_index_buffer = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBindIndexBuffer\0",
                    ),
                ),
            );
            cmd.cmd_bind_vertex_buffers = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBindVertexBuffers\0",
                    ),
                ),
            );
            cmd.cmd_draw = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdDraw\0")),
            );
            cmd.cmd_draw_indexed = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdDrawIndexed\0")),
            );
            cmd.cmd_draw_multi_ext = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdDrawMultiEXT\0")),
            );
            cmd.cmd_draw_multi_indexed_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDrawMultiIndexedEXT\0",
                    ),
                ),
            );
            cmd.cmd_draw_indirect = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdDrawIndirect\0")),
            );
            cmd.cmd_draw_indexed_indirect = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDrawIndexedIndirect\0",
                    ),
                ),
            );
            cmd.cmd_dispatch = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdDispatch\0")),
            );
            cmd.cmd_dispatch_indirect = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDispatchIndirect\0",
                    ),
                ),
            );
            cmd.cmd_subpass_shading_huawei = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSubpassShadingHUAWEI\0",
                    ),
                ),
            );
            cmd.cmd_draw_cluster_huawei = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDrawClusterHUAWEI\0",
                    ),
                ),
            );
            cmd.cmd_draw_cluster_indirect_huawei = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDrawClusterIndirectHUAWEI\0",
                    ),
                ),
            );
            cmd.cmd_update_pipeline_indirect_buffer_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdUpdatePipelineIndirectBufferNV\0",
                    ),
                ),
            );
            cmd.cmd_copy_buffer = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdCopyBuffer\0")),
            );
            cmd.cmd_copy_image = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdCopyImage\0")),
            );
            cmd.cmd_blit_image = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdBlitImage\0")),
            );
            cmd.cmd_copy_buffer_to_image = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdCopyBufferToImage\0",
                    ),
                ),
            );
            cmd.cmd_copy_image_to_buffer = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdCopyImageToBuffer\0",
                    ),
                ),
            );
            cmd.cmd_copy_memory_indirect_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdCopyMemoryIndirectNV\0",
                    ),
                ),
            );
            cmd.cmd_copy_memory_indirect_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdCopyMemoryIndirectKHR\0",
                    ),
                ),
            );
            cmd.cmd_copy_memory_to_image_indirect_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdCopyMemoryToImageIndirectNV\0",
                    ),
                ),
            );
            cmd.cmd_copy_memory_to_image_indirect_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdCopyMemoryToImageIndirectKHR\0",
                    ),
                ),
            );
            cmd.cmd_update_buffer = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdUpdateBuffer\0")),
            );
            cmd.cmd_fill_buffer = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdFillBuffer\0")),
            );
            cmd.cmd_clear_color_image = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdClearColorImage\0",
                    ),
                ),
            );
            cmd.cmd_clear_depth_stencil_image = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdClearDepthStencilImage\0",
                    ),
                ),
            );
            cmd.cmd_clear_attachments = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdClearAttachments\0",
                    ),
                ),
            );
            cmd.cmd_resolve_image = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdResolveImage\0")),
            );
            cmd.cmd_set_event = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdSetEvent\0")),
            );
            cmd.cmd_reset_event = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdResetEvent\0")),
            );
            cmd.cmd_wait_events = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdWaitEvents\0")),
            );
            cmd.cmd_pipeline_barrier = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdPipelineBarrier\0",
                    ),
                ),
            );
            cmd.cmd_begin_query = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdBeginQuery\0")),
            );
            cmd.cmd_end_query = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdEndQuery\0")),
            );
            cmd.cmd_begin_conditional_rendering_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBeginConditionalRenderingEXT\0",
                    ),
                ),
            );
            cmd.cmd_end_conditional_rendering_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdEndConditionalRenderingEXT\0",
                    ),
                ),
            );
            cmd.cmd_begin_custom_resolve_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBeginCustomResolveEXT\0",
                    ),
                ),
            );
            cmd.cmd_reset_query_pool = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdResetQueryPool\0",
                    ),
                ),
            );
            cmd.cmd_write_timestamp = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdWriteTimestamp\0",
                    ),
                ),
            );
            cmd.cmd_copy_query_pool_results = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdCopyQueryPoolResults\0",
                    ),
                ),
            );
            cmd.cmd_push_constants = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdPushConstants\0")),
            );
            cmd.cmd_begin_render_pass = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBeginRenderPass\0",
                    ),
                ),
            );
            cmd.cmd_next_subpass = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdNextSubpass\0")),
            );
            cmd.cmd_end_render_pass = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdEndRenderPass\0")),
            );
            cmd.cmd_execute_commands = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdExecuteCommands\0",
                    ),
                ),
            );
            cmd.create_shared_swapchains_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateSharedSwapchainsKHR\0",
                    ),
                ),
            );
            cmd.create_swapchain_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateSwapchainKHR\0",
                    ),
                ),
            );
            cmd.destroy_swapchain_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroySwapchainKHR\0",
                    ),
                ),
            );
            cmd.get_swapchain_images_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetSwapchainImagesKHR\0",
                    ),
                ),
            );
            cmd.acquire_next_image_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkAcquireNextImageKHR\0",
                    ),
                ),
            );
            cmd.queue_present_khr = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkQueuePresentKHR\0")),
            );
            cmd.debug_marker_set_object_name_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDebugMarkerSetObjectNameEXT\0",
                    ),
                ),
            );
            cmd.debug_marker_set_object_tag_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDebugMarkerSetObjectTagEXT\0",
                    ),
                ),
            );
            cmd.cmd_debug_marker_begin_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDebugMarkerBeginEXT\0",
                    ),
                ),
            );
            cmd.cmd_debug_marker_end_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDebugMarkerEndEXT\0",
                    ),
                ),
            );
            cmd.cmd_debug_marker_insert_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDebugMarkerInsertEXT\0",
                    ),
                ),
            );
            cmd.get_memory_win32_handle_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetMemoryWin32HandleNV\0",
                    ),
                ),
            );
            cmd.cmd_execute_generated_commands_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdExecuteGeneratedCommandsNV\0",
                    ),
                ),
            );
            cmd.cmd_preprocess_generated_commands_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdPreprocessGeneratedCommandsNV\0",
                    ),
                ),
            );
            cmd.cmd_bind_pipeline_shader_group_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBindPipelineShaderGroupNV\0",
                    ),
                ),
            );
            cmd.get_generated_commands_memory_requirements_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetGeneratedCommandsMemoryRequirementsNV\0",
                    ),
                ),
            );
            cmd.create_indirect_commands_layout_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateIndirectCommandsLayoutNV\0",
                    ),
                ),
            );
            cmd.destroy_indirect_commands_layout_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyIndirectCommandsLayoutNV\0",
                    ),
                ),
            );
            cmd.cmd_execute_generated_commands_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdExecuteGeneratedCommandsEXT\0",
                    ),
                ),
            );
            cmd.cmd_preprocess_generated_commands_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdPreprocessGeneratedCommandsEXT\0",
                    ),
                ),
            );
            cmd.get_generated_commands_memory_requirements_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetGeneratedCommandsMemoryRequirementsEXT\0",
                    ),
                ),
            );
            cmd.create_indirect_commands_layout_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateIndirectCommandsLayoutEXT\0",
                    ),
                ),
            );
            cmd.destroy_indirect_commands_layout_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyIndirectCommandsLayoutEXT\0",
                    ),
                ),
            );
            cmd.create_indirect_execution_set_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateIndirectExecutionSetEXT\0",
                    ),
                ),
            );
            cmd.destroy_indirect_execution_set_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyIndirectExecutionSetEXT\0",
                    ),
                ),
            );
            cmd.update_indirect_execution_set_pipeline_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkUpdateIndirectExecutionSetPipelineEXT\0",
                    ),
                ),
            );
            cmd.update_indirect_execution_set_shader_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkUpdateIndirectExecutionSetShaderEXT\0",
                    ),
                ),
            );
            cmd.cmd_push_descriptor_set = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdPushDescriptorSet\0",
                    ),
                ),
            );
            if cmd.cmd_push_descriptor_set.is_none() {
                cmd.cmd_push_descriptor_set = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdPushDescriptorSetKHR\0",
                        ),
                    ),
                );
            }
            cmd.trim_command_pool = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkTrimCommandPool\0")),
            );
            if cmd.trim_command_pool.is_none() {
                cmd.trim_command_pool = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkTrimCommandPoolKHR\0",
                        ),
                    ),
                );
            }
            cmd.get_memory_win32_handle_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetMemoryWin32HandleKHR\0",
                    ),
                ),
            );
            cmd.get_memory_win32_handle_properties_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetMemoryWin32HandlePropertiesKHR\0",
                    ),
                ),
            );
            cmd.get_memory_fd_khr = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkGetMemoryFdKHR\0")),
            );
            cmd.get_memory_fd_properties_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetMemoryFdPropertiesKHR\0",
                    ),
                ),
            );
            cmd.get_memory_zircon_handle_fuchsia = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetMemoryZirconHandleFUCHSIA\0",
                    ),
                ),
            );
            cmd.get_memory_zircon_handle_properties_fuchsia = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetMemoryZirconHandlePropertiesFUCHSIA\0",
                    ),
                ),
            );
            cmd.get_memory_remote_address_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetMemoryRemoteAddressNV\0",
                    ),
                ),
            );
            cmd.get_memory_sci_buf_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetMemorySciBufNV\0",
                    ),
                ),
            );
            cmd.get_semaphore_win32_handle_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetSemaphoreWin32HandleKHR\0",
                    ),
                ),
            );
            cmd.import_semaphore_win32_handle_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkImportSemaphoreWin32HandleKHR\0",
                    ),
                ),
            );
            cmd.get_semaphore_fd_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetSemaphoreFdKHR\0",
                    ),
                ),
            );
            cmd.import_semaphore_fd_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkImportSemaphoreFdKHR\0",
                    ),
                ),
            );
            cmd.get_semaphore_zircon_handle_fuchsia = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetSemaphoreZirconHandleFUCHSIA\0",
                    ),
                ),
            );
            cmd.import_semaphore_zircon_handle_fuchsia = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkImportSemaphoreZirconHandleFUCHSIA\0",
                    ),
                ),
            );
            cmd.get_fence_win32_handle_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetFenceWin32HandleKHR\0",
                    ),
                ),
            );
            cmd.import_fence_win32_handle_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkImportFenceWin32HandleKHR\0",
                    ),
                ),
            );
            cmd.get_fence_fd_khr = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkGetFenceFdKHR\0")),
            );
            cmd.import_fence_fd_khr = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkImportFenceFdKHR\0")),
            );
            cmd.get_fence_sci_sync_fence_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetFenceSciSyncFenceNV\0",
                    ),
                ),
            );
            cmd.get_fence_sci_sync_obj_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetFenceSciSyncObjNV\0",
                    ),
                ),
            );
            cmd.import_fence_sci_sync_fence_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkImportFenceSciSyncFenceNV\0",
                    ),
                ),
            );
            cmd.import_fence_sci_sync_obj_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkImportFenceSciSyncObjNV\0",
                    ),
                ),
            );
            cmd.get_semaphore_sci_sync_obj_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetSemaphoreSciSyncObjNV\0",
                    ),
                ),
            );
            cmd.import_semaphore_sci_sync_obj_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkImportSemaphoreSciSyncObjNV\0",
                    ),
                ),
            );
            cmd.create_semaphore_sci_sync_pool_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateSemaphoreSciSyncPoolNV\0",
                    ),
                ),
            );
            cmd.destroy_semaphore_sci_sync_pool_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroySemaphoreSciSyncPoolNV\0",
                    ),
                ),
            );
            cmd.display_power_control_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDisplayPowerControlEXT\0",
                    ),
                ),
            );
            cmd.register_device_event_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkRegisterDeviceEventEXT\0",
                    ),
                ),
            );
            cmd.register_display_event_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkRegisterDisplayEventEXT\0",
                    ),
                ),
            );
            cmd.get_swapchain_counter_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetSwapchainCounterEXT\0",
                    ),
                ),
            );
            cmd.get_device_group_peer_memory_features = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDeviceGroupPeerMemoryFeatures\0",
                    ),
                ),
            );
            if cmd.get_device_group_peer_memory_features.is_none() {
                cmd.get_device_group_peer_memory_features = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetDeviceGroupPeerMemoryFeaturesKHR\0",
                        ),
                    ),
                );
            }
            cmd.bind_buffer_memory2 = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkBindBufferMemory2\0",
                    ),
                ),
            );
            if cmd.bind_buffer_memory2.is_none() {
                cmd.bind_buffer_memory2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkBindBufferMemory2KHR\0",
                        ),
                    ),
                );
            }
            cmd.bind_image_memory2 = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkBindImageMemory2\0")),
            );
            if cmd.bind_image_memory2.is_none() {
                cmd.bind_image_memory2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkBindImageMemory2KHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_set_device_mask = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdSetDeviceMask\0")),
            );
            if cmd.cmd_set_device_mask.is_none() {
                cmd.cmd_set_device_mask = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdSetDeviceMaskKHR\0",
                        ),
                    ),
                );
            }
            cmd.get_device_group_present_capabilities_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDeviceGroupPresentCapabilitiesKHR\0",
                    ),
                ),
            );
            cmd.get_device_group_surface_present_modes_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDeviceGroupSurfacePresentModesKHR\0",
                    ),
                ),
            );
            cmd.acquire_next_image2_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkAcquireNextImage2KHR\0",
                    ),
                ),
            );
            cmd.cmd_dispatch_base = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdDispatchBase\0")),
            );
            if cmd.cmd_dispatch_base.is_none() {
                cmd.cmd_dispatch_base = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdDispatchBaseKHR\0",
                        ),
                    ),
                );
            }
            cmd.create_descriptor_update_template = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateDescriptorUpdateTemplate\0",
                    ),
                ),
            );
            if cmd.create_descriptor_update_template.is_none() {
                cmd.create_descriptor_update_template = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCreateDescriptorUpdateTemplateKHR\0",
                        ),
                    ),
                );
            }
            cmd.destroy_descriptor_update_template = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyDescriptorUpdateTemplate\0",
                    ),
                ),
            );
            if cmd.destroy_descriptor_update_template.is_none() {
                cmd.destroy_descriptor_update_template = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkDestroyDescriptorUpdateTemplateKHR\0",
                        ),
                    ),
                );
            }
            cmd.update_descriptor_set_with_template = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkUpdateDescriptorSetWithTemplate\0",
                    ),
                ),
            );
            if cmd.update_descriptor_set_with_template.is_none() {
                cmd.update_descriptor_set_with_template = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkUpdateDescriptorSetWithTemplateKHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_push_descriptor_set_with_template = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdPushDescriptorSetWithTemplate\0",
                    ),
                ),
            );
            if cmd.cmd_push_descriptor_set_with_template.is_none() {
                cmd.cmd_push_descriptor_set_with_template = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdPushDescriptorSetWithTemplateKHR\0",
                        ),
                    ),
                );
            }
            cmd.set_hdr_metadata_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkSetHdrMetadataEXT\0",
                    ),
                ),
            );
            cmd.get_swapchain_status_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetSwapchainStatusKHR\0",
                    ),
                ),
            );
            cmd.get_refresh_cycle_duration_google = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetRefreshCycleDurationGOOGLE\0",
                    ),
                ),
            );
            cmd.get_past_presentation_timing_google = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPastPresentationTimingGOOGLE\0",
                    ),
                ),
            );
            cmd.cmd_set_viewport_w_scaling_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetViewportWScalingNV\0",
                    ),
                ),
            );
            cmd.cmd_set_discard_rectangle_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetDiscardRectangleEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_discard_rectangle_enable_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetDiscardRectangleEnableEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_discard_rectangle_mode_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetDiscardRectangleModeEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_sample_locations_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetSampleLocationsEXT\0",
                    ),
                ),
            );
            cmd.get_buffer_memory_requirements2 = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetBufferMemoryRequirements2\0",
                    ),
                ),
            );
            if cmd.get_buffer_memory_requirements2.is_none() {
                cmd.get_buffer_memory_requirements2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetBufferMemoryRequirements2KHR\0",
                        ),
                    ),
                );
            }
            cmd.get_image_memory_requirements2 = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetImageMemoryRequirements2\0",
                    ),
                ),
            );
            if cmd.get_image_memory_requirements2.is_none() {
                cmd.get_image_memory_requirements2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetImageMemoryRequirements2KHR\0",
                        ),
                    ),
                );
            }
            cmd.get_image_sparse_memory_requirements2 = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetImageSparseMemoryRequirements2\0",
                    ),
                ),
            );
            if cmd.get_image_sparse_memory_requirements2.is_none() {
                cmd.get_image_sparse_memory_requirements2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetImageSparseMemoryRequirements2KHR\0",
                        ),
                    ),
                );
            }
            cmd.get_device_buffer_memory_requirements = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDeviceBufferMemoryRequirements\0",
                    ),
                ),
            );
            if cmd.get_device_buffer_memory_requirements.is_none() {
                cmd.get_device_buffer_memory_requirements = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetDeviceBufferMemoryRequirementsKHR\0",
                        ),
                    ),
                );
            }
            cmd.get_device_image_memory_requirements = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDeviceImageMemoryRequirements\0",
                    ),
                ),
            );
            if cmd.get_device_image_memory_requirements.is_none() {
                cmd.get_device_image_memory_requirements = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetDeviceImageMemoryRequirementsKHR\0",
                        ),
                    ),
                );
            }
            cmd.get_device_image_sparse_memory_requirements = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDeviceImageSparseMemoryRequirements\0",
                    ),
                ),
            );
            if cmd.get_device_image_sparse_memory_requirements.is_none() {
                cmd.get_device_image_sparse_memory_requirements = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetDeviceImageSparseMemoryRequirementsKHR\0",
                        ),
                    ),
                );
            }
            cmd.create_sampler_ycbcr_conversion = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateSamplerYcbcrConversion\0",
                    ),
                ),
            );
            if cmd.create_sampler_ycbcr_conversion.is_none() {
                cmd.create_sampler_ycbcr_conversion = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCreateSamplerYcbcrConversionKHR\0",
                        ),
                    ),
                );
            }
            cmd.destroy_sampler_ycbcr_conversion = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroySamplerYcbcrConversion\0",
                    ),
                ),
            );
            if cmd.destroy_sampler_ycbcr_conversion.is_none() {
                cmd.destroy_sampler_ycbcr_conversion = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkDestroySamplerYcbcrConversionKHR\0",
                        ),
                    ),
                );
            }
            cmd.get_device_queue2 = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkGetDeviceQueue2\0")),
            );
            cmd.create_validation_cache_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateValidationCacheEXT\0",
                    ),
                ),
            );
            cmd.destroy_validation_cache_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyValidationCacheEXT\0",
                    ),
                ),
            );
            cmd.get_validation_cache_data_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetValidationCacheDataEXT\0",
                    ),
                ),
            );
            cmd.merge_validation_caches_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkMergeValidationCachesEXT\0",
                    ),
                ),
            );
            cmd.get_descriptor_set_layout_support = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDescriptorSetLayoutSupport\0",
                    ),
                ),
            );
            if cmd.get_descriptor_set_layout_support.is_none() {
                cmd.get_descriptor_set_layout_support = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetDescriptorSetLayoutSupportKHR\0",
                        ),
                    ),
                );
            }
            cmd.get_swapchain_gralloc_usage_android = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetSwapchainGrallocUsageANDROID\0",
                    ),
                ),
            );
            cmd.get_swapchain_gralloc_usage2_android = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetSwapchainGrallocUsage2ANDROID\0",
                    ),
                ),
            );
            cmd.acquire_image_android = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkAcquireImageANDROID\0",
                    ),
                ),
            );
            cmd.queue_signal_release_image_android = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkQueueSignalReleaseImageANDROID\0",
                    ),
                ),
            );
            cmd.get_shader_info_amd = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkGetShaderInfoAMD\0")),
            );
            cmd.set_local_dimming_amd = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkSetLocalDimmingAMD\0",
                    ),
                ),
            );
            cmd.get_calibrated_timestamps_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetCalibratedTimestampsKHR\0",
                    ),
                ),
            );
            if cmd.get_calibrated_timestamps_khr.is_none() {
                cmd.get_calibrated_timestamps_khr = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetCalibratedTimestampsEXT\0",
                        ),
                    ),
                );
            }
            cmd.set_debug_utils_object_name_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkSetDebugUtilsObjectNameEXT\0",
                    ),
                ),
            );
            cmd.set_debug_utils_object_tag_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkSetDebugUtilsObjectTagEXT\0",
                    ),
                ),
            );
            cmd.queue_begin_debug_utils_label_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkQueueBeginDebugUtilsLabelEXT\0",
                    ),
                ),
            );
            cmd.queue_end_debug_utils_label_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkQueueEndDebugUtilsLabelEXT\0",
                    ),
                ),
            );
            cmd.queue_insert_debug_utils_label_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkQueueInsertDebugUtilsLabelEXT\0",
                    ),
                ),
            );
            cmd.cmd_begin_debug_utils_label_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBeginDebugUtilsLabelEXT\0",
                    ),
                ),
            );
            cmd.cmd_end_debug_utils_label_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdEndDebugUtilsLabelEXT\0",
                    ),
                ),
            );
            cmd.cmd_insert_debug_utils_label_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdInsertDebugUtilsLabelEXT\0",
                    ),
                ),
            );
            cmd.get_memory_host_pointer_properties_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetMemoryHostPointerPropertiesEXT\0",
                    ),
                ),
            );
            cmd.cmd_write_buffer_marker_amd = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdWriteBufferMarkerAMD\0",
                    ),
                ),
            );
            cmd.create_render_pass2 = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateRenderPass2\0",
                    ),
                ),
            );
            if cmd.create_render_pass2.is_none() {
                cmd.create_render_pass2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCreateRenderPass2KHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_begin_render_pass2 = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBeginRenderPass2\0",
                    ),
                ),
            );
            if cmd.cmd_begin_render_pass2.is_none() {
                cmd.cmd_begin_render_pass2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdBeginRenderPass2KHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_next_subpass2 = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdNextSubpass2\0")),
            );
            if cmd.cmd_next_subpass2.is_none() {
                cmd.cmd_next_subpass2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdNextSubpass2KHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_end_render_pass2 = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdEndRenderPass2\0",
                    ),
                ),
            );
            if cmd.cmd_end_render_pass2.is_none() {
                cmd.cmd_end_render_pass2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdEndRenderPass2KHR\0",
                        ),
                    ),
                );
            }
            cmd.get_semaphore_counter_value = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetSemaphoreCounterValue\0",
                    ),
                ),
            );
            if cmd.get_semaphore_counter_value.is_none() {
                cmd.get_semaphore_counter_value = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetSemaphoreCounterValueKHR\0",
                        ),
                    ),
                );
            }
            cmd.wait_semaphores = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkWaitSemaphores\0")),
            );
            if cmd.wait_semaphores.is_none() {
                cmd.wait_semaphores = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkWaitSemaphoresKHR\0",
                        ),
                    ),
                );
            }
            cmd.signal_semaphore = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkSignalSemaphore\0")),
            );
            if cmd.signal_semaphore.is_none() {
                cmd.signal_semaphore = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkSignalSemaphoreKHR\0",
                        ),
                    ),
                );
            }
            cmd.get_android_hardware_buffer_properties_android = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetAndroidHardwareBufferPropertiesANDROID\0",
                    ),
                ),
            );
            cmd.get_memory_android_hardware_buffer_android = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetMemoryAndroidHardwareBufferANDROID\0",
                    ),
                ),
            );
            cmd.cmd_draw_indirect_count = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDrawIndirectCount\0",
                    ),
                ),
            );
            if cmd.cmd_draw_indirect_count.is_none() {
                cmd.cmd_draw_indirect_count = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdDrawIndirectCountAMD\0",
                        ),
                    ),
                );
            }
            if cmd.cmd_draw_indirect_count.is_none() {
                cmd.cmd_draw_indirect_count = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdDrawIndirectCountKHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_draw_indexed_indirect_count = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDrawIndexedIndirectCount\0",
                    ),
                ),
            );
            if cmd.cmd_draw_indexed_indirect_count.is_none() {
                cmd.cmd_draw_indexed_indirect_count = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdDrawIndexedIndirectCountAMD\0",
                        ),
                    ),
                );
            }
            if cmd.cmd_draw_indexed_indirect_count.is_none() {
                cmd.cmd_draw_indexed_indirect_count = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdDrawIndexedIndirectCountKHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_set_checkpoint_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetCheckpointNV\0",
                    ),
                ),
            );
            cmd.get_queue_checkpoint_data_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetQueueCheckpointDataNV\0",
                    ),
                ),
            );
            cmd.cmd_bind_transform_feedback_buffers_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBindTransformFeedbackBuffersEXT\0",
                    ),
                ),
            );
            cmd.cmd_begin_transform_feedback_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBeginTransformFeedbackEXT\0",
                    ),
                ),
            );
            cmd.cmd_end_transform_feedback_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdEndTransformFeedbackEXT\0",
                    ),
                ),
            );
            cmd.cmd_begin_query_indexed_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBeginQueryIndexedEXT\0",
                    ),
                ),
            );
            cmd.cmd_end_query_indexed_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdEndQueryIndexedEXT\0",
                    ),
                ),
            );
            cmd.cmd_draw_indirect_byte_count_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDrawIndirectByteCountEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_exclusive_scissor_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetExclusiveScissorNV\0",
                    ),
                ),
            );
            cmd.cmd_set_exclusive_scissor_enable_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetExclusiveScissorEnableNV\0",
                    ),
                ),
            );
            cmd.cmd_bind_shading_rate_image_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBindShadingRateImageNV\0",
                    ),
                ),
            );
            cmd.cmd_set_viewport_shading_rate_palette_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetViewportShadingRatePaletteNV\0",
                    ),
                ),
            );
            cmd.cmd_set_coarse_sample_order_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetCoarseSampleOrderNV\0",
                    ),
                ),
            );
            cmd.cmd_draw_mesh_tasks_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDrawMeshTasksNV\0",
                    ),
                ),
            );
            cmd.cmd_draw_mesh_tasks_indirect_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDrawMeshTasksIndirectNV\0",
                    ),
                ),
            );
            cmd.cmd_draw_mesh_tasks_indirect_count_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDrawMeshTasksIndirectCountNV\0",
                    ),
                ),
            );
            cmd.cmd_draw_mesh_tasks_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDrawMeshTasksEXT\0",
                    ),
                ),
            );
            cmd.cmd_draw_mesh_tasks_indirect_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDrawMeshTasksIndirectEXT\0",
                    ),
                ),
            );
            cmd.cmd_draw_mesh_tasks_indirect_count_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDrawMeshTasksIndirectCountEXT\0",
                    ),
                ),
            );
            cmd.compile_deferred_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCompileDeferredNV\0",
                    ),
                ),
            );
            cmd.create_acceleration_structure_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateAccelerationStructureNV\0",
                    ),
                ),
            );
            cmd.cmd_bind_invocation_mask_huawei = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBindInvocationMaskHUAWEI\0",
                    ),
                ),
            );
            cmd.destroy_acceleration_structure_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyAccelerationStructureKHR\0",
                    ),
                ),
            );
            cmd.destroy_acceleration_structure_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyAccelerationStructureNV\0",
                    ),
                ),
            );
            cmd.get_acceleration_structure_memory_requirements_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetAccelerationStructureMemoryRequirementsNV\0",
                    ),
                ),
            );
            cmd.bind_acceleration_structure_memory_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkBindAccelerationStructureMemoryNV\0",
                    ),
                ),
            );
            cmd.cmd_copy_acceleration_structure_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdCopyAccelerationStructureNV\0",
                    ),
                ),
            );
            cmd.cmd_copy_acceleration_structure_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdCopyAccelerationStructureKHR\0",
                    ),
                ),
            );
            cmd.copy_acceleration_structure_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCopyAccelerationStructureKHR\0",
                    ),
                ),
            );
            cmd.cmd_copy_acceleration_structure_to_memory_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdCopyAccelerationStructureToMemoryKHR\0",
                    ),
                ),
            );
            cmd.copy_acceleration_structure_to_memory_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCopyAccelerationStructureToMemoryKHR\0",
                    ),
                ),
            );
            cmd.cmd_copy_memory_to_acceleration_structure_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdCopyMemoryToAccelerationStructureKHR\0",
                    ),
                ),
            );
            cmd.copy_memory_to_acceleration_structure_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCopyMemoryToAccelerationStructureKHR\0",
                    ),
                ),
            );
            cmd.cmd_write_acceleration_structures_properties_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdWriteAccelerationStructuresPropertiesKHR\0",
                    ),
                ),
            );
            cmd.cmd_write_acceleration_structures_properties_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdWriteAccelerationStructuresPropertiesNV\0",
                    ),
                ),
            );
            cmd.cmd_build_acceleration_structure_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBuildAccelerationStructureNV\0",
                    ),
                ),
            );
            cmd.write_acceleration_structures_properties_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkWriteAccelerationStructuresPropertiesKHR\0",
                    ),
                ),
            );
            cmd.cmd_trace_rays_khr = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdTraceRaysKHR\0")),
            );
            cmd.cmd_trace_rays_nv = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdTraceRaysNV\0")),
            );
            cmd.get_ray_tracing_shader_group_handles_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetRayTracingShaderGroupHandlesKHR\0",
                    ),
                ),
            );
            if cmd.get_ray_tracing_shader_group_handles_khr.is_none() {
                cmd.get_ray_tracing_shader_group_handles_khr = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetRayTracingShaderGroupHandlesNV\0",
                        ),
                    ),
                );
            }
            cmd.get_ray_tracing_capture_replay_shader_group_handles_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetRayTracingCaptureReplayShaderGroupHandlesKHR\0",
                    ),
                ),
            );
            cmd.get_acceleration_structure_handle_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetAccelerationStructureHandleNV\0",
                    ),
                ),
            );
            cmd.create_ray_tracing_pipelines_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateRayTracingPipelinesNV\0",
                    ),
                ),
            );
            cmd.create_ray_tracing_pipelines_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateRayTracingPipelinesKHR\0",
                    ),
                ),
            );
            cmd.cmd_trace_rays_indirect_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdTraceRaysIndirectKHR\0",
                    ),
                ),
            );
            cmd.cmd_trace_rays_indirect2_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdTraceRaysIndirect2KHR\0",
                    ),
                ),
            );
            cmd.get_cluster_acceleration_structure_build_sizes_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetClusterAccelerationStructureBuildSizesNV\0",
                    ),
                ),
            );
            cmd.cmd_build_cluster_acceleration_structure_indirect_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBuildClusterAccelerationStructureIndirectNV\0",
                    ),
                ),
            );
            cmd.get_device_acceleration_structure_compatibility_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDeviceAccelerationStructureCompatibilityKHR\0",
                    ),
                ),
            );
            cmd.get_ray_tracing_shader_group_stack_size_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetRayTracingShaderGroupStackSizeKHR\0",
                    ),
                ),
            );
            cmd.cmd_set_ray_tracing_pipeline_stack_size_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetRayTracingPipelineStackSizeKHR\0",
                    ),
                ),
            );
            cmd.get_image_view_handle_nvx = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetImageViewHandleNVX\0",
                    ),
                ),
            );
            cmd.get_image_view_handle64_nvx = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetImageViewHandle64NVX\0",
                    ),
                ),
            );
            cmd.get_image_view_address_nvx = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetImageViewAddressNVX\0",
                    ),
                ),
            );
            cmd.get_device_combined_image_sampler_index_nvx = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDeviceCombinedImageSamplerIndexNVX\0",
                    ),
                ),
            );
            cmd.get_device_group_surface_present_modes2_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDeviceGroupSurfacePresentModes2EXT\0",
                    ),
                ),
            );
            cmd.acquire_full_screen_exclusive_mode_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkAcquireFullScreenExclusiveModeEXT\0",
                    ),
                ),
            );
            cmd.release_full_screen_exclusive_mode_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkReleaseFullScreenExclusiveModeEXT\0",
                    ),
                ),
            );
            cmd.acquire_profiling_lock_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkAcquireProfilingLockKHR\0",
                    ),
                ),
            );
            cmd.release_profiling_lock_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkReleaseProfilingLockKHR\0",
                    ),
                ),
            );
            cmd.get_image_drm_format_modifier_properties_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetImageDrmFormatModifierPropertiesEXT\0",
                    ),
                ),
            );
            cmd.get_buffer_opaque_capture_address = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetBufferOpaqueCaptureAddress\0",
                    ),
                ),
            );
            if cmd.get_buffer_opaque_capture_address.is_none() {
                cmd.get_buffer_opaque_capture_address = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetBufferOpaqueCaptureAddressKHR\0",
                        ),
                    ),
                );
            }
            cmd.get_buffer_device_address = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetBufferDeviceAddress\0",
                    ),
                ),
            );
            if cmd.get_buffer_device_address.is_none() {
                cmd.get_buffer_device_address = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetBufferDeviceAddressKHR\0",
                        ),
                    ),
                );
            }
            if cmd.get_buffer_device_address.is_none() {
                cmd.get_buffer_device_address = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetBufferDeviceAddressEXT\0",
                        ),
                    ),
                );
            }
            cmd.initialize_performance_api_intel = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkInitializePerformanceApiINTEL\0",
                    ),
                ),
            );
            cmd.uninitialize_performance_api_intel = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkUninitializePerformanceApiINTEL\0",
                    ),
                ),
            );
            cmd.cmd_set_performance_marker_intel = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetPerformanceMarkerINTEL\0",
                    ),
                ),
            );
            cmd.cmd_set_performance_stream_marker_intel = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetPerformanceStreamMarkerINTEL\0",
                    ),
                ),
            );
            cmd.cmd_set_performance_override_intel = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetPerformanceOverrideINTEL\0",
                    ),
                ),
            );
            cmd.acquire_performance_configuration_intel = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkAcquirePerformanceConfigurationINTEL\0",
                    ),
                ),
            );
            cmd.release_performance_configuration_intel = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkReleasePerformanceConfigurationINTEL\0",
                    ),
                ),
            );
            cmd.queue_set_performance_configuration_intel = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkQueueSetPerformanceConfigurationINTEL\0",
                    ),
                ),
            );
            cmd.get_performance_parameter_intel = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPerformanceParameterINTEL\0",
                    ),
                ),
            );
            cmd.get_device_memory_opaque_capture_address = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDeviceMemoryOpaqueCaptureAddress\0",
                    ),
                ),
            );
            if cmd.get_device_memory_opaque_capture_address.is_none() {
                cmd.get_device_memory_opaque_capture_address = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetDeviceMemoryOpaqueCaptureAddressKHR\0",
                        ),
                    ),
                );
            }
            cmd.get_pipeline_executable_properties_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPipelineExecutablePropertiesKHR\0",
                    ),
                ),
            );
            cmd.get_pipeline_executable_statistics_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPipelineExecutableStatisticsKHR\0",
                    ),
                ),
            );
            cmd.get_pipeline_executable_internal_representations_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPipelineExecutableInternalRepresentationsKHR\0",
                    ),
                ),
            );
            cmd.cmd_set_line_stipple = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetLineStipple\0",
                    ),
                ),
            );
            if cmd.cmd_set_line_stipple.is_none() {
                cmd.cmd_set_line_stipple = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdSetLineStippleEXT\0",
                        ),
                    ),
                );
            }
            if cmd.cmd_set_line_stipple.is_none() {
                cmd.cmd_set_line_stipple = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdSetLineStippleKHR\0",
                        ),
                    ),
                );
            }
            cmd.get_fault_data = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkGetFaultData\0")),
            );
            cmd.create_acceleration_structure_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateAccelerationStructureKHR\0",
                    ),
                ),
            );
            cmd.cmd_build_acceleration_structures_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBuildAccelerationStructuresKHR\0",
                    ),
                ),
            );
            cmd.cmd_build_acceleration_structures_indirect_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBuildAccelerationStructuresIndirectKHR\0",
                    ),
                ),
            );
            cmd.build_acceleration_structures_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkBuildAccelerationStructuresKHR\0",
                    ),
                ),
            );
            cmd.get_acceleration_structure_device_address_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetAccelerationStructureDeviceAddressKHR\0",
                    ),
                ),
            );
            cmd.create_deferred_operation_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateDeferredOperationKHR\0",
                    ),
                ),
            );
            cmd.destroy_deferred_operation_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyDeferredOperationKHR\0",
                    ),
                ),
            );
            cmd.get_deferred_operation_max_concurrency_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDeferredOperationMaxConcurrencyKHR\0",
                    ),
                ),
            );
            cmd.get_deferred_operation_result_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDeferredOperationResultKHR\0",
                    ),
                ),
            );
            cmd.deferred_operation_join_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDeferredOperationJoinKHR\0",
                    ),
                ),
            );
            cmd.get_pipeline_indirect_memory_requirements_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPipelineIndirectMemoryRequirementsNV\0",
                    ),
                ),
            );
            cmd.get_pipeline_indirect_device_address_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPipelineIndirectDeviceAddressNV\0",
                    ),
                ),
            );
            cmd.anti_lag_update_amd = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkAntiLagUpdateAMD\0")),
            );
            cmd.cmd_set_cull_mode = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdSetCullMode\0")),
            );
            if cmd.cmd_set_cull_mode.is_none() {
                cmd.cmd_set_cull_mode = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdSetCullModeEXT\0",
                        ),
                    ),
                );
            }
            cmd.cmd_set_front_face = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdSetFrontFace\0")),
            );
            if cmd.cmd_set_front_face.is_none() {
                cmd.cmd_set_front_face = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdSetFrontFaceEXT\0",
                        ),
                    ),
                );
            }
            cmd.cmd_set_primitive_topology = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetPrimitiveTopology\0",
                    ),
                ),
            );
            if cmd.cmd_set_primitive_topology.is_none() {
                cmd.cmd_set_primitive_topology = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdSetPrimitiveTopologyEXT\0",
                        ),
                    ),
                );
            }
            cmd.cmd_set_viewport_with_count = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetViewportWithCount\0",
                    ),
                ),
            );
            if cmd.cmd_set_viewport_with_count.is_none() {
                cmd.cmd_set_viewport_with_count = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdSetViewportWithCountEXT\0",
                        ),
                    ),
                );
            }
            cmd.cmd_set_scissor_with_count = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetScissorWithCount\0",
                    ),
                ),
            );
            if cmd.cmd_set_scissor_with_count.is_none() {
                cmd.cmd_set_scissor_with_count = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdSetScissorWithCountEXT\0",
                        ),
                    ),
                );
            }
            cmd.cmd_bind_index_buffer2 = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBindIndexBuffer2\0",
                    ),
                ),
            );
            if cmd.cmd_bind_index_buffer2.is_none() {
                cmd.cmd_bind_index_buffer2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdBindIndexBuffer2KHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_bind_vertex_buffers2 = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBindVertexBuffers2\0",
                    ),
                ),
            );
            if cmd.cmd_bind_vertex_buffers2.is_none() {
                cmd.cmd_bind_vertex_buffers2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdBindVertexBuffers2EXT\0",
                        ),
                    ),
                );
            }
            cmd.cmd_set_depth_test_enable = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetDepthTestEnable\0",
                    ),
                ),
            );
            if cmd.cmd_set_depth_test_enable.is_none() {
                cmd.cmd_set_depth_test_enable = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdSetDepthTestEnableEXT\0",
                        ),
                    ),
                );
            }
            cmd.cmd_set_depth_write_enable = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetDepthWriteEnable\0",
                    ),
                ),
            );
            if cmd.cmd_set_depth_write_enable.is_none() {
                cmd.cmd_set_depth_write_enable = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdSetDepthWriteEnableEXT\0",
                        ),
                    ),
                );
            }
            cmd.cmd_set_depth_compare_op = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetDepthCompareOp\0",
                    ),
                ),
            );
            if cmd.cmd_set_depth_compare_op.is_none() {
                cmd.cmd_set_depth_compare_op = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdSetDepthCompareOpEXT\0",
                        ),
                    ),
                );
            }
            cmd.cmd_set_depth_bounds_test_enable = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetDepthBoundsTestEnable\0",
                    ),
                ),
            );
            if cmd.cmd_set_depth_bounds_test_enable.is_none() {
                cmd.cmd_set_depth_bounds_test_enable = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdSetDepthBoundsTestEnableEXT\0",
                        ),
                    ),
                );
            }
            cmd.cmd_set_stencil_test_enable = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetStencilTestEnable\0",
                    ),
                ),
            );
            if cmd.cmd_set_stencil_test_enable.is_none() {
                cmd.cmd_set_stencil_test_enable = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdSetStencilTestEnableEXT\0",
                        ),
                    ),
                );
            }
            cmd.cmd_set_stencil_op = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdSetStencilOp\0")),
            );
            if cmd.cmd_set_stencil_op.is_none() {
                cmd.cmd_set_stencil_op = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdSetStencilOpEXT\0",
                        ),
                    ),
                );
            }
            cmd.cmd_set_patch_control_points_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetPatchControlPointsEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_rasterizer_discard_enable = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetRasterizerDiscardEnable\0",
                    ),
                ),
            );
            if cmd.cmd_set_rasterizer_discard_enable.is_none() {
                cmd.cmd_set_rasterizer_discard_enable = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdSetRasterizerDiscardEnableEXT\0",
                        ),
                    ),
                );
            }
            cmd.cmd_set_depth_bias_enable = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetDepthBiasEnable\0",
                    ),
                ),
            );
            if cmd.cmd_set_depth_bias_enable.is_none() {
                cmd.cmd_set_depth_bias_enable = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdSetDepthBiasEnableEXT\0",
                        ),
                    ),
                );
            }
            cmd.cmd_set_logic_op_ext = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdSetLogicOpEXT\0")),
            );
            cmd.cmd_set_primitive_restart_enable = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetPrimitiveRestartEnable\0",
                    ),
                ),
            );
            if cmd.cmd_set_primitive_restart_enable.is_none() {
                cmd.cmd_set_primitive_restart_enable = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdSetPrimitiveRestartEnableEXT\0",
                        ),
                    ),
                );
            }
            cmd.cmd_set_tessellation_domain_origin_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetTessellationDomainOriginEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_depth_clamp_enable_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetDepthClampEnableEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_polygon_mode_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetPolygonModeEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_rasterization_samples_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetRasterizationSamplesEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_sample_mask_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetSampleMaskEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_alpha_to_coverage_enable_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetAlphaToCoverageEnableEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_alpha_to_one_enable_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetAlphaToOneEnableEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_logic_op_enable_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetLogicOpEnableEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_color_blend_enable_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetColorBlendEnableEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_color_blend_equation_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetColorBlendEquationEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_color_write_mask_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetColorWriteMaskEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_rasterization_stream_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetRasterizationStreamEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_conservative_rasterization_mode_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetConservativeRasterizationModeEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_extra_primitive_overestimation_size_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetExtraPrimitiveOverestimationSizeEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_depth_clip_enable_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetDepthClipEnableEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_sample_locations_enable_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetSampleLocationsEnableEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_color_blend_advanced_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetColorBlendAdvancedEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_provoking_vertex_mode_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetProvokingVertexModeEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_line_rasterization_mode_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetLineRasterizationModeEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_line_stipple_enable_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetLineStippleEnableEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_depth_clip_negative_one_to_one_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetDepthClipNegativeOneToOneEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_viewport_w_scaling_enable_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetViewportWScalingEnableNV\0",
                    ),
                ),
            );
            cmd.cmd_set_viewport_swizzle_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetViewportSwizzleNV\0",
                    ),
                ),
            );
            cmd.cmd_set_coverage_to_color_enable_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetCoverageToColorEnableNV\0",
                    ),
                ),
            );
            cmd.cmd_set_coverage_to_color_location_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetCoverageToColorLocationNV\0",
                    ),
                ),
            );
            cmd.cmd_set_coverage_modulation_mode_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetCoverageModulationModeNV\0",
                    ),
                ),
            );
            cmd.cmd_set_coverage_modulation_table_enable_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetCoverageModulationTableEnableNV\0",
                    ),
                ),
            );
            cmd.cmd_set_coverage_modulation_table_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetCoverageModulationTableNV\0",
                    ),
                ),
            );
            cmd.cmd_set_shading_rate_image_enable_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetShadingRateImageEnableNV\0",
                    ),
                ),
            );
            cmd.cmd_set_coverage_reduction_mode_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetCoverageReductionModeNV\0",
                    ),
                ),
            );
            cmd.cmd_set_representative_fragment_test_enable_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetRepresentativeFragmentTestEnableNV\0",
                    ),
                ),
            );
            cmd.create_private_data_slot = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreatePrivateDataSlot\0",
                    ),
                ),
            );
            if cmd.create_private_data_slot.is_none() {
                cmd.create_private_data_slot = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCreatePrivateDataSlotEXT\0",
                        ),
                    ),
                );
            }
            cmd.destroy_private_data_slot = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyPrivateDataSlot\0",
                    ),
                ),
            );
            if cmd.destroy_private_data_slot.is_none() {
                cmd.destroy_private_data_slot = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkDestroyPrivateDataSlotEXT\0",
                        ),
                    ),
                );
            }
            cmd.set_private_data = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkSetPrivateData\0")),
            );
            if cmd.set_private_data.is_none() {
                cmd.set_private_data = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkSetPrivateDataEXT\0",
                        ),
                    ),
                );
            }
            cmd.get_private_data = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkGetPrivateData\0")),
            );
            if cmd.get_private_data.is_none() {
                cmd.get_private_data = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetPrivateDataEXT\0",
                        ),
                    ),
                );
            }
            cmd.cmd_copy_buffer2 = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdCopyBuffer2\0")),
            );
            if cmd.cmd_copy_buffer2.is_none() {
                cmd.cmd_copy_buffer2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdCopyBuffer2KHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_copy_image2 = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdCopyImage2\0")),
            );
            if cmd.cmd_copy_image2.is_none() {
                cmd.cmd_copy_image2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdCopyImage2KHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_blit_image2 = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdBlitImage2\0")),
            );
            if cmd.cmd_blit_image2.is_none() {
                cmd.cmd_blit_image2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdBlitImage2KHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_copy_buffer_to_image2 = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdCopyBufferToImage2\0",
                    ),
                ),
            );
            if cmd.cmd_copy_buffer_to_image2.is_none() {
                cmd.cmd_copy_buffer_to_image2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdCopyBufferToImage2KHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_copy_image_to_buffer2 = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdCopyImageToBuffer2\0",
                    ),
                ),
            );
            if cmd.cmd_copy_image_to_buffer2.is_none() {
                cmd.cmd_copy_image_to_buffer2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdCopyImageToBuffer2KHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_resolve_image2 = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdResolveImage2\0")),
            );
            if cmd.cmd_resolve_image2.is_none() {
                cmd.cmd_resolve_image2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdResolveImage2KHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_refresh_objects_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdRefreshObjectsKHR\0",
                    ),
                ),
            );
            cmd.cmd_set_fragment_shading_rate_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetFragmentShadingRateKHR\0",
                    ),
                ),
            );
            cmd.cmd_set_fragment_shading_rate_enum_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetFragmentShadingRateEnumNV\0",
                    ),
                ),
            );
            cmd.get_acceleration_structure_build_sizes_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetAccelerationStructureBuildSizesKHR\0",
                    ),
                ),
            );
            cmd.cmd_set_vertex_input_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetVertexInputEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_color_write_enable_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetColorWriteEnableEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_event2 = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdSetEvent2\0")),
            );
            if cmd.cmd_set_event2.is_none() {
                cmd.cmd_set_event2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdSetEvent2KHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_reset_event2 = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdResetEvent2\0")),
            );
            if cmd.cmd_reset_event2.is_none() {
                cmd.cmd_reset_event2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdResetEvent2KHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_wait_events2 = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdWaitEvents2\0")),
            );
            if cmd.cmd_wait_events2.is_none() {
                cmd.cmd_wait_events2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdWaitEvents2KHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_pipeline_barrier2 = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdPipelineBarrier2\0",
                    ),
                ),
            );
            if cmd.cmd_pipeline_barrier2.is_none() {
                cmd.cmd_pipeline_barrier2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdPipelineBarrier2KHR\0",
                        ),
                    ),
                );
            }
            cmd.queue_submit2 = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkQueueSubmit2\0")),
            );
            if cmd.queue_submit2.is_none() {
                cmd.queue_submit2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkQueueSubmit2KHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_write_timestamp2 = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdWriteTimestamp2\0",
                    ),
                ),
            );
            if cmd.cmd_write_timestamp2.is_none() {
                cmd.cmd_write_timestamp2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdWriteTimestamp2KHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_write_buffer_marker2_amd = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdWriteBufferMarker2AMD\0",
                    ),
                ),
            );
            cmd.get_queue_checkpoint_data2_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetQueueCheckpointData2NV\0",
                    ),
                ),
            );
            cmd.copy_memory_to_image = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCopyMemoryToImage\0",
                    ),
                ),
            );
            if cmd.copy_memory_to_image.is_none() {
                cmd.copy_memory_to_image = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCopyMemoryToImageEXT\0",
                        ),
                    ),
                );
            }
            cmd.copy_image_to_memory = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCopyImageToMemory\0",
                    ),
                ),
            );
            if cmd.copy_image_to_memory.is_none() {
                cmd.copy_image_to_memory = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCopyImageToMemoryEXT\0",
                        ),
                    ),
                );
            }
            cmd.copy_image_to_image = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCopyImageToImage\0")),
            );
            if cmd.copy_image_to_image.is_none() {
                cmd.copy_image_to_image = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCopyImageToImageEXT\0",
                        ),
                    ),
                );
            }
            cmd.transition_image_layout = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkTransitionImageLayout\0",
                    ),
                ),
            );
            if cmd.transition_image_layout.is_none() {
                cmd.transition_image_layout = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkTransitionImageLayoutEXT\0",
                        ),
                    ),
                );
            }
            cmd.get_command_pool_memory_consumption = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetCommandPoolMemoryConsumption\0",
                    ),
                ),
            );
            cmd.create_video_session_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateVideoSessionKHR\0",
                    ),
                ),
            );
            cmd.destroy_video_session_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyVideoSessionKHR\0",
                    ),
                ),
            );
            cmd.create_video_session_parameters_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateVideoSessionParametersKHR\0",
                    ),
                ),
            );
            cmd.update_video_session_parameters_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkUpdateVideoSessionParametersKHR\0",
                    ),
                ),
            );
            cmd.get_encoded_video_session_parameters_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetEncodedVideoSessionParametersKHR\0",
                    ),
                ),
            );
            cmd.destroy_video_session_parameters_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyVideoSessionParametersKHR\0",
                    ),
                ),
            );
            cmd.get_video_session_memory_requirements_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetVideoSessionMemoryRequirementsKHR\0",
                    ),
                ),
            );
            cmd.bind_video_session_memory_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkBindVideoSessionMemoryKHR\0",
                    ),
                ),
            );
            cmd.cmd_decode_video_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDecodeVideoKHR\0",
                    ),
                ),
            );
            cmd.cmd_begin_video_coding_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBeginVideoCodingKHR\0",
                    ),
                ),
            );
            cmd.cmd_control_video_coding_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdControlVideoCodingKHR\0",
                    ),
                ),
            );
            cmd.cmd_end_video_coding_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdEndVideoCodingKHR\0",
                    ),
                ),
            );
            cmd.cmd_encode_video_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdEncodeVideoKHR\0",
                    ),
                ),
            );
            cmd.cmd_decompress_memory_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDecompressMemoryNV\0",
                    ),
                ),
            );
            cmd.cmd_decompress_memory_indirect_count_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDecompressMemoryIndirectCountNV\0",
                    ),
                ),
            );
            cmd.get_partitioned_acceleration_structures_build_sizes_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPartitionedAccelerationStructuresBuildSizesNV\0",
                    ),
                ),
            );
            cmd.cmd_build_partitioned_acceleration_structures_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBuildPartitionedAccelerationStructuresNV\0",
                    ),
                ),
            );
            cmd.cmd_decompress_memory_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDecompressMemoryEXT\0",
                    ),
                ),
            );
            cmd.cmd_decompress_memory_indirect_count_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDecompressMemoryIndirectCountEXT\0",
                    ),
                ),
            );
            cmd.create_cu_module_nvx = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateCuModuleNVX\0",
                    ),
                ),
            );
            cmd.create_cu_function_nvx = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateCuFunctionNVX\0",
                    ),
                ),
            );
            cmd.destroy_cu_module_nvx = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyCuModuleNVX\0",
                    ),
                ),
            );
            cmd.destroy_cu_function_nvx = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyCuFunctionNVX\0",
                    ),
                ),
            );
            cmd.cmd_cu_launch_kernel_nvx = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdCuLaunchKernelNVX\0",
                    ),
                ),
            );
            cmd.get_descriptor_set_layout_size_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDescriptorSetLayoutSizeEXT\0",
                    ),
                ),
            );
            cmd.get_descriptor_set_layout_binding_offset_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDescriptorSetLayoutBindingOffsetEXT\0",
                    ),
                ),
            );
            cmd.get_descriptor_ext = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkGetDescriptorEXT\0")),
            );
            cmd.cmd_bind_descriptor_buffers_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBindDescriptorBuffersEXT\0",
                    ),
                ),
            );
            cmd.cmd_set_descriptor_buffer_offsets_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetDescriptorBufferOffsetsEXT\0",
                    ),
                ),
            );
            cmd.cmd_bind_descriptor_buffer_embedded_samplers_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBindDescriptorBufferEmbeddedSamplersEXT\0",
                    ),
                ),
            );
            cmd.get_buffer_opaque_capture_descriptor_data_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetBufferOpaqueCaptureDescriptorDataEXT\0",
                    ),
                ),
            );
            cmd.get_image_opaque_capture_descriptor_data_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetImageOpaqueCaptureDescriptorDataEXT\0",
                    ),
                ),
            );
            cmd.get_image_view_opaque_capture_descriptor_data_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetImageViewOpaqueCaptureDescriptorDataEXT\0",
                    ),
                ),
            );
            cmd.get_sampler_opaque_capture_descriptor_data_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetSamplerOpaqueCaptureDescriptorDataEXT\0",
                    ),
                ),
            );
            cmd.get_acceleration_structure_opaque_capture_descriptor_data_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT\0",
                    ),
                ),
            );
            cmd.set_device_memory_priority_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkSetDeviceMemoryPriorityEXT\0",
                    ),
                ),
            );
            cmd.wait_for_present2_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkWaitForPresent2KHR\0",
                    ),
                ),
            );
            cmd.wait_for_present_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkWaitForPresentKHR\0",
                    ),
                ),
            );
            cmd.create_buffer_collection_fuchsia = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateBufferCollectionFUCHSIA\0",
                    ),
                ),
            );
            cmd.set_buffer_collection_buffer_constraints_fuchsia = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkSetBufferCollectionBufferConstraintsFUCHSIA\0",
                    ),
                ),
            );
            cmd.set_buffer_collection_image_constraints_fuchsia = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkSetBufferCollectionImageConstraintsFUCHSIA\0",
                    ),
                ),
            );
            cmd.destroy_buffer_collection_fuchsia = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyBufferCollectionFUCHSIA\0",
                    ),
                ),
            );
            cmd.get_buffer_collection_properties_fuchsia = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetBufferCollectionPropertiesFUCHSIA\0",
                    ),
                ),
            );
            cmd.create_cuda_module_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateCudaModuleNV\0",
                    ),
                ),
            );
            cmd.get_cuda_module_cache_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetCudaModuleCacheNV\0",
                    ),
                ),
            );
            cmd.create_cuda_function_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateCudaFunctionNV\0",
                    ),
                ),
            );
            cmd.destroy_cuda_module_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyCudaModuleNV\0",
                    ),
                ),
            );
            cmd.destroy_cuda_function_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyCudaFunctionNV\0",
                    ),
                ),
            );
            cmd.cmd_cuda_launch_kernel_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdCudaLaunchKernelNV\0",
                    ),
                ),
            );
            cmd.cmd_begin_rendering = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBeginRendering\0",
                    ),
                ),
            );
            if cmd.cmd_begin_rendering.is_none() {
                cmd.cmd_begin_rendering = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdBeginRenderingKHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_end_rendering = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdEndRendering\0")),
            );
            if cmd.cmd_end_rendering.is_none() {
                cmd.cmd_end_rendering = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdEndRenderingKHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_end_rendering2_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdEndRendering2KHR\0",
                    ),
                ),
            );
            if cmd.cmd_end_rendering2_khr.is_none() {
                cmd.cmd_end_rendering2_khr = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdEndRendering2EXT\0",
                        ),
                    ),
                );
            }
            cmd.get_descriptor_set_layout_host_mapping_info_valve = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDescriptorSetLayoutHostMappingInfoVALVE\0",
                    ),
                ),
            );
            cmd.get_descriptor_set_host_mapping_valve = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDescriptorSetHostMappingVALVE\0",
                    ),
                ),
            );
            cmd.create_micromap_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateMicromapEXT\0",
                    ),
                ),
            );
            cmd.cmd_build_micromaps_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBuildMicromapsEXT\0",
                    ),
                ),
            );
            cmd.build_micromaps_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkBuildMicromapsEXT\0",
                    ),
                ),
            );
            cmd.destroy_micromap_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyMicromapEXT\0",
                    ),
                ),
            );
            cmd.cmd_copy_micromap_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdCopyMicromapEXT\0",
                    ),
                ),
            );
            cmd.copy_micromap_ext = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCopyMicromapEXT\0")),
            );
            cmd.cmd_copy_micromap_to_memory_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdCopyMicromapToMemoryEXT\0",
                    ),
                ),
            );
            cmd.copy_micromap_to_memory_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCopyMicromapToMemoryEXT\0",
                    ),
                ),
            );
            cmd.cmd_copy_memory_to_micromap_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdCopyMemoryToMicromapEXT\0",
                    ),
                ),
            );
            cmd.copy_memory_to_micromap_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCopyMemoryToMicromapEXT\0",
                    ),
                ),
            );
            cmd.cmd_write_micromaps_properties_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdWriteMicromapsPropertiesEXT\0",
                    ),
                ),
            );
            cmd.write_micromaps_properties_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkWriteMicromapsPropertiesEXT\0",
                    ),
                ),
            );
            cmd.get_device_micromap_compatibility_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDeviceMicromapCompatibilityEXT\0",
                    ),
                ),
            );
            cmd.get_micromap_build_sizes_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetMicromapBuildSizesEXT\0",
                    ),
                ),
            );
            cmd.get_shader_module_identifier_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetShaderModuleIdentifierEXT\0",
                    ),
                ),
            );
            cmd.get_shader_module_create_info_identifier_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetShaderModuleCreateInfoIdentifierEXT\0",
                    ),
                ),
            );
            cmd.get_image_subresource_layout2 = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetImageSubresourceLayout2\0",
                    ),
                ),
            );
            if cmd.get_image_subresource_layout2.is_none() {
                cmd.get_image_subresource_layout2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetImageSubresourceLayout2KHR\0",
                        ),
                    ),
                );
            }
            if cmd.get_image_subresource_layout2.is_none() {
                cmd.get_image_subresource_layout2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetImageSubresourceLayout2EXT\0",
                        ),
                    ),
                );
            }
            cmd.get_pipeline_properties_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPipelinePropertiesEXT\0",
                    ),
                ),
            );
            cmd.export_metal_objects_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkExportMetalObjectsEXT\0",
                    ),
                ),
            );
            cmd.cmd_bind_tile_memory_qcom = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBindTileMemoryQCOM\0",
                    ),
                ),
            );
            cmd.get_framebuffer_tile_properties_qcom = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetFramebufferTilePropertiesQCOM\0",
                    ),
                ),
            );
            cmd.get_dynamic_rendering_tile_properties_qcom = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDynamicRenderingTilePropertiesQCOM\0",
                    ),
                ),
            );
            cmd.create_optical_flow_session_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateOpticalFlowSessionNV\0",
                    ),
                ),
            );
            cmd.destroy_optical_flow_session_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyOpticalFlowSessionNV\0",
                    ),
                ),
            );
            cmd.bind_optical_flow_session_image_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkBindOpticalFlowSessionImageNV\0",
                    ),
                ),
            );
            cmd.cmd_optical_flow_execute_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdOpticalFlowExecuteNV\0",
                    ),
                ),
            );
            cmd.get_device_fault_info_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDeviceFaultInfoEXT\0",
                    ),
                ),
            );
            cmd.get_device_fault_reports_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDeviceFaultReportsKHR\0",
                    ),
                ),
            );
            cmd.get_device_fault_debug_info_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDeviceFaultDebugInfoKHR\0",
                    ),
                ),
            );
            cmd.cmd_set_depth_bias2_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetDepthBias2EXT\0",
                    ),
                ),
            );
            cmd.release_swapchain_images_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkReleaseSwapchainImagesKHR\0",
                    ),
                ),
            );
            if cmd.release_swapchain_images_khr.is_none() {
                cmd.release_swapchain_images_khr = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkReleaseSwapchainImagesEXT\0",
                        ),
                    ),
                );
            }
            cmd.get_device_image_subresource_layout = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDeviceImageSubresourceLayout\0",
                    ),
                ),
            );
            if cmd.get_device_image_subresource_layout.is_none() {
                cmd.get_device_image_subresource_layout = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkGetDeviceImageSubresourceLayoutKHR\0",
                        ),
                    ),
                );
            }
            cmd.map_memory2 = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkMapMemory2\0")),
            );
            if cmd.map_memory2.is_none() {
                cmd.map_memory2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkMapMemory2KHR\0",
                        ),
                    ),
                );
            }
            cmd.unmap_memory2 = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkUnmapMemory2\0")),
            );
            if cmd.unmap_memory2.is_none() {
                cmd.unmap_memory2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkUnmapMemory2KHR\0",
                        ),
                    ),
                );
            }
            cmd.create_shaders_ext = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCreateShadersEXT\0")),
            );
            cmd.destroy_shader_ext = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkDestroyShaderEXT\0")),
            );
            cmd.get_shader_binary_data_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetShaderBinaryDataEXT\0",
                    ),
                ),
            );
            cmd.cmd_bind_shaders_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBindShadersEXT\0",
                    ),
                ),
            );
            cmd.set_swapchain_present_timing_queue_size_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkSetSwapchainPresentTimingQueueSizeEXT\0",
                    ),
                ),
            );
            cmd.get_swapchain_timing_properties_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetSwapchainTimingPropertiesEXT\0",
                    ),
                ),
            );
            cmd.get_swapchain_time_domain_properties_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetSwapchainTimeDomainPropertiesEXT\0",
                    ),
                ),
            );
            cmd.get_past_presentation_timing_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetPastPresentationTimingEXT\0",
                    ),
                ),
            );
            cmd.get_screen_buffer_properties_qnx = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetScreenBufferPropertiesQNX\0",
                    ),
                ),
            );
            cmd.get_execution_graph_pipeline_scratch_size_amdx = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetExecutionGraphPipelineScratchSizeAMDX\0",
                    ),
                ),
            );
            cmd.get_execution_graph_pipeline_node_index_amdx = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetExecutionGraphPipelineNodeIndexAMDX\0",
                    ),
                ),
            );
            cmd.create_execution_graph_pipelines_amdx = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateExecutionGraphPipelinesAMDX\0",
                    ),
                ),
            );
            cmd.cmd_initialize_graph_scratch_memory_amdx = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdInitializeGraphScratchMemoryAMDX\0",
                    ),
                ),
            );
            cmd.cmd_dispatch_graph_amdx = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDispatchGraphAMDX\0",
                    ),
                ),
            );
            cmd.cmd_dispatch_graph_indirect_amdx = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDispatchGraphIndirectAMDX\0",
                    ),
                ),
            );
            cmd.cmd_dispatch_graph_indirect_count_amdx = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDispatchGraphIndirectCountAMDX\0",
                    ),
                ),
            );
            cmd.cmd_bind_descriptor_sets2 = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBindDescriptorSets2\0",
                    ),
                ),
            );
            if cmd.cmd_bind_descriptor_sets2.is_none() {
                cmd.cmd_bind_descriptor_sets2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdBindDescriptorSets2KHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_push_constants2 = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdPushConstants2\0",
                    ),
                ),
            );
            if cmd.cmd_push_constants2.is_none() {
                cmd.cmd_push_constants2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdPushConstants2KHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_push_descriptor_set2 = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdPushDescriptorSet2\0",
                    ),
                ),
            );
            if cmd.cmd_push_descriptor_set2.is_none() {
                cmd.cmd_push_descriptor_set2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdPushDescriptorSet2KHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_push_descriptor_set_with_template2 = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdPushDescriptorSetWithTemplate2\0",
                    ),
                ),
            );
            if cmd.cmd_push_descriptor_set_with_template2.is_none() {
                cmd.cmd_push_descriptor_set_with_template2 = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdPushDescriptorSetWithTemplate2KHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_set_descriptor_buffer_offsets2_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetDescriptorBufferOffsets2EXT\0",
                    ),
                ),
            );
            cmd.cmd_bind_descriptor_buffer_embedded_samplers2_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBindDescriptorBufferEmbeddedSamplers2EXT\0",
                    ),
                ),
            );
            cmd.set_latency_sleep_mode_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkSetLatencySleepModeNV\0",
                    ),
                ),
            );
            cmd.latency_sleep_nv = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkLatencySleepNV\0")),
            );
            cmd.set_latency_marker_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkSetLatencyMarkerNV\0",
                    ),
                ),
            );
            cmd.get_latency_timings_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetLatencyTimingsNV\0",
                    ),
                ),
            );
            cmd.queue_notify_out_of_band_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkQueueNotifyOutOfBandNV\0",
                    ),
                ),
            );
            cmd.cmd_set_rendering_attachment_locations = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetRenderingAttachmentLocations\0",
                    ),
                ),
            );
            if cmd.cmd_set_rendering_attachment_locations.is_none() {
                cmd.cmd_set_rendering_attachment_locations = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdSetRenderingAttachmentLocationsKHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_set_rendering_input_attachment_indices = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetRenderingInputAttachmentIndices\0",
                    ),
                ),
            );
            if cmd.cmd_set_rendering_input_attachment_indices.is_none() {
                cmd.cmd_set_rendering_input_attachment_indices = std::mem::transmute(
                    f(
                        std::ffi::CStr::from_bytes_with_nul_unchecked(
                            b"vkCmdSetRenderingInputAttachmentIndicesKHR\0",
                        ),
                    ),
                );
            }
            cmd.cmd_set_depth_clamp_range_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetDepthClampRangeEXT\0",
                    ),
                ),
            );
            cmd.get_memory_metal_handle_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetMemoryMetalHandleEXT\0",
                    ),
                ),
            );
            cmd.get_memory_metal_handle_properties_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetMemoryMetalHandlePropertiesEXT\0",
                    ),
                ),
            );
            cmd.convert_cooperative_vector_matrix_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkConvertCooperativeVectorMatrixNV\0",
                    ),
                ),
            );
            cmd.cmd_convert_cooperative_vector_matrix_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdConvertCooperativeVectorMatrixNV\0",
                    ),
                ),
            );
            cmd.cmd_dispatch_tile_qcom = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDispatchTileQCOM\0",
                    ),
                ),
            );
            cmd.cmd_begin_per_tile_execution_qcom = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBeginPerTileExecutionQCOM\0",
                    ),
                ),
            );
            cmd.cmd_end_per_tile_execution_qcom = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdEndPerTileExecutionQCOM\0",
                    ),
                ),
            );
            cmd.create_external_compute_queue_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateExternalComputeQueueNV\0",
                    ),
                ),
            );
            cmd.destroy_external_compute_queue_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyExternalComputeQueueNV\0",
                    ),
                ),
            );
            cmd.create_shader_instrumentation_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateShaderInstrumentationARM\0",
                    ),
                ),
            );
            cmd.destroy_shader_instrumentation_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyShaderInstrumentationARM\0",
                    ),
                ),
            );
            cmd.cmd_begin_shader_instrumentation_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBeginShaderInstrumentationARM\0",
                    ),
                ),
            );
            cmd.cmd_end_shader_instrumentation_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdEndShaderInstrumentationARM\0",
                    ),
                ),
            );
            cmd.get_shader_instrumentation_values_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetShaderInstrumentationValuesARM\0",
                    ),
                ),
            );
            cmd.clear_shader_instrumentation_metrics_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkClearShaderInstrumentationMetricsARM\0",
                    ),
                ),
            );
            cmd.create_tensor_arm = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCreateTensorARM\0")),
            );
            cmd.destroy_tensor_arm = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkDestroyTensorARM\0")),
            );
            cmd.create_tensor_view_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateTensorViewARM\0",
                    ),
                ),
            );
            cmd.destroy_tensor_view_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyTensorViewARM\0",
                    ),
                ),
            );
            cmd.get_tensor_memory_requirements_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetTensorMemoryRequirementsARM\0",
                    ),
                ),
            );
            cmd.bind_tensor_memory_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkBindTensorMemoryARM\0",
                    ),
                ),
            );
            cmd.get_device_tensor_memory_requirements_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDeviceTensorMemoryRequirementsARM\0",
                    ),
                ),
            );
            cmd.cmd_copy_tensor_arm = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdCopyTensorARM\0")),
            );
            cmd.get_tensor_opaque_capture_descriptor_data_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetTensorOpaqueCaptureDescriptorDataARM\0",
                    ),
                ),
            );
            cmd.get_tensor_view_opaque_capture_descriptor_data_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetTensorViewOpaqueCaptureDescriptorDataARM\0",
                    ),
                ),
            );
            cmd.create_data_graph_pipelines_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateDataGraphPipelinesARM\0",
                    ),
                ),
            );
            cmd.create_data_graph_pipeline_session_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateDataGraphPipelineSessionARM\0",
                    ),
                ),
            );
            cmd.get_data_graph_pipeline_session_bind_point_requirements_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDataGraphPipelineSessionBindPointRequirementsARM\0",
                    ),
                ),
            );
            cmd.get_data_graph_pipeline_session_memory_requirements_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDataGraphPipelineSessionMemoryRequirementsARM\0",
                    ),
                ),
            );
            cmd.bind_data_graph_pipeline_session_memory_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkBindDataGraphPipelineSessionMemoryARM\0",
                    ),
                ),
            );
            cmd.destroy_data_graph_pipeline_session_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkDestroyDataGraphPipelineSessionARM\0",
                    ),
                ),
            );
            cmd.cmd_dispatch_data_graph_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDispatchDataGraphARM\0",
                    ),
                ),
            );
            cmd.get_data_graph_pipeline_available_properties_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDataGraphPipelineAvailablePropertiesARM\0",
                    ),
                ),
            );
            cmd.get_data_graph_pipeline_properties_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetDataGraphPipelinePropertiesARM\0",
                    ),
                ),
            );
            cmd.get_native_buffer_properties_ohos = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetNativeBufferPropertiesOHOS\0",
                    ),
                ),
            );
            cmd.get_memory_native_buffer_ohos = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetMemoryNativeBufferOHOS\0",
                    ),
                ),
            );
            cmd.get_swapchain_gralloc_usage_ohos = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetSwapchainGrallocUsageOHOS\0",
                    ),
                ),
            );
            cmd.acquire_image_ohos = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkAcquireImageOHOS\0")),
            );
            cmd.queue_signal_release_image_ohos = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkQueueSignalReleaseImageOHOS\0",
                    ),
                ),
            );
            cmd.cmd_set_compute_occupancy_priority_nv = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdSetComputeOccupancyPriorityNV\0",
                    ),
                ),
            );
            cmd.write_sampler_descriptors_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkWriteSamplerDescriptorsEXT\0",
                    ),
                ),
            );
            cmd.write_resource_descriptors_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkWriteResourceDescriptorsEXT\0",
                    ),
                ),
            );
            cmd.cmd_bind_sampler_heap_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBindSamplerHeapEXT\0",
                    ),
                ),
            );
            cmd.cmd_bind_resource_heap_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBindResourceHeapEXT\0",
                    ),
                ),
            );
            cmd.cmd_push_data_ext = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdPushDataEXT\0")),
            );
            cmd.register_custom_border_color_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkRegisterCustomBorderColorEXT\0",
                    ),
                ),
            );
            cmd.unregister_custom_border_color_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkUnregisterCustomBorderColorEXT\0",
                    ),
                ),
            );
            cmd.get_image_opaque_capture_data_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetImageOpaqueCaptureDataEXT\0",
                    ),
                ),
            );
            cmd.get_tensor_opaque_capture_data_arm = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkGetTensorOpaqueCaptureDataARM\0",
                    ),
                ),
            );
            cmd.cmd_copy_memory_khr = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdCopyMemoryKHR\0")),
            );
            cmd.cmd_copy_memory_to_image_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdCopyMemoryToImageKHR\0",
                    ),
                ),
            );
            cmd.cmd_copy_image_to_memory_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdCopyImageToMemoryKHR\0",
                    ),
                ),
            );
            cmd.cmd_update_memory_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdUpdateMemoryKHR\0",
                    ),
                ),
            );
            cmd.cmd_fill_memory_khr = std::mem::transmute(
                f(std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCmdFillMemoryKHR\0")),
            );
            cmd.cmd_copy_query_pool_results_to_memory_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdCopyQueryPoolResultsToMemoryKHR\0",
                    ),
                ),
            );
            cmd.cmd_begin_conditional_rendering2_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBeginConditionalRendering2EXT\0",
                    ),
                ),
            );
            cmd.cmd_bind_transform_feedback_buffers2_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBindTransformFeedbackBuffers2EXT\0",
                    ),
                ),
            );
            cmd.cmd_begin_transform_feedback2_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBeginTransformFeedback2EXT\0",
                    ),
                ),
            );
            cmd.cmd_end_transform_feedback2_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdEndTransformFeedback2EXT\0",
                    ),
                ),
            );
            cmd.cmd_draw_indirect_byte_count2_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDrawIndirectByteCount2EXT\0",
                    ),
                ),
            );
            cmd.cmd_write_marker_to_memory_amd = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdWriteMarkerToMemoryAMD\0",
                    ),
                ),
            );
            cmd.cmd_bind_index_buffer3_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBindIndexBuffer3KHR\0",
                    ),
                ),
            );
            cmd.cmd_bind_vertex_buffers3_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdBindVertexBuffers3KHR\0",
                    ),
                ),
            );
            cmd.cmd_draw_indirect2_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDrawIndirect2KHR\0",
                    ),
                ),
            );
            cmd.cmd_draw_indexed_indirect2_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDrawIndexedIndirect2KHR\0",
                    ),
                ),
            );
            cmd.cmd_draw_indirect_count2_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDrawIndirectCount2KHR\0",
                    ),
                ),
            );
            cmd.cmd_draw_indexed_indirect_count2_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDrawIndexedIndirectCount2KHR\0",
                    ),
                ),
            );
            cmd.cmd_draw_mesh_tasks_indirect2_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDrawMeshTasksIndirect2EXT\0",
                    ),
                ),
            );
            cmd.cmd_draw_mesh_tasks_indirect_count2_ext = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDrawMeshTasksIndirectCount2EXT\0",
                    ),
                ),
            );
            cmd.cmd_dispatch_indirect2_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCmdDispatchIndirect2KHR\0",
                    ),
                ),
            );
            cmd.create_acceleration_structure2_khr = std::mem::transmute(
                f(
                    std::ffi::CStr::from_bytes_with_nul_unchecked(
                        b"vkCreateAccelerationStructure2KHR\0",
                    ),
                ),
            );
            cmd
        }
    }
}
