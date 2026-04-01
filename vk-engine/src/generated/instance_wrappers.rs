#![allow(unused_imports)]
#![allow(clippy::too_many_arguments)]
use crate::error::{VkResult, check, enumerate_two_call, fill_two_call};
use crate::vk::bitmasks::*;
use crate::vk::constants::*;
use crate::vk::enums::*;
use crate::vk::handles::*;
use crate::vk::structs::*;
impl crate::Instance {
    ///Wraps [`vkDestroyInstance`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyInstance.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `instance` (self) must be valid and not destroyed.
    ///- `instance` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a Vulkan instance and frees all instance-level resources.
    ///All devices created from this instance must be destroyed first.
    ///
    ///Safe teardown order:
    ///
    ///1. Destroy all `Device` objects (which destroys all device-child
    ///   objects).
    ///2. Destroy any debug messengers or debug report callbacks.
    ///3. Destroy any surfaces.
    ///4. `destroy_instance`.
    ///
    ///After this call the instance handle and all physical device handles
    ///obtained from it are invalid.
    pub unsafe fn destroy_instance(&self, allocator: Option<&AllocationCallbacks>) {
        let fp = self
            .commands()
            .destroy_instance
            .expect("vkDestroyInstance not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), alloc_ptr) };
    }
    ///Wraps [`vkEnumeratePhysicalDevices`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkEnumeratePhysicalDevices.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `instance` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns a list of all physical devices (GPUs) available to the
    ///instance. This is typically the first call after instance creation,
    ///you need a physical device to query capabilities and create a
    ///logical device.
    ///
    ///The order of physical devices is driver-dependent and not guaranteed
    ///to be stable across runs. To pick the right GPU:
    ///
    ///1. Enumerate all devices.
    ///2. Query `get_physical_device_properties` for each.
    ///3. Prefer `PHYSICAL_DEVICE_TYPE_DISCRETE_GPU` for performance, or
    ///   `INTEGRATED_GPU` for power efficiency.
    ///4. Check queue families, memory heaps, and required features.
    ///
    ///On systems with multiple GPUs (e.g. a discrete + integrated), this
    ///returns all of them. Vulkan does not have a concept of a "default"
    ///GPU, your application must choose.
    pub unsafe fn enumerate_physical_devices(&self) -> VkResult<Vec<PhysicalDevice>> {
        let fp = self
            .commands()
            .enumerate_physical_devices
            .expect("vkEnumeratePhysicalDevices not loaded");
        enumerate_two_call(|count, data| unsafe { fp(self.handle(), count, data) })
    }
    ///Wraps [`vkGetInstanceProcAddr`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetInstanceProcAddr.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `instance` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns a function pointer for an instance-level or device-level
    ///Vulkan command. This is the root of the Vulkan function pointer
    ///loading chain.
    ///
    ///In normal usage you do not need to call this yourself, `Instance`
    ///and `Device` load all function pointers automatically. This is
    ///primarily useful for:
    ///
    ///- Loading commands that are not yet exposed as wrapper methods.
    ///- Raw interop with other Vulkan libraries (e.g. OpenXR).
    ///- Implementing custom loaders.
    ///
    ///When called with a null instance, returns pointers for global
    ///commands (`vkEnumerateInstanceVersion`,
    ///`vkEnumerateInstanceExtensionProperties`, etc.).
    ///
    ///The returned pointer may go through a loader trampoline. For
    ///device-level commands, `get_device_proc_addr` returns a
    ///driver-direct pointer that is slightly faster.
    pub unsafe fn get_instance_proc_addr(&self, p_name: *const core::ffi::c_char) {
        let fp = self
            .commands()
            .get_instance_proc_addr
            .expect("vkGetInstanceProcAddr not loaded");
        unsafe { fp(self.handle(), p_name) };
    }
    ///Wraps [`vkGetPhysicalDeviceProperties`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceProperties.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns general properties of a physical device: device name, vendor
    ///ID, driver version, API version, device type, and device limits.
    ///
    ///Key fields to check during device selection:
    ///
    ///- **`device_type`**: `DISCRETE_GPU`, `INTEGRATED_GPU`, `VIRTUAL_GPU`,
    ///  `CPU`, or `OTHER`. Discrete GPUs typically offer the best
    ///  performance.
    ///- **`api_version`**: the highest Vulkan version the device supports.
    ///- **`limits`**: contains hundreds of device limits like
    ///  `max_image_dimension_2d`, `max_push_constants_size`,
    ///  `timestamp_period`, `non_coherent_atom_size`, etc.
    ///- **`pipeline_cache_uuid`**: used to validate pipeline cache data
    ///  across sessions.
    ///
    ///For extended properties (Vulkan 1.1+), use
    ///`get_physical_device_properties2` which supports chaining additional
    ///property structs like `PhysicalDeviceVulkan12Properties`.
    pub unsafe fn get_physical_device_properties(
        &self,
        physical_device: PhysicalDevice,
    ) -> PhysicalDeviceProperties {
        let fp = self
            .commands()
            .get_physical_device_properties
            .expect("vkGetPhysicalDeviceProperties not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(physical_device, &mut out) };
        out
    }
    ///Wraps [`vkGetPhysicalDeviceQueueFamilyProperties`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceQueueFamilyProperties.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns the list of queue families supported by a physical device.
    ///Each queue family has a set of capabilities (graphics, compute,
    ///transfer, sparse binding) and a count of available queues.
    ///
    ///You must query this before creating a device to determine which
    ///queue family indices to request.
    ///
    ///**Common queue family selection**:
    ///
    ///- **Graphics + compute**: most desktop GPUs have a single family
    ///  that supports both. Use it for all rendering and compute work.
    ///- **Dedicated transfer**: some GPUs expose a transfer-only family
    ///  backed by a DMA engine. Use it for async uploads to overlap with
    ///  rendering.
    ///- **Dedicated compute**: some GPUs expose a compute-only family
    ///  for async compute.
    ///
    ///Check `queue_flags` for `QUEUE_GRAPHICS`, `QUEUE_COMPUTE`,
    ///`QUEUE_TRANSFER`, and `QUEUE_SPARSE_BINDING`. Also check
    ///`timestamp_valid_bits` if you need GPU timestamps on that queue.
    ///
    ///For extended properties (Vulkan 1.1+), use
    ///`get_physical_device_queue_family_properties2`.
    pub unsafe fn get_physical_device_queue_family_properties(
        &self,
        physical_device: PhysicalDevice,
    ) -> Vec<QueueFamilyProperties> {
        let fp = self
            .commands()
            .get_physical_device_queue_family_properties
            .expect("vkGetPhysicalDeviceQueueFamilyProperties not loaded");
        fill_two_call(|count, data| unsafe { fp(physical_device, count, data) })
    }
    ///Wraps [`vkGetPhysicalDeviceMemoryProperties`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceMemoryProperties.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns the memory heaps and memory types available on the physical
    ///device. Essential for choosing the right memory type when allocating
    ///device memory.
    ///
    ///**Memory heaps**: represent physical memory pools (e.g. VRAM, system
    ///RAM). Each heap has a size and flags (`DEVICE_LOCAL` for GPU memory).
    ///
    ///**Memory types**: each type references a heap and has property flags:
    ///
    ///- `DEVICE_LOCAL`: fast GPU access. Preferred for images, vertex
    ///  buffers, and anything the GPU reads frequently.
    ///- `HOST_VISIBLE`: can be mapped for CPU access. Required for staging
    ///  buffers and any CPU-written data.
    ///- `HOST_COHERENT`: mapped writes are automatically visible to the
    ///  GPU without explicit flushes.
    ///- `HOST_CACHED`: CPU reads are fast (cached). Useful for readback
    ///  buffers.
    ///- `LAZILY_ALLOCATED`: memory may not be backed until used. For
    ///  transient attachments on tile-based GPUs.
    ///
    ///**Choosing a memory type**: AND the `memory_type_bits` from
    ///`get_buffer_memory_requirements` or `get_image_memory_requirements`
    ///with your desired property flags. Pick the first matching type.
    ///
    ///For extended properties (Vulkan 1.1+), use
    ///`get_physical_device_memory_properties2`.
    pub unsafe fn get_physical_device_memory_properties(
        &self,
        physical_device: PhysicalDevice,
    ) -> PhysicalDeviceMemoryProperties {
        let fp = self
            .commands()
            .get_physical_device_memory_properties
            .expect("vkGetPhysicalDeviceMemoryProperties not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(physical_device, &mut out) };
        out
    }
    ///Wraps [`vkGetPhysicalDeviceFeatures`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceFeatures.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns the optional features supported by a physical device. Each
    ///field is a boolean indicating whether the feature is available.
    ///
    ///Query this before device creation to check whether features your
    ///application needs are supported. Then enable only the features you
    ///actually use in `DeviceCreateInfo::enabled_features`.
    ///
    ///Commonly checked features:
    ///
    ///- `sampler_anisotropy`: anisotropic texture filtering.
    ///- `fill_mode_non_solid`: wireframe rendering.
    ///- `wide_lines`: line widths other than 1.0.
    ///- `geometry_shader`, `tessellation_shader`: optional shader stages.
    ///- `multi_draw_indirect`: indirect draw with count > 1.
    ///- `pipeline_statistics_query`: pipeline statistics queries.
    ///
    ///Enabling a feature that is not supported causes device creation to
    ///fail. Never blindly enable all features, only request what you need.
    ///
    ///For extended features (Vulkan 1.1+), use
    ///`get_physical_device_features2` with chained feature structs.
    pub unsafe fn get_physical_device_features(
        &self,
        physical_device: PhysicalDevice,
    ) -> PhysicalDeviceFeatures {
        let fp = self
            .commands()
            .get_physical_device_features
            .expect("vkGetPhysicalDeviceFeatures not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(physical_device, &mut out) };
        out
    }
    ///Wraps [`vkGetPhysicalDeviceFormatProperties`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceFormatProperties.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries which operations a format supports on this device for
    ///linear tiling, optimal tiling, and buffer usage.
    ///
    ///The returned `FormatProperties` contains three `FormatFeatureFlags`
    ///fields:
    ///
    ///- **`linear_tiling_features`**: capabilities when the image uses
    ///  `IMAGE_TILING_LINEAR`.
    ///- **`optimal_tiling_features`**: capabilities when the image uses
    ///  `IMAGE_TILING_OPTIMAL` (the common case).
    ///- **`buffer_features`**: capabilities when used in a buffer view
    ///  (e.g. uniform texel buffer, storage texel buffer).
    ///
    ///Check the relevant feature bits before creating an image or buffer
    ///view with a particular format. For example, verify
    ///`FORMAT_FEATURE_COLOR_ATTACHMENT` before using a format as a render
    ///target, or `FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR` before
    ///enabling linear filtering.
    ///
    ///For extended format properties (Vulkan 1.1+), use
    ///`get_physical_device_format_properties2`.
    pub unsafe fn get_physical_device_format_properties(
        &self,
        physical_device: PhysicalDevice,
        format: Format,
    ) -> FormatProperties {
        let fp = self
            .commands()
            .get_physical_device_format_properties
            .expect("vkGetPhysicalDeviceFormatProperties not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(physical_device, format, &mut out) };
        out
    }
    ///Wraps [`vkGetPhysicalDeviceImageFormatProperties`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceImageFormatProperties.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_FORMAT_NOT_SUPPORTED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries whether a specific combination of image format, type, tiling,
    ///usage, and flags is supported on this device, and if so, what limits
    ///apply.
    ///
    ///Use this to validate image creation parameters before calling
    ///`create_image`. For example, check whether a format supports the
    ///`COLOR_ATTACHMENT` usage with optimal tiling at your desired
    ///resolution.
    ///
    ///The returned `ImageFormatProperties` includes:
    ///
    ///- **`max_extent`**: maximum dimensions for this combination.
    ///- **`max_mip_levels`**: maximum mipmap levels.
    ///- **`max_array_layers`**: maximum array layers.
    ///- **`sample_counts`**: supported multisample counts.
    ///- **`max_resource_size`**: maximum total bytes.
    ///
    ///Returns `VK_ERROR_FORMAT_NOT_SUPPORTED` if the combination is not
    ///supported at all, this is not a fatal error, just a "no".
    ///
    ///For extended queries (Vulkan 1.1+), use
    ///`get_physical_device_image_format_properties2` which supports
    ///chaining external memory and YCBCR properties.
    pub unsafe fn get_physical_device_image_format_properties(
        &self,
        physical_device: PhysicalDevice,
        format: Format,
        r#type: ImageType,
        tiling: ImageTiling,
        usage: ImageUsageFlags,
        flags: ImageCreateFlags,
    ) -> VkResult<ImageFormatProperties> {
        let fp = self
            .commands()
            .get_physical_device_image_format_properties
            .expect("vkGetPhysicalDeviceImageFormatProperties not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe {
            fp(
                physical_device,
                format,
                r#type,
                tiling,
                usage,
                flags,
                &mut out,
            )
        })?;
        Ok(out)
    }
    ///Wraps [`vkEnumerateDeviceLayerProperties`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkEnumerateDeviceLayerProperties.html).
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
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Enumerates the available device-level validation layers. In modern
    ///Vulkan (1.0.13+), device layers are deprecated in favour of
    ///instance layers, this function exists for backwards compatibility
    ///and typically returns the same list as instance layer enumeration.
    ///
    ///Most applications do not need to call this. Enable validation layers
    ///at instance creation via `InstanceCreateInfo::enabled_layer_names`
    ///instead.
    pub unsafe fn enumerate_device_layer_properties(
        &self,
        physical_device: PhysicalDevice,
    ) -> VkResult<Vec<LayerProperties>> {
        let fp = self
            .commands()
            .enumerate_device_layer_properties
            .expect("vkEnumerateDeviceLayerProperties not loaded");
        enumerate_two_call(|count, data| unsafe { fp(physical_device, count, data) })
    }
    ///Wraps [`vkEnumerateDeviceExtensionProperties`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkEnumerateDeviceExtensionProperties.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_LAYER_NOT_PRESENT`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns the list of extensions supported by a physical device. Call
    ///this to verify that required extensions are available before
    ///requesting them in `DeviceCreateInfo::enabled_extension_names`.
    ///
    ///Common extensions to check for:
    ///
    ///- `VK_KHR_swapchain`: presentation to a surface (required for
    ///  rendering to a window).
    ///- `VK_KHR_dynamic_rendering`: render pass-less rendering (core in
    ///  1.3).
    ///- `VK_KHR_ray_tracing_pipeline`: hardware ray tracing.
    ///- `VK_EXT_descriptor_indexing`: bindless descriptors (core in 1.2).
    ///
    ///Pass `None` for `layer_name` to enumerate extensions provided by the
    ///driver itself. Passing a layer name enumerates extensions provided by
    ///that specific layer (rarely needed).
    ///
    ///Requesting an unsupported extension at device creation causes
    ///`VK_ERROR_EXTENSION_NOT_PRESENT`.
    pub unsafe fn enumerate_device_extension_properties(
        &self,
        physical_device: PhysicalDevice,
        p_layer_name: *const core::ffi::c_char,
    ) -> VkResult<Vec<ExtensionProperties>> {
        let fp = self
            .commands()
            .enumerate_device_extension_properties
            .expect("vkEnumerateDeviceExtensionProperties not loaded");
        enumerate_two_call(|count, data| unsafe { fp(physical_device, p_layer_name, count, data) })
    }
    ///Wraps [`vkGetPhysicalDeviceSparseImageFormatProperties`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceSparseImageFormatProperties.html).
    /**
    Provided by **VK_BASE_VERSION_1_0**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the sparse image format properties for a specific format,
    ///image type, sample count, usage, and tiling combination. Returns
    ///information about the sparse texel block dimensions and flags.
    ///
    ///Only relevant if you intend to use sparse images
    ///(`IMAGE_CREATE_SPARSE_*`). For non-sparse images, this is not
    ///needed.
    ///
    ///If the combination does not support sparse residency, an empty list
    ///is returned. Check `physical_device_features.sparse_residency_*`
    ///features before attempting sparse image creation.
    ///
    ///For extended queries (Vulkan 1.1+), use
    ///`get_physical_device_sparse_image_format_properties2`.
    pub unsafe fn get_physical_device_sparse_image_format_properties(
        &self,
        physical_device: PhysicalDevice,
        format: Format,
        r#type: ImageType,
        samples: SampleCountFlagBits,
        usage: ImageUsageFlags,
        tiling: ImageTiling,
    ) -> Vec<SparseImageFormatProperties> {
        let fp = self
            .commands()
            .get_physical_device_sparse_image_format_properties
            .expect("vkGetPhysicalDeviceSparseImageFormatProperties not loaded");
        fill_two_call(|count, data| unsafe {
            fp(
                physical_device,
                format,
                r#type,
                samples,
                usage,
                tiling,
                count,
                data,
            )
        })
    }
    ///Wraps [`vkCreateSurfaceOHOS`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateSurfaceOHOS.html).
    /**
    Provided by **VK_OHOS_surface**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `instance` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a Vulkan surface for an OpenHarmony OS native window.
    ///OHOS only. The create info references the `OHNativeWindow`
    ///handle.
    ///
    ///Requires `VK_OHOS_surface`.
    pub unsafe fn create_surface_ohos(
        &self,
        p_create_info: &SurfaceCreateInfoOHOS,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<SurfaceKHR> {
        let fp = self
            .commands()
            .create_surface_ohos
            .expect("vkCreateSurfaceOHOS not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetPhysicalDeviceDisplayPropertiesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceDisplayPropertiesKHR.html).
    /**
    Provided by **VK_KHR_display**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Enumerates displays attached to a physical device. Each returned
    ///`DisplayPropertiesKHR` contains the display handle, name,
    ///physical dimensions, resolution, supported transforms, and
    ///whether the display supports per-plane reordering.
    ///
    ///This is the entry point for the `VK_KHR_display` extension,
    ///which provides direct display output without a window system
    ///(useful for embedded, VR, kiosk, and fullscreen applications).
    ///
    ///After enumerating displays, query their modes with
    ///`get_display_mode_properties_khr` and planes with
    ///`get_physical_device_display_plane_properties_khr`.
    ///
    ///Prefer `get_physical_device_display_properties2_khr` when
    ///`VK_KHR_get_display_properties2` is available, it supports
    ///extensible output via `pNext`.
    pub unsafe fn get_physical_device_display_properties_khr(
        &self,
        physical_device: PhysicalDevice,
    ) -> VkResult<Vec<DisplayPropertiesKHR>> {
        let fp = self
            .commands()
            .get_physical_device_display_properties_khr
            .expect("vkGetPhysicalDeviceDisplayPropertiesKHR not loaded");
        enumerate_two_call(|count, data| unsafe { fp(physical_device, count, data) })
    }
    ///Wraps [`vkGetPhysicalDeviceDisplayPlanePropertiesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceDisplayPlanePropertiesKHR.html).
    /**
    Provided by **VK_KHR_display**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Enumerates display planes supported by a physical device. Each
    ///plane is a compositing layer that can show a portion of a display
    ///surface, multiple planes allow hardware overlay composition.
    ///
    ///Each returned `DisplayPlanePropertiesKHR` contains the display
    ///currently connected to the plane and its current stack index.
    ///
    ///Use the plane index with `get_display_plane_supported_displays_khr`
    ///to find which displays a given plane can target, and with
    ///`get_display_plane_capabilities_khr` to query positioning and
    ///scaling limits.
    pub unsafe fn get_physical_device_display_plane_properties_khr(
        &self,
        physical_device: PhysicalDevice,
    ) -> VkResult<Vec<DisplayPlanePropertiesKHR>> {
        let fp = self
            .commands()
            .get_physical_device_display_plane_properties_khr
            .expect("vkGetPhysicalDeviceDisplayPlanePropertiesKHR not loaded");
        enumerate_two_call(|count, data| unsafe { fp(physical_device, count, data) })
    }
    ///Wraps [`vkGetDisplayPlaneSupportedDisplaysKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDisplayPlaneSupportedDisplaysKHR.html).
    /**
    Provided by **VK_KHR_display**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns the list of displays that a given display plane can be
    ///used with. Not all planes can target all displays, this query
    ///lets you find valid plane-display pairings.
    ///
    ///`plane_index` is an index into the array returned by
    ///`get_physical_device_display_plane_properties_khr`.
    ///
    ///After finding a compatible display, query its modes with
    ///`get_display_mode_properties_khr` and configure the plane via
    ///`DisplaySurfaceCreateInfoKHR`.
    pub unsafe fn get_display_plane_supported_displays_khr(
        &self,
        physical_device: PhysicalDevice,
        plane_index: u32,
    ) -> VkResult<Vec<DisplayKHR>> {
        let fp = self
            .commands()
            .get_display_plane_supported_displays_khr
            .expect("vkGetDisplayPlaneSupportedDisplaysKHR not loaded");
        enumerate_two_call(|count, data| unsafe { fp(physical_device, plane_index, count, data) })
    }
    ///Wraps [`vkGetDisplayModePropertiesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDisplayModePropertiesKHR.html).
    /**
    Provided by **VK_KHR_display**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Enumerates the display modes supported by a display. Each
    ///`DisplayModePropertiesKHR` contains a mode handle and its
    ///parameters (visible region resolution and refresh rate).
    ///
    ///Use these to select an appropriate mode for
    ///`DisplaySurfaceCreateInfoKHR`, or create a custom mode with
    ///`create_display_mode_khr` if the desired parameters are not
    ///listed.
    ///
    ///Prefer `get_display_mode_properties2_khr` when
    ///`VK_KHR_get_display_properties2` is available.
    pub unsafe fn get_display_mode_properties_khr(
        &self,
        physical_device: PhysicalDevice,
        display: DisplayKHR,
    ) -> VkResult<Vec<DisplayModePropertiesKHR>> {
        let fp = self
            .commands()
            .get_display_mode_properties_khr
            .expect("vkGetDisplayModePropertiesKHR not loaded");
        enumerate_two_call(|count, data| unsafe { fp(physical_device, display, count, data) })
    }
    ///Wraps [`vkCreateDisplayModeKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateDisplayModeKHR.html).
    /**
    Provided by **VK_KHR_display**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///- `display` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Creates a custom display mode with a specific resolution and
    ///refresh rate. Use this when the built-in modes from
    ///`get_display_mode_properties_khr` don't match your requirements.
    ///
    ///The `DisplayModeCreateInfoKHR` specifies the visible region
    ///(width/height in pixels) and refresh rate (in millihertz, e.g.,
    ///60000 for 60 Hz).
    ///
    ///Not all parameter combinations are valid, the driver may reject
    ///modes it cannot support. If creation fails, fall back to a
    ///built-in mode.
    pub unsafe fn create_display_mode_khr(
        &self,
        physical_device: PhysicalDevice,
        display: DisplayKHR,
        p_create_info: &DisplayModeCreateInfoKHR,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<DisplayModeKHR> {
        let fp = self
            .commands()
            .create_display_mode_khr
            .expect("vkCreateDisplayModeKHR not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(physical_device, display, p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetDisplayPlaneCapabilitiesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDisplayPlaneCapabilitiesKHR.html).
    /**
    Provided by **VK_KHR_display**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///- `mode` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Queries the capabilities of a display plane when used with a
    ///specific display mode. The returned `DisplayPlaneCapabilitiesKHR`
    ///describes:
    ///
    ///- Supported alpha modes (opaque, global, per-pixel).
    ///- Min/max source and destination extents, the range of scaling
    ///  the plane supports.
    ///- Min/max source and destination positions, how the plane
    ///  image can be positioned.
    ///
    ///Use these limits to configure `DisplaySurfaceCreateInfoKHR`
    ///correctly. Exceeding the reported limits results in validation
    ///errors.
    pub unsafe fn get_display_plane_capabilities_khr(
        &self,
        physical_device: PhysicalDevice,
        mode: DisplayModeKHR,
        plane_index: u32,
    ) -> VkResult<DisplayPlaneCapabilitiesKHR> {
        let fp = self
            .commands()
            .get_display_plane_capabilities_khr
            .expect("vkGetDisplayPlaneCapabilitiesKHR not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(physical_device, mode, plane_index, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkCreateDisplayPlaneSurfaceKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateDisplayPlaneSurfaceKHR.html).
    /**
    Provided by **VK_KHR_display**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `instance` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a `SurfaceKHR` for direct display output, bypassing the
    ///window system. The surface is presented directly on a display
    ///plane.
    ///
    ///Configure via `DisplaySurfaceCreateInfoKHR`:
    ///
    ///- `display_mode`: a mode from `get_display_mode_properties_khr`
    ///  or `create_display_mode_khr`.
    ///- `plane_index`: which display plane to use.
    ///- `plane_stack_index`: z-ordering among planes.
    ///- `transform`: rotation/mirroring.
    ///- `alpha_mode`: how alpha is handled (opaque, global, per-pixel).
    ///- `image_extent`: the surface resolution.
    ///
    ///After creation, use the surface with `create_swapchain_khr` like
    ///any other surface. Destroy with `destroy_surface_khr`.
    pub unsafe fn create_display_plane_surface_khr(
        &self,
        p_create_info: &DisplaySurfaceCreateInfoKHR,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<SurfaceKHR> {
        let fp = self
            .commands()
            .create_display_plane_surface_khr
            .expect("vkCreateDisplayPlaneSurfaceKHR not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetPhysicalDeviceSurfaceSupportKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceSurfaceSupportKHR.html).
    /**
    Provided by **VK_KHR_surface**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Checks whether a queue family on a physical device supports
    ///presentation to a surface. Not all queue families can present,
    ///a graphics queue that supports rendering may not support
    ///presentation on all platforms.
    ///
    ///Call this for each queue family when selecting your present queue.
    ///Often the graphics queue family also supports presentation, but
    ///this is not guaranteed.
    ///
    ///Returns `VK_TRUE` if the queue family can present to the surface.
    pub unsafe fn get_physical_device_surface_support_khr(
        &self,
        physical_device: PhysicalDevice,
        queue_family_index: u32,
        surface: SurfaceKHR,
    ) -> VkResult<u32> {
        let fp = self
            .commands()
            .get_physical_device_surface_support_khr
            .expect("vkGetPhysicalDeviceSurfaceSupportKHR not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(physical_device, queue_family_index, surface, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetPhysicalDeviceSurfaceCapabilitiesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceSurfaceCapabilitiesKHR.html).
    /**
    Provided by **VK_KHR_surface**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the surface capabilities for a physical device: supported
    ///image count range, current extent, supported transforms, composite
    ///alpha modes, and supported image usage flags.
    ///
    ///**Key fields**:
    ///
    ///- **`current_extent`**: the current surface size. If `0xFFFFFFFF`,
    ///  the surface size is determined by the swapchain extent.
    ///- **`min/max_image_count`**: the supported range for swapchain
    ///  image count.
    ///- **`current_transform`**: pass this as `pre_transform` in
    ///  swapchain creation to avoid extra composition overhead.
    ///- **`supported_usage_flags`**: which image usage bits the swapchain
    ///  images support (always includes `COLOR_ATTACHMENT`).
    ///
    ///Call this before creating a swapchain and again before recreating
    ///after a resize, the capabilities may have changed.
    pub unsafe fn get_physical_device_surface_capabilities_khr(
        &self,
        physical_device: PhysicalDevice,
        surface: SurfaceKHR,
    ) -> VkResult<SurfaceCapabilitiesKHR> {
        let fp = self
            .commands()
            .get_physical_device_surface_capabilities_khr
            .expect("vkGetPhysicalDeviceSurfaceCapabilitiesKHR not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(physical_device, surface, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetPhysicalDeviceSurfaceFormatsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceSurfaceFormatsKHR.html).
    /**
    Provided by **VK_KHR_surface**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns the list of supported format + colour space pairs for a
    ///surface. Pick one of these pairs for swapchain creation.
    ///
    ///The most portable choice is `FORMAT_B8G8R8A8_SRGB` +
    ///`COLOR_SPACE_SRGB_NONLINEAR`. If your preferred format is not
    ///listed, fall back to the first available pair.
    ///
    ///If the list contains a single entry with `FORMAT_UNDEFINED`, the
    ///surface has no preferred format and any format is acceptable.
    ///
    ///For HDR output, look for `COLOR_SPACE_HDR10_ST2084_EXT` or similar
    ///extended colour spaces.
    pub unsafe fn get_physical_device_surface_formats_khr(
        &self,
        physical_device: PhysicalDevice,
        surface: SurfaceKHR,
    ) -> VkResult<Vec<SurfaceFormatKHR>> {
        let fp = self
            .commands()
            .get_physical_device_surface_formats_khr
            .expect("vkGetPhysicalDeviceSurfaceFormatsKHR not loaded");
        enumerate_two_call(|count, data| unsafe { fp(physical_device, surface, count, data) })
    }
    ///Wraps [`vkGetPhysicalDeviceSurfacePresentModesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceSurfacePresentModesKHR.html).
    /**
    Provided by **VK_KHR_surface**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns the list of supported present modes for a surface.
    ///
    ///**Present modes**:
    ///
    ///- `FIFO`: vsync. Guaranteed to be supported on all implementations.
    ///  Frames are queued and presented at the display refresh rate.
    ///- `FIFO_RELAXED`: vsync with late frame allowance. If a frame
    ///  arrives late, it is presented immediately (may cause tearing).
    ///- `MAILBOX`: triple buffering. The driver keeps only the latest
    ///  frame in the queue, lower latency than FIFO with no tearing.
    ///- `IMMEDIATE`: no vsync. Frames are presented as soon as possible.
    ///  Lowest latency but may cause visible tearing.
    ///
    ///Common strategy: prefer `MAILBOX` for low-latency rendering, fall
    ///back to `FIFO` if unavailable.
    pub unsafe fn get_physical_device_surface_present_modes_khr(
        &self,
        physical_device: PhysicalDevice,
        surface: SurfaceKHR,
    ) -> VkResult<Vec<PresentModeKHR>> {
        let fp = self
            .commands()
            .get_physical_device_surface_present_modes_khr
            .expect("vkGetPhysicalDeviceSurfacePresentModesKHR not loaded");
        enumerate_two_call(|count, data| unsafe { fp(physical_device, surface, count, data) })
    }
    ///Wraps [`vkCreateViSurfaceNN`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateViSurfaceNN.html).
    /**
    Provided by **VK_NN_vi_surface**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_NATIVE_WINDOW_IN_USE_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `instance` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a Vulkan surface for a Nintendo Vi layer. Nintendo
    ///Switch only.
    ///
    ///Requires `VK_NN_vi_surface`.
    pub unsafe fn create_vi_surface_nn(
        &self,
        p_create_info: &ViSurfaceCreateInfoNN,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<SurfaceKHR> {
        let fp = self
            .commands()
            .create_vi_surface_nn
            .expect("vkCreateViSurfaceNN not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetPhysicalDeviceWaylandPresentationSupportKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceWaylandPresentationSupportKHR.html).
    /**
    Provided by **VK_KHR_wayland_surface**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries whether a queue family supports presentation to a
    ///Wayland compositor. Linux/Wayland only. Check this before
    ///creating a swapchain to ensure the queue can present.
    ///
    ///Requires `VK_KHR_wayland_surface`.
    pub unsafe fn get_physical_device_wayland_presentation_support_khr(
        &self,
        physical_device: PhysicalDevice,
        queue_family_index: u32,
    ) -> core::ffi::c_void {
        let fp = self
            .commands()
            .get_physical_device_wayland_presentation_support_khr
            .expect("vkGetPhysicalDeviceWaylandPresentationSupportKHR not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(physical_device, queue_family_index, &mut out) };
        out
    }
    ///Wraps [`vkCreateUbmSurfaceSEC`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateUbmSurfaceSEC.html).
    /**
    Provided by **VK_SEC_ubm_surface**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `instance` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a Vulkan surface for a Samsung UBM (Unified Buffer
    ///Manager) window. Samsung platform only.
    ///
    ///Requires `VK_SEC_ubm_surface`.
    pub unsafe fn create_ubm_surface_sec(
        &self,
        p_create_info: &UbmSurfaceCreateInfoSEC,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<SurfaceKHR> {
        let fp = self
            .commands()
            .create_ubm_surface_sec
            .expect("vkCreateUbmSurfaceSEC not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetPhysicalDeviceUbmPresentationSupportSEC`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceUbmPresentationSupportSEC.html).
    /**
    Provided by **VK_SEC_ubm_surface**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries whether a queue family supports presentation to a
    ///Samsung UBM surface. Samsung platform only.
    ///
    ///Requires `VK_SEC_ubm_surface`.
    pub unsafe fn get_physical_device_ubm_presentation_support_sec(
        &self,
        physical_device: PhysicalDevice,
        queue_family_index: u32,
    ) -> core::ffi::c_void {
        let fp = self
            .commands()
            .get_physical_device_ubm_presentation_support_sec
            .expect("vkGetPhysicalDeviceUbmPresentationSupportSEC not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(physical_device, queue_family_index, &mut out) };
        out
    }
    ///Wraps [`vkGetPhysicalDeviceWin32PresentationSupportKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceWin32PresentationSupportKHR.html).
    /**
    Provided by **VK_KHR_win32_surface**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries whether a queue family on the physical device supports
    ///presentation to the Win32 desktop compositor. Windows only.
    ///Check this before creating a swapchain to ensure the queue can
    ///present.
    ///
    ///Requires `VK_KHR_win32_surface`.
    pub unsafe fn get_physical_device_win32_presentation_support_khr(
        &self,
        physical_device: PhysicalDevice,
        queue_family_index: u32,
    ) {
        let fp = self
            .commands()
            .get_physical_device_win32_presentation_support_khr
            .expect("vkGetPhysicalDeviceWin32PresentationSupportKHR not loaded");
        unsafe { fp(physical_device, queue_family_index) };
    }
    ///Wraps [`vkGetPhysicalDeviceXlibPresentationSupportKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceXlibPresentationSupportKHR.html).
    /**
    Provided by **VK_KHR_xlib_surface**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries whether a queue family supports presentation to an X11
    ///display via Xlib for the given visual ID. Linux/X11 only.
    ///
    ///Requires `VK_KHR_xlib_surface`.
    pub unsafe fn get_physical_device_xlib_presentation_support_khr(
        &self,
        physical_device: PhysicalDevice,
        queue_family_index: u32,
        visual_id: core::ffi::c_ulong,
    ) -> core::ffi::c_void {
        let fp = self
            .commands()
            .get_physical_device_xlib_presentation_support_khr
            .expect("vkGetPhysicalDeviceXlibPresentationSupportKHR not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(physical_device, queue_family_index, &mut out, visual_id) };
        out
    }
    ///Wraps [`vkGetPhysicalDeviceXcbPresentationSupportKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceXcbPresentationSupportKHR.html).
    /**
    Provided by **VK_KHR_xcb_surface**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries whether a queue family supports presentation to an X11
    ///display via XCB for the given visual ID. Linux/X11 only.
    ///
    ///Requires `VK_KHR_xcb_surface`.
    pub unsafe fn get_physical_device_xcb_presentation_support_khr(
        &self,
        physical_device: PhysicalDevice,
        queue_family_index: u32,
        visual_id: u32,
    ) -> core::ffi::c_void {
        let fp = self
            .commands()
            .get_physical_device_xcb_presentation_support_khr
            .expect("vkGetPhysicalDeviceXcbPresentationSupportKHR not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(physical_device, queue_family_index, &mut out, visual_id) };
        out
    }
    ///Wraps [`vkCreateDirectFBSurfaceEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateDirectFBSurfaceEXT.html).
    /**
    Provided by **VK_EXT_directfb_surface**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `instance` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a Vulkan surface for a DirectFB window. Linux/DirectFB
    ///only. The create info references the DirectFB instance and
    ///surface handles.
    ///
    ///Requires `VK_EXT_directfb_surface`.
    pub unsafe fn create_direct_fb_surface_ext(
        &self,
        p_create_info: &DirectFBSurfaceCreateInfoEXT,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<SurfaceKHR> {
        let fp = self
            .commands()
            .create_direct_fb_surface_ext
            .expect("vkCreateDirectFBSurfaceEXT not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetPhysicalDeviceDirectFBPresentationSupportEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceDirectFBPresentationSupportEXT.html).
    /**
    Provided by **VK_EXT_directfb_surface**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries whether a queue family supports presentation to a
    ///DirectFB surface. Linux/DirectFB only.
    ///
    ///Requires `VK_EXT_directfb_surface`.
    pub unsafe fn get_physical_device_direct_fb_presentation_support_ext(
        &self,
        physical_device: PhysicalDevice,
        queue_family_index: u32,
    ) -> core::ffi::c_void {
        let fp = self
            .commands()
            .get_physical_device_direct_fb_presentation_support_ext
            .expect("vkGetPhysicalDeviceDirectFBPresentationSupportEXT not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(physical_device, queue_family_index, &mut out) };
        out
    }
    ///Wraps [`vkCreateImagePipeSurfaceFUCHSIA`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateImagePipeSurfaceFUCHSIA.html).
    /**
    Provided by **VK_FUCHSIA_imagepipe_surface**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `instance` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a Vulkan surface backed by a Fuchsia ImagePipe. Fuchsia
    ///OS only. The create info references the ImagePipe handle from
    ///the Fuchsia scenic compositor.
    ///
    ///Requires `VK_FUCHSIA_imagepipe_surface`.
    pub unsafe fn create_image_pipe_surface_fuchsia(
        &self,
        p_create_info: &ImagePipeSurfaceCreateInfoFUCHSIA,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<SurfaceKHR> {
        let fp = self
            .commands()
            .create_image_pipe_surface_fuchsia
            .expect("vkCreateImagePipeSurfaceFUCHSIA not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkCreateStreamDescriptorSurfaceGGP`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateStreamDescriptorSurfaceGGP.html).
    /**
    Provided by **VK_GGP_stream_descriptor_surface**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_NATIVE_WINDOW_IN_USE_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `instance` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a Vulkan surface from a Google Games Platform (Stadia)
    ///stream descriptor. GGP only. The platform has been discontinued.
    ///
    ///Requires `VK_GGP_stream_descriptor_surface`.
    pub unsafe fn create_stream_descriptor_surface_ggp(
        &self,
        p_create_info: &StreamDescriptorSurfaceCreateInfoGGP,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<SurfaceKHR> {
        let fp = self
            .commands()
            .create_stream_descriptor_surface_ggp
            .expect("vkCreateStreamDescriptorSurfaceGGP not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkCreateScreenSurfaceQNX`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateScreenSurfaceQNX.html).
    /**
    Provided by **VK_QNX_screen_surface**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `instance` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a Vulkan surface for a QNX Screen window. QNX only.
    ///The create info references the QNX screen context and window.
    ///
    ///Requires `VK_QNX_screen_surface`.
    pub unsafe fn create_screen_surface_qnx(
        &self,
        p_create_info: &ScreenSurfaceCreateInfoQNX,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<SurfaceKHR> {
        let fp = self
            .commands()
            .create_screen_surface_qnx
            .expect("vkCreateScreenSurfaceQNX not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetPhysicalDeviceScreenPresentationSupportQNX`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceScreenPresentationSupportQNX.html).
    /**
    Provided by **VK_QNX_screen_surface**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries whether a queue family supports presentation on QNX
    ///Screen. QNX only.
    ///
    ///Requires `VK_QNX_screen_surface`.
    pub unsafe fn get_physical_device_screen_presentation_support_qnx(
        &self,
        physical_device: PhysicalDevice,
        queue_family_index: u32,
    ) -> core::ffi::c_void {
        let fp = self
            .commands()
            .get_physical_device_screen_presentation_support_qnx
            .expect("vkGetPhysicalDeviceScreenPresentationSupportQNX not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(physical_device, queue_family_index, &mut out) };
        out
    }
    ///Wraps [`vkCreateDebugReportCallbackEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateDebugReportCallbackEXT.html).
    /**
    Provided by **VK_EXT_debug_report**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `instance` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a legacy debug report callback. Superseded by
    ///`create_debug_utils_messenger_ext` (`VK_EXT_debug_utils`),
    ///which provides richer message data and object labeling.
    ///
    ///The callback receives validation messages filtered by the
    ///`flags` bitmask (error, warning, performance, info, debug).
    ///
    ///Destroy with `destroy_debug_report_callback_ext`.
    ///
    ///Requires `VK_EXT_debug_report`. Prefer `VK_EXT_debug_utils`
    ///for new code.
    pub unsafe fn create_debug_report_callback_ext(
        &self,
        p_create_info: &DebugReportCallbackCreateInfoEXT,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<DebugReportCallbackEXT> {
        let fp = self
            .commands()
            .create_debug_report_callback_ext
            .expect("vkCreateDebugReportCallbackEXT not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyDebugReportCallbackEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyDebugReportCallbackEXT.html).
    /**
    Provided by **VK_EXT_debug_report**.*/
    ///
    ///# Safety
    ///- `instance` (self) must be valid and not destroyed.
    ///- `callback` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a legacy debug report callback created with
    ///`create_debug_report_callback_ext`.
    ///
    ///Destroy before the instance is destroyed.
    ///
    ///Requires `VK_EXT_debug_report`.
    pub unsafe fn destroy_debug_report_callback_ext(
        &self,
        callback: DebugReportCallbackEXT,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_debug_report_callback_ext
            .expect("vkDestroyDebugReportCallbackEXT not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), callback, alloc_ptr) };
    }
    ///Wraps [`vkDebugReportMessageEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDebugReportMessageEXT.html).
    /**
    Provided by **VK_EXT_debug_report**.*/
    ///
    ///# Safety
    ///- `instance` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Manually injects a message into the legacy debug report callback
    ///chain. All registered callbacks matching the specified `flags`
    ///will receive the message.
    ///
    ///`p_layer_prefix` and `p_message` are C strings. `object` is the
    ///raw handle of the relevant Vulkan object (or 0 if none).
    ///
    ///Superseded by `submit_debug_utils_message_ext`.
    ///
    ///Requires `VK_EXT_debug_report`.
    pub unsafe fn debug_report_message_ext(
        &self,
        flags: DebugReportFlagsEXT,
        object_type: DebugReportObjectTypeEXT,
        object: u64,
        location: usize,
        message_code: i32,
        p_layer_prefix: *const core::ffi::c_char,
        p_message: *const core::ffi::c_char,
    ) {
        let fp = self
            .commands()
            .debug_report_message_ext
            .expect("vkDebugReportMessageEXT not loaded");
        unsafe {
            fp(
                self.handle(),
                flags,
                object_type,
                object,
                location,
                message_code,
                p_layer_prefix,
                p_message,
            )
        };
    }
    ///Wraps [`vkGetPhysicalDeviceExternalImageFormatPropertiesNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceExternalImageFormatPropertiesNV.html).
    /**
    Provided by **VK_NV_external_memory_capabilities**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_FORMAT_NOT_SUPPORTED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Legacy NV path for querying external image format properties.
    ///Takes the full set of image creation parameters plus an external
    ///handle type and returns compatibility information. Prefer the
    ///core `get_physical_device_image_format_properties2` with
    ///`PhysicalDeviceExternalImageFormatInfo` in the pNext chain.
    ///
    ///Requires `VK_NV_external_memory_capabilities`.
    pub unsafe fn get_physical_device_external_image_format_properties_nv(
        &self,
        physical_device: PhysicalDevice,
        format: Format,
        r#type: ImageType,
        tiling: ImageTiling,
        usage: ImageUsageFlags,
        flags: ImageCreateFlags,
        external_handle_type: ExternalMemoryHandleTypeFlagsNV,
    ) -> VkResult<ExternalImageFormatPropertiesNV> {
        let fp = self
            .commands()
            .get_physical_device_external_image_format_properties_nv
            .expect("vkGetPhysicalDeviceExternalImageFormatPropertiesNV not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe {
            fp(
                physical_device,
                format,
                r#type,
                tiling,
                usage,
                flags,
                external_handle_type,
                &mut out,
            )
        })?;
        Ok(out)
    }
    ///Wraps [`vkGetPhysicalDeviceFeatures2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceFeatures2.html).
    /**
    Provided by **VK_BASE_VERSION_1_1**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.1 version of `get_physical_device_features` that supports
    ///chaining additional feature structs via pNext.
    ///
    ///Chain version-specific and extension feature structs to query their
    ///availability:
    ///
    ///- `PhysicalDeviceVulkan11Features`
    ///- `PhysicalDeviceVulkan12Features`
    ///- `PhysicalDeviceVulkan13Features`
    ///- Extension-specific structs like
    ///  `PhysicalDeviceRayTracingPipelineFeaturesKHR`
    ///
    ///Then pass the same chain (with desired features enabled) to
    ///`DeviceCreateInfo` to enable them at device creation.
    ///
    ///Always query before enabling, requesting an unsupported feature
    ///fails device creation.
    pub unsafe fn get_physical_device_features2(
        &self,
        physical_device: PhysicalDevice,
        p_features: *mut PhysicalDeviceFeatures2,
    ) {
        let fp = self
            .commands()
            .get_physical_device_features2
            .expect("vkGetPhysicalDeviceFeatures2 not loaded");
        unsafe { fp(physical_device, p_features) };
    }
    ///Wraps [`vkGetPhysicalDeviceProperties2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceProperties2.html).
    /**
    Provided by **VK_BASE_VERSION_1_1**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.1 version of `get_physical_device_properties` that supports
    ///chaining additional property structs via pNext.
    ///
    ///Chain version-specific and extension property structs to query
    ///extended limits and capabilities:
    ///
    ///- `PhysicalDeviceVulkan11Properties`: subgroup properties, point
    ///  clipping, protected memory.
    ///- `PhysicalDeviceVulkan12Properties`: driver conformance, denorm
    ///  behaviour, float controls, descriptor indexing limits, timeline
    ///  semaphore properties.
    ///- `PhysicalDeviceVulkan13Properties`: subgroup size control,
    ///  inline uniform block limits, dynamic rendering limits.
    ///- Extension structs like
    ///  `PhysicalDeviceRayTracingPipelinePropertiesKHR`.
    ///
    ///The base `PhysicalDeviceProperties` is identical to what
    ///`get_physical_device_properties` returns.
    pub unsafe fn get_physical_device_properties2(
        &self,
        physical_device: PhysicalDevice,
        p_properties: *mut PhysicalDeviceProperties2,
    ) {
        let fp = self
            .commands()
            .get_physical_device_properties2
            .expect("vkGetPhysicalDeviceProperties2 not loaded");
        unsafe { fp(physical_device, p_properties) };
    }
    ///Wraps [`vkGetPhysicalDeviceFormatProperties2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceFormatProperties2.html).
    /**
    Provided by **VK_BASE_VERSION_1_1**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.1 version of `get_physical_device_format_properties` that
    ///supports extensible output via pNext.
    ///
    ///Chain `DrmFormatModifierPropertiesListEXT` to query DRM format
    ///modifier support, or `FormatProperties3` (Vulkan 1.3) for extended
    ///format feature flags that do not fit in the original 32-bit fields.
    ///
    ///The base `FormatProperties` (linear, optimal, buffer features) is
    ///identical to what `get_physical_device_format_properties` returns.
    pub unsafe fn get_physical_device_format_properties2(
        &self,
        physical_device: PhysicalDevice,
        format: Format,
        p_format_properties: *mut FormatProperties2,
    ) {
        let fp = self
            .commands()
            .get_physical_device_format_properties2
            .expect("vkGetPhysicalDeviceFormatProperties2 not loaded");
        unsafe { fp(physical_device, format, p_format_properties) };
    }
    ///Wraps [`vkGetPhysicalDeviceImageFormatProperties2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceImageFormatProperties2.html).
    /**
    Provided by **VK_BASE_VERSION_1_1**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_FORMAT_NOT_SUPPORTED`
    ///- `VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR`
    ///- `VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR`
    ///- `VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR`
    ///- `VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR`
    ///- `VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.1 version of `get_physical_device_image_format_properties`
    ///that supports extensible input and output via pNext.
    ///
    ///Chain in the input:
    ///
    ///- `PhysicalDeviceExternalImageFormatInfo`: query external memory
    ///  compatibility for the image.
    ///- `PhysicalDeviceImageDrmFormatModifierInfoEXT`: query DRM modifier
    ///  support.
    ///
    ///Chain in the output:
    ///
    ///- `ExternalImageFormatProperties`: external memory capabilities.
    ///- `SamplerYcbcrConversionImageFormatProperties`: YCBCR support.
    ///
    ///Returns `VK_ERROR_FORMAT_NOT_SUPPORTED` if the combination is not
    ///supported.
    pub unsafe fn get_physical_device_image_format_properties2(
        &self,
        physical_device: PhysicalDevice,
        p_image_format_info: &PhysicalDeviceImageFormatInfo2,
        p_image_format_properties: *mut ImageFormatProperties2,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_physical_device_image_format_properties2
            .expect("vkGetPhysicalDeviceImageFormatProperties2 not loaded");
        check(unsafe {
            fp(
                physical_device,
                p_image_format_info,
                p_image_format_properties,
            )
        })
    }
    ///Wraps [`vkGetPhysicalDeviceQueueFamilyProperties2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceQueueFamilyProperties2.html).
    /**
    Provided by **VK_BASE_VERSION_1_1**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.1 version of `get_physical_device_queue_family_properties`
    ///that supports extensible output via pNext.
    ///
    ///Chain `QueueFamilyCheckpointPropertiesNV` for diagnostic checkpoint
    ///support, or `QueueFamilyGlobalPriorityPropertiesKHR` for
    ///global priority scheduling capabilities.
    ///
    ///The base `QueueFamilyProperties` is identical to what
    ///`get_physical_device_queue_family_properties` returns.
    pub unsafe fn get_physical_device_queue_family_properties2(
        &self,
        physical_device: PhysicalDevice,
    ) -> Vec<QueueFamilyProperties2> {
        let fp = self
            .commands()
            .get_physical_device_queue_family_properties2
            .expect("vkGetPhysicalDeviceQueueFamilyProperties2 not loaded");
        fill_two_call(|count, data| unsafe { fp(physical_device, count, data) })
    }
    ///Wraps [`vkGetPhysicalDeviceMemoryProperties2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceMemoryProperties2.html).
    /**
    Provided by **VK_BASE_VERSION_1_1**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.1 version of `get_physical_device_memory_properties` that
    ///supports extensible output via pNext.
    ///
    ///Chain `PhysicalDeviceMemoryBudgetPropertiesEXT` (if the
    ///`VK_EXT_memory_budget` extension is available) to query per-heap
    ///budget and current usage. This is essential for managing memory
    ///pressure on systems with unified memory or limited VRAM.
    ///
    ///The base `PhysicalDeviceMemoryProperties` (heaps and types) is
    ///identical to what `get_physical_device_memory_properties` returns.
    pub unsafe fn get_physical_device_memory_properties2(
        &self,
        physical_device: PhysicalDevice,
        p_memory_properties: *mut PhysicalDeviceMemoryProperties2,
    ) {
        let fp = self
            .commands()
            .get_physical_device_memory_properties2
            .expect("vkGetPhysicalDeviceMemoryProperties2 not loaded");
        unsafe { fp(physical_device, p_memory_properties) };
    }
    ///Wraps [`vkGetPhysicalDeviceSparseImageFormatProperties2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceSparseImageFormatProperties2.html).
    /**
    Provided by **VK_BASE_VERSION_1_1**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Vulkan 1.1 version of
    ///`get_physical_device_sparse_image_format_properties` that supports
    ///extensible output via pNext.
    ///
    ///Only relevant for sparse images. Returns the same base sparse format
    ///properties as the 1.0 version.
    pub unsafe fn get_physical_device_sparse_image_format_properties2(
        &self,
        physical_device: PhysicalDevice,
        p_format_info: &PhysicalDeviceSparseImageFormatInfo2,
    ) -> Vec<SparseImageFormatProperties2> {
        let fp = self
            .commands()
            .get_physical_device_sparse_image_format_properties2
            .expect("vkGetPhysicalDeviceSparseImageFormatProperties2 not loaded");
        fill_two_call(|count, data| unsafe { fp(physical_device, p_format_info, count, data) })
    }
    ///Wraps [`vkGetPhysicalDeviceExternalBufferProperties`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceExternalBufferProperties.html).
    /**
    Provided by **VK_BASE_VERSION_1_1**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries whether a buffer with the given usage and flags can be
    ///exported to or imported from an external handle type (e.g. POSIX
    ///file descriptor, Win32 handle, or DMA-BUF).
    ///
    ///The returned `ExternalBufferProperties` indicates:
    ///
    ///- Whether the external handle type is compatible.
    ///- Whether dedicated allocation is required.
    ///- Which other handle types the memory can be exported to
    ///  simultaneously.
    ///
    ///Use this before creating a buffer intended for cross-process or
    ///cross-API sharing to verify the external memory capabilities.
    pub unsafe fn get_physical_device_external_buffer_properties(
        &self,
        physical_device: PhysicalDevice,
        p_external_buffer_info: &PhysicalDeviceExternalBufferInfo,
        p_external_buffer_properties: *mut ExternalBufferProperties,
    ) {
        let fp = self
            .commands()
            .get_physical_device_external_buffer_properties
            .expect("vkGetPhysicalDeviceExternalBufferProperties not loaded");
        unsafe {
            fp(
                physical_device,
                p_external_buffer_info,
                p_external_buffer_properties,
            )
        };
    }
    ///Wraps [`vkGetPhysicalDeviceExternalMemorySciBufPropertiesNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceExternalMemorySciBufPropertiesNV.html).
    ///
    ///# Errors
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_INVALID_EXTERNAL_HANDLE`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries compatibility of external NvSciBuf memory handles with
    ///Vulkan on NVIDIA platforms (primarily automotive / embedded). Returns
    ///the compatible memory types for a given NvSciBuf attribute list.
    ///
    ///This is a platform-specific extension for NVIDIA's Safety Critical
    ///ecosystem. Not available on desktop or mobile platforms.
    pub unsafe fn get_physical_device_external_memory_sci_buf_properties_nv(
        &self,
        physical_device: PhysicalDevice,
        handle_type: ExternalMemoryHandleTypeFlagBits,
        handle: *const core::ffi::c_void,
        p_memory_sci_buf_properties: *mut MemorySciBufPropertiesNV,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_physical_device_external_memory_sci_buf_properties_nv
            .expect("vkGetPhysicalDeviceExternalMemorySciBufPropertiesNV not loaded");
        check(unsafe {
            fp(
                physical_device,
                handle_type,
                handle,
                p_memory_sci_buf_properties,
            )
        })
    }
    ///Wraps [`vkGetPhysicalDeviceSciBufAttributesNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceSciBufAttributesNV.html).
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Fills an NvSciBuf attribute list with Vulkan's requirements for
    ///a given image or buffer creation info. Used to negotiate buffer
    ///attributes when sharing memory between Vulkan and other NvSciBuf
    ///consumers.
    ///
    ///This is a platform-specific extension for NVIDIA's Safety Critical
    ///ecosystem. Not available on desktop or mobile platforms.
    pub unsafe fn get_physical_device_sci_buf_attributes_nv(
        &self,
        physical_device: PhysicalDevice,
        p_attributes: *const core::ffi::c_void,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_physical_device_sci_buf_attributes_nv
            .expect("vkGetPhysicalDeviceSciBufAttributesNV not loaded");
        check(unsafe { fp(physical_device, p_attributes) })
    }
    ///Wraps [`vkGetPhysicalDeviceExternalSemaphoreProperties`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceExternalSemaphoreProperties.html).
    /**
    Provided by **VK_BASE_VERSION_1_1**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries whether a semaphore can be exported to or imported from an
    ///external handle type (e.g. POSIX file descriptor, Win32 handle,
    ///sync file, or Zircon event).
    ///
    ///The returned properties indicate:
    ///
    ///- Whether the external handle type is compatible with semaphores.
    ///- Whether the handle is a copy or a reference.
    ///- Which other handle types the semaphore can be exported to.
    ///
    ///External semaphores are the primary cross-process and cross-API
    ///synchronisation mechanism, for example, synchronising Vulkan
    ///rendering with an OpenGL or DirectX consumer.
    pub unsafe fn get_physical_device_external_semaphore_properties(
        &self,
        physical_device: PhysicalDevice,
        p_external_semaphore_info: &PhysicalDeviceExternalSemaphoreInfo,
        p_external_semaphore_properties: *mut ExternalSemaphoreProperties,
    ) {
        let fp = self
            .commands()
            .get_physical_device_external_semaphore_properties
            .expect("vkGetPhysicalDeviceExternalSemaphoreProperties not loaded");
        unsafe {
            fp(
                physical_device,
                p_external_semaphore_info,
                p_external_semaphore_properties,
            )
        };
    }
    ///Wraps [`vkGetPhysicalDeviceExternalFenceProperties`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceExternalFenceProperties.html).
    /**
    Provided by **VK_BASE_VERSION_1_1**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries whether a fence can be exported to or imported from an
    ///external handle type (e.g. POSIX file descriptor, Win32 handle,
    ///or sync file).
    ///
    ///The returned properties indicate:
    ///
    ///- Whether the external handle type is compatible with fences.
    ///- Whether the handle is a copy or a reference to the fence state.
    ///- Which other handle types the fence can be exported to.
    ///
    ///Use this before creating a fence intended for cross-process
    ///synchronisation.
    pub unsafe fn get_physical_device_external_fence_properties(
        &self,
        physical_device: PhysicalDevice,
        p_external_fence_info: &PhysicalDeviceExternalFenceInfo,
        p_external_fence_properties: *mut ExternalFenceProperties,
    ) {
        let fp = self
            .commands()
            .get_physical_device_external_fence_properties
            .expect("vkGetPhysicalDeviceExternalFenceProperties not loaded");
        unsafe {
            fp(
                physical_device,
                p_external_fence_info,
                p_external_fence_properties,
            )
        };
    }
    ///Wraps [`vkGetPhysicalDeviceSciSyncAttributesNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceSciSyncAttributesNV.html).
    ///
    ///# Errors
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Fills an NvSciSync attribute list with Vulkan's requirements for
    ///synchronisation. Used to negotiate sync object attributes when
    ///sharing synchronisation primitives between Vulkan and other NvSciSync
    ///consumers (e.g. camera, display, or compute pipelines).
    ///
    ///This is a platform-specific extension for NVIDIA's Safety Critical
    ///ecosystem. Not available on desktop or mobile platforms.
    pub unsafe fn get_physical_device_sci_sync_attributes_nv(
        &self,
        physical_device: PhysicalDevice,
        p_sci_sync_attributes_info: &SciSyncAttributesInfoNV,
        p_attributes: *const core::ffi::c_void,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_physical_device_sci_sync_attributes_nv
            .expect("vkGetPhysicalDeviceSciSyncAttributesNV not loaded");
        check(unsafe { fp(physical_device, p_sci_sync_attributes_info, p_attributes) })
    }
    ///Wraps [`vkReleaseDisplayEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkReleaseDisplayEXT.html).
    /**
    Provided by **VK_EXT_direct_mode_display**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Releases a previously acquired display, returning control to the
    ///platform's display manager. Call this when done with direct
    ///display rendering.
    ///
    ///Requires `VK_EXT_direct_mode_display`.
    pub unsafe fn release_display_ext(
        &self,
        physical_device: PhysicalDevice,
        display: DisplayKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .release_display_ext
            .expect("vkReleaseDisplayEXT not loaded");
        check(unsafe { fp(physical_device, display) })
    }
    ///Wraps [`vkAcquireXlibDisplayEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkAcquireXlibDisplayEXT.html).
    /**
    Provided by **VK_EXT_acquire_xlib_display**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Acquires exclusive control of an X11 display for direct rendering,
    ///bypassing the X server's compositor. The display must be released
    ///with `release_display_ext` when finished.
    ///
    ///Requires `VK_EXT_acquire_xlib_display`. Linux/X11 only.
    pub unsafe fn acquire_xlib_display_ext(
        &self,
        physical_device: PhysicalDevice,
        display: DisplayKHR,
    ) -> VkResult<core::ffi::c_void> {
        let fp = self
            .commands()
            .acquire_xlib_display_ext
            .expect("vkAcquireXlibDisplayEXT not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(physical_device, &mut out, display) })?;
        Ok(out)
    }
    ///Wraps [`vkGetRandROutputDisplayEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetRandROutputDisplayEXT.html).
    /**
    Provided by **VK_EXT_acquire_xlib_display**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Maps an X11 RandR output to a Vulkan `DisplayKHR` handle. Use
    ///this to identify which Vulkan display corresponds to a specific
    ///RandR output when doing direct display rendering.
    ///
    ///Requires `VK_EXT_acquire_xlib_display`. Linux/X11 only.
    pub unsafe fn get_rand_r_output_display_ext(
        &self,
        physical_device: PhysicalDevice,
        dpy: *mut core::ffi::c_void,
        rr_output: core::ffi::c_ulong,
    ) -> VkResult<DisplayKHR> {
        let fp = self
            .commands()
            .get_rand_r_output_display_ext
            .expect("vkGetRandROutputDisplayEXT not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(physical_device, dpy, rr_output, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkAcquireWinrtDisplayNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkAcquireWinrtDisplayNV.html).
    /**
    Provided by **VK_NV_acquire_winrt_display**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_DEVICE_LOST`
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Acquires exclusive ownership of a display using the Windows
    ///Runtime (WinRT) display interface. Windows only. The display
    ///must be released before another application can use it.
    ///
    ///Requires `VK_NV_acquire_winrt_display`.
    pub unsafe fn acquire_winrt_display_nv(
        &self,
        physical_device: PhysicalDevice,
        display: DisplayKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .acquire_winrt_display_nv
            .expect("vkAcquireWinrtDisplayNV not loaded");
        check(unsafe { fp(physical_device, display) })
    }
    ///Wraps [`vkGetWinrtDisplayNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetWinrtDisplayNV.html).
    /**
    Provided by **VK_NV_acquire_winrt_display**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_DEVICE_LOST`
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Gets a `DisplayKHR` handle for a WinRT display identified by
    ///its device-relative ID. Windows only. Use with
    ///`acquire_winrt_display_nv` for direct display access.
    ///
    ///Requires `VK_NV_acquire_winrt_display`.
    pub unsafe fn get_winrt_display_nv(
        &self,
        physical_device: PhysicalDevice,
        device_relative_id: u32,
    ) -> VkResult<DisplayKHR> {
        let fp = self
            .commands()
            .get_winrt_display_nv
            .expect("vkGetWinrtDisplayNV not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(physical_device, device_relative_id, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetPhysicalDeviceSurfaceCapabilities2EXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceSurfaceCapabilities2EXT.html).
    /**
    Provided by **VK_EXT_display_surface_counter**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries surface capabilities with additional output fields
    ///compared to `get_physical_device_surface_capabilities_khr`.
    ///Returns a `SurfaceCapabilities2EXT` that includes shared
    ///present mode support flags.
    ///
    ///Prefer `get_physical_device_surface_capabilities2_khr` (KHR)
    ///for general use; this EXT variant is primarily for
    ///`VK_EXT_display_surface_counter` integration.
    ///
    ///Requires `VK_EXT_display_surface_counter`.
    pub unsafe fn get_physical_device_surface_capabilities2_ext(
        &self,
        physical_device: PhysicalDevice,
        surface: SurfaceKHR,
        p_surface_capabilities: *mut SurfaceCapabilities2EXT,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_physical_device_surface_capabilities2_ext
            .expect("vkGetPhysicalDeviceSurfaceCapabilities2EXT not loaded");
        check(unsafe { fp(physical_device, surface, p_surface_capabilities) })
    }
    ///Wraps [`vkEnumeratePhysicalDeviceGroups`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkEnumeratePhysicalDeviceGroups.html).
    /**
    Provided by **VK_BASE_VERSION_1_1**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `instance` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Enumerates the available device groups, sets of physical devices
    ///that can be used together as a single logical device for multi-GPU
    ///rendering (e.g. SLI/CrossFire).
    ///
    ///Each `PhysicalDeviceGroupProperties` lists the physical devices in
    ///the group and whether the group supports memory allocation that
    ///spans all devices (`subset_allocation`).
    ///
    ///On single-GPU systems, this returns one group containing one device.
    ///
    ///To use a device group, pass `DeviceGroupDeviceCreateInfo` in the
    ///pNext chain of `DeviceCreateInfo` with the desired physical devices.
    ///This is an advanced multi-GPU feature; most applications use a
    ///single physical device.
    pub unsafe fn enumerate_physical_device_groups(
        &self,
    ) -> VkResult<Vec<PhysicalDeviceGroupProperties>> {
        let fp = self
            .commands()
            .enumerate_physical_device_groups
            .expect("vkEnumeratePhysicalDeviceGroups not loaded");
        enumerate_two_call(|count, data| unsafe { fp(self.handle(), count, data) })
    }
    ///Wraps [`vkGetPhysicalDevicePresentRectanglesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDevicePresentRectanglesKHR.html).
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
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///- `surface` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Returns the set of rectangular regions that cover the presentable
    ///area of a surface for a device group. Each rectangle represents a
    ///region that one physical device in the group is responsible for
    ///presenting.
    ///
    ///Only relevant for multi-GPU device groups with
    ///`DEVICE_GROUP_PRESENT_MODE_LOCAL_MULTI_DEVICE`. On single-GPU
    ///systems, this returns a single rectangle covering the entire
    ///surface.
    pub unsafe fn get_physical_device_present_rectangles_khr(
        &self,
        physical_device: PhysicalDevice,
        surface: SurfaceKHR,
    ) -> VkResult<Vec<Rect2D>> {
        let fp = self
            .commands()
            .get_physical_device_present_rectangles_khr
            .expect("vkGetPhysicalDevicePresentRectanglesKHR not loaded");
        enumerate_two_call(|count, data| unsafe { fp(physical_device, surface, count, data) })
    }
    ///Wraps [`vkCreateIOSSurfaceMVK`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateIOSSurfaceMVK.html).
    /**
    Provided by **VK_MVK_ios_surface**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_NATIVE_WINDOW_IN_USE_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `instance` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a Vulkan surface for an iOS `UIView`. iOS only. Legacy
    ///MoltenVK path, prefer `VK_EXT_metal_surface` with
    ///`create_metal_surface_ext` on modern MoltenVK.
    ///
    ///Requires `VK_MVK_ios_surface`.
    pub unsafe fn create_ios_surface_mvk(
        &self,
        p_create_info: &IOSSurfaceCreateInfoMVK,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<SurfaceKHR> {
        let fp = self
            .commands()
            .create_ios_surface_mvk
            .expect("vkCreateIOSSurfaceMVK not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkCreateMacOSSurfaceMVK`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateMacOSSurfaceMVK.html).
    /**
    Provided by **VK_MVK_macos_surface**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_NATIVE_WINDOW_IN_USE_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `instance` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a Vulkan surface for a macOS `NSView`. macOS only.
    ///Legacy MoltenVK path, prefer `VK_EXT_metal_surface` with
    ///`create_metal_surface_ext` on modern MoltenVK.
    ///
    ///Requires `VK_MVK_macos_surface`.
    pub unsafe fn create_mac_os_surface_mvk(
        &self,
        p_create_info: &MacOSSurfaceCreateInfoMVK,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<SurfaceKHR> {
        let fp = self
            .commands()
            .create_mac_os_surface_mvk
            .expect("vkCreateMacOSSurfaceMVK not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetPhysicalDeviceMultisamplePropertiesEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceMultisamplePropertiesEXT.html).
    /**
    Provided by **VK_EXT_sample_locations**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the multisample properties for a specific sample count
    ///on the physical device. Returns the maximum sample location grid
    ///size in `MultisamplePropertiesEXT`.
    ///
    ///Use this to determine valid grid sizes for
    ///`cmd_set_sample_locations_ext`.
    ///
    ///Requires `VK_EXT_sample_locations`.
    pub unsafe fn get_physical_device_multisample_properties_ext(
        &self,
        physical_device: PhysicalDevice,
        samples: SampleCountFlagBits,
        p_multisample_properties: *mut MultisamplePropertiesEXT,
    ) {
        let fp = self
            .commands()
            .get_physical_device_multisample_properties_ext
            .expect("vkGetPhysicalDeviceMultisamplePropertiesEXT not loaded");
        unsafe { fp(physical_device, samples, p_multisample_properties) };
    }
    ///Wraps [`vkGetPhysicalDeviceSurfaceCapabilities2KHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceSurfaceCapabilities2KHR.html).
    /**
    Provided by **VK_KHR_get_surface_capabilities2**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Extensible version of
    ///`get_physical_device_surface_capabilities_khr`. Takes a
    ///`PhysicalDeviceSurfaceInfo2KHR` input and writes to
    ///`SurfaceCapabilities2KHR`, both supporting `pNext` chains.
    ///
    ///Chain `SurfaceProtectedCapabilitiesKHR` or other extension
    ///structs into the output `pNext` to query additional capabilities
    ///not available through the v1 query.
    ///
    ///Provided by `VK_KHR_get_surface_capabilities2`. Prefer this over
    ///the v1 query when available.
    pub unsafe fn get_physical_device_surface_capabilities2_khr(
        &self,
        physical_device: PhysicalDevice,
        p_surface_info: &PhysicalDeviceSurfaceInfo2KHR,
        p_surface_capabilities: *mut SurfaceCapabilities2KHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_physical_device_surface_capabilities2_khr
            .expect("vkGetPhysicalDeviceSurfaceCapabilities2KHR not loaded");
        check(unsafe { fp(physical_device, p_surface_info, p_surface_capabilities) })
    }
    ///Wraps [`vkGetPhysicalDeviceSurfaceFormats2KHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceSurfaceFormats2KHR.html).
    /**
    Provided by **VK_KHR_get_surface_capabilities2**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_SURFACE_LOST_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Extensible version of
    ///`get_physical_device_surface_formats_khr`. Returns
    ///`SurfaceFormat2KHR` with `pNext` support, allowing extensions
    ///to attach additional per-format information.
    ///
    ///Takes `PhysicalDeviceSurfaceInfo2KHR` as input so you can query
    ///formats for a specific surface configuration (e.g., with
    ///full-screen exclusive info chained in).
    ///
    ///Provided by `VK_KHR_get_surface_capabilities2`. Prefer this over
    ///the v1 query when available.
    pub unsafe fn get_physical_device_surface_formats2_khr(
        &self,
        physical_device: PhysicalDevice,
        p_surface_info: &PhysicalDeviceSurfaceInfo2KHR,
    ) -> VkResult<Vec<SurfaceFormat2KHR>> {
        let fp = self
            .commands()
            .get_physical_device_surface_formats2_khr
            .expect("vkGetPhysicalDeviceSurfaceFormats2KHR not loaded");
        enumerate_two_call(|count, data| unsafe {
            fp(physical_device, p_surface_info, count, data)
        })
    }
    ///Wraps [`vkGetPhysicalDeviceDisplayProperties2KHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceDisplayProperties2KHR.html).
    /**
    Provided by **VK_KHR_get_display_properties2**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Extensible version of `get_physical_device_display_properties_khr`.
    ///Returns `DisplayProperties2KHR` which wraps the original
    ///properties and supports `pNext` extensions.
    ///
    ///Provided by `VK_KHR_get_display_properties2`. Prefer this over
    ///the v1 query when available.
    pub unsafe fn get_physical_device_display_properties2_khr(
        &self,
        physical_device: PhysicalDevice,
    ) -> VkResult<Vec<DisplayProperties2KHR>> {
        let fp = self
            .commands()
            .get_physical_device_display_properties2_khr
            .expect("vkGetPhysicalDeviceDisplayProperties2KHR not loaded");
        enumerate_two_call(|count, data| unsafe { fp(physical_device, count, data) })
    }
    ///Wraps [`vkGetPhysicalDeviceDisplayPlaneProperties2KHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceDisplayPlaneProperties2KHR.html).
    /**
    Provided by **VK_KHR_get_display_properties2**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Extensible version of
    ///`get_physical_device_display_plane_properties_khr`. Returns
    ///`DisplayPlaneProperties2KHR` with `pNext` support.
    ///
    ///Provided by `VK_KHR_get_display_properties2`. Prefer this over
    ///the v1 query when available.
    pub unsafe fn get_physical_device_display_plane_properties2_khr(
        &self,
        physical_device: PhysicalDevice,
    ) -> VkResult<Vec<DisplayPlaneProperties2KHR>> {
        let fp = self
            .commands()
            .get_physical_device_display_plane_properties2_khr
            .expect("vkGetPhysicalDeviceDisplayPlaneProperties2KHR not loaded");
        enumerate_two_call(|count, data| unsafe { fp(physical_device, count, data) })
    }
    ///Wraps [`vkGetDisplayModeProperties2KHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDisplayModeProperties2KHR.html).
    /**
    Provided by **VK_KHR_get_display_properties2**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Extensible version of `get_display_mode_properties_khr`. Returns
    ///`DisplayModeProperties2KHR` with `pNext` support.
    ///
    ///Provided by `VK_KHR_get_display_properties2`. Prefer this over
    ///the v1 query when available.
    pub unsafe fn get_display_mode_properties2_khr(
        &self,
        physical_device: PhysicalDevice,
        display: DisplayKHR,
    ) -> VkResult<Vec<DisplayModeProperties2KHR>> {
        let fp = self
            .commands()
            .get_display_mode_properties2_khr
            .expect("vkGetDisplayModeProperties2KHR not loaded");
        enumerate_two_call(|count, data| unsafe { fp(physical_device, display, count, data) })
    }
    ///Wraps [`vkGetDisplayPlaneCapabilities2KHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDisplayPlaneCapabilities2KHR.html).
    /**
    Provided by **VK_KHR_get_display_properties2**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Extensible version of `get_display_plane_capabilities_khr`.
    ///Takes `DisplayPlaneInfo2KHR` (with `pNext` for input extensions)
    ///and writes to `DisplayPlaneCapabilities2KHR` (with `pNext` for
    ///output extensions).
    ///
    ///Provided by `VK_KHR_get_display_properties2`. Prefer this over
    ///the v1 query when available.
    pub unsafe fn get_display_plane_capabilities2_khr(
        &self,
        physical_device: PhysicalDevice,
        p_display_plane_info: &DisplayPlaneInfo2KHR,
        p_capabilities: *mut DisplayPlaneCapabilities2KHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_display_plane_capabilities2_khr
            .expect("vkGetDisplayPlaneCapabilities2KHR not loaded");
        check(unsafe { fp(physical_device, p_display_plane_info, p_capabilities) })
    }
    ///Wraps [`vkGetPhysicalDeviceCalibrateableTimeDomainsKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceCalibrateableTimeDomainsKHR.html).
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
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Enumerates the time domains that can be used with
    ///`get_calibrated_timestamps_khr` on this physical device.
    ///
    ///Common time domains include:
    ///
    ///- `DEVICE`: GPU timestamp counter (same as `cmd_write_timestamp2`).
    ///- `CLOCK_MONOTONIC` / `CLOCK_MONOTONIC_RAW`: Linux monotonic
    ///  clocks.
    ///- `QUERY_PERFORMANCE_COUNTER`: Windows high-resolution timer.
    ///
    ///The device time domain is always available. Host time domains
    ///depend on the platform.
    pub unsafe fn get_physical_device_calibrateable_time_domains_khr(
        &self,
        physical_device: PhysicalDevice,
    ) -> VkResult<Vec<TimeDomainKHR>> {
        let fp = self
            .commands()
            .get_physical_device_calibrateable_time_domains_khr
            .expect("vkGetPhysicalDeviceCalibrateableTimeDomainsKHR not loaded");
        enumerate_two_call(|count, data| unsafe { fp(physical_device, count, data) })
    }
    ///Wraps [`vkCreateDebugUtilsMessengerEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateDebugUtilsMessengerEXT.html).
    /**
    Provided by **VK_EXT_debug_utils**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `instance` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a debug messenger that receives validation layer messages,
    ///performance warnings, and general debug info via a user-provided
    ///callback.
    ///
    ///`DebugUtilsMessengerCreateInfoEXT` configures:
    ///- `message_severity`: which severities to receive (verbose, info,
    ///  warning, error).
    ///- `message_type`: which categories (general, validation,
    ///  performance).
    ///- `pfn_user_callback`: your callback function.
    ///
    ///Create the messenger immediately after the instance for maximum
    ///coverage. Destroy with `destroy_debug_utils_messenger_ext`.
    ///
    ///Requires `VK_EXT_debug_utils`. Supersedes `VK_EXT_debug_report`.
    pub unsafe fn create_debug_utils_messenger_ext(
        &self,
        p_create_info: &DebugUtilsMessengerCreateInfoEXT,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<DebugUtilsMessengerEXT> {
        let fp = self
            .commands()
            .create_debug_utils_messenger_ext
            .expect("vkCreateDebugUtilsMessengerEXT not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkDestroyDebugUtilsMessengerEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyDebugUtilsMessengerEXT.html).
    /**
    Provided by **VK_EXT_debug_utils**.*/
    ///
    ///# Safety
    ///- `instance` (self) must be valid and not destroyed.
    ///- `messenger` must be externally synchronized.
    ///
    ///# Usage Notes
    ///
    ///Destroys a debug messenger created with
    ///`create_debug_utils_messenger_ext`. After this call, the
    ///messenger's callback will no longer be invoked.
    ///
    ///Destroy before the instance is destroyed.
    ///
    ///Requires `VK_EXT_debug_utils`.
    pub unsafe fn destroy_debug_utils_messenger_ext(
        &self,
        messenger: DebugUtilsMessengerEXT,
        allocator: Option<&AllocationCallbacks>,
    ) {
        let fp = self
            .commands()
            .destroy_debug_utils_messenger_ext
            .expect("vkDestroyDebugUtilsMessengerEXT not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        unsafe { fp(self.handle(), messenger, alloc_ptr) };
    }
    ///Wraps [`vkSubmitDebugUtilsMessageEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkSubmitDebugUtilsMessageEXT.html).
    /**
    Provided by **VK_EXT_debug_utils**.*/
    ///
    ///# Safety
    ///- `instance` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Manually injects a debug message into the debug utils callback
    ///chain. The message is delivered to all active debug messengers
    ///that match the specified severity and type flags.
    ///
    ///Useful for application-level diagnostics, e.g., logging a
    ///warning when a resource limit is approached.
    ///
    ///The `DebugUtilsMessengerCallbackDataEXT` carries the message
    ///string, message ID, and optional object labels/queue labels for
    ///context.
    ///
    ///Requires `VK_EXT_debug_utils`.
    pub unsafe fn submit_debug_utils_message_ext(
        &self,
        message_severity: DebugUtilsMessageSeverityFlagBitsEXT,
        message_types: DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: &DebugUtilsMessengerCallbackDataEXT,
    ) {
        let fp = self
            .commands()
            .submit_debug_utils_message_ext
            .expect("vkSubmitDebugUtilsMessageEXT not loaded");
        unsafe {
            fp(
                self.handle(),
                message_severity,
                message_types,
                p_callback_data,
            )
        };
    }
    ///Wraps [`vkGetPhysicalDeviceCooperativeMatrixPropertiesNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceCooperativeMatrixPropertiesNV.html).
    /**
    Provided by **VK_NV_cooperative_matrix**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Enumerates the cooperative matrix types and sizes supported by
    ///the physical device. Uses the two-call idiom. Legacy NV path,
    ///prefer `get_physical_device_cooperative_matrix_properties_khr`.
    ///
    ///Requires `VK_NV_cooperative_matrix`.
    pub unsafe fn get_physical_device_cooperative_matrix_properties_nv(
        &self,
        physical_device: PhysicalDevice,
    ) -> VkResult<Vec<CooperativeMatrixPropertiesNV>> {
        let fp = self
            .commands()
            .get_physical_device_cooperative_matrix_properties_nv
            .expect("vkGetPhysicalDeviceCooperativeMatrixPropertiesNV not loaded");
        enumerate_two_call(|count, data| unsafe { fp(physical_device, count, data) })
    }
    ///Wraps [`vkGetPhysicalDeviceSurfacePresentModes2EXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceSurfacePresentModes2EXT.html).
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
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the supported present modes for a physical device and
    ///surface, using the extended surface info structure. This is the
    ///`VK_EXT_full_screen_exclusive` variant of
    ///`get_physical_device_surface_present_modes_khr`, allowing
    ///full-screen exclusive configuration to influence the result.
    ///
    ///Requires `VK_EXT_full_screen_exclusive`. Windows only.
    pub unsafe fn get_physical_device_surface_present_modes2_ext(
        &self,
        physical_device: PhysicalDevice,
        p_surface_info: &PhysicalDeviceSurfaceInfo2KHR,
    ) -> VkResult<Vec<PresentModeKHR>> {
        let fp = self
            .commands()
            .get_physical_device_surface_present_modes2_ext
            .expect("vkGetPhysicalDeviceSurfacePresentModes2EXT not loaded");
        enumerate_two_call(|count, data| unsafe {
            fp(physical_device, p_surface_info, count, data)
        })
    }
    ///Wraps [`vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR.html).
    /**
    Provided by **VK_KHR_performance_query**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Enumerates the performance counters available for a specific queue
    ///family on a physical device. Returns `PerformanceCounterKHR`
    ///structs (counter unit, storage type, UUID) and optionally fills
    ///`PerformanceCounterDescriptionKHR` with human-readable names and
    ///descriptions.
    ///
    ///Use the counter indices when creating a performance query pool
    ///with `QueryPoolPerformanceCreateInfoKHR`.
    ///
    ///Requires `VK_KHR_performance_query`.
    pub unsafe fn enumerate_physical_device_queue_family_performance_query_counters_khr(
        &self,
        physical_device: PhysicalDevice,
        queue_family_index: u32,
        p_counter_descriptions: *mut PerformanceCounterDescriptionKHR,
    ) -> VkResult<Vec<PerformanceCounterKHR>> {
        let fp = self
            .commands()
            .enumerate_physical_device_queue_family_performance_query_counters_khr
            .expect("vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR not loaded");
        enumerate_two_call(|count, data| unsafe {
            fp(
                physical_device,
                queue_family_index,
                count,
                data,
                p_counter_descriptions,
            )
        })
    }
    ///Wraps [`vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR.html).
    /**
    Provided by **VK_KHR_performance_query**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns the number of passes required to collect all the
    ///performance counters specified in
    ///`QueryPoolPerformanceCreateInfoKHR`.
    ///
    ///Hardware can typically sample only a limited number of counters
    ///per pass. If this returns N, you must submit the same command
    ///buffer N times (each with a different pass index set via
    ///`PerformanceQuerySubmitInfoKHR` in the pNext of
    ///`SubmitInfo`/`SubmitInfo2`) to collect all results.
    ///
    ///Requires `VK_KHR_performance_query`.
    pub unsafe fn get_physical_device_queue_family_performance_query_passes_khr(
        &self,
        physical_device: PhysicalDevice,
        p_performance_query_create_info: &QueryPoolPerformanceCreateInfoKHR,
    ) -> u32 {
        let fp = self
            .commands()
            .get_physical_device_queue_family_performance_query_passes_khr
            .expect("vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(physical_device, p_performance_query_create_info, &mut out) };
        out
    }
    ///Wraps [`vkCreateHeadlessSurfaceEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateHeadlessSurfaceEXT.html).
    /**
    Provided by **VK_EXT_headless_surface**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `instance` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Creates a headless surface that is not associated with any window
    ///system. Useful for off-screen rendering, compute-only workloads,
    ///and automated testing where no display is available.
    ///
    ///Destroy with `destroy_surface_khr`.
    ///
    ///Requires `VK_EXT_headless_surface`.
    pub unsafe fn create_headless_surface_ext(
        &self,
        p_create_info: &HeadlessSurfaceCreateInfoEXT,
        allocator: Option<&AllocationCallbacks>,
    ) -> VkResult<SurfaceKHR> {
        let fp = self
            .commands()
            .create_headless_surface_ext
            .expect("vkCreateHeadlessSurfaceEXT not loaded");
        let alloc_ptr = allocator.map_or(core::ptr::null(), core::ptr::from_ref);
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(self.handle(), p_create_info, alloc_ptr, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV.html).
    /**
    Provided by **VK_NV_coverage_reduction_mode**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Enumerates the supported combinations of coverage reduction
    ///mode, rasterisation samples, and colour/depth sample counts for
    ///mixed-sample rendering. Uses the two-call idiom.
    ///
    ///Requires `VK_NV_coverage_reduction_mode`.
    pub unsafe fn get_physical_device_supported_framebuffer_mixed_samples_combinations_nv(
        &self,
        physical_device: PhysicalDevice,
    ) -> VkResult<Vec<FramebufferMixedSamplesCombinationNV>> {
        let fp = self
            .commands()
            .get_physical_device_supported_framebuffer_mixed_samples_combinations_nv
            .expect("vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV not loaded");
        enumerate_two_call(|count, data| unsafe { fp(physical_device, count, data) })
    }
    ///Wraps [`vkGetPhysicalDeviceToolProperties`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceToolProperties.html).
    /**
    Provided by **VK_BASE_VERSION_1_3**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns a list of active tools (validation layers, profilers,
    ///debuggers, crash dump utilities) that are currently intercepting
    ///Vulkan calls for this physical device.
    ///
    ///Each `PhysicalDeviceToolProperties` includes the tool's name,
    ///version, purposes (validation, profiling, tracing, etc.), and a
    ///description.
    ///
    ///Useful for diagnostics, log the active tools at startup to help
    ///debug performance issues or unexpected validation messages. If no
    ///tools are active, the list is empty.
    pub unsafe fn get_physical_device_tool_properties(
        &self,
        physical_device: PhysicalDevice,
    ) -> VkResult<Vec<PhysicalDeviceToolProperties>> {
        let fp = self
            .commands()
            .get_physical_device_tool_properties
            .expect("vkGetPhysicalDeviceToolProperties not loaded");
        enumerate_two_call(|count, data| unsafe { fp(physical_device, count, data) })
    }
    ///Wraps [`vkGetPhysicalDeviceRefreshableObjectTypesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceRefreshableObjectTypesKHR.html).
    ///
    ///# Errors
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Enumerates the Vulkan object types that can be refreshed on the
    ///physical device. Part of Vulkan SC (Safety Critical) and the
    ///object refresh mechanism for long-running safety applications.
    ///Uses the two-call idiom.
    ///
    ///Requires `VK_KHR_object_refresh`.
    pub unsafe fn get_physical_device_refreshable_object_types_khr(
        &self,
        physical_device: PhysicalDevice,
    ) -> VkResult<Vec<ObjectType>> {
        let fp = self
            .commands()
            .get_physical_device_refreshable_object_types_khr
            .expect("vkGetPhysicalDeviceRefreshableObjectTypesKHR not loaded");
        enumerate_two_call(|count, data| unsafe { fp(physical_device, count, data) })
    }
    ///Wraps [`vkGetPhysicalDeviceFragmentShadingRatesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceFragmentShadingRatesKHR.html).
    /**
    Provided by **VK_KHR_fragment_shading_rate**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Enumerates the fragment shading rates supported by the physical
    ///device. Each entry reports a fragment size (e.g., 2x2, 4x4) and
    ///which sample counts are compatible with it.
    ///
    ///The results are sorted from largest to smallest fragment size.
    ///1x1 (full-rate shading) is always supported.
    ///
    ///Use these results to validate fragment sizes passed to
    ///`cmd_set_fragment_shading_rate_khr` or configured in a shading
    ///rate attachment.
    pub unsafe fn get_physical_device_fragment_shading_rates_khr(
        &self,
        physical_device: PhysicalDevice,
    ) -> VkResult<Vec<PhysicalDeviceFragmentShadingRateKHR>> {
        let fp = self
            .commands()
            .get_physical_device_fragment_shading_rates_khr
            .expect("vkGetPhysicalDeviceFragmentShadingRatesKHR not loaded");
        enumerate_two_call(|count, data| unsafe { fp(physical_device, count, data) })
    }
    ///Wraps [`vkGetPhysicalDeviceVideoCapabilitiesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceVideoCapabilitiesKHR.html).
    /**
    Provided by **VK_KHR_video_queue**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR`
    ///- `VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR`
    ///- `VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR`
    ///- `VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries video codec capabilities for a given video profile on a
    ///physical device. Returns `VideoCapabilitiesKHR` describing:
    ///
    ///- Supported coded extent range (min/max resolution).
    ///- Maximum DPB slot and active reference picture counts.
    ///- Bitstream buffer offset and size alignment requirements.
    ///- Supported standard header version.
    ///
    ///Chain codec-specific capability structs (e.g.,
    ///`VideoDecodeH264CapabilitiesKHR`) into the `pNext` of
    ///`p_capabilities` to receive additional codec details.
    ///
    ///This is the first query in the video workflow, use it to
    ///determine whether a codec profile is supported and what limits
    ///apply before creating a video session.
    pub unsafe fn get_physical_device_video_capabilities_khr(
        &self,
        physical_device: PhysicalDevice,
        p_video_profile: &VideoProfileInfoKHR,
        p_capabilities: *mut VideoCapabilitiesKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_physical_device_video_capabilities_khr
            .expect("vkGetPhysicalDeviceVideoCapabilitiesKHR not loaded");
        check(unsafe { fp(physical_device, p_video_profile, p_capabilities) })
    }
    ///Wraps [`vkGetPhysicalDeviceVideoFormatPropertiesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceVideoFormatPropertiesKHR.html).
    /**
    Provided by **VK_KHR_video_queue**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR`
    ///- `VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR`
    ///- `VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR`
    ///- `VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR`
    ///- `VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the image formats compatible with a video profile for
    ///decoded output, DPB reference, or encode input images.
    ///
    ///Specify the intended usage in `PhysicalDeviceVideoFormatInfoKHR`
    ///(image usage flags indicating decode output, DPB, or encode
    ///input). The returned `VideoFormatPropertiesKHR` list the
    ///compatible formats, image types, tiling modes, and usage flags.
    ///
    ///Use these results to create images that are compatible with the
    ///video session. Using an unsupported format results in validation
    ///errors.
    pub unsafe fn get_physical_device_video_format_properties_khr(
        &self,
        physical_device: PhysicalDevice,
        p_video_format_info: &PhysicalDeviceVideoFormatInfoKHR,
    ) -> VkResult<Vec<VideoFormatPropertiesKHR>> {
        let fp = self
            .commands()
            .get_physical_device_video_format_properties_khr
            .expect("vkGetPhysicalDeviceVideoFormatPropertiesKHR not loaded");
        enumerate_two_call(|count, data| unsafe {
            fp(physical_device, p_video_format_info, count, data)
        })
    }
    ///Wraps [`vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR.html).
    /**
    Provided by **VK_KHR_video_encode_queue**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR`
    ///- `VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR`
    ///- `VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR`
    ///- `VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the properties of a specific video encode quality level.
    ///Quality levels range from 0 (lowest quality, fastest) to
    ///`maxQualityLevels - 1` (highest quality, slowest), as reported
    ///by `VideoEncodeCapabilitiesKHR`.
    ///
    ///The output `VideoEncodeQualityLevelPropertiesKHR` provides
    ///recommended encode settings for the requested quality level.
    ///Chain codec-specific quality level info (e.g.,
    ///`VideoEncodeH264QualityLevelPropertiesKHR`) into `pNext` to get
    ///codec-specific recommended parameters.
    ///
    ///Use these recommended settings as a starting point for
    ///`VideoEncodeInfoKHR` and rate control configuration.
    pub unsafe fn get_physical_device_video_encode_quality_level_properties_khr(
        &self,
        physical_device: PhysicalDevice,
        p_quality_level_info: &PhysicalDeviceVideoEncodeQualityLevelInfoKHR,
        p_quality_level_properties: *mut VideoEncodeQualityLevelPropertiesKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .get_physical_device_video_encode_quality_level_properties_khr
            .expect("vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR not loaded");
        check(unsafe {
            fp(
                physical_device,
                p_quality_level_info,
                p_quality_level_properties,
            )
        })
    }
    ///Wraps [`vkAcquireDrmDisplayEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkAcquireDrmDisplayEXT.html).
    /**
    Provided by **VK_EXT_acquire_drm_display**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Acquires exclusive control of a DRM display for direct rendering.
    ///Takes a DRM file descriptor and a display handle. Release with
    ///`release_display_ext` when finished.
    ///
    ///Requires `VK_EXT_acquire_drm_display`. Linux only.
    pub unsafe fn acquire_drm_display_ext(
        &self,
        physical_device: PhysicalDevice,
        drm_fd: i32,
        display: DisplayKHR,
    ) -> VkResult<()> {
        let fp = self
            .commands()
            .acquire_drm_display_ext
            .expect("vkAcquireDrmDisplayEXT not loaded");
        check(unsafe { fp(physical_device, drm_fd, display) })
    }
    ///Wraps [`vkGetDrmDisplayEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDrmDisplayEXT.html).
    /**
    Provided by **VK_EXT_acquire_drm_display**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Maps a DRM connector to a Vulkan `DisplayKHR` handle. Takes a
    ///DRM file descriptor and connector ID, and returns the
    ///corresponding display. Use this to identify which Vulkan display
    ///corresponds to a specific DRM output.
    ///
    ///Requires `VK_EXT_acquire_drm_display`. Linux only.
    pub unsafe fn get_drm_display_ext(
        &self,
        physical_device: PhysicalDevice,
        drm_fd: i32,
        connector_id: u32,
    ) -> VkResult<DisplayKHR> {
        let fp = self
            .commands()
            .get_drm_display_ext
            .expect("vkGetDrmDisplayEXT not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        check(unsafe { fp(physical_device, drm_fd, connector_id, &mut out) })?;
        Ok(out)
    }
    ///Wraps [`vkGetPhysicalDeviceOpticalFlowImageFormatsNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceOpticalFlowImageFormatsNV.html).
    /**
    Provided by **VK_NV_optical_flow**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_EXTENSION_NOT_PRESENT`
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_FORMAT_NOT_SUPPORTED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries which image formats are supported for optical flow at a
    ///given resolution and usage. Use this to select compatible formats
    ///before creating images for the optical flow session.
    ///
    ///Requires `VK_NV_optical_flow`.
    pub unsafe fn get_physical_device_optical_flow_image_formats_nv(
        &self,
        physical_device: PhysicalDevice,
        p_optical_flow_image_format_info: &OpticalFlowImageFormatInfoNV,
    ) -> VkResult<Vec<OpticalFlowImageFormatPropertiesNV>> {
        let fp = self
            .commands()
            .get_physical_device_optical_flow_image_formats_nv
            .expect("vkGetPhysicalDeviceOpticalFlowImageFormatsNV not loaded");
        enumerate_two_call(|count, data| unsafe {
            fp(
                physical_device,
                p_optical_flow_image_format_info,
                count,
                data,
            )
        })
    }
    ///Wraps [`vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR.html).
    /**
    Provided by **VK_KHR_cooperative_matrix**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Enumerates the cooperative matrix types and configurations
    ///supported by the physical device. Each returned
    ///`CooperativeMatrixPropertiesKHR` describes a supported combination
    ///of matrix dimensions (M, N, K), component types (A, B, C, Result),
    ///and scope (subgroup or workgroup).
    ///
    ///Use these results to select valid cooperative matrix parameters
    ///for SPIR-V `OpCooperativeMatrixMulAddKHR` operations.
    ///
    ///Requires `VK_KHR_cooperative_matrix`.
    pub unsafe fn get_physical_device_cooperative_matrix_properties_khr(
        &self,
        physical_device: PhysicalDevice,
    ) -> VkResult<Vec<CooperativeMatrixPropertiesKHR>> {
        let fp = self
            .commands()
            .get_physical_device_cooperative_matrix_properties_khr
            .expect("vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR not loaded");
        enumerate_two_call(|count, data| unsafe { fp(physical_device, count, data) })
    }
    ///Wraps [`vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV.html).
    /**
    Provided by **VK_NV_cooperative_matrix2**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Enumerates the supported flexible-dimension cooperative matrix
    ///configurations on the physical device. Uses the two-call idiom.
    ///Flexible dimensions allow non-power-of-two matrix sizes for
    ///better utilisation of hardware matrix units.
    ///
    ///Requires `VK_NV_cooperative_matrix2`.
    pub unsafe fn get_physical_device_cooperative_matrix_flexible_dimensions_properties_nv(
        &self,
        physical_device: PhysicalDevice,
    ) -> VkResult<Vec<CooperativeMatrixFlexibleDimensionsPropertiesNV>> {
        let fp = self
            .commands()
            .get_physical_device_cooperative_matrix_flexible_dimensions_properties_nv
            .expect(
                "vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV not loaded",
            );
        enumerate_two_call(|count, data| unsafe { fp(physical_device, count, data) })
    }
    ///Wraps [`vkGetPhysicalDeviceCooperativeVectorPropertiesNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceCooperativeVectorPropertiesNV.html).
    /**
    Provided by **VK_NV_cooperative_vector**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the supported cooperative vector properties (data types,
    ///matrix dimensions, operations) for a physical device. Use this
    ///to determine what cooperative vector configurations are available
    ///before creating pipelines that use them.
    ///
    ///Requires `VK_NV_cooperative_vector`.
    pub unsafe fn get_physical_device_cooperative_vector_properties_nv(
        &self,
        physical_device: PhysicalDevice,
    ) -> VkResult<Vec<CooperativeVectorPropertiesNV>> {
        let fp = self
            .commands()
            .get_physical_device_cooperative_vector_properties_nv
            .expect("vkGetPhysicalDeviceCooperativeVectorPropertiesNV not loaded");
        enumerate_two_call(|count, data| unsafe { fp(physical_device, count, data) })
    }
    ///Wraps [`vkEnumeratePhysicalDeviceShaderInstrumentationMetricsARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkEnumeratePhysicalDeviceShaderInstrumentationMetricsARM.html).
    /**
    Provided by **VK_ARM_shader_instrumentation**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Enumerates the shader instrumentation metrics supported by a
    ///physical device. Uses the two-call idiom. Returns metric
    ///descriptions that can be selected when creating a shader
    ///instrumentation object.
    ///
    ///Requires `VK_ARM_shader_instrumentation`.
    pub unsafe fn enumerate_physical_device_shader_instrumentation_metrics_arm(
        &self,
        physical_device: PhysicalDevice,
    ) -> VkResult<Vec<ShaderInstrumentationMetricDescriptionARM>> {
        let fp = self
            .commands()
            .enumerate_physical_device_shader_instrumentation_metrics_arm
            .expect("vkEnumeratePhysicalDeviceShaderInstrumentationMetricsARM not loaded");
        enumerate_two_call(|count, data| unsafe { fp(physical_device, count, data) })
    }
    ///Wraps [`vkGetPhysicalDeviceExternalTensorPropertiesARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceExternalTensorPropertiesARM.html).
    /**
    Provided by **VK_ARM_tensors**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries whether a tensor with the given parameters can be
    ///exported to or imported from an external handle type. Returns
    ///compatibility and feature flags for external tensor sharing.
    ///
    ///Requires `VK_ARM_tensors`.
    pub unsafe fn get_physical_device_external_tensor_properties_arm(
        &self,
        physical_device: PhysicalDevice,
        p_external_tensor_info: &PhysicalDeviceExternalTensorInfoARM,
        p_external_tensor_properties: *mut ExternalTensorPropertiesARM,
    ) {
        let fp = self
            .commands()
            .get_physical_device_external_tensor_properties_arm
            .expect("vkGetPhysicalDeviceExternalTensorPropertiesARM not loaded");
        unsafe {
            fp(
                physical_device,
                p_external_tensor_info,
                p_external_tensor_properties,
            )
        };
    }
    ///Wraps [`vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM.html).
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
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Enumerates the data graph pipeline properties supported by a
    ///specific queue family. Uses the two-call idiom. Use to determine
    ///which queue families can execute data graph pipelines.
    ///
    ///Requires `VK_ARM_data_graph`.
    pub unsafe fn get_physical_device_queue_family_data_graph_properties_arm(
        &self,
        physical_device: PhysicalDevice,
        queue_family_index: u32,
    ) -> VkResult<Vec<QueueFamilyDataGraphPropertiesARM>> {
        let fp = self
            .commands()
            .get_physical_device_queue_family_data_graph_properties_arm
            .expect("vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM not loaded");
        enumerate_two_call(|count, data| unsafe {
            fp(physical_device, queue_family_index, count, data)
        })
    }
    ///Wraps [`vkGetPhysicalDeviceQueueFamilyDataGraphProcessingEnginePropertiesARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceQueueFamilyDataGraphProcessingEnginePropertiesARM.html).
    /**
    Provided by **VK_ARM_data_graph**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Queries the data graph processing engine properties for a
    ///specific queue family. Returns hardware-specific engine
    ///capabilities such as supported data types and operations.
    ///
    ///Requires `VK_ARM_data_graph`.
    pub unsafe fn get_physical_device_queue_family_data_graph_processing_engine_properties_arm(
        &self,
        physical_device: PhysicalDevice,
        p_queue_family_data_graph_processing_engine_info: &PhysicalDeviceQueueFamilyDataGraphProcessingEngineInfoARM,
        p_queue_family_data_graph_processing_engine_properties: *mut QueueFamilyDataGraphProcessingEnginePropertiesARM,
    ) {
        let fp = self
            .commands()
            .get_physical_device_queue_family_data_graph_processing_engine_properties_arm
            .expect(
                "vkGetPhysicalDeviceQueueFamilyDataGraphProcessingEnginePropertiesARM not loaded",
            );
        unsafe {
            fp(
                physical_device,
                p_queue_family_data_graph_processing_engine_info,
                p_queue_family_data_graph_processing_engine_properties,
            )
        };
    }
    ///Wraps [`vkEnumeratePhysicalDeviceQueueFamilyPerformanceCountersByRegionARM`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkEnumeratePhysicalDeviceQueueFamilyPerformanceCountersByRegionARM.html).
    /**
    Provided by **VK_ARM_performance_counters_by_region**.*/
    ///
    ///# Errors
    ///- `VK_ERROR_OUT_OF_HOST_MEMORY`
    ///- `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    ///- `VK_ERROR_INITIALIZATION_FAILED`
    ///- `VK_ERROR_UNKNOWN`
    ///- `VK_ERROR_VALIDATION_FAILED`
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Enumerates performance counters for a specific queue family on
    ///ARM GPUs, grouped by hardware region. Uses the two-call idiom
    ///for the counter array and also writes corresponding descriptions.
    ///
    ///Requires `VK_ARM_shader_instrumentation`.
    pub unsafe fn enumerate_physical_device_queue_family_performance_counters_by_region_arm(
        &self,
        physical_device: PhysicalDevice,
        queue_family_index: u32,
        p_counter_descriptions: *mut PerformanceCounterDescriptionARM,
    ) -> VkResult<Vec<PerformanceCounterARM>> {
        let fp = self
            .commands()
            .enumerate_physical_device_queue_family_performance_counters_by_region_arm
            .expect(
                "vkEnumeratePhysicalDeviceQueueFamilyPerformanceCountersByRegionARM not loaded",
            );
        enumerate_two_call(|count, data| unsafe {
            fp(
                physical_device,
                queue_family_index,
                count,
                data,
                p_counter_descriptions,
            )
        })
    }
    ///Wraps [`vkGetPhysicalDeviceDescriptorSizeEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceDescriptorSizeEXT.html).
    /**
    Provided by **VK_EXT_descriptor_heap**.*/
    ///
    ///# Safety
    ///- `physicalDevice` (self) must be valid and not destroyed.
    ///
    ///# Usage Notes
    ///
    ///Returns the byte size of a single descriptor of the specified
    ///type on this physical device.
    ///
    ///Use this to compute buffer sizes and offsets when working with
    ///the descriptor heap model.
    ///
    ///Provided by `VK_EXT_descriptor_heap`.
    pub unsafe fn get_physical_device_descriptor_size_ext(
        &self,
        physical_device: PhysicalDevice,
        descriptor_type: DescriptorType,
    ) {
        let fp = self
            .commands()
            .get_physical_device_descriptor_size_ext
            .expect("vkGetPhysicalDeviceDescriptorSizeEXT not loaded");
        unsafe { fp(physical_device, descriptor_type) };
    }
}
