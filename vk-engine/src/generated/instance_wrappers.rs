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
