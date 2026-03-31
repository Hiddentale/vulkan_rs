#![allow(unused_imports)]
#![allow(clippy::too_many_arguments)]
use crate::error::{VkResult, check, enumerate_two_call, fill_two_call};
use crate::vk::bitmasks::*;
use crate::vk::constants::*;
use crate::vk::enums::*;
use crate::vk::handles::*;
use crate::vk::structs::*;
impl crate::Entry {
    ///Wraps [`vkGetExternalComputeQueueDataNV`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetExternalComputeQueueDataNV.html).
    /**
    Provided by **VK_NV_external_compute_queue**.*/
    ///
    ///# Safety
    ///- `externalQueue` (self) must be valid and not destroyed.
    pub unsafe fn get_external_compute_queue_data_nv(
        &self,
        external_queue: ExternalComputeQueueNV,
        params: *mut ExternalComputeQueueDataParamsNV,
    ) -> core::ffi::c_void {
        let fp = self
            .commands()
            .get_external_compute_queue_data_nv
            .expect("vkGetExternalComputeQueueDataNV not loaded");
        let mut out = unsafe { core::mem::zeroed() };
        unsafe { fp(external_queue, params, &mut out) };
        out
    }
}
