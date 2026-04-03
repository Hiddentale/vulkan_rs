//! Hidden helpers for doc examples and integration tests.
//!
//! These reduce boilerplate in `no_run` doctests. Not part of the public API.

use crate::device::Device;
use crate::entry::Entry;
use crate::instance::Instance;
use crate::loader::LibloadingLoader;
use crate::vk;

/// Load the Vulkan library and create an [`Entry`].
///
/// # Errors
///
/// Returns an error if the Vulkan shared library cannot be found.
pub fn create_test_entry() -> Result<Entry, Box<dyn std::error::Error>> {
    let loader =
        LibloadingLoader::new().map_err(|e| -> Box<dyn std::error::Error> { Box::new(e) })?;
    let entry =
        unsafe { Entry::new(loader) }.map_err(|e| -> Box<dyn std::error::Error> { Box::new(e) })?;
    Ok(entry)
}

/// Create an [`Entry`] and a minimal [`Instance`] with no extensions.
///
/// # Errors
///
/// Returns an error if the Vulkan runtime is not available.
pub fn create_test_instance() -> Result<(Entry, Instance), Box<dyn std::error::Error>> {
    let entry = create_test_entry()?;
    let create_info = vk::InstanceCreateInfo::builder();
    let instance = unsafe { entry.create_instance(&create_info, None) }
        .map_err(|e| -> Box<dyn std::error::Error> { Box::new(crate::VkError(e)) })?;
    Ok((entry, instance))
}

/// Create an [`Entry`], [`Instance`], and [`Device`] with one queue.
///
/// Picks the first physical device and first queue family.
///
/// # Errors
///
/// Returns an error if the Vulkan runtime is not available or has no devices.
pub fn create_test_device() -> Result<(Entry, Instance, Device), Box<dyn std::error::Error>> {
    let (entry, instance) = create_test_instance()?;

    let physical_devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|e| -> Box<dyn std::error::Error> { Box::new(crate::VkError(e)) })?;
    let physical_device = *physical_devices
        .first()
        .ok_or("no Vulkan physical devices found")?;

    let priorities = [1.0f32];
    let queue_info = vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(0)
        .queue_priorities(&priorities);
    let queue_infos = [*queue_info];
    let device_info = vk::DeviceCreateInfo::builder().queue_create_infos(&queue_infos);

    let device = unsafe { instance.create_device(physical_device, &device_info, None) }
        .map_err(|e| -> Box<dyn std::error::Error> { Box::new(crate::VkError(e)) })?;

    Ok((entry, instance, device))
}
