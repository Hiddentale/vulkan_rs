// Exact copy of the "Putting it all together" section from
// <https://hiddentale.github.io/vulkan_rs/getting-started/hello-triangle-1.html>

use vk::structs::*;
use vulkan_rs::vk;
use vulkan_rs::{Entry, LibloadingLoader, Version};

fn main() {
    // ── Step 1: Load Vulkan ────────────────────────────────────
    let loader = LibloadingLoader::new().expect("Vulkan library not found");
    let entry = unsafe { Entry::new(loader) }.expect("Failed to load Vulkan");

    let version = entry.version().expect("Failed to query version");
    println!(
        "Vulkan {}.{}.{}",
        version.major, version.minor, version.patch
    );

    // ── Step 2: Create Instance ────────────────────────────────
    let app_info = ApplicationInfo::builder()
        .p_application_name(c"Hello Triangle")
        .application_version(1)
        .p_engine_name(c"No Engine")
        .engine_version(1)
        .api_version(Version::new(1, 0, 0).to_raw());

    let create_info = InstanceCreateInfo::builder().p_application_info(&*app_info);

    let instance =
        unsafe { entry.create_instance(&create_info, None) }.expect("Failed to create instance");

    // ── Step 3: Pick a GPU ─────────────────────────────────────
    let physical_devices =
        unsafe { instance.enumerate_physical_devices() }.expect("Failed to enumerate GPUs");

    let physical_device = physical_devices[0];

    let props = unsafe { instance.get_physical_device_properties(physical_device) };
    let name_bytes: Vec<u8> = props
        .device_name
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    println!("GPU: {}", String::from_utf8_lossy(&name_bytes));

    // ── Step 4: Find a graphics queue family ───────────────────
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

    let graphics_family_index = queue_families
        .iter()
        .enumerate()
        .find(|(_, family)| family.queue_flags & QueueFlags::GRAPHICS != QueueFlags::empty())
        .map(|(index, _)| index as u32)
        .expect("No graphics queue family found");

    // ── Step 5: Create Device ──────────────────────────────────
    let queue_priority = 1.0_f32;
    let queue_info = DeviceQueueCreateInfo::builder()
        .queue_family_index(graphics_family_index)
        .queue_priorities(std::slice::from_ref(&queue_priority));

    let device_info =
        DeviceCreateInfo::builder().queue_create_infos(std::slice::from_ref(&*queue_info));

    let device = unsafe { instance.create_device(physical_device, &device_info, None) }
        .expect("Failed to create device");

    // ── Step 6: Get the graphics queue ─────────────────────────
    let _graphics_queue = unsafe { device.get_device_queue(graphics_family_index, 0) };

    println!("Vulkan initialized successfully!");
    println!("Ready for Part 2: Swapchain & Surface");

    // ── Step 7: Clean up ───────────────────────────────────────
    unsafe {
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
}
