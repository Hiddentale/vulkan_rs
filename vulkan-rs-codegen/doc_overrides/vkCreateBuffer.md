# Examples

```no_run
# let (_, instance) = vulkan_rs::test_helpers::create_test_instance().expect("test setup");
# let phys = unsafe { instance.enumerate_physical_devices() }.expect("no devices");
# let p = [1.0f32];
# let qi = vulkan_rs::vk::structs::DeviceQueueCreateInfo::builder().queue_priorities(&p);
# let qis = [*qi];
# let di = vulkan_rs::vk::structs::DeviceCreateInfo::builder().queue_create_infos(&qis);
# let device = unsafe { instance.create_device(phys[0], &di, None) }.expect("device creation");
use vulkan_rs::vk::structs::*;
use vulkan_rs::vk::bitmasks::*;

let info = BufferCreateInfo::builder()
    .size(1024)
    .usage(BufferUsageFlagBits::VERTEX_BUFFER)
    .sharing_mode(vulkan_rs::vk::enums::SharingMode::EXCLUSIVE);
let buffer = unsafe { device.create_buffer(&info, None) }
    .expect("buffer creation failed");
// Use buffer...
unsafe { device.destroy_buffer(buffer, None) };
# unsafe { device.destroy_device(None) };
# unsafe { instance.destroy_instance(None) };
```

# Guide

See [Memory Management](https://hiddentale.github.io/vulkan_rs/concepts/memory.html) in the vulkan_rs guide.
