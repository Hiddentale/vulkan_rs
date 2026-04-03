# Examples

```no_run
# let (_, instance) = vulkan_rust::test_helpers::create_test_instance().expect("test setup");
# let phys = unsafe { instance.enumerate_physical_devices() }.expect("no devices");
# let p = [1.0f32];
# let qi = vulkan_rust::vk::DeviceQueueCreateInfo::builder().queue_priorities(&p);
# let qis = [*qi];
# let di = vulkan_rust::vk::DeviceCreateInfo::builder().queue_create_infos(&qis);
# let device = unsafe { instance.create_device(phys[0], &di, None) }.expect("device creation");
use vulkan_rust::vk::*;

let info = BufferCreateInfo::builder()
    .size(1024)
    .usage(BufferUsageFlagBits::VERTEX_BUFFER)
    .sharing_mode(vulkan_rust::vk::SharingMode::EXCLUSIVE);
let buffer = unsafe { device.create_buffer(&info, None) }
    .expect("buffer creation failed");
// Use buffer...
unsafe { device.destroy_buffer(buffer, None) };
# unsafe { device.destroy_device(None) };
# unsafe { instance.destroy_instance(None) };
```

# Guide

See [Memory Management](https://hiddentale.github.io/vulkan_rust/concepts/memory.html) in the vulkan_rust guide.
