# Examples

```no_run
# let (_, instance) = vk_engine::test_helpers::create_test_instance().unwrap();
# let phys = unsafe { instance.enumerate_physical_devices() }.unwrap();
# let p = [1.0f32];
# let qi = vk_engine::vk::structs::DeviceQueueCreateInfo::builder().queue_priorities(&p);
# let qis = [*qi];
# let di = vk_engine::vk::structs::DeviceCreateInfo::builder().queue_create_infos(&qis);
# let device = unsafe { instance.create_device(phys[0], &di, None) }.unwrap();
use vk_engine::vk::structs::*;
use vk_engine::vk::bitmasks::*;

let info = BufferCreateInfo::builder()
    .size(1024)
    .usage(BufferUsageFlagBits::VERTEX_BUFFER)
    .sharing_mode(vk_engine::vk::enums::SharingMode::EXCLUSIVE);
let buffer = unsafe { device.create_buffer(&info, None) }
    .expect("buffer creation failed");
// Use buffer...
unsafe { device.destroy_buffer(buffer, None) };
# unsafe { device.destroy_device(None) };
# unsafe { instance.destroy_instance(None) };
```
