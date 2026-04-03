use vk::handles::Handle;
use vulkan_rust::{Entry, LibloadingLoader, vk};

fn create_entry() -> Entry {
    let loader = LibloadingLoader::new().expect("failed to load Vulkan library");
    unsafe { Entry::new(loader) }.expect("failed to create Entry")
}

fn minimal_instance_create_info(
    app_info: &vk::structs::ApplicationInfo,
) -> vk::structs::InstanceCreateInfo {
    vk::structs::InstanceCreateInfo {
        s_type: vk::enums::StructureType::INSTANCE_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: vk::bitmasks::InstanceCreateFlagBits::empty(),
        p_application_info: app_info,
        enabled_layer_count: 0,
        pp_enabled_layer_names: std::ptr::null(),
        enabled_extension_count: 0,
        pp_enabled_extension_names: std::ptr::null(),
    }
}

fn minimal_app_info() -> vk::structs::ApplicationInfo {
    vk::structs::ApplicationInfo {
        s_type: vk::enums::StructureType::APPLICATION_INFO,
        p_next: std::ptr::null(),
        p_application_name: std::ptr::null(),
        application_version: 0,
        p_engine_name: std::ptr::null(),
        engine_version: 0,
        api_version: vulkan_rust::Version::new(1, 0, 0).to_raw(),
    }
}

#[test]
#[ignore] // requires Vulkan runtime
fn enumerate_physical_devices_returns_at_least_one() {
    let entry = create_entry();
    let app_info = minimal_app_info();
    let create_info = minimal_instance_create_info(&app_info);
    let instance =
        unsafe { entry.create_instance(&create_info, None) }.expect("failed to create instance");

    let devices = unsafe { instance.enumerate_physical_devices() }
        .expect("enumerate_physical_devices failed");
    assert!(!devices.is_empty(), "expected at least one physical device");

    unsafe { instance.destroy_instance(None) };
}

#[test]
#[ignore] // requires Vulkan runtime
fn get_physical_device_properties_reports_device_name() {
    let entry = create_entry();
    let app_info = minimal_app_info();
    let create_info = minimal_instance_create_info(&app_info);
    let instance =
        unsafe { entry.create_instance(&create_info, None) }.expect("failed to create instance");

    let devices = unsafe { instance.enumerate_physical_devices() }
        .expect("enumerate_physical_devices failed");
    let props = unsafe { instance.get_physical_device_properties(devices[0]) };

    let name_bytes: Vec<u8> = props
        .device_name
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    let name = String::from_utf8_lossy(&name_bytes);
    println!("GPU: {name}");
    assert!(!name.is_empty());

    unsafe { instance.destroy_instance(None) };
}

#[test]
#[ignore] // requires Vulkan runtime
fn get_physical_device_queue_family_properties_returns_at_least_one() {
    let entry = create_entry();
    let app_info = minimal_app_info();
    let create_info = minimal_instance_create_info(&app_info);
    let instance =
        unsafe { entry.create_instance(&create_info, None) }.expect("failed to create instance");

    let devices = unsafe { instance.enumerate_physical_devices() }
        .expect("enumerate_physical_devices failed");
    let families = unsafe { instance.get_physical_device_queue_family_properties(devices[0]) };
    assert!(!families.is_empty(), "expected at least one queue family");

    unsafe { instance.destroy_instance(None) };
}

#[test]
#[ignore] // requires Vulkan runtime
fn full_lifecycle() {
    let entry = create_entry();
    let app_info = minimal_app_info();
    let create_info = minimal_instance_create_info(&app_info);
    let instance =
        unsafe { entry.create_instance(&create_info, None) }.expect("failed to create instance");

    let physical_devices = unsafe { instance.enumerate_physical_devices() }
        .expect("enumerate_physical_devices failed");
    assert!(!physical_devices.is_empty());
    let physical_device = physical_devices[0];

    let props = unsafe { instance.get_physical_device_properties(physical_device) };
    let name_bytes: Vec<u8> = props
        .device_name
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    println!("GPU: {}", String::from_utf8_lossy(&name_bytes));

    let families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let graphics_family = families
        .iter()
        .position(|f| {
            f.queue_flags
                .contains(vk::bitmasks::QueueFlagBits::GRAPHICS)
        })
        .expect("no graphics queue family") as u32;

    let queue_priority = 1.0f32;
    let queue_create_info = vk::structs::DeviceQueueCreateInfo {
        s_type: vk::enums::StructureType::DEVICE_QUEUE_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: vk::bitmasks::DeviceQueueCreateFlagBits::empty(),
        queue_family_index: graphics_family,
        queue_count: 1,
        p_queue_priorities: &queue_priority,
    };
    let device_create_info = vk::structs::DeviceCreateInfo {
        s_type: vk::enums::StructureType::DEVICE_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: 0,
        queue_create_info_count: 1,
        p_queue_create_infos: &queue_create_info,
        enabled_layer_count: 0,
        pp_enabled_layer_names: std::ptr::null(),
        enabled_extension_count: 0,
        pp_enabled_extension_names: std::ptr::null(),
        p_enabled_features: std::ptr::null(),
    };
    let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }
        .expect("failed to create device");

    let queue = unsafe { device.get_device_queue(graphics_family, 0) };
    assert!(!queue.is_null(), "expected non-null queue handle");

    unsafe { device.device_wait_idle() }.expect("device_wait_idle failed");

    unsafe { device.destroy_device(None) };
    unsafe { instance.destroy_instance(None) };
}
