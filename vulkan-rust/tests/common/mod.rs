use vulkan_rust::{Entry, Instance, LibloadingLoader, vk};

pub fn create_entry() -> Entry {
    let loader = LibloadingLoader::new().expect("failed to load Vulkan library");
    unsafe { Entry::new(loader) }.expect("failed to create Entry")
}

pub fn create_instance() -> (Entry, Instance) {
    let entry = create_entry();
    let app_info = vk::ApplicationInfo {
        s_type: vk::StructureType::APPLICATION_INFO,
        p_next: std::ptr::null(),
        p_application_name: std::ptr::null(),
        application_version: 0,
        p_engine_name: std::ptr::null(),
        engine_version: 0,
        api_version: vulkan_rust::Version::new(1, 0, 0).to_raw(),
    };
    let create_info = vk::InstanceCreateInfo {
        s_type: vk::StructureType::INSTANCE_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: vk::InstanceCreateFlagBits::empty(),
        p_application_info: &app_info,
        enabled_layer_count: 0,
        pp_enabled_layer_names: std::ptr::null(),
        enabled_extension_count: 0,
        pp_enabled_extension_names: std::ptr::null(),
    };
    let instance =
        unsafe { entry.create_instance(&create_info, None) }.expect("failed to create instance");
    (entry, instance)
}
