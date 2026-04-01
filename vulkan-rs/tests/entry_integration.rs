use vulkan_rs::{Entry, LibloadingLoader, Version};

#[test]
#[ignore] // requires Vulkan runtime
fn entry_loads_and_queries_version() {
    let loader = LibloadingLoader::new().expect("failed to load Vulkan library");
    let entry = unsafe { Entry::new(loader) }.expect("failed to create Entry");
    let version = entry.version().expect("failed to query version");
    assert!(version.major >= 1);
    println!("Vulkan {version}");
}

#[test]
#[ignore] // requires Vulkan runtime
fn entry_enumerates_layers_and_extensions() {
    let loader = LibloadingLoader::new().expect("failed to load Vulkan library");
    let entry = unsafe { Entry::new(loader) }.expect("failed to create Entry");

    let layers =
        unsafe { entry.enumerate_instance_layer_properties() }.expect("failed to enumerate layers");
    println!("found {} layers", layers.len());

    let extensions = unsafe { entry.enumerate_instance_extension_properties(None) }
        .expect("failed to enumerate extensions");
    assert!(!extensions.is_empty());
    println!("found {} extensions", extensions.len());
}

#[test]
#[ignore] // requires Vulkan runtime
fn entry_version_is_at_least_1_0() {
    let loader = LibloadingLoader::new().expect("failed to load Vulkan library");
    let entry = unsafe { Entry::new(loader) }.expect("failed to create Entry");
    let version = entry.version().expect("failed to query version");
    assert!(
        version
            >= Version {
                major: 1,
                minor: 0,
                patch: 0
            }
    );
}
