#![allow(dead_code)]

mod parse;
mod type_map;

use parse::DispatchLevel;
use std::path::Path;

fn main() {
    let vk_xml = Path::new(env!("CARGO_MANIFEST_DIR")).join("vk.xml");
    let registry = parse::parse_registry(&vk_xml);

    let unions = registry.structs.iter().filter(|s| s.is_union).count();
    let structs = registry.structs.len() - unions;
    let bitmasks_64 = registry
        .bitmasks
        .iter()
        .filter(|b| b.bitwidth == 64)
        .count();
    let entry_cmds = registry
        .commands
        .iter()
        .filter(|c| c.dispatch_level == DispatchLevel::Entry)
        .count();
    let instance_cmds = registry
        .commands
        .iter()
        .filter(|c| c.dispatch_level == DispatchLevel::Instance)
        .count();
    let device_cmds = registry
        .commands
        .iter()
        .filter(|c| c.dispatch_level == DispatchLevel::Device)
        .count();

    println!("=== vk.xml parse summary ===");
    println!("  handles:       {}", registry.handles.len());
    println!("  structs:       {structs}");
    println!("  unions:        {unions}");
    println!("  enums:         {}", registry.enums.len());
    println!(
        "  bitmasks:      {} ({bitmasks_64} are 64-bit)",
        registry.bitmasks.len()
    );
    println!(
        "  commands:      {} (entry={entry_cmds}, instance={instance_cmds}, device={device_cmds})",
        registry.commands.len()
    );
    println!("  constants:     {}", registry.constants.len());
    println!("  funcpointers:  {}", registry.func_pointers.len());
    println!("  extensions:    {}", registry.extensions.len());
    println!("  platforms:     {}", registry.platforms.len());
    println!("  aliases:       {}", registry.aliases.len());

    println!();
    println!("=== spot checks ===");

    // Handles
    let instance = registry.handles.iter().find(|h| h.name == "Instance");
    println!(
        "  Instance handle:      {} (dispatchable={}, obj_type={:?}, provided_by={:?})",
        instance.is_some(),
        instance.is_some_and(|h| h.dispatchable),
        instance.and_then(|h| h.object_type.as_deref()),
        instance.and_then(|h| h.provided_by.as_deref()),
    );

    // Structs vs unions
    let has_buffer_create = registry
        .structs
        .iter()
        .any(|s| s.name == "BufferCreateInfo" && !s.is_union);
    let has_clear_color = registry
        .structs
        .iter()
        .any(|s| s.name == "ClearColorValue" && s.is_union);
    println!("  BufferCreateInfo:     {has_buffer_create}");
    println!("  ClearColorValue:      {has_clear_color} (union)");

    // Enums & bitmasks
    let has_format = registry.enums.iter().any(|e| e.name == "Format");
    let pipeline_stage_2 = registry
        .bitmasks
        .iter()
        .find(|b| b.name == "PipelineStageFlagBits2");
    println!("  Format enum:          {has_format}");
    println!(
        "  PipelineStageFlagBits2: {} (bitwidth={})",
        pipeline_stage_2.is_some(),
        pipeline_stage_2.map_or(0, |b| b.bitwidth),
    );

    // Commands & dispatch levels
    let create_instance = registry
        .commands
        .iter()
        .find(|c| c.name == "vkCreateInstance");
    let create_device = registry
        .commands
        .iter()
        .find(|c| c.name == "vkCreateDevice");
    let create_buffer = registry
        .commands
        .iter()
        .find(|c| c.name == "vkCreateBuffer");
    println!(
        "  vkCreateInstance:     {} (dispatch={:?})",
        create_instance.is_some(),
        create_instance.map(|c| c.dispatch_level),
    );
    println!(
        "  vkCreateDevice:      {} (dispatch={:?})",
        create_device.is_some(),
        create_device.map(|c| c.dispatch_level),
    );
    println!(
        "  vkCreateBuffer:      {} (dispatch={:?})",
        create_buffer.is_some(),
        create_buffer.map(|c| c.dispatch_level),
    );

    // Funcpointers
    let has_alloc_fn = registry
        .func_pointers
        .iter()
        .any(|f| f.name == "PFN_vkAllocationFunction");
    let has_debug_cb = registry
        .func_pointers
        .iter()
        .any(|f| f.name == "PFN_vkDebugUtilsMessengerCallbackEXT");
    println!("  PFN_vkAllocationFunction:             {has_alloc_fn}");
    println!("  PFN_vkDebugUtilsMessengerCallbackEXT: {has_debug_cb}");

    // Extensions & platforms
    let xlib_surface = registry
        .extensions
        .iter()
        .find(|e| e.name == "VK_KHR_xlib_surface");
    println!(
        "  VK_KHR_xlib_surface:  {} (platform={:?})",
        xlib_surface.is_some(),
        xlib_surface.and_then(|e| e.platform.as_deref()),
    );
    let has_xlib_platform = registry.platforms.iter().any(|p| p.name == "xlib");
    println!("  xlib platform:        {has_xlib_platform}");

    // Provenance
    let swapchain_create = registry
        .structs
        .iter()
        .find(|s| s.name == "SwapchainCreateInfoKHR");
    println!(
        "  SwapchainCreateInfoKHR provided_by: {:?}",
        swapchain_create.and_then(|s| s.provided_by.as_deref()),
    );

    // Externsync
    let has_externsync = registry
        .structs
        .iter()
        .flat_map(|s| &s.members)
        .any(|m| m.extern_sync.is_some());
    println!("  Has externsync members: {has_externsync}");
}
