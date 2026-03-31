//! Parses vk.xml into categorized intermediate types for code generation.
//!
//! Fields that appear unused are consumed by emitters in later phases.

mod commands;
mod enums;
mod extensions;
mod types;

use std::collections::{HashMap, HashSet};
use vk_parse::{self, Enum, Extension, RegistryChild};

// ---------------------------------------------------------------------------
// Intermediate types — what emitters consume
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct VkRegistry {
    pub handles: Vec<HandleDef>,
    pub structs: Vec<StructDef>,
    pub enums: Vec<EnumDef>,
    pub bitmasks: Vec<BitmaskDef>,
    pub commands: Vec<CommandDef>,
    pub constants: Vec<ConstantDef>,
    pub func_pointers: Vec<FuncPointerDef>,
    pub extensions: Vec<ExtensionDef>,
    pub platforms: Vec<PlatformDef>,
    pub aliases: Vec<AliasDef>,
    pub base_types: HashMap<String, String>,
}

#[derive(Debug)]
pub struct HandleDef {
    pub name: String,
    pub dispatchable: bool,
    pub parent: Option<String>,
    pub object_type: Option<String>,
    pub provided_by: Option<String>,
}

#[derive(Debug)]
pub struct StructDef {
    pub name: String,
    pub members: Vec<MemberDef>,
    pub extends: Vec<String>,
    pub returned_only: bool,
    pub is_union: bool,
    pub provided_by: Option<String>,
}

#[derive(Debug, Clone)]
pub struct MemberDef {
    pub name: String,
    pub type_name: String,
    pub is_pointer: bool,
    pub is_const: bool,
    pub is_double_pointer: bool,
    pub array_size: Option<String>,
    pub optional: bool,
    pub values: Option<String>,
    pub len: Option<String>,
    pub extern_sync: Option<String>,
    pub is_bitfield: bool,
    pub bitwidth: Option<u8>,
}

#[derive(Debug)]
pub struct EnumDef {
    pub name: String,
    pub variants: Vec<EnumVariant>,
}

#[derive(Debug)]
pub struct EnumVariant {
    pub name: String,
    pub value: EnumValue,
    pub comment: Option<String>,
}

#[derive(Debug)]
pub enum EnumValue {
    I32(i32),
    Alias(String),
}

#[derive(Debug)]
pub struct BitmaskDef {
    pub name: String,
    pub flags_name: String,
    pub bitwidth: u32,
    pub bits: Vec<BitmaskBit>,
}

#[derive(Debug)]
pub struct BitmaskBit {
    pub name: String,
    pub value: BitmaskValue,
}

#[derive(Debug)]
pub enum BitmaskValue {
    Bitpos(u32),
    Value(u64),
    Alias(String),
}

#[derive(Debug)]
pub struct CommandDef {
    pub name: String,
    pub return_type: String,
    pub params: Vec<ParamDef>,
    pub success_codes: Vec<String>,
    pub error_codes: Vec<String>,
    pub dispatch_level: DispatchLevel,
    pub provided_by: Option<String>,
}

#[derive(Debug)]
pub struct ParamDef {
    pub name: String,
    pub type_name: String,
    pub is_pointer: bool,
    pub is_const: bool,
    pub is_double_pointer: bool,
    pub array_size: Option<String>,
    pub optional: bool,
    pub len: Option<String>,
    pub extern_sync: Option<String>,
}

#[derive(Debug)]
pub struct ConstantDef {
    pub name: String,
    pub value: String,
    pub ty: Option<String>,
    pub comment: Option<String>,
}

#[derive(Debug)]
pub struct FuncPointerDef {
    pub name: String,
    pub return_type: String,
    pub is_return_pointer: bool,
    pub params: Vec<ParamDef>,
}

#[derive(Debug, Clone)]
pub struct ExtensionDef {
    pub name: String,
    pub number: i64,
    pub ext_type: Option<String>,
    pub platform: Option<String>,
    pub depends: Option<String>,
    pub promoted_to: Option<String>,
    pub deprecated_by: Option<String>,
    pub supported: String,
    pub items: Vec<ExtensionItem>,
}

#[derive(Debug, Clone)]
pub enum ExtensionItem {
    Type(String),
    Command(String),
}

#[derive(Debug, Clone)]
pub struct PlatformDef {
    pub name: String,
    pub protect: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchLevel {
    Entry,
    Instance,
    Device,
}

#[derive(Debug, Clone)]
pub struct AliasDef {
    pub name: String,
    pub target: String,
    pub kind: AliasKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AliasKind {
    /// Type aliases: structs, handles, enums (from `<type ... alias="...">`)
    Type,
    /// Command aliases: promoted extension commands (from `<command ... alias="...">`)
    Command,
    /// Bitmask Flags→FlagBits aliases (from bitmask type definitions)
    Bitmask,
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

pub fn parse_registry(path: &std::path::Path) -> VkRegistry {
    let (registry, errors) = vk_parse::parse_file(path).expect("failed to parse vk.xml");

    if !errors.is_empty() {
        eprintln!("vk.xml parse warnings ({}):", errors.len());
        for e in errors.iter().take(5) {
            eprintln!("  {e:?}");
        }
    }

    let mut reg = VkRegistry {
        handles: Vec::new(),
        structs: Vec::new(),
        enums: Vec::new(),
        bitmasks: Vec::new(),
        commands: Vec::new(),
        constants: Vec::new(),
        func_pointers: Vec::new(),
        extensions: Vec::new(),
        platforms: Vec::new(),
        aliases: Vec::new(),
        base_types: HashMap::new(),
    };

    // Collect enum/bitmask groups by name so we can append extension values later.
    let mut enum_map: HashMap<String, Vec<EnumVariant>> = HashMap::new();
    let mut bitmask_map: HashMap<String, Vec<BitmaskBit>> = HashMap::new();
    let mut bitmask_meta: HashMap<String, (String, u32)> = HashMap::new(); // enum_name → (flags_name, bitwidth)

    // Track which names are bitmask enums vs value enums.
    let mut bitmask_enum_names: HashSet<String> = HashSet::new();

    // First pass: collect types, commands, top-level enums.
    for child in &registry.0 {
        match child {
            RegistryChild::Types(t) => {
                types::collect_types(
                    &t.children,
                    &mut reg,
                    &mut bitmask_meta,
                    &mut bitmask_enum_names,
                );
            }
            RegistryChild::Enums(e) => {
                enums::collect_enums(e, &mut enum_map, &mut bitmask_map, &bitmask_enum_names);
            }
            RegistryChild::Commands(cmds) => {
                commands::collect_commands(&cmds.children, &mut reg);
            }
            _ => {}
        }
    }

    // Second pass: gather enum/bitmask extensions from features and extensions.
    for child in &registry.0 {
        match child {
            RegistryChild::Feature(feature) => {
                enums::collect_feature_enums(
                    feature,
                    &mut enum_map,
                    &mut bitmask_map,
                    &bitmask_enum_names,
                );
            }
            RegistryChild::Extensions(extensions) => {
                for ext in &extensions.children {
                    if !is_vulkan_extension(ext) {
                        continue;
                    }
                    enums::collect_extension_enums(
                        ext,
                        &mut enum_map,
                        &mut bitmask_map,
                        &bitmask_enum_names,
                    );
                }
            }
            _ => {}
        }
    }

    // Build final enum/bitmask defs. Sort by name for deterministic output.
    let mut enum_entries: Vec<_> = enum_map.into_iter().collect();
    enum_entries.sort_by(|(a, _), (b, _)| a.cmp(b));
    for (name, variants) in enum_entries {
        reg.enums.push(EnumDef { name, variants });
    }

    let mut bitmask_entries: Vec<_> = bitmask_map.into_iter().collect();
    bitmask_entries.sort_by(|(a, _), (b, _)| a.cmp(b));
    for (enum_name, bits) in bitmask_entries {
        let (flags_name, bitwidth) = bitmask_meta
            .get(&enum_name)
            .cloned()
            .unwrap_or_else(|| (enum_name.to_string(), 32));
        reg.bitmasks.push(BitmaskDef {
            name: enum_name,
            flags_name,
            bitwidth,
            bits,
        });
    }

    // Collect API constants.
    for child in &registry.0 {
        if let RegistryChild::Enums(enums) = child
            && enums.name.as_deref() == Some("API Constants")
        {
            commands::collect_constants(enums, &mut reg);
        }
    }

    // Funcpointers: vk-parse doesn't parse proto/param for these, so we
    // parse them from the raw XML text in a separate pass.
    let xml_text = std::fs::read_to_string(path).expect("failed to read vk.xml");
    reg.func_pointers = extensions::parse_func_pointers(&xml_text);

    // Collect platforms and extensions.
    extensions::collect_platforms(&registry, &mut reg);
    extensions::collect_extensions(&registry, &mut reg);

    // Stamp provenance (which feature/extension introduced each type/command).
    extensions::stamp_provenance(&registry, &mut reg);

    reg
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the name from TypeCodeMarkup (for types with inline `<name>`).
fn extract_markup_name(markup: &[vk_parse::TypeCodeMarkup]) -> Option<String> {
    markup.iter().find_map(|m| match m {
        vk_parse::TypeCodeMarkup::Name(n) => Some(n.clone()),
        _ => None,
    })
}

/// Strip the `Vk` prefix from a Vulkan type name.
pub fn strip_vk(name: &str) -> String {
    name.strip_prefix("Vk").unwrap_or(name).to_string()
}

/// Compute extension enum value: base + (ext_number - 1) * 1000 + offset.
fn compute_enum_offset(ext_number: i64, offset: i64, negative: bool) -> i32 {
    const BASE: i64 = 1_000_000_000;
    const RANGE: i64 = 1000;
    let value = BASE + (ext_number - 1) * RANGE + offset;
    if negative {
        -(value as i32)
    } else {
        value as i32
    }
}

fn parse_c_literal(s: &str) -> i32 {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        i32::from_str_radix(hex, 16).unwrap_or(0)
    } else if s.starts_with('-') {
        s.parse().unwrap_or(0)
    } else {
        // Strip suffixes like U, UL.
        let num: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
        num.parse().unwrap_or(0)
    }
}

fn parse_c_literal_u64(s: &str) -> u64 {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        let hex: String = hex.chars().take_while(|c| c.is_ascii_hexdigit()).collect();
        u64::from_str_radix(&hex, 16).unwrap_or(0)
    } else {
        let num: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
        num.parse().unwrap_or(0)
    }
}

fn is_non_vulkan(e: &Enum) -> bool {
    is_non_vulkan_api(e.api.as_deref())
}

fn is_non_vulkan_api(api: Option<&str>) -> bool {
    match api {
        Some(a) => !a
            .split(',')
            .any(|part| matches!(part.trim(), "vulkan" | "vulkanbase")),
        None => false,
    }
}

fn is_vulkan_extension(ext: &Extension) -> bool {
    is_vulkan_supported(ext.supported.as_deref())
}

fn is_vulkan_supported(supported: Option<&str>) -> bool {
    match supported {
        Some(s) => s.split(',').any(|part| part.trim() == "vulkan"),
        None => true,
    }
}

// Made visible for testing.
#[cfg(test)]
pub(crate) fn test_parse_c_literal(s: &str) -> i32 {
    parse_c_literal(s)
}

#[cfg(test)]
pub(crate) fn test_compute_enum_offset(ext_number: i64, offset: i64, negative: bool) -> i32 {
    compute_enum_offset(ext_number, offset, negative)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::OnceLock;

    /// Parse vk.xml once and cache for all tests.
    fn registry() -> &'static VkRegistry {
        static REG: OnceLock<VkRegistry> = OnceLock::new();
        REG.get_or_init(|| {
            let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("vk.xml");
            parse_registry(&path)
        })
    }

    // -----------------------------------------------------------------------
    // Pure helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn strip_vk_removes_prefix() {
        assert_eq!(strip_vk("VkInstance"), "Instance");
        assert_eq!(strip_vk("VkPhysicalDevice"), "PhysicalDevice");
    }

    #[test]
    fn strip_vk_no_prefix_unchanged() {
        assert_eq!(strip_vk("uint32_t"), "uint32_t");
    }

    #[test]
    fn compute_enum_offset_basic() {
        // Extension 1, offset 0 → 1_000_000_000.
        assert_eq!(test_compute_enum_offset(1, 0, false), 1_000_000_000);
    }

    #[test]
    fn compute_enum_offset_negative() {
        assert_eq!(test_compute_enum_offset(1, 0, true), -1_000_000_000);
    }

    #[test]
    fn compute_enum_offset_extension_number() {
        // Extension 2, offset 3 → 1_000_000_000 + 1000 + 3 = 1_000_001_003.
        assert_eq!(test_compute_enum_offset(2, 3, false), 1_000_001_003);
    }

    #[test]
    fn parse_c_literal_decimal() {
        assert_eq!(test_parse_c_literal("42"), 42);
    }

    #[test]
    fn parse_c_literal_hex() {
        assert_eq!(test_parse_c_literal("0x7FFFFFFF"), 0x7FFFFFFF);
    }

    #[test]
    fn parse_c_literal_negative() {
        assert_eq!(test_parse_c_literal("-1"), -1);
    }

    #[test]
    fn parse_c_literal_with_suffix() {
        assert_eq!(test_parse_c_literal("256U"), 256);
    }

    // -----------------------------------------------------------------------
    // is_vulkan_supported
    // -----------------------------------------------------------------------

    #[test]
    fn vulkan_supported_standard() {
        assert!(is_vulkan_supported(Some("vulkan")));
    }

    #[test]
    fn vulkan_supported_comma_list() {
        assert!(is_vulkan_supported(Some("vulkan,vulkansc")));
    }

    #[test]
    fn vulkan_supported_rejects_vulkansc_only() {
        assert!(!is_vulkan_supported(Some("vulkansc")));
    }

    #[test]
    fn vulkan_supported_rejects_disabled() {
        assert!(!is_vulkan_supported(Some("disabled")));
    }

    #[test]
    fn vulkan_supported_none_defaults_true() {
        assert!(is_vulkan_supported(None));
    }

    // -----------------------------------------------------------------------
    // is_non_vulkan_api
    // -----------------------------------------------------------------------

    #[test]
    fn non_vulkan_api_none_is_vulkan() {
        assert!(!is_non_vulkan_api(None));
    }

    #[test]
    fn non_vulkan_api_vulkan() {
        assert!(!is_non_vulkan_api(Some("vulkan")));
    }

    #[test]
    fn non_vulkan_api_vulkansc_only() {
        assert!(is_non_vulkan_api(Some("vulkansc")));
    }

    #[test]
    fn non_vulkan_api_comma_list_with_vulkan() {
        assert!(!is_non_vulkan_api(Some("vulkan,vulkansc")));
    }

    // -----------------------------------------------------------------------
    // Registry-level: counts and existence
    // -----------------------------------------------------------------------

    #[test]
    fn registry_has_handles() {
        assert!(registry().handles.len() > 50, "expected 50+ handles");
    }

    #[test]
    fn registry_has_enums() {
        assert!(registry().enums.len() > 100, "expected 100+ enums");
    }

    #[test]
    fn registry_has_bitmasks() {
        assert!(registry().bitmasks.len() > 100, "expected 100+ bitmasks");
    }

    #[test]
    fn registry_has_commands() {
        assert!(registry().commands.len() > 700, "expected 700+ commands");
    }

    #[test]
    fn registry_has_constants() {
        assert!(registry().constants.len() > 30, "expected 30+ constants");
    }

    #[test]
    fn registry_has_extensions() {
        assert!(
            registry().extensions.len() > 400,
            "expected 400+ extensions"
        );
    }

    #[test]
    fn registry_has_platforms() {
        assert!(registry().platforms.len() > 10, "expected 10+ platforms");
    }

    #[test]
    fn no_vulkansc_only_extensions_collected() {
        let reg = registry();
        for ext in &reg.extensions {
            assert!(
                ext.supported.split(',').any(|p| p.trim() == "vulkan"),
                "vulkansc-only extension collected: {} (supported={})",
                ext.name,
                ext.supported,
            );
        }
    }

    // -----------------------------------------------------------------------
    // Handles
    // -----------------------------------------------------------------------

    #[test]
    fn handle_instance_is_dispatchable() {
        let h = registry().handles.iter().find(|h| h.name == "Instance");
        assert!(h.is_some(), "Instance handle not found");
        assert!(h.expect("Instance handle must exist").dispatchable);
    }

    #[test]
    fn handle_buffer_is_non_dispatchable() {
        let h = registry().handles.iter().find(|h| h.name == "Buffer");
        assert!(h.is_some(), "Buffer handle not found");
        assert!(!h.expect("Buffer handle must exist").dispatchable);
    }

    #[test]
    fn handle_has_parent() {
        let h = registry()
            .handles
            .iter()
            .find(|h| h.name == "Device")
            .expect("Device handle not found");
        assert_eq!(h.parent.as_deref(), Some("PhysicalDevice"));
    }

    #[test]
    fn handle_has_object_type() {
        let h = registry()
            .handles
            .iter()
            .find(|h| h.name == "Instance")
            .expect("Instance handle not found");
        assert!(h.object_type.is_some());
    }

    // -----------------------------------------------------------------------
    // Structs
    // -----------------------------------------------------------------------

    #[test]
    fn struct_buffer_create_info_exists() {
        let s = registry()
            .structs
            .iter()
            .find(|s| s.name == "BufferCreateInfo");
        assert!(s.is_some());
        let s = s.expect("BufferCreateInfo struct not found");
        assert!(!s.is_union);
        assert!(!s.members.is_empty());
    }

    #[test]
    fn struct_has_s_type_member() {
        let s = registry()
            .structs
            .iter()
            .find(|s| s.name == "BufferCreateInfo")
            .expect("BufferCreateInfo struct not found");
        let s_type = s.members.iter().find(|m| m.name == "sType");
        assert!(
            s_type.is_some(),
            "BufferCreateInfo should have sType member"
        );
    }

    #[test]
    fn struct_has_p_next_member() {
        let s = registry()
            .structs
            .iter()
            .find(|s| s.name == "BufferCreateInfo")
            .expect("BufferCreateInfo struct not found");
        let p_next = s.members.iter().find(|m| m.name == "pNext");
        assert!(p_next.is_some());
        let p_next = p_next.expect("pNext member not found");
        assert!(p_next.is_pointer);
        assert!(p_next.is_const);
    }

    #[test]
    fn union_clear_color_value_exists() {
        let u = registry()
            .structs
            .iter()
            .find(|s| s.name == "ClearColorValue");
        assert!(u.is_some());
        assert!(u.expect("ClearColorValue union not found").is_union);
    }

    #[test]
    fn struct_extends_parsed() {
        // PhysicalDeviceFeatures2 is extended by many structs.
        let extending = registry()
            .structs
            .iter()
            .filter(|s| s.extends.contains(&"PhysicalDeviceFeatures2".to_string()));
        assert!(
            extending.count() > 10,
            "expected many structs extending PhysicalDeviceFeatures2"
        );
    }

    // -----------------------------------------------------------------------
    // Enums
    // -----------------------------------------------------------------------

    #[test]
    fn enum_format_exists_with_variants() {
        let e = registry()
            .enums
            .iter()
            .find(|e| e.name == "Format")
            .expect("Format enum not found");
        assert!(e.variants.len() > 100, "Format should have 100+ variants");

        let undefined = e.variants.iter().find(|v| v.name == "VK_FORMAT_UNDEFINED");
        assert!(undefined.is_some());
        match &undefined
            .expect("VK_FORMAT_UNDEFINED variant not found")
            .value
        {
            EnumValue::I32(0) => {}
            other => panic!("expected I32(0), got {other:?}"),
        }
    }

    #[test]
    fn enum_result_has_error_codes() {
        let e = registry()
            .enums
            .iter()
            .find(|e| e.name == "Result")
            .expect("Result enum not found");

        let success = e.variants.iter().find(|v| v.name == "VK_SUCCESS");
        assert!(success.is_some());

        let oom = e
            .variants
            .iter()
            .find(|v| v.name == "VK_ERROR_OUT_OF_HOST_MEMORY");
        assert!(oom.is_some());
        match &oom
            .expect("VK_ERROR_OUT_OF_HOST_MEMORY variant not found")
            .value
        {
            EnumValue::I32(v) => assert!(*v < 0, "error codes should be negative"),
            other => panic!("expected I32, got {other:?}"),
        }
    }

    #[test]
    fn enum_result_promoted_error_codes_are_negative() {
        let e = registry()
            .enums
            .iter()
            .find(|e| e.name == "Result")
            .expect("Result enum not found");

        // ERROR_OUT_OF_POOL_MEMORY is promoted from VK_KHR_maintenance1 (ext 70)
        // In vk.xml it has dir="-", so the value should be -1000069000.
        let oom_pool = e
            .variants
            .iter()
            .find(|v| v.name == "VK_ERROR_OUT_OF_POOL_MEMORY");
        assert!(oom_pool.is_some(), "VK_ERROR_OUT_OF_POOL_MEMORY not found");
        match &oom_pool
            .expect("VK_ERROR_OUT_OF_POOL_MEMORY variant not found")
            .value
        {
            EnumValue::I32(v) => assert!(
                *v < 0,
                "VK_ERROR_OUT_OF_POOL_MEMORY should be negative, got {v}"
            ),
            other => panic!("expected I32, got {other:?}"),
        }
    }

    #[test]
    fn enum_structure_type_has_many_variants() {
        // StructureType is the largest enum — it gains variants from every extension.
        let e = registry()
            .enums
            .iter()
            .find(|e| e.name == "StructureType")
            .expect("StructureType not found");
        assert!(
            e.variants.len() > 500,
            "StructureType should have 500+ variants, got {}",
            e.variants.len()
        );
    }

    // -----------------------------------------------------------------------
    // Bitmasks
    // -----------------------------------------------------------------------

    #[test]
    fn bitmask_buffer_usage_exists() {
        let b = registry()
            .bitmasks
            .iter()
            .find(|b| b.name == "BufferUsageFlagBits");
        assert!(b.is_some());
        let b = b.expect("BufferUsageFlagBits bitmask not found");
        assert_eq!(b.bitwidth, 32);
        assert!(!b.bits.is_empty());
    }

    #[test]
    fn bitmask_64bit_pipeline_stage() {
        let b = registry()
            .bitmasks
            .iter()
            .find(|b| b.name == "PipelineStageFlagBits2");
        assert!(b.is_some());
        assert_eq!(
            b.expect("PipelineStageFlagBits2 bitmask not found")
                .bitwidth,
            64
        );
    }

    #[test]
    fn bitmask_bit_values_are_powers_of_two() {
        let b = registry()
            .bitmasks
            .iter()
            .find(|b| b.name == "BufferUsageFlagBits")
            .expect("BufferUsageFlagBits bitmask not found");
        for bit in &b.bits {
            match &bit.value {
                BitmaskValue::Bitpos(pos) => assert!(*pos < 64, "bitpos too large: {pos}"),
                BitmaskValue::Value(_) => {}
                BitmaskValue::Alias(_) => {}
            }
        }
    }

    // -----------------------------------------------------------------------
    // Commands
    // -----------------------------------------------------------------------

    #[test]
    fn command_create_instance_is_entry_level() {
        let c = registry()
            .commands
            .iter()
            .find(|c| c.name == "vkCreateInstance")
            .expect("vkCreateInstance not found");
        assert_eq!(c.dispatch_level, DispatchLevel::Entry);
        assert_eq!(c.return_type, "VkResult");
    }

    #[test]
    fn command_create_device_is_instance_level() {
        let c = registry()
            .commands
            .iter()
            .find(|c| c.name == "vkCreateDevice")
            .expect("vkCreateDevice not found");
        assert_eq!(c.dispatch_level, DispatchLevel::Instance);
    }

    #[test]
    fn command_create_buffer_is_device_level() {
        let c = registry()
            .commands
            .iter()
            .find(|c| c.name == "vkCreateBuffer")
            .expect("vkCreateBuffer not found");
        assert_eq!(c.dispatch_level, DispatchLevel::Device);
    }

    #[test]
    fn command_has_params() {
        let c = registry()
            .commands
            .iter()
            .find(|c| c.name == "vkCreateBuffer")
            .expect("vkCreateBuffer command not found");
        assert!(c.params.len() >= 3, "vkCreateBuffer should have 4 params");
        assert_eq!(c.params[0].type_name, "VkDevice");
    }

    #[test]
    fn command_has_success_and_error_codes() {
        let c = registry()
            .commands
            .iter()
            .find(|c| c.name == "vkCreateInstance")
            .expect("vkCreateInstance command not found");
        assert!(!c.success_codes.is_empty());
        assert!(!c.error_codes.is_empty());
    }

    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------

    #[test]
    fn constant_max_physical_device_name_size() {
        let c = registry()
            .constants
            .iter()
            .find(|c| c.name == "VK_MAX_PHYSICAL_DEVICE_NAME_SIZE");
        assert!(c.is_some());
        assert_eq!(
            c.expect("VK_MAX_PHYSICAL_DEVICE_NAME_SIZE not found").value,
            "256"
        );
    }

    #[test]
    fn constant_whole_size_is_u64() {
        let c = registry()
            .constants
            .iter()
            .find(|c| c.name == "VK_WHOLE_SIZE")
            .expect("VK_WHOLE_SIZE not found");
        assert_eq!(c.ty.as_deref(), Some("uint64_t"));
    }

    // -----------------------------------------------------------------------
    // Func pointers
    // -----------------------------------------------------------------------

    #[test]
    fn funcpointer_allocation_function() {
        let fp = registry()
            .func_pointers
            .iter()
            .find(|f| f.name == "PFN_vkAllocationFunction");
        assert!(fp.is_some());
        let fp = fp.expect("PFN_vkAllocationFunction not found");
        assert!(!fp.params.is_empty());
    }

    // -----------------------------------------------------------------------
    // Extensions & platforms
    // -----------------------------------------------------------------------

    #[test]
    fn extension_khr_swapchain_exists() {
        let ext = registry()
            .extensions
            .iter()
            .find(|e| e.name == "VK_KHR_swapchain");
        assert!(ext.is_some());
        let ext = ext.expect("VK_KHR_swapchain extension not found");
        assert!(!ext.items.is_empty());
    }

    #[test]
    fn extension_platform_xlib() {
        let ext = registry()
            .extensions
            .iter()
            .find(|e| e.name == "VK_KHR_xlib_surface")
            .expect("VK_KHR_xlib_surface not found");
        assert_eq!(ext.platform.as_deref(), Some("xlib"));
    }

    #[test]
    fn platform_has_protect_guard() {
        let p = registry()
            .platforms
            .iter()
            .find(|p| p.name == "win32")
            .expect("win32 platform not found");
        assert_eq!(p.protect, "VK_USE_PLATFORM_WIN32_KHR");
    }

    // -----------------------------------------------------------------------
    // Provenance
    // -----------------------------------------------------------------------

    #[test]
    fn handle_provenance_stamped() {
        let h = registry()
            .handles
            .iter()
            .find(|h| h.name == "Instance")
            .expect("Instance handle not found");
        assert!(h.provided_by.is_some(), "Instance should have provided_by");
    }

    #[test]
    fn extension_type_provenance() {
        let s = registry()
            .structs
            .iter()
            .find(|s| s.name == "SwapchainCreateInfoKHR");
        assert!(s.is_some());
        assert_eq!(
            s.expect("SwapchainCreateInfoKHR struct not found")
                .provided_by
                .as_deref(),
            Some("VK_KHR_swapchain")
        );
    }

    // -----------------------------------------------------------------------
    // Aliases
    // -----------------------------------------------------------------------

    #[test]
    fn aliases_populated() {
        let reg = registry();
        assert!(reg.aliases.len() > 500, "expected 500+ aliases");
        // Verify aliases have correct kinds.
        assert!(
            reg.aliases.iter().any(|a| a.kind == AliasKind::Command),
            "should have command aliases"
        );
        assert!(
            reg.aliases.iter().any(|a| a.kind == AliasKind::Type),
            "should have type aliases"
        );
        assert!(
            reg.aliases.iter().any(|a| a.kind == AliasKind::Bitmask),
            "should have bitmask aliases"
        );
    }
}
