//! Command and constant collection from vk.xml.

use vk_parse::{Command, CommandDefinition, CommandParam, EnumSpec, EnumsChild};

use super::{
    AliasDef, AliasKind, CommandDef, ConstantDef, DispatchLevel, ParamDef, VkRegistry,
    is_non_vulkan, is_non_vulkan_api, strip_vk,
};

pub(super) fn collect_commands(commands: &[Command], reg: &mut VkRegistry) {
    for cmd in commands {
        match cmd {
            Command::Alias { name, alias } => {
                reg.aliases.push(AliasDef {
                    name: strip_vk(name),
                    target: strip_vk(alias),
                    kind: AliasKind::Command,
                });
            }
            Command::Definition(def) => {
                if let Some(cmd_def) = parse_command(def) {
                    reg.commands.push(cmd_def);
                }
            }
            _ => {}
        }
    }
}

fn parse_command(def: &CommandDefinition) -> Option<CommandDef> {
    if is_non_vulkan_api(def.api.as_deref()) {
        return None;
    }

    let name = def.proto.name.clone();
    let return_type = def.proto.type_name.as_deref().unwrap_or("void").to_string();

    let params: Vec<ParamDef> = def
        .params
        .iter()
        .filter(|p| !is_non_vulkan_api(p.api.as_deref()))
        .map(parse_param)
        .collect();

    let dispatch_level = classify_dispatch_level(&params);

    let success_codes = def
        .successcodes
        .as_deref()
        .map(|s| s.split(',').map(str::to_string).collect())
        .unwrap_or_default();

    let error_codes = def
        .errorcodes
        .as_deref()
        .map(|s| s.split(',').map(str::to_string).collect())
        .unwrap_or_default();

    Some(CommandDef {
        name,
        return_type,
        params,
        success_codes,
        error_codes,
        dispatch_level,
        provided_by: None,
    })
}

fn classify_dispatch_level(params: &[ParamDef]) -> DispatchLevel {
    let first_type = match params.first() {
        Some(p) => p.type_name.as_str(),
        None => return DispatchLevel::Entry,
    };
    match first_type {
        "VkDevice" | "VkCommandBuffer" | "VkQueue" => DispatchLevel::Device,
        "VkInstance" | "VkPhysicalDevice" => DispatchLevel::Instance,
        _ => DispatchLevel::Entry,
    }
}

fn parse_param(p: &CommandParam) -> ParamDef {
    let code = &p.definition.code;
    let pointer_count = code.matches('*').count();
    let is_const = code.contains("const");

    ParamDef {
        name: p.definition.name.clone(),
        type_name: p.definition.type_name.clone().unwrap_or_default(),
        is_pointer: pointer_count >= 1,
        is_const,
        is_double_pointer: pointer_count >= 2,
        array_size: None,
        optional: p.optional.as_deref().is_some_and(|o| o.contains("true")),
        len: p.len.clone(),
        extern_sync: p.externsync.clone(),
    }
}

pub(super) fn collect_constants(enums: &vk_parse::Enums, reg: &mut VkRegistry) {
    for child in &enums.children {
        let e = match child {
            EnumsChild::Enum(e) => e,
            _ => continue,
        };
        if is_non_vulkan(e) {
            continue;
        }
        match &e.spec {
            EnumSpec::Value { value, .. } => {
                reg.constants.push(ConstantDef {
                    name: e.name.clone(),
                    value: value.clone(),
                    ty: e.type_suffix.clone(),
                    comment: e.comment.clone(),
                });
            }
            EnumSpec::Alias { alias, .. } => {
                reg.constants.push(ConstantDef {
                    name: e.name.clone(),
                    value: alias.clone(),
                    ty: None,
                    comment: e.comment.clone(),
                });
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_param_def(name: &str, type_name: &str) -> ParamDef {
        ParamDef {
            name: name.to_string(),
            type_name: type_name.to_string(),
            is_pointer: false,
            is_const: false,
            is_double_pointer: false,
            array_size: None,
            optional: false,
            len: None,
            extern_sync: None,
        }
    }

    fn make_command_param(name: &str, type_name: &str, code: &str) -> CommandParam {
        let mut p = CommandParam::default();
        p.definition.name = name.to_string();
        p.definition.type_name = Some(type_name.to_string());
        p.definition.code = code.to_string();
        p
    }

    // -- classify_dispatch_level ------------------------------------------------

    #[test]
    fn dispatch_level_device_for_vk_device() {
        let params = vec![make_param_def("device", "VkDevice")];
        assert_eq!(classify_dispatch_level(&params), DispatchLevel::Device);
    }

    #[test]
    fn dispatch_level_device_for_command_buffer() {
        let params = vec![make_param_def("commandBuffer", "VkCommandBuffer")];
        assert_eq!(classify_dispatch_level(&params), DispatchLevel::Device);
    }

    #[test]
    fn dispatch_level_device_for_queue() {
        let params = vec![make_param_def("queue", "VkQueue")];
        assert_eq!(classify_dispatch_level(&params), DispatchLevel::Device);
    }

    #[test]
    fn dispatch_level_instance_for_vk_instance() {
        let params = vec![make_param_def("instance", "VkInstance")];
        assert_eq!(classify_dispatch_level(&params), DispatchLevel::Instance);
    }

    #[test]
    fn dispatch_level_instance_for_physical_device() {
        let params = vec![make_param_def("physicalDevice", "VkPhysicalDevice")];
        assert_eq!(classify_dispatch_level(&params), DispatchLevel::Instance);
    }

    #[test]
    fn dispatch_level_entry_for_empty_params() {
        assert_eq!(classify_dispatch_level(&[]), DispatchLevel::Entry);
    }

    #[test]
    fn dispatch_level_entry_for_unknown_first_param() {
        let params = vec![make_param_def("pAllocator", "VkAllocationCallbacks")];
        assert_eq!(classify_dispatch_level(&params), DispatchLevel::Entry);
    }

    // -- parse_param -----------------------------------------------------------

    #[test]
    fn parse_param_detects_pointer() {
        let p = make_command_param("pData", "void", "void* pData");
        let result = parse_param(&p);
        assert!(result.is_pointer);
        assert!(!result.is_const);
        assert!(!result.is_double_pointer);
    }

    #[test]
    fn parse_param_detects_const_pointer() {
        let p = make_command_param(
            "pCreateInfo",
            "VkBufferCreateInfo",
            "const VkBufferCreateInfo* pCreateInfo",
        );
        let result = parse_param(&p);
        assert!(result.is_pointer);
        assert!(result.is_const);
        assert!(!result.is_double_pointer);
    }

    #[test]
    fn parse_param_detects_double_pointer() {
        let p = make_command_param("ppData", "void", "void** ppData");
        let result = parse_param(&p);
        assert!(result.is_pointer);
        assert!(result.is_double_pointer);
    }

    #[test]
    fn parse_param_non_pointer() {
        let p = make_command_param("flags", "VkFlags", "VkFlags flags");
        let result = parse_param(&p);
        assert!(!result.is_pointer);
        assert!(!result.is_const);
    }

    #[test]
    fn parse_param_optional_true() {
        let mut p = make_command_param(
            "pAllocator",
            "VkAllocationCallbacks",
            "const VkAllocationCallbacks* pAllocator",
        );
        p.optional = Some("true".to_string());
        let result = parse_param(&p);
        assert!(result.optional);
    }

    #[test]
    fn parse_param_optional_false_or_missing() {
        let p = make_command_param("device", "VkDevice", "VkDevice device");
        let result = parse_param(&p);
        assert!(!result.optional);
    }

    #[test]
    fn parse_param_optional_true_false_combo() {
        let mut p = make_command_param(
            "pAllocator",
            "VkAllocationCallbacks",
            "const VkAllocationCallbacks* pAllocator",
        );
        p.optional = Some("true,false".to_string());
        let result = parse_param(&p);
        assert!(
            result.optional,
            "should detect true in comma-separated optional"
        );
    }

    #[test]
    fn parse_param_preserves_len() {
        let mut p = make_command_param("pData", "void", "void* pData");
        p.len = Some("dataSize".to_string());
        let result = parse_param(&p);
        assert_eq!(result.len.as_deref(), Some("dataSize"));
    }

    #[test]
    fn parse_param_preserves_extern_sync() {
        let mut p = make_command_param("fence", "VkFence", "VkFence fence");
        p.externsync = Some("true".to_string());
        let result = parse_param(&p);
        assert_eq!(result.extern_sync.as_deref(), Some("true"));
    }

    // -- parse_command ---------------------------------------------------------

    fn make_command_def(
        name: &str,
        return_type: Option<&str>,
        params: Vec<CommandParam>,
    ) -> CommandDefinition {
        let mut def = CommandDefinition::default();
        def.proto.name = name.to_string();
        def.proto.type_name = return_type.map(str::to_string);
        def.proto.code = format!("{} {name}", return_type.unwrap_or("void"));
        def.params = params;
        def
    }

    #[test]
    fn parse_command_extracts_return_type() {
        let def = make_command_def(
            "vkCreateBuffer",
            Some("VkResult"),
            vec![make_command_param("device", "VkDevice", "VkDevice device")],
        );
        let result = parse_command(&def).expect("should parse command");
        assert_eq!(result.return_type, "VkResult");
        assert_eq!(result.name, "vkCreateBuffer");
    }

    #[test]
    fn parse_command_defaults_void_return() {
        let def = make_command_def("vkDestroyBuffer", None, vec![]);
        let result = parse_command(&def).expect("should parse void command");
        assert_eq!(result.return_type, "void");
    }

    #[test]
    fn parse_command_splits_success_codes() {
        let mut def = make_command_def("vkFoo", Some("VkResult"), vec![]);
        def.successcodes = Some("VK_SUCCESS,VK_INCOMPLETE".to_string());
        def.errorcodes =
            Some("VK_ERROR_OUT_OF_HOST_MEMORY,VK_ERROR_OUT_OF_DEVICE_MEMORY".to_string());
        let result = parse_command(&def).expect("should parse command with codes");
        assert_eq!(result.success_codes, vec!["VK_SUCCESS", "VK_INCOMPLETE"]);
        assert_eq!(
            result.error_codes,
            vec![
                "VK_ERROR_OUT_OF_HOST_MEMORY",
                "VK_ERROR_OUT_OF_DEVICE_MEMORY"
            ]
        );
    }

    #[test]
    fn parse_command_skips_non_vulkan_api() {
        let mut def = make_command_def("vkFoo", Some("VkResult"), vec![]);
        def.api = Some("vulkansc".to_string());
        assert!(parse_command(&def).is_none());
    }

    #[test]
    fn parse_command_filters_non_vulkan_params() {
        let mut sc_param = make_command_param("scOnly", "VkFoo", "VkFoo scOnly");
        sc_param.api = Some("vulkansc".to_string());

        let def = make_command_def(
            "vkBar",
            Some("void"),
            vec![
                make_command_param("device", "VkDevice", "VkDevice device"),
                sc_param,
            ],
        );
        let result = parse_command(&def).expect("should parse with filtered params");
        assert_eq!(result.params.len(), 1);
        assert_eq!(result.params[0].name, "device");
    }
}
