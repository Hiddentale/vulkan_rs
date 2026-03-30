//! Command and constant collection from vk.xml.

use vk_parse::{Command, CommandDefinition, CommandParam, EnumSpec, EnumsChild};

use super::{
    is_non_vulkan, is_non_vulkan_api, strip_vk, AliasDef, AliasKind, CommandDef, ConstantDef,
    DispatchLevel, ParamDef, VkRegistry,
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

    let params: Vec<ParamDef> = def.params.iter().map(parse_param).collect();

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
