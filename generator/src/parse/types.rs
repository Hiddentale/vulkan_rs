//! Type collection: handles, structs, unions, bitmasks, basetypes.

use std::collections::{HashMap, HashSet};

use vk_parse::{Type, TypeMember, TypeMemberDefinition, TypeMemberMarkup, TypeSpec};

use super::{
    extract_markup_name, strip_vk, AliasDef, AliasKind, HandleDef, MemberDef, StructDef,
    VkRegistry,
};

pub(super) fn collect_types(
    types: &[vk_parse::TypesChild],
    reg: &mut VkRegistry,
    bitmask_meta: &mut HashMap<String, (String, u32)>,
    bitmask_enum_names: &mut HashSet<String>,
) {
    for child in types {
        let ty = match child {
            vk_parse::TypesChild::Type(ty) => ty,
            _ => continue,
        };

        // Skip non-vulkan API types.
        if let Some(ref api) = ty.api
            && !api.contains("vulkan")
        {
            continue;
        }

        // Handle aliases first.
        if let Some(ref alias) = ty.alias
            && let Some(ref name) = ty.name
        {
            reg.aliases.push(AliasDef {
                name: name.clone(),
                target: alias.clone(),
                kind: AliasKind::Type,
            });
            continue;
        }

        let category = match ty.category.as_deref() {
            Some(c) => c,
            None => continue,
        };

        match category {
            "handle" => collect_handle(ty, reg),
            "struct" => collect_struct(ty, reg, false),
            "union" => collect_struct(ty, reg, true),
            "bitmask" => collect_bitmask_type(ty, bitmask_meta, bitmask_enum_names, reg),
            "basetype" => collect_basetype(ty, reg),
            _ => {}
        }
    }
}

fn collect_handle(ty: &Type, reg: &mut VkRegistry) {
    let code = match &ty.spec {
        TypeSpec::Code(code) => code,
        _ => return,
    };

    let name = ty.name.as_deref().or_else(|| {
        code.markup.iter().find_map(|m| match m {
            vk_parse::TypeCodeMarkup::Name(n) => Some(n.as_str()),
            _ => None,
        })
    });

    let name = match name {
        Some(n) => strip_vk(n),
        None => return,
    };

    let dispatchable = code.code.contains("VK_DEFINE_HANDLE(");

    reg.handles.push(HandleDef {
        name,
        dispatchable,
        parent: ty.parent.as_ref().map(|p| strip_vk(p)),
        object_type: ty.objtypeenum.clone(),
        provided_by: None,
    });
}

fn collect_struct(ty: &Type, reg: &mut VkRegistry, is_union: bool) {
    let name = match ty.name {
        Some(ref n) => strip_vk(n),
        None => return,
    };

    let members = match &ty.spec {
        TypeSpec::Members(members) => parse_members(members),
        _ => return,
    };

    let extends = ty
        .structextends
        .as_deref()
        .map(|s| s.split(',').map(|e| strip_vk(e.trim())).collect())
        .unwrap_or_default();

    let returned_only = ty.returnedonly.as_deref() == Some("true");

    reg.structs.push(StructDef {
        name,
        members,
        extends,
        returned_only,
        is_union,
        provided_by: None,
    });
}

fn parse_members(members: &[TypeMember]) -> Vec<MemberDef> {
    members
        .iter()
        .filter_map(|m| match m {
            TypeMember::Definition(def) => Some(parse_member_def(def)),
            _ => None,
        })
        .collect()
}

fn parse_member_def(def: &TypeMemberDefinition) -> MemberDef {
    let type_name = def
        .markup
        .iter()
        .find_map(|m| match m {
            TypeMemberMarkup::Type(t) => Some(t.clone()),
            _ => None,
        })
        .unwrap_or_default();

    let name = def
        .markup
        .iter()
        .find_map(|m| match m {
            TypeMemberMarkup::Name(n) => Some(n.clone()),
            _ => None,
        })
        .unwrap_or_default();

    let array_size = def.markup.iter().find_map(|m| match m {
        TypeMemberMarkup::Enum(e) => Some(e.clone()),
        _ => None,
    });

    let code = &def.code;
    let pointer_count = code.matches('*').count();
    let is_const = code.contains("const");

    let array_size = array_size.or_else(|| parse_fixed_array_size(code));

    MemberDef {
        name,
        type_name,
        is_pointer: pointer_count >= 1,
        is_const,
        is_double_pointer: pointer_count >= 2,
        array_size,
        optional: def.optional.as_deref().is_some_and(|o| o.contains("true")),
        values: def.values.clone(),
        len: def.len.clone(),
        extern_sync: def.externsync.clone(),
    }
}

/// Extract fixed array size from C code like `float color[4]`.
fn parse_fixed_array_size(code: &str) -> Option<String> {
    let start = code.find('[')?;
    let end = code.find(']')?;
    let inner = code[start + 1..end].trim();
    if !inner.is_empty() && inner.chars().all(|c| c.is_ascii_digit()) {
        Some(inner.to_string())
    } else {
        None
    }
}

fn collect_bitmask_type(
    ty: &Type,
    bitmask_meta: &mut HashMap<String, (String, u32)>,
    bitmask_enum_names: &mut HashSet<String>,
    reg: &mut VkRegistry,
) {
    let flags_name = match &ty.name {
        Some(n) => strip_vk(n),
        None => match &ty.spec {
            TypeSpec::Code(code) => match extract_markup_name(&code.markup) {
                Some(n) => strip_vk(&n),
                None => return,
            },
            _ => return,
        },
    };

    let bitwidth = match &ty.spec {
        TypeSpec::Code(code) if code.code.contains("VkFlags64") => 64,
        _ => 32,
    };

    let enum_name = ty
        .requires
        .as_ref()
        .or(ty.bitvalues.as_ref())
        .map(|n| strip_vk(n));

    if let Some(ref enum_name) = enum_name {
        bitmask_meta.insert(enum_name.clone(), (flags_name.clone(), bitwidth));
        bitmask_enum_names.insert(enum_name.clone());
    }

    if let Some(enum_name) = enum_name {
        reg.aliases.push(AliasDef {
            name: flags_name,
            target: enum_name,
            kind: AliasKind::Bitmask,
        });
    }
}

fn collect_basetype(ty: &Type, reg: &mut VkRegistry) {
    if let (Some(name), TypeSpec::Code(code)) = (&ty.name, &ty.spec) {
        reg.base_types.insert(name.clone(), code.code.clone());
    }
}
