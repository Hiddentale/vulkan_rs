//! Type collection: handles, structs, unions, bitmasks, basetypes.

use std::collections::{HashMap, HashSet};

use vk_parse::{Type, TypeMember, TypeMemberDefinition, TypeMemberMarkup, TypeSpec};

use super::{
    AliasDef, AliasKind, HandleDef, MemberDef, StructDef, VkRegistry, extract_markup_name, strip_vk,
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

        // Skip non-vulkan API types (exact match to avoid "vulkansc" substring hit).
        if let Some(ref api) = ty.api
            && !api.split(',').any(|part| part.trim() == "vulkan")
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

    // Bit-fields appear as "uint32_t name:24" in the code string.
    let bitwidth = code.rsplit_once(':').and_then(|(_, bits)| {
        let bits = bits.trim();
        if !bits.is_empty() && bits.chars().all(|c| c.is_ascii_digit()) {
            bits.parse::<u8>().ok()
        } else {
            None
        }
    });
    let is_bitfield = bitwidth.is_some();

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
        is_bitfield,
        bitwidth,
    }
}

/// Extract fixed array dimensions from C code like `float color[4]` or `float matrix[3][4]`.
/// Returns dimensions joined by `:` for multi-dimensional arrays (e.g. `"3:4"`).
fn parse_fixed_array_size(code: &str) -> Option<String> {
    let mut dims = Vec::new();
    let mut rest = code;
    while let Some(start) = rest.find('[') {
        let end = rest[start..].find(']')?;
        let inner = rest[start + 1..start + end].trim();
        if inner.is_empty() || !inner.chars().all(|c| c.is_ascii_digit()) {
            break;
        }
        dims.push(inner.to_string());
        rest = &rest[start + end + 1..];
    }
    if dims.is_empty() {
        None
    } else {
        Some(dims.join(":"))
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

#[cfg(test)]
mod tests {
    use super::*;
    use vk_parse::{TypeCode, TypeCodeMarkup};

    fn empty_registry() -> VkRegistry {
        VkRegistry {
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
        }
    }

    fn make_member_def(code: &str, markup: Vec<TypeMemberMarkup>) -> TypeMemberDefinition {
        let mut def = TypeMemberDefinition::default();
        def.code = code.to_string();
        def.markup = markup;
        def
    }

    fn make_type(name: &str, category: &str, spec: TypeSpec) -> Type {
        let mut ty = Type::default();
        ty.name = Some(name.to_string());
        ty.category = Some(category.to_string());
        ty.spec = spec;
        ty
    }

    fn make_type_code(code: &str, markup: Vec<TypeCodeMarkup>) -> TypeSpec {
        let mut tc = TypeCode::default();
        tc.code = code.to_string();
        tc.markup = markup;
        TypeSpec::Code(tc)
    }

    // -- parse_fixed_array_size ------------------------------------------------

    #[test]
    fn fixed_array_digits() {
        assert_eq!(
            parse_fixed_array_size("float color[4]"),
            Some("4".to_string())
        );
    }

    #[test]
    fn fixed_array_larger() {
        assert_eq!(
            parse_fixed_array_size("uint32_t data[256]"),
            Some("256".to_string())
        );
    }

    #[test]
    fn fixed_array_rejects_enum_reference() {
        assert_eq!(
            parse_fixed_array_size("char name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE]"),
            None
        );
    }

    #[test]
    fn fixed_array_rejects_empty_brackets() {
        assert_eq!(parse_fixed_array_size("void* data[]"), None);
    }

    #[test]
    fn fixed_array_no_brackets() {
        assert_eq!(parse_fixed_array_size("uint32_t value"), None);
    }

    #[test]
    fn fixed_array_trims_whitespace() {
        assert_eq!(
            parse_fixed_array_size("float v[ 3 ]"),
            Some("3".to_string())
        );
    }

    // -- parse_member_def ------------------------------------------------------

    #[test]
    fn member_def_simple_value() {
        let def = make_member_def(
            "uint32_t flags",
            vec![
                TypeMemberMarkup::Type("uint32_t".to_string()),
                TypeMemberMarkup::Name("flags".to_string()),
            ],
        );
        let m = parse_member_def(&def);
        assert_eq!(m.name, "flags");
        assert_eq!(m.type_name, "uint32_t");
        assert!(!m.is_pointer);
        assert!(!m.is_const);
        assert!(m.array_size.is_none());
    }

    #[test]
    fn member_def_const_pointer() {
        let def = make_member_def(
            "const void* pNext",
            vec![
                TypeMemberMarkup::Type("void".to_string()),
                TypeMemberMarkup::Name("pNext".to_string()),
            ],
        );
        let m = parse_member_def(&def);
        assert!(m.is_pointer);
        assert!(m.is_const);
        assert!(!m.is_double_pointer);
    }

    #[test]
    fn member_def_double_pointer() {
        let def = make_member_def(
            "const char* const* ppEnabledLayerNames",
            vec![
                TypeMemberMarkup::Type("char".to_string()),
                TypeMemberMarkup::Name("ppEnabledLayerNames".to_string()),
            ],
        );
        let m = parse_member_def(&def);
        assert!(m.is_double_pointer);
    }

    #[test]
    fn member_def_enum_array_size() {
        let def = make_member_def(
            "char deviceName[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE]",
            vec![
                TypeMemberMarkup::Type("char".to_string()),
                TypeMemberMarkup::Name("deviceName".to_string()),
                TypeMemberMarkup::Enum("VK_MAX_PHYSICAL_DEVICE_NAME_SIZE".to_string()),
            ],
        );
        let m = parse_member_def(&def);
        assert_eq!(
            m.array_size.as_deref(),
            Some("VK_MAX_PHYSICAL_DEVICE_NAME_SIZE")
        );
    }

    #[test]
    fn member_def_fixed_numeric_array() {
        let def = make_member_def(
            "float color[4]",
            vec![
                TypeMemberMarkup::Type("float".to_string()),
                TypeMemberMarkup::Name("color".to_string()),
            ],
        );
        let m = parse_member_def(&def);
        assert_eq!(m.array_size.as_deref(), Some("4"));
    }

    #[test]
    fn member_def_optional() {
        let mut def = make_member_def(
            "VkFence fence",
            vec![
                TypeMemberMarkup::Type("VkFence".to_string()),
                TypeMemberMarkup::Name("fence".to_string()),
            ],
        );
        def.optional = Some("true".to_string());
        let m = parse_member_def(&def);
        assert!(m.optional);
    }

    #[test]
    fn member_def_preserves_values() {
        let mut def = make_member_def(
            "VkStructureType sType",
            vec![
                TypeMemberMarkup::Type("VkStructureType".to_string()),
                TypeMemberMarkup::Name("sType".to_string()),
            ],
        );
        def.values = Some("VK_STRUCTURE_TYPE_APPLICATION_INFO".to_string());
        let m = parse_member_def(&def);
        assert_eq!(
            m.values.as_deref(),
            Some("VK_STRUCTURE_TYPE_APPLICATION_INFO")
        );
    }

    // -- collect_handle --------------------------------------------------------

    #[test]
    fn collect_handle_dispatchable() {
        let mut reg = empty_registry();
        let ty = make_type(
            "VkInstance",
            "handle",
            make_type_code(
                "VK_DEFINE_HANDLE(VkInstance)",
                vec![TypeCodeMarkup::Name("VkInstance".to_string())],
            ),
        );
        collect_handle(&ty, &mut reg);
        assert_eq!(reg.handles.len(), 1);
        assert_eq!(reg.handles[0].name, "Instance");
        assert!(reg.handles[0].dispatchable);
    }

    #[test]
    fn collect_handle_non_dispatchable() {
        let mut reg = empty_registry();
        let ty = make_type(
            "VkBuffer",
            "handle",
            make_type_code(
                "VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkBuffer)",
                vec![TypeCodeMarkup::Name("VkBuffer".to_string())],
            ),
        );
        collect_handle(&ty, &mut reg);
        assert_eq!(reg.handles.len(), 1);
        assert_eq!(reg.handles[0].name, "Buffer");
        assert!(!reg.handles[0].dispatchable);
    }

    #[test]
    fn collect_handle_with_parent() {
        let mut reg = empty_registry();
        let mut ty = make_type(
            "VkBuffer",
            "handle",
            make_type_code(
                "VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkBuffer)",
                vec![TypeCodeMarkup::Name("VkBuffer".to_string())],
            ),
        );
        ty.parent = Some("VkDevice".to_string());
        collect_handle(&ty, &mut reg);
        assert_eq!(reg.handles[0].parent.as_deref(), Some("Device"));
    }

    #[test]
    fn collect_handle_prefers_ty_name() {
        let mut reg = empty_registry();
        let ty = make_type(
            "VkFence",
            "handle",
            make_type_code(
                "VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkFence)",
                vec![TypeCodeMarkup::Name("VkFence".to_string())],
            ),
        );
        collect_handle(&ty, &mut reg);
        assert_eq!(reg.handles[0].name, "Fence");
    }

    // -- collect_struct --------------------------------------------------------

    #[test]
    fn collect_struct_basic() {
        let mut reg = empty_registry();
        let ty = make_type(
            "VkExtent2D",
            "struct",
            TypeSpec::Members(vec![
                TypeMember::Definition(make_member_def(
                    "uint32_t width",
                    vec![
                        TypeMemberMarkup::Type("uint32_t".to_string()),
                        TypeMemberMarkup::Name("width".to_string()),
                    ],
                )),
                TypeMember::Definition(make_member_def(
                    "uint32_t height",
                    vec![
                        TypeMemberMarkup::Type("uint32_t".to_string()),
                        TypeMemberMarkup::Name("height".to_string()),
                    ],
                )),
            ]),
        );
        collect_struct(&ty, &mut reg, false);
        assert_eq!(reg.structs.len(), 1);
        assert_eq!(reg.structs[0].name, "Extent2D");
        assert_eq!(reg.structs[0].members.len(), 2);
        assert!(!reg.structs[0].is_union);
        assert!(!reg.structs[0].returned_only);
    }

    #[test]
    fn collect_struct_union() {
        let mut reg = empty_registry();
        let ty = make_type(
            "VkClearColorValue",
            "union",
            TypeSpec::Members(vec![TypeMember::Definition(make_member_def(
                "float float32[4]",
                vec![
                    TypeMemberMarkup::Type("float".to_string()),
                    TypeMemberMarkup::Name("float32".to_string()),
                ],
            ))]),
        );
        collect_struct(&ty, &mut reg, true);
        assert!(reg.structs[0].is_union);
    }

    #[test]
    fn collect_struct_returned_only() {
        let mut reg = empty_registry();
        let mut ty = make_type(
            "VkPhysicalDeviceProperties",
            "struct",
            TypeSpec::Members(vec![]),
        );
        ty.returnedonly = Some("true".to_string());
        collect_struct(&ty, &mut reg, false);
        assert!(reg.structs[0].returned_only);
    }

    #[test]
    fn collect_struct_extends_parsed() {
        let mut reg = empty_registry();
        let mut ty = make_type("VkFoo", "struct", TypeSpec::Members(vec![]));
        ty.structextends = Some("VkDeviceCreateInfo,VkInstanceCreateInfo".to_string());
        collect_struct(&ty, &mut reg, false);
        assert_eq!(
            reg.structs[0].extends,
            vec!["DeviceCreateInfo", "InstanceCreateInfo"]
        );
    }

    // -- collect_bitmask_type --------------------------------------------------

    #[test]
    fn collect_bitmask_type_32bit() {
        let mut reg = empty_registry();
        let mut meta = HashMap::new();
        let mut names = HashSet::new();
        let mut ty = make_type(
            "VkCullModeFlags",
            "bitmask",
            make_type_code("typedef VkFlags VkCullModeFlags;", vec![]),
        );
        ty.requires = Some("VkCullModeFlagBits".to_string());
        collect_bitmask_type(&ty, &mut meta, &mut names, &mut reg);
        assert!(names.contains("CullModeFlagBits"));
        assert_eq!(meta["CullModeFlagBits"], ("CullModeFlags".to_string(), 32));
    }

    #[test]
    fn collect_bitmask_type_64bit() {
        let mut reg = empty_registry();
        let mut meta = HashMap::new();
        let mut names = HashSet::new();
        let mut ty = make_type(
            "VkPipelineStageFlagBits2",
            "bitmask",
            make_type_code("typedef VkFlags64 VkPipelineStageFlagBits2;", vec![]),
        );
        ty.bitvalues = Some("VkPipelineStageFlagBits2".to_string());
        collect_bitmask_type(&ty, &mut meta, &mut names, &mut reg);
        let (_, bitwidth) = &meta["PipelineStageFlagBits2"];
        assert_eq!(*bitwidth, 64);
    }
}
