//! Parses vk.xml into categorized intermediate types for code generation.

use std::collections::HashMap;
use vk_parse::{
    self, Command, CommandDefinition, CommandParam, Enum, EnumSpec, EnumsChild, Extension,
    ExtensionChild, Feature, FeatureChild, InterfaceItem, Registry, RegistryChild, Type,
    TypeMember, TypeMemberDefinition, TypeMemberMarkup, TypeSpec,
};

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
    pub aliases: HashMap<String, String>,
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

#[derive(Debug)]
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
        aliases: HashMap::new(),
        base_types: HashMap::new(),
    };

    // Collect enum/bitmask groups by name so we can append extension values later.
    let mut enum_map: HashMap<String, Vec<EnumVariant>> = HashMap::new();
    let mut bitmask_map: HashMap<String, Vec<BitmaskBit>> = HashMap::new();
    let mut bitmask_meta: HashMap<String, (String, u32)> = HashMap::new(); // enum_name → (flags_name, bitwidth)

    // Track which names are bitmask enums vs value enums.
    let mut bitmask_enum_names: HashMap<String, ()> = HashMap::new();

    // First pass: collect types, commands, top-level enums.
    for child in &registry.0 {
        match child {
            RegistryChild::Types(types) => {
                collect_types(
                    &types.children,
                    &mut reg,
                    &mut bitmask_meta,
                    &mut bitmask_enum_names,
                );
            }
            RegistryChild::Enums(enums) => {
                collect_enums(enums, &mut enum_map, &mut bitmask_map, &bitmask_enum_names);
            }
            RegistryChild::Commands(commands) => {
                collect_commands(&commands.children, &mut reg);
            }
            _ => {}
        }
    }

    // Second pass: gather enum/bitmask extensions from features and extensions.
    for child in &registry.0 {
        match child {
            RegistryChild::Feature(feature) => {
                collect_feature_enums(
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
                    collect_extension_enums(
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

    // Build final enum/bitmask defs.
    for (name, variants) in enum_map {
        reg.enums.push(EnumDef { name, variants });
    }

    for (enum_name, bits) in bitmask_map {
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
            collect_constants(enums, &mut reg);
        }
    }

    // Funcpointers: vk-parse doesn't parse proto/param for these, so we
    // parse them from the raw XML text in a separate pass.
    let xml_text = std::fs::read_to_string(path).expect("failed to read vk.xml");
    reg.func_pointers = parse_func_pointers(&xml_text);

    // Collect platforms and extensions.
    collect_platforms(&registry, &mut reg);
    collect_extensions(&registry, &mut reg);

    // Stamp provenance (which feature/extension introduced each type/command).
    stamp_provenance(&registry, &mut reg);

    reg
}

// ---------------------------------------------------------------------------
// Type collection
// ---------------------------------------------------------------------------

fn collect_types(
    types: &[vk_parse::TypesChild],
    reg: &mut VkRegistry,
    bitmask_meta: &mut HashMap<String, (String, u32)>,
    bitmask_enum_names: &mut HashMap<String, ()>,
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
            reg.aliases.insert(name.clone(), alias.clone());
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

    // Name may be in ty.name or inside the code markup.
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

    // Fixed-size array from code like `[4]`, `[3]`.
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
    bitmask_enum_names: &mut HashMap<String, ()>,
    reg: &mut VkRegistry,
) {
    // Name may be in ty.name (attribute) or inline in the code markup.
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

    // Detect 64-bit bitmasks from the underlying type (VkFlags vs VkFlags64).
    let bitwidth = match &ty.spec {
        TypeSpec::Code(code) if code.code.contains("VkFlags64") => 64,
        _ => 32,
    };

    // The `requires` or `bitvalues` field points to the FlagBits enum.
    // Also check code markup for the referenced type when bitvalues is absent.
    let enum_name = ty
        .requires
        .as_ref()
        .or(ty.bitvalues.as_ref())
        .map(|n| strip_vk(n));

    if let Some(ref enum_name) = enum_name {
        bitmask_meta.insert(enum_name.clone(), (flags_name.clone(), bitwidth));
        bitmask_enum_names.insert(enum_name.clone(), ());
    }

    // Also record as alias: FooFlags → FooFlagBits for cross-referencing.
    if let Some(enum_name) = enum_name {
        reg.aliases.insert(flags_name, enum_name);
    }
}

fn collect_basetype(ty: &Type, reg: &mut VkRegistry) {
    if let (Some(name), TypeSpec::Code(code)) = (&ty.name, &ty.spec) {
        reg.base_types.insert(name.clone(), code.code.clone());
    }
}

// ---------------------------------------------------------------------------
// Enum/bitmask collection
// ---------------------------------------------------------------------------

fn collect_enums(
    enums: &vk_parse::Enums,
    enum_map: &mut HashMap<String, Vec<EnumVariant>>,
    bitmask_map: &mut HashMap<String, Vec<BitmaskBit>>,
    bitmask_enum_names: &HashMap<String, ()>,
) {
    let name = match enums.name {
        Some(ref n) => n.clone(),
        None => return,
    };

    // Skip API Constants — handled separately.
    if name == "API Constants" {
        return;
    }

    let stripped = strip_vk(&name);
    let is_bitmask =
        enums.kind.as_deref() == Some("bitmask") || bitmask_enum_names.contains_key(&stripped);

    if is_bitmask {
        let bits: Vec<BitmaskBit> = enums
            .children
            .iter()
            .filter_map(|c| match c {
                EnumsChild::Enum(e) => parse_bitmask_bit(e),
                _ => None,
            })
            .collect();
        bitmask_map.entry(stripped).or_default().extend(bits);
    } else {
        let variants: Vec<EnumVariant> = enums
            .children
            .iter()
            .filter_map(|c| match c {
                EnumsChild::Enum(e) => parse_enum_variant(e),
                _ => None,
            })
            .collect();
        enum_map.entry(stripped).or_default().extend(variants);
    }
}

fn parse_enum_variant(e: &Enum) -> Option<EnumVariant> {
    if is_non_vulkan(e) {
        return None;
    }
    let value = match &e.spec {
        EnumSpec::Value { value, .. } => EnumValue::I32(parse_c_literal(value)),
        EnumSpec::Alias { alias, .. } => EnumValue::Alias(alias.clone()),
        EnumSpec::Offset {
            offset,
            extnumber,
            dir,
            ..
        } => {
            let ext_num = extnumber.unwrap_or(0);
            let val = compute_enum_offset(ext_num, *offset, *dir);
            EnumValue::I32(val)
        }
        EnumSpec::Bitpos { bitpos, .. } => EnumValue::I32(1i32 << bitpos),
        EnumSpec::None => return None,
        _ => return None,
    };
    Some(EnumVariant {
        name: e.name.clone(),
        value,
    })
}

fn parse_bitmask_bit(e: &Enum) -> Option<BitmaskBit> {
    if is_non_vulkan(e) {
        return None;
    }
    let value = match &e.spec {
        EnumSpec::Bitpos { bitpos, .. } => BitmaskValue::Bitpos(*bitpos as u32),
        EnumSpec::Value { value, .. } => BitmaskValue::Value(parse_c_literal_u64(value)),
        EnumSpec::Alias { alias, .. } => BitmaskValue::Alias(alias.clone()),
        _ => return None,
    };
    Some(BitmaskBit {
        name: e.name.clone(),
        value,
    })
}

// ---------------------------------------------------------------------------
// Feature/extension enum gathering
// ---------------------------------------------------------------------------

fn collect_feature_enums(
    feature: &Feature,
    enum_map: &mut HashMap<String, Vec<EnumVariant>>,
    bitmask_map: &mut HashMap<String, Vec<BitmaskBit>>,
    bitmask_enum_names: &HashMap<String, ()>,
) {
    for child in &feature.children {
        let items = match child {
            FeatureChild::Require { items, .. } => items,
            _ => continue,
        };
        for item in items {
            if let InterfaceItem::Enum(e) = item {
                add_extension_enum(e, None, enum_map, bitmask_map, bitmask_enum_names);
            }
        }
    }
}

fn collect_extension_enums(
    ext: &Extension,
    enum_map: &mut HashMap<String, Vec<EnumVariant>>,
    bitmask_map: &mut HashMap<String, Vec<BitmaskBit>>,
    bitmask_enum_names: &HashMap<String, ()>,
) {
    let ext_number = ext.number;
    for child in &ext.children {
        let items = match child {
            ExtensionChild::Require { items, .. } => items,
            _ => continue,
        };
        for item in items {
            if let InterfaceItem::Enum(e) = item {
                add_extension_enum(e, ext_number, enum_map, bitmask_map, bitmask_enum_names);
            }
        }
    }
}

fn add_extension_enum(
    e: &Enum,
    ext_number: Option<i64>,
    enum_map: &mut HashMap<String, Vec<EnumVariant>>,
    bitmask_map: &mut HashMap<String, Vec<BitmaskBit>>,
    bitmask_enum_names: &HashMap<String, ()>,
) {
    if is_non_vulkan(e) {
        return;
    }

    let extends = match &e.spec {
        EnumSpec::Offset { extends, .. }
        | EnumSpec::Bitpos {
            extends: Some(extends),
            ..
        }
        | EnumSpec::Value {
            extends: Some(extends),
            ..
        }
        | EnumSpec::Alias {
            extends: Some(extends),
            ..
        } => strip_vk(extends),
        _ => return, // No extends = not extending an enum/bitmask group.
    };

    let is_bitmask = bitmask_enum_names.contains_key(&extends);

    if is_bitmask {
        if let Some(bit) = parse_extension_bitmask_bit(e, ext_number) {
            bitmask_map.entry(extends).or_default().push(bit);
        }
    } else if let Some(variant) = parse_extension_enum_variant(e, ext_number) {
        enum_map.entry(extends).or_default().push(variant);
    }
}

fn parse_extension_enum_variant(e: &Enum, ext_number: Option<i64>) -> Option<EnumVariant> {
    let value = match &e.spec {
        EnumSpec::Offset {
            offset,
            extnumber,
            dir,
            ..
        } => {
            let num = extnumber.or(ext_number).unwrap_or(0);
            EnumValue::I32(compute_enum_offset(num, *offset, *dir))
        }
        EnumSpec::Value { value, .. } => EnumValue::I32(parse_c_literal(value)),
        EnumSpec::Alias { alias, .. } => EnumValue::Alias(alias.clone()),
        EnumSpec::Bitpos { bitpos, .. } => EnumValue::I32(1i32 << bitpos),
        EnumSpec::None | _ => return None,
    };
    Some(EnumVariant {
        name: e.name.clone(),
        value,
    })
}

fn parse_extension_bitmask_bit(e: &Enum, ext_number: Option<i64>) -> Option<BitmaskBit> {
    let value = match &e.spec {
        EnumSpec::Bitpos { bitpos, .. } => BitmaskValue::Bitpos(*bitpos as u32),
        EnumSpec::Value { value, .. } => BitmaskValue::Value(parse_c_literal_u64(value)),
        EnumSpec::Alias { alias, .. } => BitmaskValue::Alias(alias.clone()),
        EnumSpec::Offset {
            offset,
            extnumber,
            dir,
            ..
        } => {
            // Rare: some extensions add numeric values to bitmask enums.
            let num = extnumber.or(ext_number).unwrap_or(0);
            let val = compute_enum_offset(num, *offset, *dir);
            BitmaskValue::Value(val as u64)
        }
        EnumSpec::None | _ => return None,
    };
    Some(BitmaskBit {
        name: e.name.clone(),
        value,
    })
}

// ---------------------------------------------------------------------------
// Command collection
// ---------------------------------------------------------------------------

fn collect_commands(commands: &[Command], reg: &mut VkRegistry) {
    for cmd in commands {
        match cmd {
            Command::Alias { name, alias } => {
                reg.aliases.insert(strip_vk(name), strip_vk(alias));
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

// ---------------------------------------------------------------------------
// Constants collection
// ---------------------------------------------------------------------------

fn collect_constants(enums: &vk_parse::Enums, reg: &mut VkRegistry) {
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
        Some(a) => !a.contains("vulkan"),
        None => false,
    }
}

fn is_vulkan_extension(ext: &Extension) -> bool {
    match ext.supported.as_deref() {
        Some(s) => s.contains("vulkan"),
        None => true,
    }
}

// ---------------------------------------------------------------------------
// Provenance stamping
// ---------------------------------------------------------------------------

fn stamp_provenance(registry: &Registry, reg: &mut VkRegistry) {
    // Build a map: type/command name → provider string.
    let mut provider_map: HashMap<String, String> = HashMap::new();

    for child in &registry.0 {
        match child {
            RegistryChild::Feature(feature) => {
                let provider = feature.name.clone();
                for fc in &feature.children {
                    let items = match fc {
                        FeatureChild::Require { items, .. } => items,
                        _ => continue,
                    };
                    for item in items {
                        match item {
                            InterfaceItem::Type { name, .. } => {
                                provider_map
                                    .entry(strip_vk(name))
                                    .or_insert(provider.clone());
                            }
                            InterfaceItem::Command { name, .. } => {
                                provider_map.entry(name.clone()).or_insert(provider.clone());
                            }
                            _ => {}
                        }
                    }
                }
            }
            RegistryChild::Extensions(extensions) => {
                for ext in &extensions.children {
                    if !is_vulkan_extension(ext) {
                        continue;
                    }
                    for ec in &ext.children {
                        let items = match ec {
                            ExtensionChild::Require { items, .. } => items,
                            _ => continue,
                        };
                        for item in items {
                            match item {
                                InterfaceItem::Type { name, .. } => {
                                    provider_map
                                        .entry(strip_vk(name))
                                        .or_insert(ext.name.clone());
                                }
                                InterfaceItem::Command { name, .. } => {
                                    provider_map.entry(name.clone()).or_insert(ext.name.clone());
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // Apply to parsed types.
    for h in &mut reg.handles {
        h.provided_by = provider_map.get(&h.name).cloned();
    }
    for s in &mut reg.structs {
        s.provided_by = provider_map.get(&s.name).cloned();
    }
    for c in &mut reg.commands {
        c.provided_by = provider_map.get(&c.name).cloned();
    }
}

// ---------------------------------------------------------------------------
// Platform and extension collection
// ---------------------------------------------------------------------------

fn collect_platforms(registry: &Registry, reg: &mut VkRegistry) {
    for child in &registry.0 {
        if let RegistryChild::Platforms(platforms) = child {
            for p in &platforms.children {
                reg.platforms.push(PlatformDef {
                    name: p.name.clone(),
                    protect: p.protect.clone(),
                });
            }
        }
    }
}

fn collect_extensions(registry: &Registry, reg: &mut VkRegistry) {
    for child in &registry.0 {
        let extensions = match child {
            RegistryChild::Extensions(exts) => &exts.children,
            _ => continue,
        };
        for ext in extensions {
            if !is_vulkan_extension(ext) {
                continue;
            }

            let mut ext_items = Vec::new();
            for child in &ext.children {
                let req_items = match child {
                    ExtensionChild::Require { items, .. } => items,
                    _ => continue,
                };
                for item in req_items {
                    match item {
                        InterfaceItem::Type { name, .. } => {
                            ext_items.push(ExtensionItem::Type(name.clone()));
                        }
                        InterfaceItem::Command { name, .. } => {
                            ext_items.push(ExtensionItem::Command(name.clone()));
                        }
                        _ => {}
                    }
                }
            }

            reg.extensions.push(ExtensionDef {
                name: ext.name.clone(),
                number: ext.number.unwrap_or(0),
                ext_type: ext.ext_type.clone(),
                platform: ext.platform.clone(),
                depends: ext.depends.clone(),
                promoted_to: ext.promotedto.clone(),
                deprecated_by: ext.deprecatedby.clone(),
                supported: ext.supported.clone().unwrap_or_default(),
                items: ext_items,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Funcpointer parsing (from raw XML, since vk-parse skips proto/param)
// ---------------------------------------------------------------------------

/// Extract content between `<tag>` and `</tag>` (non-greedy, first match).
fn extract_tag<'a>(xml: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);
    let start = xml.find(&open)?;
    let after_open = &xml[start..];
    let content_start = after_open.find('>')? + 1;
    let content = &xml[start + content_start..];
    let end = content.find(&close)?;
    Some(&content[..end])
}

fn parse_funcpointer_xml(block: &str) -> Option<FuncPointerDef> {
    let proto = extract_tag(block, "proto")?;
    let name = extract_tag(proto, "name")?.to_string();
    let return_type = extract_tag(proto, "type")?.to_string();
    let is_return_pointer = proto.contains('*');

    let mut params = Vec::new();
    let mut search = block;
    while let Some(param_start) = search.find("<param") {
        let rest = &search[param_start..];
        let param_end = match rest.find("</param>") {
            Some(e) => e + "</param>".len(),
            None => break,
        };
        let param_xml = &rest[..param_end];

        if let (Some(type_name), Some(param_name)) = (
            extract_tag(param_xml, "type"),
            extract_tag(param_xml, "name"),
        ) {
            let code = param_xml;
            let pointer_count = code.matches('*').count();
            let is_const = code.contains("const");

            params.push(ParamDef {
                name: param_name.to_string(),
                type_name: type_name.to_string(),
                is_pointer: pointer_count >= 1,
                is_const,
                is_double_pointer: pointer_count >= 2,
                array_size: None,
                optional: false,
                len: None,
                extern_sync: None,
            });
        }

        search = &search[param_start + param_end..];
    }

    Some(FuncPointerDef {
        name,
        return_type,
        is_return_pointer,
        params,
    })
}

pub fn parse_func_pointers(xml_text: &str) -> Vec<FuncPointerDef> {
    let mut result = Vec::new();

    // Split on funcpointer markers and extract each block.
    let marker = "category=\"funcpointer\"";
    let mut rest = xml_text;
    while let Some(marker_pos) = rest.find(marker) {
        // Walk back to find the opening `<type` for this block.
        let before = &rest[..marker_pos];
        let type_start = match before.rfind("<type") {
            Some(s) => s,
            None => {
                rest = &rest[marker_pos + marker.len()..];
                continue;
            }
        };

        let from_block = &rest[type_start..];

        // Find the outermost </type> by counting nested <type> / </type> pairs.
        if let Some(block_end) = find_closing_type_tag(from_block) {
            let block = &from_block[..block_end];
            if let Some(fp) = parse_funcpointer_xml(block) {
                result.push(fp);
            }
        }

        rest = &rest[marker_pos + marker.len()..];
    }

    result
}

/// Find the position after the outermost `</type>` tag, handling nested `<type>`.
fn find_closing_type_tag(xml: &str) -> Option<usize> {
    let mut depth = 0i32;
    let mut pos = 0;
    while pos < xml.len() {
        if xml[pos..].starts_with("</type>") {
            depth -= 1;
            if depth == 0 {
                return Some(pos + "</type>".len());
            }
            pos += "</type>".len();
        } else if xml[pos..].starts_with("<type") {
            // Check it's actually a tag open (followed by space, > or /)
            let after = pos + "<type".len();
            if after < xml.len() {
                let ch = xml.as_bytes()[after];
                if ch == b' ' || ch == b'>' || ch == b'/' {
                    depth += 1;
                }
            }
            pos += 1;
        } else {
            pos += 1;
        }
    }
    None
}
