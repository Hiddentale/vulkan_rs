//! Enum and bitmask variant collection from top-level enums, features, and extensions.

use std::collections::{HashMap, HashSet};

use vk_parse::{
    Enum, EnumSpec, EnumsChild, Extension, ExtensionChild, Feature, FeatureChild, InterfaceItem,
};

use super::{
    compute_enum_offset, is_non_vulkan, parse_c_literal, parse_c_literal_u64, strip_vk,
    BitmaskBit, BitmaskValue, EnumValue, EnumVariant,
};

pub(super) fn collect_enums(
    enums: &vk_parse::Enums,
    enum_map: &mut HashMap<String, Vec<EnumVariant>>,
    bitmask_map: &mut HashMap<String, Vec<BitmaskBit>>,
    bitmask_enum_names: &HashSet<String>,
) {
    let name = match enums.name {
        Some(ref n) => n.clone(),
        None => return,
    };

    if name == "API Constants" {
        return;
    }

    let stripped = strip_vk(&name);
    let is_bitmask =
        enums.kind.as_deref() == Some("bitmask") || bitmask_enum_names.contains(&stripped);

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

pub(super) fn collect_feature_enums(
    feature: &Feature,
    enum_map: &mut HashMap<String, Vec<EnumVariant>>,
    bitmask_map: &mut HashMap<String, Vec<BitmaskBit>>,
    bitmask_enum_names: &HashSet<String>,
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

pub(super) fn collect_extension_enums(
    ext: &Extension,
    enum_map: &mut HashMap<String, Vec<EnumVariant>>,
    bitmask_map: &mut HashMap<String, Vec<BitmaskBit>>,
    bitmask_enum_names: &HashSet<String>,
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
    bitmask_enum_names: &HashSet<String>,
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
        _ => return,
    };

    let is_bitmask = bitmask_enum_names.contains(&extends);

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
