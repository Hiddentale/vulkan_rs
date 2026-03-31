//! Enum and bitmask variant collection from top-level enums, features, and extensions.

use std::collections::{HashMap, HashSet};

use vk_parse::{
    Enum, EnumSpec, EnumsChild, Extension, ExtensionChild, Feature, FeatureChild, InterfaceItem,
};

use super::{
    BitmaskBit, BitmaskValue, EnumValue, EnumVariant, compute_enum_offset, is_non_vulkan,
    parse_c_literal, parse_c_literal_u64, strip_vk,
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
            let val = compute_enum_offset(ext_num, *offset, !*dir);
            EnumValue::I32(val)
        }
        EnumSpec::Bitpos { bitpos, .. } => EnumValue::I32(1i32 << bitpos),
        EnumSpec::None => return None,
        _ => return None,
    };
    Some(EnumVariant {
        name: e.name.clone(),
        value,
        comment: e.comment.clone(),
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
            EnumValue::I32(compute_enum_offset(num, *offset, !*dir))
        }
        EnumSpec::Value { value, .. } => EnumValue::I32(parse_c_literal(value)),
        EnumSpec::Alias { alias, .. } => EnumValue::Alias(alias.clone()),
        EnumSpec::Bitpos { bitpos, .. } => EnumValue::I32(1i32 << bitpos),
        EnumSpec::None | _ => return None,
    };
    Some(EnumVariant {
        name: e.name.clone(),
        value,
        comment: e.comment.clone(),
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
            let val = compute_enum_offset(num, *offset, !*dir);
            BitmaskValue::Value(val as u64)
        }
        EnumSpec::None | _ => return None,
    };
    Some(BitmaskBit {
        name: e.name.clone(),
        value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_value_enum(name: &str, value: &str) -> Enum {
        let mut e = Enum::default();
        e.name = name.to_string();
        e.spec = EnumSpec::Value {
            value: value.to_string(),
            extends: None,
        };
        e
    }

    fn make_bitpos_enum(name: &str, bitpos: i64) -> Enum {
        let mut e = Enum::default();
        e.name = name.to_string();
        e.spec = EnumSpec::Bitpos {
            bitpos,
            extends: None,
        };
        e
    }

    fn make_alias_enum(name: &str, alias: &str) -> Enum {
        let mut e = Enum::default();
        e.name = name.to_string();
        e.spec = EnumSpec::Alias {
            alias: alias.to_string(),
            extends: None,
        };
        e
    }

    fn make_offset_enum(name: &str, offset: i64, extnumber: Option<i64>, dir: bool) -> Enum {
        let mut e = Enum::default();
        e.name = name.to_string();
        e.spec = EnumSpec::Offset {
            offset,
            extends: "VkResult".to_string(),
            extnumber,
            dir,
        };
        e
    }

    // -- parse_enum_variant ----------------------------------------------------

    #[test]
    fn enum_variant_value_parses_decimal() {
        let e = make_value_enum("VK_SUCCESS", "0");
        let v = parse_enum_variant(&e).expect("should parse value variant");
        assert_eq!(v.name, "VK_SUCCESS");
        assert!(matches!(v.value, EnumValue::I32(0)));
    }

    #[test]
    fn enum_variant_alias() {
        let e = make_alias_enum(
            "VK_COLORSPACE_SRGB_NONLINEAR_KHR",
            "VK_COLOR_SPACE_SRGB_NONLINEAR_KHR",
        );
        let v = parse_enum_variant(&e).expect("should parse alias variant");
        assert!(
            matches!(v.value, EnumValue::Alias(ref a) if a == "VK_COLOR_SPACE_SRGB_NONLINEAR_KHR")
        );
    }

    #[test]
    fn enum_variant_bitpos_shift() {
        let e = make_bitpos_enum("VK_QUEUE_GRAPHICS_BIT", 0);
        let v = parse_enum_variant(&e).expect("should parse bitpos variant");
        assert!(matches!(v.value, EnumValue::I32(1)));
    }

    #[test]
    fn enum_variant_bitpos_high() {
        let e = make_bitpos_enum("VK_SOME_BIT", 4);
        let v = parse_enum_variant(&e).expect("should parse bitpos=4");
        assert!(matches!(v.value, EnumValue::I32(16)));
    }

    #[test]
    fn enum_variant_offset_positive() {
        // dir=true → !dir=false → positive. Ext 1, offset 0 → BASE + 0 = 1_000_000_000
        let e = make_offset_enum("VK_SOMETHING", 0, Some(1), true);
        let v = parse_enum_variant(&e).expect("should parse offset variant");
        assert!(matches!(v.value, EnumValue::I32(1_000_000_000)));
    }

    #[test]
    fn enum_variant_offset_negative() {
        // dir=false → !dir=true → negative. Ext 1, offset 0 → -1_000_000_000
        let e = make_offset_enum("VK_ERROR_SOMETHING", 0, Some(1), false);
        let v = parse_enum_variant(&e).expect("should parse negative offset");
        assert!(matches!(v.value, EnumValue::I32(-1_000_000_000)));
    }

    #[test]
    fn enum_variant_offset_ext_number_2() {
        // dir=true → positive. Ext 2, offset 0 → BASE + 1*1000 = 1_000_001_000
        let e = make_offset_enum("VK_EXT2_THING", 0, Some(2), true);
        let v = parse_enum_variant(&e).expect("should parse ext_number=2");
        assert!(matches!(v.value, EnumValue::I32(1_000_001_000)));
    }

    #[test]
    fn enum_variant_none_spec_returns_none() {
        let mut e = Enum::default();
        e.name = "VK_UNUSED".to_string();
        e.spec = EnumSpec::None;
        assert!(parse_enum_variant(&e).is_none());
    }

    #[test]
    fn enum_variant_skips_non_vulkan_api() {
        let mut e = make_value_enum("VK_SC_THING", "5");
        e.api = Some("vulkansc".to_string());
        assert!(parse_enum_variant(&e).is_none());
    }

    #[test]
    fn enum_variant_preserves_comment() {
        let mut e = make_value_enum("VK_SUCCESS", "0");
        e.comment = Some("Command completed successfully".to_string());
        let v = parse_enum_variant(&e).expect("should parse");
        assert_eq!(v.comment.as_deref(), Some("Command completed successfully"));
    }

    // -- parse_bitmask_bit -----------------------------------------------------

    #[test]
    fn bitmask_bit_bitpos() {
        let e = make_bitpos_enum("VK_QUEUE_GRAPHICS_BIT", 0);
        let b = parse_bitmask_bit(&e).expect("should parse bitmask bitpos");
        assert!(matches!(b.value, BitmaskValue::Bitpos(0)));
    }

    #[test]
    fn bitmask_bit_bitpos_high() {
        let e = make_bitpos_enum("VK_SOME_BIT", 31);
        let b = parse_bitmask_bit(&e).expect("should parse bitpos=31");
        assert!(matches!(b.value, BitmaskValue::Bitpos(31)));
    }

    #[test]
    fn bitmask_bit_value() {
        let e = make_value_enum("VK_SOMETHING", "0x7FFFFFFF");
        let b = parse_bitmask_bit(&e).expect("should parse bitmask value");
        assert!(matches!(b.value, BitmaskValue::Value(0x7FFFFFFF)));
    }

    #[test]
    fn bitmask_bit_alias() {
        let e = make_alias_enum("VK_NEW_NAME", "VK_OLD_NAME");
        let b = parse_bitmask_bit(&e).expect("should parse bitmask alias");
        assert!(matches!(b.value, BitmaskValue::Alias(ref a) if a == "VK_OLD_NAME"));
    }

    #[test]
    fn bitmask_bit_skips_non_vulkan() {
        let mut e = make_bitpos_enum("VK_SC_BIT", 0);
        e.api = Some("vulkansc".to_string());
        assert!(parse_bitmask_bit(&e).is_none());
    }

    // -- collect_enums dispatch ------------------------------------------------

    fn make_enums_group(
        name: &str,
        kind: Option<&str>,
        children: Vec<EnumsChild>,
    ) -> vk_parse::Enums {
        let mut enums = vk_parse::Enums::default();
        enums.name = Some(name.to_string());
        enums.kind = kind.map(str::to_string);
        enums.children = children;
        enums
    }

    #[test]
    fn collect_enums_dispatches_to_enum_map() {
        let enums = make_enums_group(
            "VkResult",
            Some("enum"),
            vec![EnumsChild::Enum(make_value_enum("VK_SUCCESS", "0"))],
        );
        let mut enum_map = HashMap::new();
        let mut bitmask_map = HashMap::new();
        let bitmask_names = HashSet::new();

        collect_enums(&enums, &mut enum_map, &mut bitmask_map, &bitmask_names);

        assert!(enum_map.contains_key("Result"));
        assert_eq!(enum_map["Result"].len(), 1);
        assert!(bitmask_map.is_empty());
    }

    #[test]
    fn collect_enums_dispatches_to_bitmask_map() {
        let enums = make_enums_group(
            "VkQueueFlagBits",
            Some("bitmask"),
            vec![EnumsChild::Enum(make_bitpos_enum(
                "VK_QUEUE_GRAPHICS_BIT",
                0,
            ))],
        );
        let mut enum_map = HashMap::new();
        let mut bitmask_map = HashMap::new();
        let bitmask_names = HashSet::new();

        collect_enums(&enums, &mut enum_map, &mut bitmask_map, &bitmask_names);

        assert!(bitmask_map.contains_key("QueueFlagBits"));
        assert_eq!(bitmask_map["QueueFlagBits"].len(), 1);
        assert!(enum_map.is_empty());
    }

    #[test]
    fn collect_enums_skips_api_constants() {
        let enums = make_enums_group(
            "API Constants",
            None,
            vec![EnumsChild::Enum(make_value_enum("VK_TRUE", "1"))],
        );
        let mut enum_map = HashMap::new();
        let mut bitmask_map = HashMap::new();
        let bitmask_names = HashSet::new();

        collect_enums(&enums, &mut enum_map, &mut bitmask_map, &bitmask_names);

        assert!(enum_map.is_empty());
        assert!(bitmask_map.is_empty());
    }

    #[test]
    fn collect_enums_recognises_bitmask_by_name_set() {
        let enums = make_enums_group(
            "VkCustomFlagBits",
            Some("enum"),
            vec![EnumsChild::Enum(make_bitpos_enum("VK_CUSTOM_BIT", 2))],
        );
        let mut enum_map = HashMap::new();
        let mut bitmask_map = HashMap::new();
        let mut bitmask_names = HashSet::new();
        bitmask_names.insert("CustomFlagBits".to_string());

        collect_enums(&enums, &mut enum_map, &mut bitmask_map, &bitmask_names);

        assert!(bitmask_map.contains_key("CustomFlagBits"));
        assert!(enum_map.is_empty());
    }
}
