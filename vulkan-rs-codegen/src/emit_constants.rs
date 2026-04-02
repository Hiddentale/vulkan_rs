//! Emits `pub const` items for Vulkan API constants.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::parse::{ConstantDef, VkRegistry};

/// Emit all API constants.
pub fn emit_constants(registry: &VkRegistry) -> TokenStream {
    let constants: Vec<TokenStream> = registry
        .constants
        .iter()
        .filter_map(emit_constant)
        .collect();
    quote! { #(#constants)* }
}

fn emit_constant(def: &ConstantDef) -> Option<TokenStream> {
    let rust_name = def.name.strip_prefix("VK_").unwrap_or(&def.name);
    let ident = format_ident!("{}", rust_name);
    let vk_name = &def.name;
    let (ty_tokens, val_tokens) = resolve_constant_type_and_value(def)?;

    let comment_doc: Vec<TokenStream> = def
        .comment
        .as_deref()
        .map(|c| {
            let sanitized = c.replace("<<", "`").replace(">>", "`");
            vec![quote! { #[doc = #sanitized] }]
        })
        .unwrap_or_default();

    Some(quote! {
        #(#comment_doc)*
        #[doc(alias = #vk_name)]
        pub const #ident: #ty_tokens = #val_tokens;
    })
}

fn resolve_constant_type_and_value(def: &ConstantDef) -> Option<(TokenStream, TokenStream)> {
    let value = def.value.trim();
    let c_type = def.ty.as_deref().unwrap_or("");

    // Use the explicit type from vk.xml when available.
    match c_type {
        "uint64_t" => {
            let val = parse_c_value_u64(value)?;
            Some((quote! { u64 }, val))
        }
        "uint32_t" => {
            let val = parse_c_value_u32(value)?;
            Some((quote! { u32 }, val))
        }
        "float" => {
            let val = parse_c_float(value)?;
            Some((quote! { f32 }, val))
        }
        _ => {
            // Fallback: infer from the value string.
            if value.contains("ULL") {
                let val = parse_c_value_u64(value)?;
                Some((quote! { u64 }, val))
            } else if value.contains('U') || value.contains("0x") {
                let val = parse_c_value_u32(value)?;
                Some((quote! { u32 }, val))
            } else if value.contains('.') || value.contains('f') || value.contains('F') {
                let val = parse_c_float(value)?;
                Some((quote! { f32 }, val))
            } else {
                // Default to i32 for plain integers.
                let v: i32 = value.parse().ok()?;
                let lit = proc_macro2::Literal::i32_suffixed(v);
                Some((quote! { i32 }, quote! { #lit }))
            }
        }
    }
}

/// Parse a C u32 value like `256`, `(~0U)`, `(~1U)`, `0x7FFFFFFFU`.
fn parse_c_value_u32(s: &str) -> Option<TokenStream> {
    let s = s.trim();

    // Bitwise NOT: (~0U), (~1U), (~2U)
    if let Some(inner) = extract_bitwise_not(s) {
        let inner = inner.trim_end_matches('U').trim_end_matches('u');
        let n: u32 = inner.parse().ok()?;
        let lit = proc_macro2::Literal::u32_suffixed(n);
        return Some(quote! { !#lit });
    }

    // Hex
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        let hex = hex.trim_end_matches(|c: char| !c.is_ascii_hexdigit());
        let val = u32::from_str_radix(hex, 16).ok()?;
        let lit = proc_macro2::Literal::u32_suffixed(val);
        return Some(quote! { #lit });
    }

    // Plain integer
    let num: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
    let val: u32 = num.parse().ok()?;
    let lit = proc_macro2::Literal::u32_suffixed(val);
    Some(quote! { #lit })
}

/// Parse a C u64 value like `(~0ULL)`, `0xFFFFFFFFFFFFFFFFULL`.
fn parse_c_value_u64(s: &str) -> Option<TokenStream> {
    let s = s.trim();

    if let Some(inner) = extract_bitwise_not(s) {
        let inner = inner
            .trim_end_matches("ULL")
            .trim_end_matches("ull")
            .trim_end_matches('U')
            .trim_end_matches('u');
        let n: u64 = inner.parse().ok()?;
        let lit = proc_macro2::Literal::u64_suffixed(n);
        return Some(quote! { !#lit });
    }

    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        let hex: String = hex.chars().take_while(|c| c.is_ascii_hexdigit()).collect();
        let val = u64::from_str_radix(&hex, 16).ok()?;
        let lit = proc_macro2::Literal::u64_suffixed(val);
        return Some(quote! { #lit });
    }

    let num: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
    let val: u64 = num.parse().ok()?;
    let lit = proc_macro2::Literal::u64_suffixed(val);
    Some(quote! { #lit })
}

/// Parse a C float like `1000.0F`, `0.25f`.
fn parse_c_float(s: &str) -> Option<TokenStream> {
    let s = s.trim().trim_end_matches(['f', 'F']);
    let val: f32 = s.parse().ok()?;
    let lit = proc_macro2::Literal::f32_suffixed(val);
    Some(quote! { #lit })
}

/// Extract the inner value from `(~NNN)` patterns.
fn extract_bitwise_not(s: &str) -> Option<&str> {
    let s = s.trim();
    let inner = s.strip_prefix('(')?.strip_suffix(')')?.trim();
    let inner = inner.strip_prefix('~')?;
    Some(inner)
}

#[cfg(test)]
fn assert_valid_rust(tokens: &TokenStream) {
    syn::parse2::<syn::File>(tokens.clone())
        .unwrap_or_else(|e| panic!("generated code is not valid Rust: {e}\n\n{tokens}"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::ConstantDef;

    fn make_constant(name: &str, value: &str, ty: Option<&str>) -> ConstantDef {
        ConstantDef {
            name: name.to_string(),
            value: value.to_string(),
            ty: ty.map(str::to_string),
            comment: None,
        }
    }

    #[test]
    fn constant_u32_plain() {
        let def = make_constant("VK_MAX_MEMORY_TYPES", "32", Some("uint32_t"));
        let tokens = emit_constant(&def).expect("should emit u32 constant");
        assert_valid_rust(&tokens);
        let code = tokens.to_string();
        assert!(code.contains("MAX_MEMORY_TYPES"));
        assert!(code.contains("u32"));
    }

    #[test]
    fn constant_u32_bitwise_not() {
        let def = make_constant("VK_ATTACHMENT_UNUSED", "(~0U)", Some("uint32_t"));
        let tokens = emit_constant(&def).expect("should emit bitwise-not constant");
        assert_valid_rust(&tokens);
        let code = tokens.to_string();
        assert!(code.contains("ATTACHMENT_UNUSED"));
        assert!(code.contains("!"));
    }

    #[test]
    fn constant_u64() {
        let def = make_constant("VK_WHOLE_SIZE", "(~0ULL)", Some("uint64_t"));
        let tokens = emit_constant(&def).expect("should emit u64 constant");
        assert_valid_rust(&tokens);
        let code = tokens.to_string();
        assert!(code.contains("WHOLE_SIZE"));
        assert!(code.contains("u64"));
    }

    #[test]
    fn constant_float() {
        let def = make_constant("VK_LOD_CLAMP_NONE", "1000.0F", Some("float"));
        let tokens = emit_constant(&def).expect("should emit float constant");
        assert_valid_rust(&tokens);
        let code = tokens.to_string();
        assert!(code.contains("LOD_CLAMP_NONE"));
        assert!(code.contains("f32"));
    }

    #[test]
    fn constant_has_doc_alias() {
        let def = make_constant("VK_TRUE", "1", Some("uint32_t"));
        let code = emit_constant(&def)
            .expect("should emit constant")
            .to_string();
        assert!(code.contains("VK_TRUE"), "expected doc(alias)");
    }

    #[test]
    fn test_parse_c_value_u32_plain() {
        let tokens = parse_c_value_u32("256").expect("should parse plain u32");
        assert_eq!(tokens.to_string(), "256u32");
    }

    #[test]
    fn test_parse_c_value_u32_bitwise_not() {
        let tokens = parse_c_value_u32("(~0U)").expect("should parse bitwise-not u32");
        assert_eq!(tokens.to_string(), "! 0u32");
    }

    #[test]
    fn test_parse_c_value_u64_bitwise_not() {
        let tokens = parse_c_value_u64("(~0ULL)").expect("should parse bitwise-not u64");
        assert_eq!(tokens.to_string(), "! 0u64");
    }

    #[test]
    fn test_parse_c_float() {
        let tokens = parse_c_float("1000.0F").expect("should parse float");
        assert_eq!(tokens.to_string(), "1000f32");
    }

    #[test]
    fn test_parse_c_float_lowercase() {
        let tokens = parse_c_float("0.25f").expect("should parse lowercase float");
        assert_eq!(tokens.to_string(), "0.25f32");
    }

    // --- Type inference fallback (no explicit ty) ---

    #[test]
    fn infer_u64_from_ull_suffix() {
        let def = make_constant("VK_WHOLE_SIZE", "(~0ULL)", None);
        let tokens = emit_constant(&def).expect("should infer u64 from ULL");
        assert_valid_rust(&tokens);
        let code = tokens.to_string();
        assert!(code.contains("u64"), "expected u64 type");
    }

    #[test]
    fn infer_u32_from_u_suffix() {
        let def = make_constant("VK_QUEUE_FAMILY_IGNORED", "(~0U)", None);
        let tokens = emit_constant(&def).expect("should infer u32 from U");
        assert_valid_rust(&tokens);
        let code = tokens.to_string();
        assert!(code.contains("u32"), "expected u32 type");
    }

    #[test]
    fn infer_u32_from_hex_prefix() {
        let def = make_constant("VK_SOME_HEX", "0xFF", None);
        let tokens = emit_constant(&def).expect("should infer u32 from 0x prefix");
        assert_valid_rust(&tokens);
        let code = tokens.to_string();
        assert!(code.contains("u32"), "expected u32 type");
        assert!(code.contains("255u32"));
    }

    #[test]
    fn infer_f32_from_dot() {
        let def = make_constant("VK_SOME_FLOAT", "3.14", None);
        let tokens = emit_constant(&def).expect("should infer f32 from decimal point");
        assert_valid_rust(&tokens);
        let code = tokens.to_string();
        assert!(code.contains("f32"), "expected f32 type");
    }

    #[test]
    fn infer_f32_from_f_suffix() {
        let def = make_constant("VK_SOME_FLOAT", "42f", None);
        let tokens = emit_constant(&def).expect("should infer f32 from f suffix");
        assert_valid_rust(&tokens);
        let code = tokens.to_string();
        assert!(code.contains("f32"), "expected f32 type");
    }

    #[test]
    fn infer_i32_for_plain_integer() {
        let def = make_constant("VK_SUBPASS_EXTERNAL", "256", None);
        let tokens = emit_constant(&def).expect("should default to i32");
        assert_valid_rust(&tokens);
        let code = tokens.to_string();
        assert!(code.contains("i32"), "expected i32 type");
        assert!(code.contains("256i32"));
    }

    #[test]
    fn infer_i32_for_negative_integer() {
        let def = make_constant("VK_NEG", "-1", None);
        let tokens = emit_constant(&def).expect("should parse negative i32");
        assert_valid_rust(&tokens);
        let code = tokens.to_string();
        assert!(code.contains("i32"), "expected i32 type");
    }

    // --- Hex parsing ---

    #[test]
    fn parse_u32_hex() {
        let tokens = parse_c_value_u32("0x7FFFFFFFU").expect("should parse hex u32");
        assert_eq!(tokens.to_string(), "2147483647u32");
    }

    #[test]
    fn parse_u32_hex_uppercase_prefix() {
        let tokens = parse_c_value_u32("0XFF").expect("should parse 0X prefix");
        assert_eq!(tokens.to_string(), "255u32");
    }

    #[test]
    fn parse_u64_hex() {
        let tokens = parse_c_value_u64("0xFFFFFFFFFFFFFFFFULL").expect("should parse hex u64");
        assert_eq!(tokens.to_string(), "18446744073709551615u64");
    }

    #[test]
    fn parse_u64_plain() {
        let tokens = parse_c_value_u64("42").expect("should parse plain u64");
        assert_eq!(tokens.to_string(), "42u64");
    }

    #[test]
    fn parse_u64_bitwise_not_lowercase() {
        let tokens = parse_c_value_u64("(~0ull)").expect("should parse lowercase ull");
        assert_eq!(tokens.to_string(), "! 0u64");
    }

    // --- Comment handling ---

    #[test]
    fn constant_with_comment() {
        let def = ConstantDef {
            name: "VK_TRUE".to_string(),
            value: "1".to_string(),
            ty: Some("uint32_t".to_string()),
            comment: Some("Boolean true value".to_string()),
        };
        let tokens = emit_constant(&def).expect("should emit constant with comment");
        assert_valid_rust(&tokens);
        let code = tokens.to_string();
        assert!(code.contains("Boolean true value"));
    }

    #[test]
    fn constant_comment_sanitizes_angle_brackets() {
        let def = ConstantDef {
            name: "VK_LIMIT".to_string(),
            value: "256".to_string(),
            ty: Some("uint32_t".to_string()),
            comment: Some("See <<features>> for details".to_string()),
        };
        let tokens = emit_constant(&def).expect("should emit constant");
        let code = tokens.to_string();
        assert!(!code.contains("<<"), "angle brackets should be sanitized");
        assert!(code.contains("`features`"));
    }

    // --- Name handling ---

    #[test]
    fn name_without_vk_prefix() {
        let def = make_constant("CUSTOM_CONSTANT", "1", Some("uint32_t"));
        let tokens = emit_constant(&def).expect("should handle name without VK_ prefix");
        assert_valid_rust(&tokens);
        let code = tokens.to_string();
        assert!(code.contains("CUSTOM_CONSTANT"));
    }

    // --- Top-level emit_constants ---

    #[test]
    fn emit_constants_empty_registry() {
        let registry = VkRegistry {
            handles: vec![],
            structs: vec![],
            enums: vec![],
            bitmasks: vec![],
            commands: vec![],
            constants: vec![],
            func_pointers: vec![],
            extensions: vec![],
            platforms: vec![],
            aliases: vec![],
            base_types: Default::default(),
        };
        let tokens = emit_constants(&registry);
        assert_eq!(tokens.to_string(), "");
    }

    #[test]
    fn emit_constants_multiple() {
        let registry = VkRegistry {
            handles: vec![],
            structs: vec![],
            enums: vec![],
            bitmasks: vec![],
            commands: vec![],
            constants: vec![
                make_constant("VK_TRUE", "1", Some("uint32_t")),
                make_constant("VK_FALSE", "0", Some("uint32_t")),
                make_constant("VK_WHOLE_SIZE", "(~0ULL)", Some("uint64_t")),
            ],
            func_pointers: vec![],
            extensions: vec![],
            platforms: vec![],
            aliases: vec![],
            base_types: Default::default(),
        };
        let tokens = emit_constants(&registry);
        assert_valid_rust(&tokens);
        let code = tokens.to_string();
        assert!(code.contains("TRUE"));
        assert!(code.contains("FALSE"));
        assert!(code.contains("WHOLE_SIZE"));
    }

    // --- extract_bitwise_not edge cases ---

    #[test]
    fn extract_bitwise_not_returns_none_without_parens() {
        assert!(extract_bitwise_not("~0U").is_none());
    }

    #[test]
    fn extract_bitwise_not_returns_none_without_tilde() {
        assert!(extract_bitwise_not("(0U)").is_none());
    }

    #[test]
    fn extract_bitwise_not_with_whitespace() {
        assert_eq!(extract_bitwise_not("( ~0U )").unwrap(), "0U");
    }
}
