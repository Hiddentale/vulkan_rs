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

    Some(quote! {
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

    let num: String = s
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
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
        let tokens = emit_constant(&def).unwrap();
        assert_valid_rust(&tokens);
        let code = tokens.to_string();
        assert!(code.contains("MAX_MEMORY_TYPES"));
        assert!(code.contains("u32"));
    }

    #[test]
    fn constant_u32_bitwise_not() {
        let def = make_constant("VK_ATTACHMENT_UNUSED", "(~0U)", Some("uint32_t"));
        let tokens = emit_constant(&def).unwrap();
        assert_valid_rust(&tokens);
        let code = tokens.to_string();
        assert!(code.contains("ATTACHMENT_UNUSED"));
        assert!(code.contains("!"));
    }

    #[test]
    fn constant_u64() {
        let def = make_constant("VK_WHOLE_SIZE", "(~0ULL)", Some("uint64_t"));
        let tokens = emit_constant(&def).unwrap();
        assert_valid_rust(&tokens);
        let code = tokens.to_string();
        assert!(code.contains("WHOLE_SIZE"));
        assert!(code.contains("u64"));
    }

    #[test]
    fn constant_float() {
        let def = make_constant("VK_LOD_CLAMP_NONE", "1000.0F", Some("float"));
        let tokens = emit_constant(&def).unwrap();
        assert_valid_rust(&tokens);
        let code = tokens.to_string();
        assert!(code.contains("LOD_CLAMP_NONE"));
        assert!(code.contains("f32"));
    }

    #[test]
    fn constant_has_doc_alias() {
        let def = make_constant("VK_TRUE", "1", Some("uint32_t"));
        let code = emit_constant(&def).unwrap().to_string();
        assert!(code.contains("VK_TRUE"), "expected doc(alias)");
    }

    #[test]
    fn test_parse_c_value_u32_plain() {
        let tokens = parse_c_value_u32("256").unwrap();
        assert_eq!(tokens.to_string(), "256u32");
    }

    #[test]
    fn test_parse_c_value_u32_bitwise_not() {
        let tokens = parse_c_value_u32("(~0U)").unwrap();
        assert_eq!(tokens.to_string(), "! 0u32");
    }

    #[test]
    fn test_parse_c_value_u64_bitwise_not() {
        let tokens = parse_c_value_u64("(~0ULL)").unwrap();
        assert_eq!(tokens.to_string(), "! 0u64");
    }

    #[test]
    fn test_parse_c_float() {
        let tokens = parse_c_float("1000.0F").unwrap();
        assert_eq!(tokens.to_string(), "1000f32");
    }

    #[test]
    fn test_parse_c_float_lowercase() {
        let tokens = parse_c_float("0.25f").unwrap();
        assert_eq!(tokens.to_string(), "0.25f32");
    }
}
