//! Emits `#[repr(transparent)]` enum newtypes with associated constants.

use std::collections::HashSet;

use heck::ToShoutySnakeCase;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::parse::{EnumDef, EnumValue, EnumVariant, VkRegistry};

/// Emit all enum types.
pub fn emit_enums(registry: &VkRegistry) -> TokenStream {
    let enums: Vec<TokenStream> = registry.enums.iter().map(emit_enum).collect();
    quote! { #(#enums)* }
}

fn emit_enum(def: &EnumDef) -> TokenStream {
    let name = format_ident!("{}", &def.name);
    let vk_name = format!("Vk{}", &def.name);
    let prefix = enum_variant_prefix(&def.name);

    let mut seen = HashSet::new();
    let constants: Vec<TokenStream> = def
        .variants
        .iter()
        .filter_map(|v| {
            let rust_name = strip_variant_prefix(&v.name, &prefix)?;
            if !seen.insert(rust_name) {
                return None;
            }
            emit_variant(v, &prefix)
        })
        .collect();

    let mut seen_debug = HashSet::new();
    let debug_arms: Vec<TokenStream> = def
        .variants
        .iter()
        .filter_map(|v| {
            let rust_name = strip_variant_prefix(&v.name, &prefix)?;
            if !seen_debug.insert(rust_name) {
                return None;
            }
            emit_debug_arm(v, &prefix)
        })
        .collect();

    quote! {
        #[repr(transparent)]
        #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
        #[doc(alias = #vk_name)]
        pub struct #name(i32);

        impl #name {
            #(#constants)*

            #[inline]
            pub const fn from_raw(value: i32) -> Self { Self(value) }

            #[inline]
            pub const fn as_raw(self) -> i32 { self.0 }
        }

        impl core::fmt::Debug for #name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                match self.0 {
                    #(#debug_arms)*
                    other => write!(f, "{}({})", stringify!(#name), other),
                }
            }
        }
    }
}

fn emit_variant(variant: &EnumVariant, prefix: &str) -> Option<TokenStream> {
    let rust_name = strip_variant_prefix(&variant.name, prefix)?;
    let ident = format_ident!("{}", rust_name);

    match &variant.value {
        EnumValue::I32(val) => Some(quote! {
            pub const #ident: Self = Self(#val);
        }),
        EnumValue::Alias(alias) => {
            let alias_name = strip_variant_prefix(alias, prefix)?;
            // Skip self-referential aliases (variant and target strip to the same name).
            if alias_name == rust_name {
                return None;
            }
            let alias_ident = format_ident!("{}", alias_name);
            Some(quote! {
                pub const #ident: Self = Self::#alias_ident;
            })
        }
    }
}

fn emit_debug_arm(variant: &EnumVariant, prefix: &str) -> Option<TokenStream> {
    let rust_name = strip_variant_prefix(&variant.name, prefix)?;
    match &variant.value {
        EnumValue::I32(val) => {
            let label = rust_name.to_string();
            Some(quote! { #val => f.write_str(#label), })
        }
        EnumValue::Alias(_) => None, // skip aliases in debug to avoid duplicate arms
    }
}

// ---------------------------------------------------------------------------
// Name stripping
// ---------------------------------------------------------------------------

/// Compute the SHOUTY_SNAKE prefix to strip from C variant names.
///
/// `Format` → `VK_FORMAT_`, `Result` → `VK_RESULT_`,
/// `DebugReportObjectTypeEXT` → `VK_DEBUG_REPORT_OBJECT_TYPE_`.
///
/// Extension suffixes (KHR, EXT, etc.) are removed from the prefix
/// so both core and extension variants match.
pub fn enum_variant_prefix(rust_type_name: &str) -> String {
    let base = strip_extension_suffix(rust_type_name);
    format!("VK_{}_", base.to_shouty_snake_case())
}

/// Strip a C variant name like `VK_FORMAT_R8G8B8A8_SRGB` given prefix `VK_FORMAT_`.
/// Returns the Rust constant name like `R8G8B8A8_SRGB`.
///
/// Also strips trailing extension suffixes from the result if present.
pub fn strip_variant_prefix(c_name: &str, prefix: &str) -> Option<String> {
    let stripped = if let Some(s) = c_name.strip_prefix(prefix) {
        s
    } else {
        // Some variants don't match the expected prefix (e.g. VK_RESULT_MAX_ENUM).
        // Skip sentinel values.
        if c_name.contains("_MAX_ENUM") {
            return None;
        }
        // Try without the full prefix — fall back to stripping just `VK_`.
        c_name.strip_prefix("VK_")?
    };

    if stripped.is_empty() || stripped == "MAX_ENUM" {
        return None;
    }

    // Strip trailing extension tag from the variant name if present.
    let result = strip_extension_suffix(stripped);
    if result.is_empty() {
        return None;
    }

    // Ensure it starts with a letter or underscore (valid Rust ident).
    // Numeric-leading names get a leading underscore.
    if result.starts_with(|c: char| c.is_ascii_digit()) {
        Some(format!("_{result}"))
    } else {
        Some(result.to_string())
    }
}

pub const EXTENSION_SUFFIXES: &[&str] = &[
    "KHR", "EXT", "NV", "AMD", "INTEL", "ARM", "QCOM", "HUAWEI", "MESA", "VALVE", "GOOGLE",
    "FUCHSIA", "GGP", "MVK", "NN", "NVX", "AMDX", "SEC", "MSFT", "IMG", "LUNARG", "QNX", "ANDROID",
    "KDAB", "OHOS",
];

pub fn strip_extension_suffix(name: &str) -> String {
    let mut s = name.to_string();

    // For SHOUTY_SNAKE names, check `_KHR`, `_EXT`, etc.
    for suffix in EXTENSION_SUFFIXES {
        let snake_suffix = format!("_{suffix}");
        if s.ends_with(&snake_suffix) {
            s.truncate(s.len() - snake_suffix.len());
            break;
        }
    }

    // For PascalCase names, check trailing `KHR`, `EXT`, etc.
    for suffix in EXTENSION_SUFFIXES {
        if s.ends_with(suffix) {
            s.truncate(s.len() - suffix.len());
            break;
        }
    }

    s
}

/// Parses a TokenStream as a Rust file, panicking with a clear message on failure.
#[cfg(test)]
fn assert_valid_rust(tokens: &TokenStream) {
    syn::parse2::<syn::File>(tokens.clone())
        .unwrap_or_else(|e| panic!("generated code is not valid Rust: {e}\n\n{tokens}"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::{EnumDef, EnumValue, EnumVariant};

    fn make_enum(name: &str, variants: Vec<(&str, i32)>) -> EnumDef {
        EnumDef {
            name: name.to_string(),
            variants: variants
                .into_iter()
                .map(|(vname, val)| EnumVariant {
                    name: vname.to_string(),
                    value: EnumValue::I32(val),
                })
                .collect(),
        }
    }

    #[test]
    fn enum_emits_valid_rust() {
        let def = make_enum(
            "Format",
            vec![("VK_FORMAT_UNDEFINED", 0), ("VK_FORMAT_R8G8B8A8_SRGB", 43)],
        );
        let tokens = emit_enum(&def);
        assert_valid_rust(&tokens);
    }

    #[test]
    fn enum_has_expected_constants() {
        let def = make_enum(
            "Format",
            vec![("VK_FORMAT_UNDEFINED", 0), ("VK_FORMAT_R8G8B8A8_SRGB", 43)],
        );
        let code = emit_enum(&def).to_string();
        assert!(code.contains("pub const UNDEFINED"), "missing UNDEFINED");
        assert!(
            code.contains("pub const R8G8B8A8_SRGB"),
            "missing R8G8B8A8_SRGB"
        );
    }

    #[test]
    fn enum_has_from_raw_and_as_raw() {
        let def = make_enum("Result", vec![("VK_RESULT_SUCCESS", 0)]);
        let code = emit_enum(&def).to_string();
        assert!(code.contains("fn from_raw"));
        assert!(code.contains("fn as_raw"));
    }

    #[test]
    fn enum_has_debug_impl_with_named_arms() {
        let def = make_enum(
            "Format",
            vec![("VK_FORMAT_UNDEFINED", 0), ("VK_FORMAT_R8_UNORM", 9)],
        );
        let code = emit_enum(&def).to_string();
        assert!(code.contains("UNDEFINED"), "debug should print UNDEFINED");
        assert!(code.contains("R8_UNORM"), "debug should print R8_UNORM");
    }

    #[test]
    fn enum_deduplicates_extension_variants() {
        let def = EnumDef {
            name: "ColorSpaceKHR".to_string(),
            variants: vec![
                EnumVariant {
                    name: "VK_COLOR_SPACE_SRGB_NONLINEAR_KHR".to_string(),
                    value: EnumValue::I32(0),
                },
                EnumVariant {
                    name: "VK_COLORSPACE_SRGB_NONLINEAR_KHR".to_string(),
                    value: EnumValue::Alias("VK_COLOR_SPACE_SRGB_NONLINEAR_KHR".to_string()),
                },
            ],
        };
        let tokens = emit_enum(&def);
        assert_valid_rust(&tokens);
    }

    #[test]
    fn enum_skips_self_referential_alias() {
        let def = EnumDef {
            name: "PresentModeKHR".to_string(),
            variants: vec![
                EnumVariant {
                    name: "VK_PRESENT_MODE_FIFO_KHR".to_string(),
                    value: EnumValue::I32(2),
                },
                // This alias strips to the same name as the original after suffix removal.
                EnumVariant {
                    name: "VK_PRESENT_MODE_FIFO_LATEST_READY_KHR".to_string(),
                    value: EnumValue::I32(1000361000),
                },
                EnumVariant {
                    name: "VK_PRESENT_MODE_FIFO_LATEST_READY_EXT".to_string(),
                    value: EnumValue::Alias("VK_PRESENT_MODE_FIFO_LATEST_READY_KHR".to_string()),
                },
            ],
        };
        let tokens = emit_enum(&def);
        assert_valid_rust(&tokens);
    }

    #[test]
    fn enum_has_doc_alias() {
        let def = make_enum("Format", vec![("VK_FORMAT_UNDEFINED", 0)]);
        let code = emit_enum(&def).to_string();
        assert!(
            code.contains("VkFormat"),
            "expected doc(alias) with VkFormat"
        );
    }

    #[test]
    fn test_enum_variant_prefix() {
        assert_eq!(enum_variant_prefix("Format"), "VK_FORMAT_");
        assert_eq!(enum_variant_prefix("Result"), "VK_RESULT_");
        assert_eq!(
            enum_variant_prefix("DebugReportObjectTypeEXT"),
            "VK_DEBUG_REPORT_OBJECT_TYPE_"
        );
    }

    #[test]
    fn test_strip_variant_prefix() {
        let prefix = "VK_FORMAT_";
        assert_eq!(
            strip_variant_prefix("VK_FORMAT_R8G8B8A8_SRGB", prefix),
            Some("R8G8B8A8_SRGB".to_string())
        );
        assert_eq!(
            strip_variant_prefix("VK_FORMAT_UNDEFINED", prefix),
            Some("UNDEFINED".to_string())
        );
        assert_eq!(strip_variant_prefix("VK_FORMAT_MAX_ENUM", prefix), None);
    }

    #[test]
    fn test_strip_variant_extension_suffix() {
        let prefix = "VK_COLOR_SPACE_";
        assert_eq!(
            strip_variant_prefix("VK_COLOR_SPACE_SRGB_NONLINEAR_KHR", prefix),
            Some("SRGB_NONLINEAR".to_string())
        );
    }

    #[test]
    fn test_numeric_leading_variant() {
        let prefix = "VK_FORMAT_";
        let result = strip_variant_prefix("VK_FORMAT_2PACK16", prefix);
        assert_eq!(result, Some("_2PACK16".to_string()));
    }
}
