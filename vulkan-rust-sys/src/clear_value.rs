//! Safe constructors for [`ClearValue`] and [`ClearColorValue`] unions.

use crate::structs::{ClearColorValue, ClearDepthStencilValue, ClearValue};

impl ClearColorValue {
    /// Create from RGBA float values.
    #[inline]
    pub const fn from_float32(rgba: [f32; 4]) -> Self {
        Self { float32: rgba }
    }

    /// Create from RGBA signed integer values.
    #[inline]
    pub const fn from_int32(rgba: [i32; 4]) -> Self {
        Self { int32: rgba }
    }

    /// Create from RGBA unsigned integer values.
    #[inline]
    pub const fn from_uint32(rgba: [u32; 4]) -> Self {
        Self { uint32: rgba }
    }
}

impl ClearValue {
    /// Create a color clear value from RGBA floats.
    #[inline]
    pub const fn color_f32(rgba: [f32; 4]) -> Self {
        Self {
            color: ClearColorValue::from_float32(rgba),
        }
    }

    /// Create a color clear value from RGBA signed integers.
    #[inline]
    pub const fn color_i32(rgba: [i32; 4]) -> Self {
        Self {
            color: ClearColorValue::from_int32(rgba),
        }
    }

    /// Create a color clear value from RGBA unsigned integers.
    #[inline]
    pub const fn color_u32(rgba: [u32; 4]) -> Self {
        Self {
            color: ClearColorValue::from_uint32(rgba),
        }
    }

    /// Create a depth/stencil clear value.
    #[inline]
    pub const fn depth_stencil(depth: f32, stencil: u32) -> Self {
        Self {
            depth_stencil: ClearDepthStencilValue { depth, stencil },
        }
    }
}
