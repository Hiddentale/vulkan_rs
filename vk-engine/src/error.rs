use crate::vk;

/// Two-call enumerate pattern used by many Vulkan commands that return `VkResult`.
///
/// First call with null data pointer to get the count, allocate, then call
/// again to fill the buffer.
pub(crate) fn enumerate_two_call<T>(
    call: impl Fn(*mut u32, *mut T) -> vk::enums::Result,
) -> VkResult<Vec<T>> {
    let mut count = 0u32;
    check(call(&mut count, std::ptr::null_mut()))?;
    let mut data = Vec::with_capacity(count as usize);
    let result = call(&mut count, data.as_mut_ptr());
    check(result)?;
    // SAFETY: the Vulkan command wrote `count` elements into `data`'s spare capacity.
    unsafe { data.set_len(count as usize) };
    Ok(data)
}

/// Two-call fill pattern for Vulkan commands that return `void` (no `VkResult`).
///
/// Used by commands like `vkGetPhysicalDeviceQueueFamilyProperties` which write
/// directly into the output buffer without a result code.
pub(crate) fn fill_two_call<T>(call: impl Fn(*mut u32, *mut T)) -> Vec<T> {
    let mut count = 0u32;
    call(&mut count, std::ptr::null_mut());
    let mut data = Vec::with_capacity(count as usize);
    call(&mut count, data.as_mut_ptr());
    // SAFETY: the Vulkan command wrote `count` elements into `data`'s spare capacity.
    unsafe { data.set_len(count as usize) };
    data
}

/// Vulkan API result type.
///
/// The `Err` variant is any negative `vk::enums::Result` (an error code).
/// Non-negative codes (including `SUCCESS`, `INCOMPLETE`, `SUBOPTIMAL`)
/// are treated as success.
///
/// # Examples
///
/// ```
/// use vk_engine::VkResult;
/// use vk_engine::vk;
///
/// fn do_vulkan_work() -> VkResult<u32> {
///     // Simulate a successful Vulkan call.
///     Ok(42)
/// }
///
/// let result = do_vulkan_work();
/// assert!(result.is_ok());
/// ```
pub type VkResult<T> = std::result::Result<T, vk::enums::Result>;

/// Convert a raw `vk::enums::Result` into `VkResult<()>`.
///
/// Vulkan defines success codes as non-negative and error codes as negative.
/// Commands that need to distinguish specific success codes (e.g. `INCOMPLETE`
/// for enumeration) handle that explicitly after calling this.
pub(crate) fn check(result: vk::enums::Result) -> VkResult<()> {
    if result.as_raw() >= 0 {
        Ok(())
    } else {
        Err(result)
    }
}

/// Error returned when the Vulkan shared library cannot be loaded.
///
/// This is distinct from `vk::enums::Result`, it represents a failure to reach
/// the Vulkan API at all, not a Vulkan API error.
///
/// # Examples
///
/// ```
/// use vk_engine::LoadError;
///
/// let err = LoadError::MissingEntryPoint;
/// assert_eq!(
///     err.to_string(),
///     "vkGetInstanceProcAddr not found in Vulkan library",
/// );
/// ```
#[derive(Debug)]
pub enum LoadError {
    /// The Vulkan shared library could not be found or opened.
    Library(libloading::Error),
    /// `vkGetInstanceProcAddr` could not be resolved from the library.
    MissingEntryPoint,
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Library(e) => write!(f, "failed to load Vulkan library: {e}"),
            LoadError::MissingEntryPoint => {
                f.write_str("vkGetInstanceProcAddr not found in Vulkan library")
            }
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoadError::Library(e) => Some(e),
            LoadError::MissingEntryPoint => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_success_returns_ok() {
        assert!(check(vk::enums::Result::SUCCESS).is_ok());
    }

    #[test]
    fn check_negative_returns_err() {
        let result = check(vk::enums::Result::ERROR_OUT_OF_HOST_MEMORY);
        assert_eq!(result, Err(vk::enums::Result::ERROR_OUT_OF_HOST_MEMORY));
    }

    #[test]
    fn check_non_zero_success_codes_return_ok() {
        // INCOMPLETE and SUBOPTIMAL are non-negative success codes.
        assert!(check(vk::enums::Result::INCOMPLETE).is_ok());
        assert!(check(vk::enums::Result::SUBOPTIMAL).is_ok());
    }

    #[test]
    fn check_extension_error_codes_return_err() {
        // Extension error codes (promoted to core) must be negative.
        assert!(check(vk::enums::Result::ERROR_OUT_OF_POOL_MEMORY).is_err());
        assert!(check(vk::enums::Result::ERROR_SURFACE_LOST).is_err());
        assert!(check(vk::enums::Result::ERROR_VALIDATION_FAILED).is_err());
    }

    #[test]
    fn enumerate_two_call_returns_items() {
        // Simulate a Vulkan enumerate command that returns 3 u32 values.
        let result = enumerate_two_call(|count, data: *mut u32| {
            unsafe { *count = 3 };
            if !data.is_null() {
                unsafe {
                    *data = 10;
                    *data.add(1) = 20;
                    *data.add(2) = 30;
                }
            }
            vk::enums::Result::SUCCESS
        });
        assert_eq!(result.expect("should succeed"), vec![10u32, 20, 30]);
    }

    #[test]
    fn enumerate_two_call_returns_empty_on_zero_count() {
        let result = enumerate_two_call::<u32>(|count, _data| {
            unsafe { *count = 0 };
            vk::enums::Result::SUCCESS
        });
        assert!(result.expect("should succeed").is_empty());
    }

    #[test]
    fn enumerate_two_call_propagates_error() {
        let result =
            enumerate_two_call::<u32>(|_count, _data| vk::enums::Result::ERROR_OUT_OF_HOST_MEMORY);
        assert_eq!(
            result.unwrap_err(),
            vk::enums::Result::ERROR_OUT_OF_HOST_MEMORY
        );
    }

    #[test]
    fn fill_two_call_returns_items() {
        let result = fill_two_call(|count, data: *mut u64| {
            unsafe { *count = 2 };
            if !data.is_null() {
                unsafe {
                    *data = 42u64;
                    *data.add(1) = 99;
                }
            }
        });
        assert_eq!(result, vec![42u64, 99]);
    }

    #[test]
    fn fill_two_call_returns_empty_on_zero_count() {
        let result = fill_two_call::<u32>(|count, _data| {
            unsafe { *count = 0 };
        });
        assert!(result.is_empty());
    }

    #[test]
    fn load_error_source_library_returns_some() {
        let lib_err =
            unsafe { libloading::Library::new("nonexistent_vulkan_lib.dll") }.unwrap_err();
        let err = LoadError::Library(lib_err);
        assert!(
            std::error::Error::source(&err).is_some(),
            "Library variant should have a source"
        );
    }

    #[test]
    fn load_error_source_missing_entry_point_returns_none() {
        let err = LoadError::MissingEntryPoint;
        assert!(
            std::error::Error::source(&err).is_none(),
            "MissingEntryPoint should have no source"
        );
    }

    #[test]
    fn load_error_display_missing_entry_point() {
        let err = LoadError::MissingEntryPoint;
        assert_eq!(
            err.to_string(),
            "vkGetInstanceProcAddr not found in Vulkan library"
        );
    }

    #[test]
    fn load_error_display_library() {
        // Trigger a real libloading error by loading a nonexistent library.
        let lib_err =
            unsafe { libloading::Library::new("nonexistent_vulkan_lib.dll") }.unwrap_err();
        let err = LoadError::Library(lib_err);
        assert!(err.to_string().contains("failed to load Vulkan library"));
    }
}
