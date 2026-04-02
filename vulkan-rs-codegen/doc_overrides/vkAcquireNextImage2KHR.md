# Usage Notes

Extended version of `acquire_next_image_khr` that takes an
`AcquireNextImageInfoKHR` struct with pNext support.

The key addition is `device_mask` for device groups (multi-GPU),
specifying which physical devices the acquired image will be used
on.

For single-GPU usage, this is functionally identical to
`acquire_next_image_khr`.
