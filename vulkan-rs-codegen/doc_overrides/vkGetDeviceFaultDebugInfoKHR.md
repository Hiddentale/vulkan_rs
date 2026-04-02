# Usage Notes

Retrieves debug information after a device fault (GPU crash or
hang). The returned `DeviceFaultDebugInfoKHR` contains
vendor-specific binary data that can be passed to GPU vendor
diagnostic tools.

Call `get_device_fault_reports_khr` first to get the high-level
fault reports; this call provides the raw debug blob.

Requires `VK_KHR_device_fault`.
