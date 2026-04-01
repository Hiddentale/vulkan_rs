# Usage Notes

Retrieves fault reports after a device loss (`ERROR_DEVICE_LOST`).
Returns a list of `DeviceFaultInfoKHR` structs describing what
went wrong, address faults, vendor-specific fault codes, etc.

`timeout` specifies how long to wait (in nanoseconds) for the
driver to collect fault data. Use `UINT64_MAX` to wait
indefinitely.

For raw debug data suitable for vendor tools, follow up with
`get_device_fault_debug_info_khr`.

Requires `VK_KHR_device_fault`.
