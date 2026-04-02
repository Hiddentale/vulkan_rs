# Usage Notes

Retrieves diagnostic information after a device-lost error. Call
once with a null info pointer to query the fault counts, then
again with allocated structures to retrieve the fault details.

The returned data is vendor-specific and intended for crash
reporting and post-mortem debugging.

Requires `VK_EXT_device_fault`.
