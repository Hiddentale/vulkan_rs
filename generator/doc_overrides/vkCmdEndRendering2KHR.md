# Usage Notes

Extended version of `cmd_end_rendering` (core 1.3) that accepts
an optional `RenderingEndInfoKHR` with pNext extensibility.

Ends the current dynamic rendering pass. If `p_rendering_end_info`
is `None`, behaves identically to `cmd_end_rendering`.

Provided by `VK_KHR_maintenance7`.
