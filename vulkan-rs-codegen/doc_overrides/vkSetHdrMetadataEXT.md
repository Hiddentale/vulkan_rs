# Usage Notes

Sets HDR metadata (mastering display color volume, content light
levels) for one or more swapchains. The compositor uses this to
tone-map content appropriately for the connected display.

Call whenever the content characteristics change (e.g., switching
between SDR UI and HDR scene rendering).

Requires `VK_EXT_hdr_metadata`.
