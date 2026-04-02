# Usage Notes

Registers a fence to be signaled when a device event occurs.
`DeviceEventInfoEXT` specifies the event type (e.g.,
`DISPLAY_HOTPLUG`).

Returns a fence that will be signaled when the event fires.

Requires `VK_EXT_display_control`.
