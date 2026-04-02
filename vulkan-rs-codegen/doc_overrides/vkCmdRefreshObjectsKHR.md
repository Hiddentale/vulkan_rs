# Usage Notes

Refreshes a set of Vulkan objects managed by a refreshable object
type, resetting their internal state. Used in safety-critical
Vulkan SC environments to periodically refresh objects and detect
hardware faults.

Not relevant for desktop Vulkan. This is part of the
`VK_KHR_object_refresh` extension used in automotive and embedded
safety-critical environments.
