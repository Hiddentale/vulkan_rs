# Usage Notes

Retrieves a queue handle for a queue created with specific flags.
This is the Vulkan 1.1 version of `get_device_queue` that supports
`DeviceQueueInfo2` with queue creation flags.

Use this instead of `get_device_queue` when you created queues with
non-zero `DeviceQueueCreateFlags` (e.g. `PROTECTED` for protected
content processing). For queues created without flags, both
`get_device_queue` and `get_device_queue2` work.
