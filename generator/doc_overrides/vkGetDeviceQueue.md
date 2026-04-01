# Usage Notes

Retrieves a queue handle for a queue that was requested at device
creation time. The `queue_family_index` and `queue_index` must match
a family and index that was included in the `DeviceCreateInfo`'s
`queue_create_infos`.

Queue handles are implicitly owned by the device, they do not need
to be destroyed and become invalid when the device is destroyed.

Queues retrieved this way have no special flags. If you created
queues with `DeviceQueueCreateFlags` (e.g. protected queues), use
`get_device_queue2` instead.

It is common to retrieve queues once after device creation and store
them for the lifetime of the device.
