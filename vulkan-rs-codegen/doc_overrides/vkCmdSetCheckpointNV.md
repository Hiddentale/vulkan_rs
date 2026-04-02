# Usage Notes

Inserts a checkpoint marker into the command buffer for
diagnostic purposes. If the device is lost, the most recently
executed checkpoint can be retrieved with
`get_queue_checkpoint_data_nv` to identify which commands
completed before the failure.

Requires `VK_NV_device_diagnostic_checkpoints`.
