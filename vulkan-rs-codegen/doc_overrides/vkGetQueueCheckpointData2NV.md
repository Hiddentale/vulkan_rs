# Usage Notes

Extended version of `get_queue_checkpoint_data_nv` that returns
pipeline stage information alongside the checkpoint markers.
Use for finer-grained post-mortem debugging after device loss.

Requires `VK_NV_device_diagnostic_checkpoints` +
`VK_KHR_synchronization2`.
