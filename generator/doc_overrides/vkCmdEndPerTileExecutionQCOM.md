# Usage Notes

Ends a per-tile execution region started by
`cmd_begin_per_tile_execution_qcom`. After this call, the
command buffer returns to normal (non-tile-local) recording.

Requires `VK_QCOM_tile_shading`.
