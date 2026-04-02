# Usage Notes

Dispatches compute work within a per-tile execution region on
Qualcomm tile-based GPUs. Must be called between
`cmd_begin_per_tile_execution_qcom` and
`cmd_end_per_tile_execution_qcom`.

Requires `VK_QCOM_tile_shading`.
