# Usage Notes

Queries tile properties (tile dimensions and extents) for a
framebuffer on Qualcomm tile-based GPUs. Uses the two-call
idiom. Useful for optimising rendering to match the hardware
tile layout.

Requires `VK_QCOM_tile_properties`.
