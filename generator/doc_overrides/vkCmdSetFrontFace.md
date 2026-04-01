# Usage Notes

Dynamically sets which triangle winding order is considered
front-facing. Only takes effect if the pipeline was created with
`DYNAMIC_STATE_FRONT_FACE`.

Values: `FRONT_FACE_COUNTER_CLOCKWISE` (the Vulkan default) or
`FRONT_FACE_CLOCKWISE`.

Useful when rendering mirrored or reflected geometry where the
winding order is flipped, without needing a separate pipeline.

Core dynamic state in Vulkan 1.3.
