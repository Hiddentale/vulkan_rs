# Usage Notes

Dynamically sets the primitive topology. Only takes effect if the
pipeline was created with `DYNAMIC_STATE_PRIMITIVE_TOPOLOGY`.

Values include `POINT_LIST`, `LINE_LIST`, `LINE_STRIP`,
`TRIANGLE_LIST`, `TRIANGLE_STRIP`, `TRIANGLE_FAN`,
`LINE_LIST_WITH_ADJACENCY`, `PATCH_LIST`, etc.

The dynamic topology must be in the same topology class as the
pipeline's static topology (e.g. you can switch between
`TRIANGLE_LIST` and `TRIANGLE_STRIP` since both are triangle
topologies, but not between `TRIANGLE_LIST` and `LINE_LIST`).

Core dynamic state in Vulkan 1.3.
