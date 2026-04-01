# Usage Notes

Sets depth bias parameters dynamically. Only takes effect if the
pipeline was created with `DYNAMIC_STATE_DEPTH_BIAS` and depth bias
is enabled in the rasterisation state.

Depth bias adds a computed offset to each fragment's depth value
before the depth test. The primary use case is **shadow mapping**, 
biasing shadow caster geometry slightly away from the light prevents
self-shadowing artifacts (shadow acne).

The final bias is computed as:

```text
bias = constant_factor * r + slope_factor * max_slope
```

where `r` is the minimum resolvable depth difference and `max_slope`
is the maximum depth slope of the triangle.

**`depth_bias_clamp`** limits the maximum bias value. Requires the
`depth_bias_clamp` device feature. A clamp of 0.0 disables clamping.

Typical shadow map values: `constant_factor` = 1.25,
`slope_factor` = 1.75, `clamp` = 0.0. Tune per scene.
