# Usage Notes

Dynamically sets the sample mask. The sample mask is ANDed with
the coverage mask to determine which samples are written. Bits
that are zero disable the corresponding sample.

The slice length must match `ceil(rasterizationSamples / 32)`.

Provided by `VK_EXT_extended_dynamic_state3`.
