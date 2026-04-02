# Usage Notes

Enumerates the fragment shading rates supported by the physical
device. Each entry reports a fragment size (e.g., 2x2, 4x4) and
which sample counts are compatible with it.

The results are sorted from largest to smallest fragment size.
1x1 (full-rate shading) is always supported.

Use these results to validate fragment sizes passed to
`cmd_set_fragment_shading_rate_khr` or configured in a shading
rate attachment.
