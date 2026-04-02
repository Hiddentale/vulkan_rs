# Usage Notes

Creates a sampler YCBCR conversion object that describes how to
convert YCBCR-encoded image data to RGB during sampling. Required
for multi-planar formats like `G8_B8_R8_3PLANE_420_UNORM` commonly
used in video decoding and camera capture.

The conversion parameters specify:

- **Format**: the multi-planar format being converted.
- **YCBCR model**: `RGB_IDENTITY`, `YCBCR_IDENTITY`,
  `YCBCR_709`, `YCBCR_601`, `YCBCR_2020`.
- **Range**: `ITU_FULL` or `ITU_NARROW`.
- **Chroma location**: where subsampled chroma samples are located
  relative to luma samples.
- **Chroma filter**: `NEAREST` or `LINEAR` for chroma upsampling.

The conversion object is attached to a sampler via
`SamplerYcbcrConversionInfo` in the pNext chain, and that sampler
must be used as an immutable sampler in the descriptor set layout.
