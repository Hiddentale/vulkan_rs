# Usage Notes

Creates a sampler that controls how shaders read image data:
filtering, addressing, mip level selection, and anisotropy.

**Common configurations**:

- **Nearest/point**: `MIN_FILTER_NEAREST`, `MAG_FILTER_NEAREST`.
  No interpolation, pixel art, data textures, or shadow map
  comparison.
- **Bilinear**: `MIN_FILTER_LINEAR`, `MAG_FILTER_LINEAR`,
  `MIPMAP_MODE_NEAREST`. Smooth within a mip level but snaps
  between levels.
- **Trilinear**: same as bilinear but with `MIPMAP_MODE_LINEAR`.
  Smooth transitions between mip levels. The default choice for
  most 3D textures.
- **Anisotropic**: enable `anisotropy_enable` and set
  `max_anisotropy` (commonly 4–16). Improves quality at oblique
  viewing angles at a small GPU cost.

**Address modes** (`REPEAT`, `MIRRORED_REPEAT`, `CLAMP_TO_EDGE`,
`CLAMP_TO_BORDER`) control what happens when UVs go outside [0, 1].

Samplers are immutable after creation and can be shared across any
number of descriptor sets. Most applications need only a handful of
samplers.
