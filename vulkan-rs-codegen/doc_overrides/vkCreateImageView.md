# Usage Notes

An image view selects a subset of an image's subresources and
reinterprets them for a specific use (sampling, color attachment, etc.).

Common pitfalls:

- **Aspect mask**: use `COLOR` for color formats, `DEPTH` and/or
  `STENCIL` for depth/stencil formats. Getting this wrong causes
  validation errors that are not always obvious.
- **Format compatibility**: the view format must be compatible with the
  image's format. Using `IMAGE_CREATE_MUTABLE_FORMAT` on the image
  relaxes this to any format in the same size-compatibility class.
- **View type vs image type**: a 2D image can back a `VIEW_TYPE_2D` or
  `VIEW_TYPE_2D_ARRAY`. A 3D image cannot be viewed as 2D without
  `VK_EXT_image_2d_view_of_3d`.

Destroy with `destroy_image_view` when no longer needed. Destroy image
views *before* destroying the underlying image.
