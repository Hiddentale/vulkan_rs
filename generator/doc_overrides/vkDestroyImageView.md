# Usage Notes

Destroys an image view. The view must not be referenced by any
pending GPU work or by any framebuffer that is still in use.

Destroy image views **before** the underlying image. Destroy any
framebuffers that reference the view before destroying the view
itself.
