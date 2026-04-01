# Usage Notes

Destroys a render pass. All framebuffers and pipelines created with
this render pass must no longer be in use.

Render passes are typically long-lived, created once at startup
and destroyed during shutdown. Destroying a render pass does not
affect pipelines that were created with a compatible render pass
(same attachment count, formats, and sample counts).
