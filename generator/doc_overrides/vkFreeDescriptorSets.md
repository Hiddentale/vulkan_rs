# Usage Notes

Returns individual descriptor sets back to their parent pool. The
pool must have been created with `FREE_DESCRIPTOR_SET`, without
that flag this call is invalid.

For most applications, resetting the entire pool with
`reset_descriptor_pool` is simpler and faster than tracking and
freeing individual sets. Use `free_descriptor_sets` only when you
need fine-grained lifetime control over specific sets.

Freed sets must not be referenced by any pending command buffer.
