# Usage Notes

Destroys a descriptor pool and implicitly frees all descriptor sets
allocated from it. You do not need to free individual sets before
destroying the pool.

Ensure no command buffer that references any set from this pool is
still pending execution.
