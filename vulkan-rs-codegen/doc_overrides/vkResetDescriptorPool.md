# Usage Notes

Recycles all descriptor sets allocated from this pool back to the
pool, without destroying the pool itself. After a reset, all
previously allocated sets are invalid and must not be used.

This is the fastest way to reclaim descriptor sets, much cheaper
than freeing them individually. Ideal for the per-frame pool pattern
where you allocate fresh sets every frame and reset the pool at the
start of the next frame.

No command buffer that references any set from this pool may be
pending execution when you reset.
