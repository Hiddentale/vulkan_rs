# Usage Notes

Writes a small amount of data inline into a buffer from the command
stream. The data is embedded directly in the command buffer, no
staging buffer needed.

**Size limit**: the data size must be ≤ 65536 bytes and a multiple
of 4. For larger uploads, use `cmd_copy_buffer` with a staging
buffer instead.

This is convenient for small per-frame updates (e.g. a uniform
buffer with a single matrix) but should not be used for bulk data, 
it inflates command buffer size and the data is uploaded through the
command stream, which is slower than a DMA transfer.

Must be recorded outside a render pass. The destination buffer must
have `BUFFER_USAGE_TRANSFER_DST`.
