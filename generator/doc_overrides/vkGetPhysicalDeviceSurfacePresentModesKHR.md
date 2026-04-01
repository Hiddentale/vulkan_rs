# Usage Notes

Returns the list of supported present modes for a surface.

**Present modes**:

- `FIFO`: vsync. Guaranteed to be supported on all implementations.
  Frames are queued and presented at the display refresh rate.
- `FIFO_RELAXED`: vsync with late frame allowance. If a frame
  arrives late, it is presented immediately (may cause tearing).
- `MAILBOX`: triple buffering. The driver keeps only the latest
  frame in the queue, lower latency than FIFO with no tearing.
- `IMMEDIATE`: no vsync. Frames are presented as soon as possible.
  Lowest latency but may cause visible tearing.

Common strategy: prefer `MAILBOX` for low-latency rendering, fall
back to `FIFO` if unavailable.
