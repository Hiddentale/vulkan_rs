# Usage Notes

Creates an event, a fine-grained synchronisation primitive that
can be signaled and waited on from both the host (CPU) and the
device (GPU).

Events are most useful for split barriers: signal an event at one
point in a command buffer, do other work, then wait on it later.
This gives the GPU more flexibility to overlap execution compared to
a single `cmd_pipeline_barrier`.

**Host-side usage**: `set_event` and `reset_event` signal and reset
from the CPU. `get_event_status` polls the current state. However,
host-signaled events cannot be reliably waited on by the GPU on all
implementations, use them primarily for GPU–GPU sync within a
queue.

Events are lightweight and cheap to create.
