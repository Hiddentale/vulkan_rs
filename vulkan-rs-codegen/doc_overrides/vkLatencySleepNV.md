# Usage Notes

Sleeps the calling thread until the optimal time to begin the
next frame, as determined by the NVIDIA Reflex low-latency
system. Reduces input-to-display latency by preventing the CPU
from running too far ahead of the GPU.

Requires `VK_NV_low_latency2`.
