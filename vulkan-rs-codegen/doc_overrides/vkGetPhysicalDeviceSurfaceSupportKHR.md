# Usage Notes

Checks whether a queue family on a physical device supports
presentation to a surface. Not all queue families can present, 
a graphics queue that supports rendering may not support
presentation on all platforms.

Call this for each queue family when selecting your present queue.
Often the graphics queue family also supports presentation, but
this is not guaranteed.

Returns `VK_TRUE` if the queue family can present to the surface.
