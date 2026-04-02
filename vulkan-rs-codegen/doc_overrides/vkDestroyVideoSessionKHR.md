# Usage Notes

Destroys a video session and releases its internal resources.
Any video session parameters created against this session become
invalid, destroy them first.

All command buffers referencing this session must have completed
execution before destruction.
