# Usage Notes

Destroys a video session parameters object. All command buffers
referencing these parameters must have completed execution before
destruction.

The video session itself is not affected, other parameter objects
associated with the same session remain valid.
