# Usage Notes

Destroys a sampler YCBCR conversion object. Any sampler that was
created with this conversion must already be destroyed.

After destruction, any descriptor set layout that used the associated
sampler as an immutable sampler remains valid but cannot be used to
allocate new descriptor sets.
