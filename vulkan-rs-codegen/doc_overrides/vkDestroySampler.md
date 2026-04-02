# Usage Notes

Destroys a sampler. The sampler must not be referenced by any
descriptor set that is bound in a pending command buffer.

Since most applications use a small fixed set of samplers, they are
typically created once at startup and destroyed only during
application shutdown.
