# Usage Notes

Queries whether a descriptor set layout with the given bindings can
be created on this device, and returns information about the
variable descriptor count limit if applicable.

Use this before creating layouts with very large descriptor counts
or update-after-bind bindings to verify they are within device
limits. The call is lightweight and does not allocate anything.

Chain `DescriptorSetVariableDescriptorCountLayoutSupport` in the
output to query the maximum variable descriptor count for layouts
that use `DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT`.
