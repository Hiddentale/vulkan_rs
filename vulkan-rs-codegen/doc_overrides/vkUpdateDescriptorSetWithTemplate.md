# Usage Notes

Updates a descriptor set using a descriptor update template and a
raw pointer to a block of host data. The template defines the mapping
from the data block to descriptor bindings.

This is faster than `update_descriptor_sets` for repeated updates
with the same layout, the driver has pre-compiled the update path.

The `data` pointer must point to a block of memory laid out according
to the template's entry offsets and strides. The data is consumed
immediately; the pointer does not need to remain valid after the
call returns.
