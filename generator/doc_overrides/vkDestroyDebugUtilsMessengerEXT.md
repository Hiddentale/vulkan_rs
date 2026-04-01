# Usage Notes

Destroys a debug messenger created with
`create_debug_utils_messenger_ext`. After this call, the
messenger's callback will no longer be invoked.

Destroy before the instance is destroyed.

Requires `VK_EXT_debug_utils`.
