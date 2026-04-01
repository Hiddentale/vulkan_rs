# Usage Notes

Destroys a descriptor set layout. The layout must not be referenced
by any pipeline layout or pending descriptor set allocation that is
still in use.

Descriptor set layouts are lightweight and typically long-lived.
Destroy them during application shutdown after all dependent
pipeline layouts and descriptor pools have been destroyed.
