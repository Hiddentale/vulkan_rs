# Usage Notes

Dynamically enables or disables writes to the depth buffer. Only
takes effect if the pipeline was created with
`DYNAMIC_STATE_DEPTH_WRITE_ENABLE`.

Disable depth writes when:

- Drawing transparent objects (they should test against depth but
  not write to it).
- Drawing skyboxes after the opaque pass.
- Performing post-processing with depth reads but no depth output.

Depth testing and depth writing are independent controls, you can
test without writing, or write without testing.

Core dynamic state in Vulkan 1.3.
