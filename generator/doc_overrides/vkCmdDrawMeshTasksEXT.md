# Usage Notes

Dispatches mesh shader work groups. `group_count_x/y/z` specify
the number of task or mesh shader work groups to launch.

Each work group runs the mesh shader (or task shader, if bound)
which emits primitives directly without traditional vertex/index
buffers.

Requires `VK_EXT_mesh_shader` and the `meshShader` feature.
