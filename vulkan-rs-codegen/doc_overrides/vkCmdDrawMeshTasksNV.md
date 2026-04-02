# Usage Notes

Dispatches mesh shader work using the NV mesh shader model.
Launches `task_count` task shader workgroups starting at
`first_task`.

This is the legacy NV path; prefer `cmd_draw_mesh_tasks_ext`
for new code.

Requires `VK_NV_mesh_shader`.
