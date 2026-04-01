# Usage Notes

Inserts an execution and memory dependency between commands recorded
before and after the barrier. This is the primary synchronisation
tool inside a command buffer.

**Three types of barrier**:

- **Memory barriers** (`MemoryBarrier`): global, affects all
  resources. Rarely needed, prefer the more specific variants.
- **Buffer memory barriers** (`BufferMemoryBarrier`): targets a
  specific buffer region. Use for storage buffer read-after-write.
- **Image memory barriers** (`ImageMemoryBarrier`): targets a
  specific image subresource range. Also performs layout transitions.

**Layout transitions**: image memory barriers are the primary way to
transition images between layouts (e.g.
`TRANSFER_DST_OPTIMAL` → `SHADER_READ_ONLY_OPTIMAL`). The old and
new layouts in the barrier define the transition.

**Stage masks**: `src_stage_mask` is the set of stages that must
complete before the barrier. `dst_stage_mask` is the set of stages
that must wait for the barrier. Choose the narrowest stages possible
to minimise stalls.

Inside a render pass, only self-dependencies are allowed (barriers
within a single subpass). Outside a render pass, there are no
restrictions.

For Vulkan 1.3+, prefer `cmd_pipeline_barrier2` which uses
extensible structs.
