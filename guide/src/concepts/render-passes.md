# Render Passes & Framebuffers

## Motivation

A render pass tells Vulkan the *structure* of your rendering: what
attachments you use (color, depth), how they are loaded and stored, and
how subpasses depend on each other. This information lets the driver
make hardware-specific optimizations, especially on tile-based GPUs
(mobile) where the render pass boundaries determine what fits in on-chip
memory.

If you skip this concept and just try to render, the validation layers
will immediately tell you: "you need a render pass." Understanding *why*
it exists will save you from cargo-culting boilerplate you don't
understand.

## Intuition

### Blueprint and canvas

A render pass is a *blueprint* for a painting session. It describes:
- What surfaces you'll paint on (attachments: color, depth, stencil)
- How each surface is prepared before painting (load ops)
- What happens to each surface after painting (store ops)
- If there are multiple phases (subpasses) and how they depend on each other

A framebuffer is the *specific canvas*, the actual images that match
the blueprint's description.

```text
Render Pass (blueprint)              Framebuffer (canvas)
┌───────────────────────┐            ┌────────────────────────┐
│ Attachment 0:         │            │ Attachment 0:          │
│   format: B8G8R8A8    │───matches──│   swapchain_image_view │
│   load: CLEAR         │            │                        │
│   store: STORE        │            │ Attachment 1:          │
│   layout: → PRESENT   │───matches──│   depth_image_view     │
│                       │            │                        │
│ Attachment 1:         │            │ width: 1920            │
│   format: D32_SFLOAT  │            │ height: 1080           │
│   load: CLEAR         │            │ layers: 1              │
│   store: DONT_CARE    │            └────────────────────────┘
│   layout: → DEPTH_OPT │
│                       │
│ Subpass 0:            │
│   color: [0]          │
│   depth: [1]          │
└───────────────────────┘
```

You create the render pass once. You create a framebuffer for each
set of images you render to (typically one per swapchain image).

### Load and store ops: why they matter

When a render pass begins, the driver needs to know what to do with
each attachment's existing contents:

| Load Op | Meaning | When to use |
|---------|---------|-------------|
| `CLEAR` | Fill with a clear value | Start of frame, you want a clean slate |
| `LOAD` | Preserve the existing contents | Continuing previous rendering |
| `DONT_CARE` | Contents are undefined | You will overwrite every pixel anyway |

When the render pass ends:

| Store Op | Meaning | When to use |
|----------|---------|-------------|
| `STORE` | Write results to memory | You need the results (color for present, etc.) |
| `DONT_CARE` | Results may be discarded | Transient data (depth buffer you won't read later) |

> *Before reading on: on a tile-based mobile GPU, rendering happens
> in small tiles stored in fast on-chip memory. The load op controls
> whether tile data is loaded from main memory, and the store op
> controls whether it is written back. Why would `DONT_CARE` be
> significantly faster than `LOAD` on such hardware?*
>
> Answer: `DONT_CARE` lets the driver skip the expensive memory transfer
> entirely. On a mobile GPU, loading a full-screen depth buffer from
> main memory into tile memory can take milliseconds. If you are
> clearing it anyway, `CLEAR` tells the driver to fill tiles on-chip
> without touching main memory. `DONT_CARE` is even cheaper: it does
> nothing at all.

## Worked example: a single-subpass render pass

This is the most common setup: one color attachment (the swapchain
image) and one depth attachment.

### Step 1: Describe the attachments

```rust,ignore
use vulkan_rust::vk;
use vk::*;

// Color attachment: the swapchain image we render into.
let color_attachment = AttachmentDescription {
    flags: AttachmentDescriptionFlags::empty(),
    format: swapchain_format,           // e.g. B8G8R8A8_SRGB
    samples: SampleCountFlagBits::_1,
    load_op: AttachmentLoadOp::CLEAR,       // clear at start
    store_op: AttachmentStoreOp::STORE,      // keep the result
    stencil_load_op: AttachmentLoadOp::DONT_CARE,
    stencil_store_op: AttachmentStoreOp::DONT_CARE,
    initial_layout: ImageLayout::UNDEFINED,  // we don't care about previous contents
    final_layout: ImageLayout::PRESENT_SRC,  // ready for presentation after the pass
};

// Depth attachment: used for depth testing, discarded after.
let depth_attachment = AttachmentDescription {
    flags: AttachmentDescriptionFlags::empty(),
    format: Format::D32_SFLOAT,
    samples: SampleCountFlagBits::_1,
    load_op: AttachmentLoadOp::CLEAR,
    store_op: AttachmentStoreOp::DONT_CARE,  // we won't read it later
    stencil_load_op: AttachmentLoadOp::DONT_CARE,
    stencil_store_op: AttachmentStoreOp::DONT_CARE,
    initial_layout: ImageLayout::UNDEFINED,
    final_layout: ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
};
```

### Step 2: Define the subpass

```rust,ignore
use vulkan_rust::vk;
use vk::*;

// Subpass 0 uses attachment 0 as color output and attachment 1 as depth.
let color_ref = AttachmentReference {
    attachment: 0,    // index into the attachments array
    layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
};
let depth_ref = AttachmentReference {
    attachment: 1,
    layout: ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
};

let subpass = SubpassDescription {
    flags: SubpassDescriptionFlags::empty(),
    pipeline_bind_point: PipelineBindPoint::GRAPHICS,
    input_attachment_count: 0,
    p_input_attachments: core::ptr::null(),
    color_attachment_count: 1,
    p_color_attachments: &color_ref,
    p_resolve_attachments: core::ptr::null(),
    p_depth_stencil_attachment: &depth_ref,
    preserve_attachment_count: 0,
    p_preserve_attachments: core::ptr::null(),
};
```

### Step 3: Add a subpass dependency

```rust,ignore
use vulkan_rust::vk;
use vk::*;

// This dependency ensures that the image layout transition
// (from the previous frame's PRESENT_SRC to our UNDEFINED→COLOR_ATTACHMENT)
// happens before we start writing color output.
let dependency = SubpassDependency {
    src_subpass: SUBPASS_EXTERNAL,  // operations before the render pass
    dst_subpass: 0,                      // our subpass
    src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
        | PipelineStageFlags::EARLY_FRAGMENT_TESTS,
    dst_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
        | PipelineStageFlags::EARLY_FRAGMENT_TESTS,
    src_access_mask: AccessFlags::NONE,
    dst_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE
        | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
    dependency_flags: DependencyFlags::empty(),
};
```

### Step 4: Create the render pass

```rust,ignore
use vulkan_rust::vk;
use vk::*;

let attachments = [color_attachment, depth_attachment];

let render_pass_info = RenderPassCreateInfo::builder()
    .attachments(&attachments)
    .subpasses(&[subpass])
    .dependencies(&[dependency]);

let render_pass = unsafe {
    device.create_render_pass(&render_pass_info, None)?
};
```

### Step 5: Create framebuffers (one per swapchain image)

```rust,ignore
use vulkan_rust::vk;
use vk::*;

let framebuffers: Vec<Framebuffer> = swapchain_image_views
    .iter()
    .map(|&view| {
        // Each framebuffer uses a different swapchain image view
        // but the same depth image view (shared across frames).
        let attachments = [view, depth_image_view];

        let fb_info = FramebufferCreateInfo::builder()
            .render_pass(render_pass)     // must be compatible
            .attachments(&attachments)
            .width(swapchain_extent.width)
            .height(swapchain_extent.height)
            .layers(1);

        unsafe { device.create_framebuffer(&fb_info, None).unwrap() }
    })
    .collect();
```

### Step 6: Use in command recording

```rust,ignore
use vulkan_rust::vk;
use vk::*;

let clear_values = [
    ClearValue {
        color: ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],  // black
        },
    },
    ClearValue {
        depth_stencil: ClearDepthStencilValue {
            depth: 1.0,
            stencil: 0,
        },
    },
];

let begin_info = RenderPassBeginInfo::builder()
    .render_pass(render_pass)
    .framebuffer(framebuffers[image_index as usize])
    .render_area(Rect2D {
        offset: Offset2D { x: 0, y: 0 },
        extent: swapchain_extent,
    })
    .clear_values(&clear_values);

unsafe {
    // INLINE means we record drawing commands directly in this
    // primary command buffer (not via secondary command buffers).
    device.cmd_begin_render_pass(
        command_buffer,
        &begin_info,
        SubpassContents::INLINE,
    );

    // ... bind pipeline, bindescriptor sets, draw ...

    device.cmd_end_render_pass(command_buffer);
};
```

## Dynamic rendering (Vulkan 1.3)

Vulkan 1.3 introduced `cmd_begin_rendering` / `cmd_end_rendering`,
which lets you skip render pass and framebuffer objects entirely.
You specify attachments inline at recording time:

```rust,ignore
use vulkan_rust::vk;
use vk::*;

let color_attachment = RenderingAttachmentInfo::builder()
    .image_view(swapchain_image_view)
    .image_layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
    .load_op(AttachmentLoadOp::CLEAR)
    .store_op(AttachmentStoreOp::STORE)
    .clear_value(ClearValue {
        color: ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],
        },
    });

let rendering_info = RenderingInfo::builder()
    .render_area(Rect2D {
        offset: Offset2D { x: 0, y: 0 },
        extent: swapchain_extent,
    })
    .layer_count(1)
    .color_attachments(&[*color_attachment]);

unsafe {
    device.cmd_begin_rendering(command_buffer, &rendering_info);
    // ... draw ...
    device.cmd_end_rendering(command_buffer);
};
```

Dynamic rendering is simpler for most use cases. Use traditional render
passes when you need subpass dependencies, input attachments, or
compatibility with Vulkan 1.0/1.1/1.2.

## Formal reference

### Key structs

| Struct | Purpose |
|--------|---------|
| `AttachmentDescription` | Describes one attachment: format, samples, load/store ops, layouts |
| `AttachmentReference` | Points a subpass to an attachment by index + desired layout |
| `SubpassDescription` | Lists which attachments a subpass uses (color, depth, input, preserve) |
| `SubpassDependency` | Synchronization between subpasses (same fields as a pipeline barrier) |
| `RenderPassCreateInfo` | Combines attachments + subpasses + dependencies |
| `FramebufferCreateInfo` | Binds specific image views to a render pass |
| `RenderPassBeginInfo` | Starts a render pass instance with a framebuffer + clear values |

### Subpass dependencies are barriers

A `SubpassDependency` has the same fields as a pipeline barrier:
`src_stage_mask`, `dst_stage_mask`, `src_access_mask`, `dst_access_mask`.
The special value `SUBPASS_EXTERNAL` refers to commands outside the
render pass (before it starts or after it ends).

If you understood [Synchronization](synchronization.md), subpass
dependencies will feel familiar. They are barriers that the driver
inserts automatically at subpass transitions.

### Layout transitions are automatic

The render pass handles image layout transitions for you. Each
attachment has an `initial_layout` and `final_layout`. The driver
transitions the image at render pass begin/end. Within a subpass, the
image is in the layout specified by the `AttachmentReference`.

This is one of the render pass's biggest conveniences: you do not need
to insert manual `cmd_pipeline_barrier` calls for attachment layout
transitions inside a render pass.

### API reference links

- [`RenderPass`](https://docs.rs/vulkan-rust/latest/vulkan_rust/vk/struct.RenderPass.html)
- [`Framebuffer`](https://docs.rs/vulkan-rust/latest/vulkan_rust/vk/struct.Framebuffer.html)
- [`AttachmentDescription`](https://docs.rs/vulkan-rust/latest/vulkan_rust/vk/struct.AttachmentDescription.html)
- [`RenderingInfo`](https://docs.rs/vulkan-rust/latest/vulkan_rust/vk/struct.RenderingInfo.html)
- [Vulkan spec: Render Pass](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#renderpass)

## Key takeaways

- A render pass is a blueprint describing attachments, subpasses, and
  dependencies. A framebuffer binds specific images to that blueprint.
- Load and store ops tell the driver how to handle attachment data at
  the start and end of the pass. Choosing `DONT_CARE` or `CLEAR` over
  `LOAD` can dramatically improve performance on mobile GPUs.
- Most applications need only a single subpass. Multiple subpasses are
  for advanced techniques (deferred rendering, input attachments).
- Vulkan 1.3 dynamic rendering (`cmd_begin_rendering`) eliminates the
  need for render pass and framebuffer objects in simple cases.
- Render passes handle layout transitions automatically. You do not
  need manual barriers for attachment images inside a render pass.
