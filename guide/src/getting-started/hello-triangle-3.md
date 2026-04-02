# Hello Triangle, Part 3: Render Pass & Pipeline

In [Part 2](hello-triangle-2.md) we opened a window, created a surface
and swapchain, and retrieved image views. We have somewhere to render,
but no instructions for *how* to render.

**What we build in this part:**

```text
Write shaders ──> Create Render Pass ──> Create Pipeline ──> Create Framebuffers
```

> **Threshold concept.** The graphics pipeline is one of Vulkan's biggest
> conceptual shifts. Instead of setting state one call at a time (like
> OpenGL's `glEnable(GL_DEPTH_TEST)`), you define *all* rendering state
> in a single pipeline object. This is verbose, but it means the driver
> has complete information at creation time and compiles everything to
> GPU machine code once, not at draw time.

## Step 1: Write shaders

We need a vertex shader (positions the triangle) and a fragment shader
(colors it). Write these as GLSL and compile to SPIR-V.

**`triangle.vert`:**
```glsl
#version 450

// Hard-coded triangle vertices (no vertex buffer needed).
vec2 positions[3] = vec2[](
    vec2( 0.0, -0.5),
    vec2( 0.5,  0.5),
    vec2(-0.5,  0.5)
);

vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),   // red
    vec3(0.0, 1.0, 0.0),   // green
    vec3(0.0, 0.0, 1.0)    // blue
);

layout(location = 0) out vec3 frag_color;

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    frag_color = colors[gl_VertexIndex];
}
```

**`triangle.frag`:**
```glsl
#version 450

layout(location = 0) in vec3 frag_color;
layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(frag_color, 1.0);
}
```

Compile them with `glslc` (included in the Vulkan SDK):

```bash
glslc triangle.vert -o triangle.vert.spv
glslc triangle.frag -o triangle.frag.spv
```

Place the `.spv` files in your project's `src/` directory (or wherever
you prefer, adjust the path in the code below).

> *Before reading on: this vertex shader hard-codes the triangle
> positions inside the shader rather than reading them from a vertex
> buffer. Why might this be useful for a first example?*
>
> It eliminates the need for vertex buffers, memory allocation, and
> buffer binding, letting us focus on the pipeline and render pass
> without those distractions. A real application reads vertices from
> buffers (covered in the [Memory Management](../concepts/memory.md)
> chapter).

## Step 2: Load SPIR-V and create shader modules

```rust,ignore
use vulkan_rs::vk;
use vulkan_rs::cast_to_u32;
use vk::structs::*;

// ── Load SPIR-V bytecode ───────────────────────────────────────
let vert_bytes = include_bytes!("triangle.vert.spv");
let frag_bytes = include_bytes!("triangle.frag.spv");

// SPIR-V must be aligned to 4 bytes. cast_to_u32 checks alignment.
let vert_code = cast_to_u32(vert_bytes)
    .expect("Vertex shader SPIR-V is not 4-byte aligned");
let frag_code = cast_to_u32(frag_bytes)
    .expect("Fragment shader SPIR-V is not 4-byte aligned");

// ── Create shader modules ──────────────────────────────────────
let vert_info = ShaderModuleCreateInfo::builder()
    .code_size(vert_code.len() * 4)
    .p_code(vert_code.as_ptr());
let frag_info = ShaderModuleCreateInfo::builder()
    .code_size(frag_code.len() * 4)
    .p_code(frag_code.as_ptr());

let vert_module = unsafe { device.create_shader_module(&vert_info, None) }
    .expect("Failed to create vertex shader module");
let frag_module = unsafe { device.create_shader_module(&frag_info, None) }
    .expect("Failed to create fragment shader module");
```

Shader modules are temporary containers. After the pipeline is created,
we can destroy them.

## Step 3: Create the render pass

The render pass describes *what* attachments we render to and how they
are handled. See [Render Passes & Framebuffers](../concepts/render-passes.md)
for the full concept.

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;
use vk::enums::*;
use vk::bitmasks::*;

// ── Color attachment: the swapchain image ──────────────────────
let color_attachment = AttachmentDescription {
    flags: AttachmentDescriptionFlags::empty(),
    format: surface_format.format,  // from Part 2
    samples: SampleCountFlagBits::_1,
    load_op: AttachmentLoadOp::CLEAR,       // clear to black
    store_op: AttachmentStoreOp::STORE,      // keep the result
    stencil_load_op: AttachmentLoadOp::DONT_CARE,
    stencil_store_op: AttachmentStoreOp::DONT_CARE,
    initial_layout: ImageLayout::UNDEFINED,
    final_layout: ImageLayout::PRESENT_SRC,  // ready for display
};

// ── Subpass: use the color attachment ──────────────────────────
let color_ref = AttachmentReference {
    attachment: 0,
    layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
};

let subpass = SubpassDescription {
    flags: SubpassDescriptionFlags::empty(),
    pipeline_bind_point: PipelineBindPoint::GRAPHICS,
    input_attachment_count: 0,
    p_input_attachments: core::ptr::null(),
    color_attachment_count: 1,
    p_color_attachments: &color_ref,
    p_resolve_attachments: core::ptr::null(),
    p_depth_stencil_attachment: core::ptr::null(),
    preserve_attachment_count: 0,
    p_preserve_attachments: core::ptr::null(),
};

// ── Subpass dependency ─────────────────────────────────────────
//
// Ensure the image layout transition happens before we write color.
let dependency = SubpassDependency {
    src_subpass: vk::constants::SUBPASS_EXTERNAL,
    dst_subpass: 0,
    src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
    dst_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
    src_access_mask: AccessFlags::NONE,
    dst_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE,
    dependency_flags: DependencyFlags::empty(),
};

let render_pass_info = RenderPassCreateInfo::builder()
    .attachments(std::slice::from_ref(&color_attachment))
    .subpasses(std::slice::from_ref(&subpass))
    .dependencies(std::slice::from_ref(&dependency));

let render_pass = unsafe {
    device.create_render_pass(&render_pass_info, None)
}
.expect("Failed to create render pass");
```

## Step 4: Create the pipeline layout

Our shaders don't use any descriptors or push constants, so the layout
is empty.

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;

let layout_info = PipelineLayoutCreateInfo::builder();
let pipeline_layout = unsafe {
    device.create_pipeline_layout(&layout_info, None)
}
.expect("Failed to create pipeline layout");
```

## Step 5: Create the graphics pipeline

This is the largest struct in the Vulkan API. Every piece of rendering
state is specified here.

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;
use vk::enums::*;
use vk::bitmasks::*;
use vk::handles::*;

// ── Shader stages ──────────────────────────────────────────────
let entry_name = c"main";
let stages = [
    *PipelineShaderStageCreateInfo::builder()
        .stage(ShaderStageFlags::VERTEX)
        .module(vert_module)
        .p_name(entry_name.as_ptr()),
    *PipelineShaderStageCreateInfo::builder()
        .stage(ShaderStageFlags::FRAGMENT)
        .module(frag_module)
        .p_name(entry_name.as_ptr()),
];

// ── Vertex input: empty (positions are hard-coded in shader) ───
let vertex_input = PipelineVertexInputStateCreateInfo::builder();

// ── Input assembly: triangle list ──────────────────────────────
let input_assembly = PipelineInputAssemblyStateCreateInfo::builder()
    .topology(PrimitiveTopology::TRIANGLE_LIST);

// ── Viewport and scissor: dynamic (set at draw time) ───────────
let mut viewport_state = PipelineViewportStateCreateInfo::builder();
viewport_state.viewport_count = 1;
viewport_state.scissor_count = 1;

// ── Rasterization ──────────────────────────────────────────────
let rasterizer = PipelineRasterizationStateCreateInfo::builder()
    .polygon_mode(PolygonMode::FILL)
    .cull_mode(CullModeFlags::BACK)
    .front_face(FrontFace::CLOCKWISE)
    .line_width(1.0);

// ── Multisampling: off ─────────────────────────────────────────
let multisampling = PipelineMultisampleStateCreateInfo::builder()
    .rasterization_samples(SampleCountFlagBits::_1);

// ── Color blending: no blending, write all channels ────────────
let blend_attachment = PipelineColorBlendAttachmentState {
    blend_enable: 0,
    color_write_mask: ColorComponentFlags::R
        | ColorComponentFlags::G
        | ColorComponentFlags::B
        | ColorComponentFlags::A,
    ..unsafe { core::mem::zeroed() }
};

let color_blending = PipelineColorBlendStateCreateInfo::builder()
    .attachments(std::slice::from_ref(&blend_attachment));

// ── Dynamic state ──────────────────────────────────────────────
let dynamic_states = [DynamicState::VIEWPORT, DynamicState::SCISSOR];
let dynamic_state = PipelineDynamicStateCreateInfo::builder()
    .dynamic_states(&dynamic_states);

// ── Assemble the pipeline ──────────────────────────────────────
let pipeline_info = GraphicsPipelineCreateInfo::builder()
    .stages(&stages)
    .p_vertex_input_state(&vertex_input)
    .p_input_assembly_state(&input_assembly)
    .p_viewport_state(&viewport_state)
    .p_rasterization_state(&rasterizer)
    .p_multisample_state(&multisampling)
    .p_color_blend_state(&color_blending)
    .p_dynamic_state(&dynamic_state)
    .layout(pipeline_layout)
    .render_pass(render_pass)
    .subpass(0);

let pipeline = unsafe {
    device.create_graphics_pipelines(
        PipelineCache::null(),
        &[*pipeline_info],
        None,
    )
}
.expect("Failed to create graphics pipeline")[0];

// ── Shader modules are no longer needed ────────────────────────
unsafe {
    device.destroy_shader_module(vert_module, None);
    device.destroy_shader_module(frag_module, None);
};
```

> *Before reading on: we set `cull_mode` to BACK and `front_face`
> to CLOCKWISE. What happens if the triangle vertices are wound
> counter-clockwise? What would you see?*
>
> The triangle would be culled (invisible). Back-face culling
> discards triangles whose vertices appear in the wrong winding
> order from the camera's perspective. If your triangle is invisible,
> try switching to `COUNTER_CLOCKWISE` or disabling culling with
> `CullModeFlags::NONE`.

## Step 6: Create framebuffers

A framebuffer binds specific image views to a render pass. We need one
per swapchain image.

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;
use vk::handles::*;

let framebuffers: Vec<Framebuffer> = swapchain_image_views
    .iter()
    .map(|&view| {
        let views = [view];
        let fb_info = FramebufferCreateInfo::builder()
            .render_pass(render_pass)
            .attachments(&views)
            .width(extent.width)
            .height(extent.height)
            .layers(1);

        unsafe { device.create_framebuffer(&fb_info, None) }
            .expect("Failed to create framebuffer")
    })
    .collect();
```

## Where we are now

```text
Render Pass        "clear to black, store the result, present"
     │
Pipeline           "use these shaders, fill triangles, no blending"
     │
Framebuffers       [swapchain image 0, swapchain image 1, ...]
```

We have everything needed to describe *what* to draw and *how*. In
Part 4, we record commands that use the pipeline and render pass,
submit them, and present the result.

## Clean up (new objects)

Add these to the cleanup sequence from Part 2, before device destruction:

```rust,ignore
unsafe {
    for &fb in &framebuffers {
        device.destroy_framebuffer(fb, None);
    }
    device.destroy_pipeline(pipeline, None);
    device.destroy_pipeline_layout(pipeline_layout, None);
    device.destroy_render_pass(render_pass, None);
    // ... then image views, swapchain, device, surface, instance
}
```

## What we learned

| Step | What | Why |
|------|------|-----|
| Shaders | GLSL → SPIR-V → `ShaderModule` | GPU programs that position and color pixels |
| Render pass | `create_render_pass` | Declares attachments and how they are loaded/stored |
| Pipeline layout | `create_pipeline_layout` | Declares what resources shaders expect (none for now) |
| Graphics pipeline | `create_graphics_pipelines` | Bakes all rendering state into one compiled object |
| Framebuffers | `create_framebuffer` | Binds specific images to a render pass |

## Concepts to explore

- [Pipelines](../concepts/pipelines.md), dynamic state, compute
  pipelines, pipeline cache.
- [Render Passes & Framebuffers](../concepts/render-passes.md), load
  ops, subpass dependencies, dynamic rendering.

## Exercises

1. **Change the clear color.** Modify the render pass begin info (in
   Part 4) to clear to a different color. The clear value is passed
   when beginning the render pass, not when creating it.
2. **Add a depth attachment.** Create a depth image and image view, add
   a second attachment to the render pass, and enable depth testing in
   the pipeline.
3. **Try `PolygonMode::LINE`.** Change the polygon mode to LINE to see
   the triangle as wireframe. (Requires the `fillModeNonSolid` device
   feature.)

## Next

[Part 4: Command Buffers & Drawing](hello-triangle-4.md) records the
draw commands, submits them, and presents the triangle to the screen.
