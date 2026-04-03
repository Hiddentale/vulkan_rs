# Installation

<!-- Phase 6.2.1: Cargo.toml setup, feature flags, platform requirements -->

## Add vulkan_rust to your project

```toml
[dependencies]
vulkan-rust = "0.9"
```

## Platform requirements

### Windows

Install the [LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home). This
provides `vulkan-1.dll` and the validation layers.

### Linux

Install your distribution's Vulkan packages:

```bash
# Ubuntu / Debian
sudo apt install libvulkan-dev vulkan-validationlayers

# Fedora
sudo dnf install vulkan-loader-devel vulkan-validation-layers

# Arch
sudo pacman -S vulkan-icd-loader vulkan-validation-layers
```

### macOS

Install the [LunarG Vulkan SDK for macOS](https://vulkan.lunarg.com/sdk/home),
which includes MoltenVK for Vulkan-on-Metal translation.

## Verify your setup

After installing, run this to confirm Vulkan is available:

```bash
# If you installed the Vulkan SDK:
vulkaninfo --summary
```

You should see your GPU listed with a supported Vulkan version.

## Next steps

Ready to write code? Continue to
[Hello Triangle, Part 1](hello-triangle-1.md).
