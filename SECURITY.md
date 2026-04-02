# Security Policy

## Reporting a Vulnerability

If you discover a security issue in vulkan-rs, please report it privately
by emailing 55276630+Hiddentale@users.noreply.github.com or using GitHub's
[private vulnerability reporting](https://github.com/Hiddentale/vulkan_rs/security/advisories/new).

Please do not open a public issue for security vulnerabilities.

I will acknowledge receipt within 48 hours and aim to release a fix
within 7 days for critical issues.

## Scope

vulkan-rs is a thin FFI wrapper. Most `unsafe` code is inherent to the Vulkan
API contract. Security-relevant issues include:

- Soundness bugs in safe-facing APIs (e.g., `cast_to_u32`, builder patterns)
- Undefined behavior reachable without the caller violating documented safety contracts
- Memory safety issues in the code generator that produce incorrect FFI bindings
