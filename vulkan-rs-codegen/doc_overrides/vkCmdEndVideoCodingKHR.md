# Usage Notes

Ends a video coding scope previously started with
`cmd_begin_video_coding_khr`. After this call, video decode and
encode commands can no longer be recorded until a new scope is
started.

The `VideoEndCodingInfoKHR` struct is currently reserved for
future use (no flags defined).
