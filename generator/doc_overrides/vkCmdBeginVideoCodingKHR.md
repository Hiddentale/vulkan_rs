# Usage Notes

Begins a video coding scope within a command buffer. All video
decode and encode commands must be recorded between
`cmd_begin_video_coding_khr` and `cmd_end_video_coding_khr`.

`VideoBeginCodingInfoKHR` specifies:

- `video_session`: the session to use.
- `video_session_parameters`: codec parameters (SPS/PPS, etc.).
- `reference_slots`: DPB (decoded picture buffer) slots and their
  associated image views for reference pictures.

The command buffer must be allocated from a queue family that
supports the appropriate video operations (decode or encode),
as reported by `QueueFamilyVideoPropertiesKHR`.
