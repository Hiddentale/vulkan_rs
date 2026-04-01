# Usage Notes

Creates a video session for hardware-accelerated video decoding
or encoding. The session defines the video codec profile,
resolution range, format, and maximum reference picture count.

Key fields in `VideoSessionCreateInfoKHR`:

- `video_profile`: codec (H.264, H.265, AV1) and profile/level.
- `max_coded_extent`: maximum frame resolution.
- `picture_format` / `reference_picture_format`: image formats
  for decoded pictures and DPB (decoded picture buffer) slots.
- `max_dpb_slots` / `max_active_reference_pictures`: reference
  frame capacity.

After creation, query memory requirements with
`get_video_session_memory_requirements_khr`, allocate and bind
memory with `bind_video_session_memory_khr`, then create session
parameters with `create_video_session_parameters_khr`.
