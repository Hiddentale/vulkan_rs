# Usage Notes

Issues control commands within a video coding scope. Used to
reset the video session state or set encode quality/rate control
parameters.

`VideoCodingControlInfoKHR` flags include:

- `RESET`: resets the video session to a clean state, clearing
  all DPB slots and internal codec state.
- `ENCODE_RATE_CONTROL`: applies rate control settings (chain
  `VideoEncodeRateControlInfoKHR` into `pNext`).
- `ENCODE_QUALITY_LEVEL`: sets the encode quality level.

Must be recorded between `cmd_begin_video_coding_khr` and
`cmd_end_video_coding_khr`.
