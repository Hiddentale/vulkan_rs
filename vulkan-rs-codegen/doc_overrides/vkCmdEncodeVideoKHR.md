# Usage Notes

Encodes a single video frame into a compressed bitstream.
Must be recorded within a video coding scope
(`cmd_begin_video_coding_khr` / `cmd_end_video_coding_khr`).

`VideoEncodeInfoKHR` specifies:

- `dst_buffer` / `dst_buffer_offset` / `dst_buffer_range`: where
  to write the compressed output.
- `src_picture_resource`: the input image view to encode.
- `setup_reference_slot`: DPB slot to store the reconstructed
  frame for future reference.
- `reference_slots`: reference frames for inter-prediction.

Chain codec-specific encode info (e.g.,
`VideoEncodeH264PictureInfoKHR`) into `pNext`. Configure rate
control beforehand with `cmd_control_video_coding_khr`.
