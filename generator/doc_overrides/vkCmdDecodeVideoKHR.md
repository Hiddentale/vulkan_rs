# Usage Notes

Decodes a single video frame from a compressed bitstream.
Must be recorded within a video coding scope
(`cmd_begin_video_coding_khr` / `cmd_end_video_coding_khr`).

`VideoDecodeInfoKHR` specifies:

- `src_buffer` / `src_buffer_offset` / `src_buffer_range`: the
  bitstream data containing the compressed frame.
- `dst_picture_resource`: the output image view for the decoded
  frame.
- `setup_reference_slot`: DPB slot to store this frame for use
  as a reference by future frames.
- `reference_slots`: previously decoded reference frames needed
  to decode this frame.

Chain codec-specific decode info (e.g.,
`VideoDecodeH264PictureInfoKHR`) into `pNext`.
