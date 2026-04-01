# Usage Notes

Creates a video session parameters object that holds codec-specific
parameter sets (SPS, PPS for H.264/H.265, sequence headers for
AV1). These are referenced during decode/encode operations.

Chain the appropriate codec-specific struct into `pNext`:

- `VideoDecodeH264SessionParametersCreateInfoKHR` for H.264 decode.
- `VideoDecodeH265SessionParametersCreateInfoKHR` for H.265 decode.
- `VideoEncodeH264SessionParametersCreateInfoKHR` for H.264 encode.

Parameters can be added incrementally with
`update_video_session_parameters_khr`. A template parameter object
can be specified to inherit existing parameters.
