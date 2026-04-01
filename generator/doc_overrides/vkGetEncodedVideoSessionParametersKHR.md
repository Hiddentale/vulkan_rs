# Usage Notes

Retrieves the encoded (serialized) form of video session
parameters, typically codec headers (SPS/PPS for H.264/H.265,
sequence header for AV1) that must be prepended to the encoded
bitstream.

Uses the two-call pattern: call with null `p_data` to query
the size, allocate, then call again to fill the buffer.

The `p_feedback_info` output indicates whether the driver
modified or overrode any parameters relative to what was
requested (check `has_overrides`).

This data is the codec parameter payload that decoders need to
initialize before processing encoded frames.
