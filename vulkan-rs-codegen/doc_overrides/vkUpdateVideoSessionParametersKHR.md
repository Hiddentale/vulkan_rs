# Usage Notes

Adds new codec-specific parameter sets to an existing video
session parameters object. For example, adding new SPS/PPS
entries for H.264 as they are encountered in the bitstream.

Chain the codec-specific update struct into the `pNext` of
`VideoSessionParametersUpdateInfoKHR`. The `update_sequence_count`
must increment monotonically with each update.

Parameters cannot be removed or modified, only new entries can
be added. If a parameter set with the same ID already exists,
the update fails.
