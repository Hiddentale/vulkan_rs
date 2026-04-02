# Usage Notes

Queries video codec capabilities for a given video profile on a
physical device. Returns `VideoCapabilitiesKHR` describing:

- Supported coded extent range (min/max resolution).
- Maximum DPB slot and active reference picture counts.
- Bitstream buffer offset and size alignment requirements.
- Supported standard header version.

Chain codec-specific capability structs (e.g.,
`VideoDecodeH264CapabilitiesKHR`) into the `pNext` of
`p_capabilities` to receive additional codec details.

This is the first query in the video workflow, use it to
determine whether a codec profile is supported and what limits
apply before creating a video session.
