# Usage Notes

Queries the properties of a specific video encode quality level.
Quality levels range from 0 (lowest quality, fastest) to
`maxQualityLevels - 1` (highest quality, slowest), as reported
by `VideoEncodeCapabilitiesKHR`.

The output `VideoEncodeQualityLevelPropertiesKHR` provides
recommended encode settings for the requested quality level.
Chain codec-specific quality level info (e.g.,
`VideoEncodeH264QualityLevelPropertiesKHR`) into `pNext` to get
codec-specific recommended parameters.

Use these recommended settings as a starting point for
`VideoEncodeInfoKHR` and rate control configuration.
