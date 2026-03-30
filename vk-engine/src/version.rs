use core::fmt;

/// Decoded Vulkan API version (major.minor.patch).
///
/// Vulkan packs versions into a `u32`: major (bits 31–22), minor (21–12),
/// patch (11–0). This type provides named fields and a `Display` impl.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl Version {
    /// Decode a packed Vulkan version `u32` into its components.
    pub const fn from_raw(raw: u32) -> Self {
        Self {
            major: raw >> 22,
            minor: (raw >> 12) & 0x3FF,
            patch: raw & 0xFFF,
        }
    }

    /// Encode this version back into the packed `u32` representation.
    pub const fn to_raw(self) -> u32 {
        (self.major << 22) | (self.minor << 12) | self.patch
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let v = Version { major: 1, minor: 3, patch: 250 };
        assert_eq!(Version::from_raw(v.to_raw()), v);
    }

    #[test]
    fn from_raw_known_versions() {
        // VK_API_VERSION_1_0 = VK_MAKE_API_VERSION(0, 1, 0, 0)
        let v10 = Version::from_raw((1 << 22) | (0 << 12) | 0);
        assert_eq!(v10, Version { major: 1, minor: 0, patch: 0 });

        // VK_API_VERSION_1_3 = 0x00403000
        let v13 = Version::from_raw(0x00403000);
        assert_eq!(v13, Version { major: 1, minor: 3, patch: 0 });
    }

    #[test]
    fn display_format() {
        let v = Version { major: 1, minor: 2, patch: 195 };
        assert_eq!(v.to_string(), "1.2.195");
    }

    #[test]
    fn ordering() {
        let v10 = Version { major: 1, minor: 0, patch: 0 };
        let v12 = Version { major: 1, minor: 2, patch: 0 };
        let v13 = Version { major: 1, minor: 3, patch: 0 };
        assert!(v10 < v12);
        assert!(v12 < v13);
    }

    #[test]
    fn to_raw_known_versions() {
        let v13 = Version { major: 1, minor: 3, patch: 0 };
        assert_eq!(v13.to_raw(), 0x00403000);
    }
}
