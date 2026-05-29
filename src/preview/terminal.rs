//! Terminal image rendering helpers.

use super::PreviewImage;

/// Terminal preview display mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayMode {
    /// Prefer Sixel when supported, otherwise ANSI truecolor blocks.
    Auto,
    /// Emit Sixel escape sequences.
    Sixel,
    /// Emit ANSI truecolor half-blocks.
    Ansi,
    /// Do not render image pixels.
    None,
}

/// Runtime terminal capability hints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerminalCapabilities {
    /// Whether Sixel appears to be supported.
    pub sixel: bool,
    /// Whether ANSI truecolor appears to be supported.
    pub truecolor: bool,
}

/// Detects terminal capabilities from environment hints.
pub fn terminal_capabilities() -> TerminalCapabilities {
    let term = std::env::var("TERM")
        .unwrap_or_default()
        .to_ascii_lowercase();
    let term_program = std::env::var("TERM_PROGRAM")
        .unwrap_or_default()
        .to_ascii_lowercase();
    let colorterm = std::env::var("COLORTERM")
        .unwrap_or_default()
        .to_ascii_lowercase();
    let wt_profile = std::env::var("WT_PROFILE_ID")
        .unwrap_or_default()
        .to_ascii_lowercase();
    let sixel = std::env::var_os("IMQ_NO_SIXEL").is_none()
        && (std::env::var_os("IMQ_SIXEL").is_some()
            || term.contains("sixel")
            || term.contains("mlterm")
            || term_program.contains("ghostty")
            || term_program.contains("wezterm")
            || term_program.contains("windows_terminal")
            || wt_profile.contains("windows")
            || std::env::var_os("WT_SESSION").is_some()
            || std::env::var_os("KONSOLE_VERSION").is_some()
            || std::env::var_os("MLTERM").is_some()
            || std::env::var_os("RLOGIN").is_some());
    let truecolor = colorterm.contains("truecolor")
        || colorterm.contains("24bit")
        || term_program.contains("wezterm")
        || term_program.contains("iterm")
        || std::env::var_os("WT_SESSION").is_some();
    TerminalCapabilities { sixel, truecolor }
}

/// Renders an image as ANSI truecolor half-blocks.
pub fn render_ansi_blocks(image: &PreviewImage) -> String {
    let mut out = String::new();
    let mut y = 0;
    while y < image.height {
        for x in 0..image.width {
            let top = image.pixel(x, y);
            let bottom = if y + 1 < image.height {
                image.pixel(x, y + 1)
            } else {
                [0, 0, 0]
            };
            out.push_str(&format!(
                "\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m▀",
                top[0], top[1], top[2], bottom[0], bottom[1], bottom[2]
            ));
        }
        out.push_str("\x1b[0m\n");
        y += 2;
    }
    out
}

/// Renders an image as Sixel using a fixed 6x6x6 RGB color cube.
pub fn render_sixel(image: &PreviewImage) -> String {
    let mut out = String::from("\x1bPq");
    for r in 0..6 {
        for g in 0..6 {
            for b in 0..6 {
                let idx = color_index(r, g, b);
                out.push_str(&format!("#{idx};2;{};{};{}", r * 20, g * 20, b * 20));
            }
        }
    }
    let bands = image.height.div_ceil(6);
    for band in 0..bands {
        let y0 = band * 6;
        for r in 0..6 {
            for g in 0..6 {
                for b in 0..6 {
                    let idx = color_index(r, g, b);
                    let mut line_has_color = false;
                    let mut encoded = String::new();
                    for x in 0..image.width {
                        let mut bits = 0u8;
                        for bit in 0..6 {
                            let y = y0 + bit;
                            if y >= image.height {
                                continue;
                            }
                            let [pr, pg, pb] = quantize(image.pixel(x, y));
                            if pr == r && pg == g && pb == b {
                                bits |= 1 << bit;
                            }
                        }
                        if bits != 0 {
                            line_has_color = true;
                        }
                        encoded.push(char::from(63 + bits));
                    }
                    if line_has_color {
                        out.push_str(&format!("#{idx}{encoded}$"));
                    }
                }
            }
        }
        out.push('-');
    }
    out.push_str("\x1b\\");
    out
}

fn quantize(rgb: [u8; 3]) -> [u8; 3] {
    [
        ((u16::from(rgb[0]) * 5 + 127) / 255) as u8,
        ((u16::from(rgb[1]) * 5 + 127) / 255) as u8,
        ((u16::from(rgb[2]) * 5 + 127) / 255) as u8,
    ]
}

fn color_index(r: u8, g: u8, b: u8) -> u16 {
    u16::from(r) * 36 + u16::from(g) * 6 + u16::from(b)
}
