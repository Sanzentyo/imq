//! Terminal image rendering helpers.

use super::PreviewImage;
use base64::Engine;
use image::ImageEncoder;

/// Terminal preview display mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayMode {
    /// Prefer native terminal graphics, then Sixel, otherwise ANSI truecolor blocks.
    Auto,
    /// Emit Kitty graphics protocol escape sequences.
    Kitty,
    /// Emit Sixel escape sequences.
    Sixel,
    /// Emit iTerm2 inline image escape sequences.
    Iterm2,
    /// Emit ANSI truecolor half-blocks.
    Ansi,
    /// Do not render image pixels.
    None,
}

/// Runtime terminal capability hints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerminalCapabilities {
    /// Whether Kitty graphics protocol appears to be supported.
    pub kitty: bool,
    /// Whether Sixel appears to be supported.
    pub sixel: bool,
    /// Whether iTerm2 inline image protocol appears to be supported.
    pub iterm2: bool,
    /// Whether ANSI truecolor appears to be supported.
    pub truecolor: bool,
}

/// Detects terminal capabilities from environment hints.
pub fn terminal_capabilities() -> TerminalCapabilities {
    terminal_capabilities_from_env(|key| std::env::var(key).ok())
}

/// Selects the display mode used by [`DisplayMode::Auto`].
pub fn auto_display_mode() -> DisplayMode {
    auto_display_mode_from_env(|key| std::env::var(key).ok())
}

fn terminal_capabilities_from_env(getenv: impl Fn(&str) -> Option<String>) -> TerminalCapabilities {
    let term = env_lc(&getenv, "TERM").unwrap_or_default();
    let term_program = env_lc(&getenv, "TERM_PROGRAM").unwrap_or_default();
    let colorterm = env_lc(&getenv, "COLORTERM").unwrap_or_default();
    let wt_profile = env_lc(&getenv, "WT_PROFILE_ID").unwrap_or_default();
    let imq_protocol = env_lc(&getenv, "IMQ_IMAGE_PROTOCOL")
        .or_else(|| env_lc(&getenv, "IMQ_DISPLAY"))
        .unwrap_or_default();
    let kitty = !env_flag(&getenv, "IMQ_NO_KITTY")
        && (imq_protocol == "kitty"
            || env_flag(&getenv, "IMQ_KITTY")
            || env_present(&getenv, "KITTY_WINDOW_ID")
            || term.contains("xterm-kitty")
            || term.contains("ghostty")
            || term_program.contains("ghostty")
            || term_program.contains("kitty"));
    let sixel = !env_flag(&getenv, "IMQ_NO_SIXEL")
        && (imq_protocol == "sixel"
            || env_flag(&getenv, "IMQ_SIXEL")
            || term.contains("sixel")
            || term.contains("mlterm")
            || term.contains("foot")
            || term.contains("rio")
            || term.contains("contour")
            || term.contains("mintty")
            || term_program.contains("wezterm")
            || term_program.contains("windows_terminal")
            || wt_profile.contains("windows")
            || env_present(&getenv, "WT_SESSION")
            || env_present(&getenv, "WEZTERM_EXECUTABLE")
            || env_present(&getenv, "KONSOLE_VERSION")
            || env_present(&getenv, "MLTERM")
            || env_present(&getenv, "RLOGIN"));
    let iterm2 = !env_flag(&getenv, "IMQ_NO_ITERM2")
        && (imq_protocol == "iterm2"
            || imq_protocol == "iterm"
            || env_flag(&getenv, "IMQ_ITERM2")
            || term_program.contains("iterm")
            || env_lc(&getenv, "LC_TERMINAL").is_some_and(|value| value.contains("iterm")));
    let truecolor = colorterm.contains("truecolor")
        || colorterm.contains("24bit")
        || term_program.contains("wezterm")
        || term_program.contains("iterm")
        || env_present(&getenv, "WT_SESSION");
    TerminalCapabilities {
        kitty,
        sixel,
        iterm2,
        truecolor,
    }
}

fn auto_display_mode_from_env(getenv: impl Fn(&str) -> Option<String>) -> DisplayMode {
    let forced = env_lc(&getenv, "IMQ_IMAGE_PROTOCOL")
        .or_else(|| env_lc(&getenv, "IMQ_DISPLAY"))
        .unwrap_or_default();
    match forced.as_str() {
        "kitty" => return DisplayMode::Kitty,
        "sixel" => return DisplayMode::Sixel,
        "iterm" | "iterm2" => return DisplayMode::Iterm2,
        "ansi" | "blocks" | "halfblocks" => return DisplayMode::Ansi,
        "none" | "off" => return DisplayMode::None,
        _ => {}
    }
    let capabilities = terminal_capabilities_from_env(getenv);
    if capabilities.kitty {
        DisplayMode::Kitty
    } else if capabilities.sixel {
        DisplayMode::Sixel
    } else if capabilities.iterm2 {
        DisplayMode::Iterm2
    } else {
        DisplayMode::Ansi
    }
}

fn env_lc(getenv: &impl Fn(&str) -> Option<String>, key: &str) -> Option<String> {
    getenv(key).map(|value| value.to_ascii_lowercase())
}

fn env_present(getenv: &impl Fn(&str) -> Option<String>, key: &str) -> bool {
    getenv(key).is_some_and(|value| !value.is_empty())
}

fn env_flag(getenv: &impl Fn(&str) -> Option<String>, key: &str) -> bool {
    getenv(key).is_some_and(|value| {
        let value = value.to_ascii_lowercase();
        !matches!(value.as_str(), "" | "0" | "false" | "no" | "off")
    })
}

/// Renders an image through the Kitty graphics protocol as inline PNG data.
pub fn render_kitty(image: &PreviewImage) -> String {
    let png = encode_png(image);
    let encoded = base64::engine::general_purpose::STANDARD.encode(png);
    let mut out = String::new();
    let mut offset = 0;
    let chunk_size = 4096;
    while offset < encoded.len() {
        let end = (offset + chunk_size).min(encoded.len());
        let more = u8::from(end < encoded.len());
        if offset == 0 {
            out.push_str(&format!("\x1b_Ga=T,f=100,m={more};"));
        } else {
            out.push_str(&format!("\x1b_Gm={more};"));
        }
        out.push_str(&encoded[offset..end]);
        out.push_str("\x1b\\");
        offset = end;
    }
    out.push('\n');
    out
}

/// Renders an image through the iTerm2 inline image protocol as inline PNG data.
pub fn render_iterm2(image: &PreviewImage) -> String {
    let png = encode_png(image);
    let encoded = base64::engine::general_purpose::STANDARD.encode(&png);
    format!(
        "\x1b]1337;File=inline=1;size={};preserveAspectRatio=1:{}\x07\n",
        png.len(),
        encoded
    )
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
    let mut out = format!(
        "\x1bP9;1;0q\"1;1;{};{}",
        image.width.max(1),
        image.height.max(1)
    );
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

fn encode_png(image: &PreviewImage) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(image.pixels.len() * 3);
    image
        .pixels
        .iter()
        .for_each(|pixel| rgb.extend_from_slice(pixel));
    let mut png = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new(&mut png);
    encoder
        .write_image(
            &rgb,
            image.width,
            image.height,
            image::ExtendedColorType::Rgb8,
        )
        .expect("encoding RGB preview as PNG should not fail");
    png
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |key| {
            pairs
                .iter()
                .find_map(|(env_key, value)| (*env_key == key).then(|| (*value).to_string()))
        }
    }

    #[test]
    fn sixel_contains_dcs_markers() {
        let image = test_image();
        let out = render_sixel(&image);
        assert!(out.starts_with("\x1bP9;1;0q\"1;1;1;1"));
        assert!(out.ends_with("\x1b\\"));
    }

    #[test]
    fn sixel_declares_square_pixels_and_raster_size() {
        let image = PreviewImage {
            width: 16,
            height: 9,
            source_width: 16,
            source_height: 9,
            pixels: vec![[255, 0, 0]; 16 * 9],
            source: "test".to_string(),
        };
        let out = render_sixel(&image);

        assert!(out.starts_with("\x1bP9;1;0q"));
        assert!(out.contains("\"1;1;16;9"));
    }

    #[test]
    fn iterm2_contains_osc_1337_marker() {
        let image = test_image();
        let out = render_iterm2(&image);
        assert!(out.starts_with("\x1b]1337;File=inline=1;"));
        assert!(out.ends_with("\x07\n"));
    }

    #[test]
    fn auto_prefers_kitty_for_ghostty_over_ssh_term() {
        assert_eq!(
            auto_display_mode_from_env(env(&[("TERM", "xterm-ghostty")])),
            DisplayMode::Kitty
        );
    }

    #[test]
    fn auto_uses_sixel_for_windows_terminal_hint() {
        assert_eq!(
            auto_display_mode_from_env(env(&[("WT_SESSION", "abc")])),
            DisplayMode::Sixel
        );
    }

    #[test]
    fn auto_uses_sixel_for_windows_terminal_program_hint() {
        assert_eq!(
            auto_display_mode_from_env(env(&[("TERM_PROGRAM", "Windows_Terminal")])),
            DisplayMode::Sixel
        );
    }

    #[test]
    fn auto_falls_back_to_ansi_when_ssh_hides_terminal_identity() {
        assert_eq!(
            auto_display_mode_from_env(env(&[
                ("SSH_CONNECTION", "192.0.2.1 1 192.0.2.2 2"),
                ("TERM", "xterm-256color")
            ])),
            DisplayMode::Ansi
        );
    }

    #[test]
    fn auto_can_be_forced_for_unforwarded_ssh_environment() {
        assert_eq!(
            auto_display_mode_from_env(env(&[("IMQ_IMAGE_PROTOCOL", "sixel")])),
            DisplayMode::Sixel
        );
    }

    fn test_image() -> PreviewImage {
        PreviewImage {
            width: 1,
            height: 1,
            source_width: 1,
            source_height: 1,
            pixels: vec![[255, 0, 0]],
            source: "test".to_string(),
        }
    }
}
