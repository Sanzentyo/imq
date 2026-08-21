//! Remote input and ordered video frame-pair helpers.

use crate::{Error, Result};
use std::path::PathBuf;

/// Local, stdin, or SSH-backed input.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum InputSpec {
    /// Local filesystem path.
    Local(PathBuf),
    /// Standard input.
    Stdin,
    /// SSH URI input.
    Ssh(SshInput),
}

impl InputSpec {
    /// Parses a CLI input string.
    pub fn parse(input: &str) -> Result<Self> {
        if input == "-" {
            return Ok(Self::Stdin);
        }
        if let Some(rest) = input.strip_prefix("ssh://") {
            return Ok(Self::Ssh(SshInput::parse_uri_rest(rest)?));
        }
        Ok(Self::Local(PathBuf::from(input)))
    }

    /// User-facing label.
    pub fn display_label(&self) -> String {
        match self {
            Self::Local(path) => path.display().to_string(),
            Self::Stdin => "stdin".to_string(),
            Self::Ssh(spec) => spec.uri(),
        }
    }

    /// Returns true when the input is local.
    pub fn is_local(&self) -> bool {
        matches!(self, Self::Local(_))
    }

    /// Returns the file extension when available.
    pub fn extension(&self) -> Option<&str> {
        match self {
            Self::Local(path) => path.extension().and_then(|ext| ext.to_str()),
            Self::Ssh(spec) => spec.path.rsplit_once('.').map(|(_, ext)| ext),
            Self::Stdin => None,
        }
    }
}

/// SSH URI input.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SshInput {
    /// Optional user name.
    pub user: Option<String>,
    /// Host name or address.
    pub host: String,
    /// Optional SSH port.
    pub port: Option<u16>,
    /// Absolute remote path.
    pub path: String,
}

impl SshInput {
    fn parse_uri_rest(rest: &str) -> Result<Self> {
        let (authority, path) = rest.split_once('/').ok_or_else(|| {
            Error::unsupported("ssh URI must include an absolute path, such as ssh://host/path")
        })?;
        if authority.is_empty() {
            return Err(Error::unsupported("ssh URI host must not be empty"));
        }
        let path = format!("/{path}");
        let (user, host_port) = authority
            .rsplit_once('@')
            .map_or((None, authority), |(user, host_port)| {
                (Some(user.to_string()), host_port)
            });
        let (host, port) = parse_host_port(host_port)?;
        let input = Self {
            user,
            host,
            port,
            path,
        };
        input.validate()?;
        Ok(input)
    }

    /// Validates values before they are passed to SSH/SCP command-line tools.
    pub fn validate(&self) -> Result<()> {
        validate_ssh_component("host", &self.host)?;
        if let Some(user) = &self.user {
            validate_ssh_component("user", user)?;
            if user.contains('@') {
                return Err(Error::unsupported("ssh user must not contain `@`"));
            }
        }
        if self.port == Some(0) {
            return Err(Error::unsupported("ssh port must be greater than zero"));
        }
        if !self.path.starts_with('/') {
            return Err(Error::unsupported("ssh path must be absolute"));
        }
        if self.path.contains('\0') {
            return Err(Error::unsupported("ssh path must not contain NUL"));
        }
        Ok(())
    }

    /// Returns the canonical URI string.
    pub fn uri(&self) -> String {
        let mut uri = String::from("ssh://");
        if let Some(user) = &self.user {
            uri.push_str(user);
            uri.push('@');
        }
        if self.host.contains(':') {
            uri.push('[');
            uri.push_str(&self.host);
            uri.push(']');
        } else {
            uri.push_str(&self.host);
        }
        if let Some(port) = self.port {
            uri.push(':');
            uri.push_str(&port.to_string());
        }
        uri.push_str(&self.path);
        uri
    }

    /// Host target suitable for `ssh`.
    pub fn ssh_target(&self) -> String {
        self.user
            .as_ref()
            .map_or_else(|| self.host.clone(), |user| format!("{user}@{}", self.host))
    }

    /// Host prefix suitable for an SCP remote-path argument.
    pub fn scp_target(&self) -> String {
        let host = if self.host.contains(':') {
            format!("[{}]", self.host)
        } else {
            self.host.clone()
        };
        self.user
            .as_ref()
            .map_or(host.clone(), |user| format!("{user}@{host}"))
    }
}

fn parse_host_port(input: &str) -> Result<(String, Option<u16>)> {
    if let Some(bracketed) = input.strip_prefix('[') {
        let (host, rest) = bracketed
            .split_once(']')
            .ok_or_else(|| Error::unsupported("invalid bracketed IPv6 ssh host"))?;
        if host.is_empty() {
            return Err(Error::unsupported("ssh URI host must not be empty"));
        }
        let port = match rest {
            "" => None,
            rest if rest.starts_with(':') => Some(parse_ssh_port(&rest[1..])?),
            _ => {
                return Err(Error::unsupported(
                    "unexpected characters after bracketed ssh host",
                ));
            }
        };
        return Ok((host.to_string(), port));
    }
    if input.matches(':').count() > 1 {
        return Ok((input.to_string(), None));
    }
    let Some((host, port)) = input.rsplit_once(':') else {
        return Ok((input.to_string(), None));
    };
    if host.is_empty() || port.is_empty() {
        return Ok((input.to_string(), None));
    }
    if port.chars().all(|c| c.is_ascii_digit()) {
        Ok((host.to_string(), Some(parse_ssh_port(port)?)))
    } else {
        Ok((input.to_string(), None))
    }
}

fn parse_ssh_port(port: &str) -> Result<u16> {
    let port = port
        .parse::<u16>()
        .map_err(|_| Error::unsupported(format!("invalid ssh URI port `{port}`")))?;
    if port == 0 {
        Err(Error::unsupported("ssh port must be greater than zero"))
    } else {
        Ok(port)
    }
}

fn validate_ssh_component(label: &str, value: &str) -> Result<()> {
    if value.is_empty() {
        return Err(Error::unsupported(format!("ssh {label} must not be empty")));
    }
    if value.starts_with('-') {
        return Err(Error::unsupported(format!(
            "ssh {label} must not start with `-`"
        )));
    }
    if value.chars().any(char::is_whitespace) || value.chars().any(char::is_control) {
        return Err(Error::unsupported(format!(
            "ssh {label} must not contain whitespace or control characters"
        )));
    }
    Ok(())
}

/// Remote file transfer behavior.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum RemoteTransferMode {
    /// Stream bytes over SSH stdout. This is the safe default.
    #[default]
    Stream,
    /// Copy remote still-image inputs to a local temporary file.
    CopyInput,
    /// Copy extracted remote frame images, not source videos.
    CopyFrame,
    /// Copy whole remote source files explicitly.
    CopySource,
}

/// Remote video frame stream format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum RemoteFrameFormat {
    /// PNG image pipe.
    #[default]
    Png,
    /// Raw RGBA pipe. Reserved for a future dimension-probed path.
    Rgba,
}

/// Remote command options.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RemoteOptions {
    /// Transfer mode.
    pub transfer: RemoteTransferMode,
    /// Remote frame stream format.
    pub frame_format: RemoteFrameFormat,
    /// SSH executable.
    pub ssh: PathBuf,
    /// SCP executable.
    pub scp: PathBuf,
    /// Optional connect timeout in seconds.
    pub connect_timeout_seconds: Option<u64>,
    /// Whether to pass BatchMode=yes to SSH.
    pub batch_mode: bool,
    /// Local copy directory for explicit copy modes.
    pub copy_dir: Option<PathBuf>,
    /// Preserve temporary files.
    pub keep_temp: bool,
    /// Maximum captured stdout bytes.
    pub max_bytes: usize,
}

impl Default for RemoteOptions {
    fn default() -> Self {
        Self {
            transfer: RemoteTransferMode::Stream,
            frame_format: RemoteFrameFormat::Png,
            ssh: PathBuf::from("ssh"),
            scp: PathBuf::from("scp"),
            connect_timeout_seconds: None,
            batch_mode: false,
            copy_dir: None,
            keep_temp: false,
            max_bytes: 512 * 1024 * 1024,
        }
    }
}

/// Ordered frame pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VideoFramePair {
    /// Reference frame index.
    pub reference: u64,
    /// Distorted/candidate frame index.
    pub distorted: u64,
    /// Optional canonical/reporting frame index from 3-tuple syntax.
    pub label_frame_index: Option<u64>,
}

impl VideoFramePair {
    /// Creates a pair without an explicit label frame.
    pub fn new(reference: u64, distorted: u64) -> Self {
        Self {
            reference,
            distorted,
            label_frame_index: None,
        }
    }

    /// Creates a pair with an explicit label frame.
    pub fn with_label(reference: u64, distorted: u64, label_frame_index: u64) -> Self {
        Self {
            reference,
            distorted,
            label_frame_index: Some(label_frame_index),
        }
    }
}

/// Parses `--video-frames`.
pub fn parse_video_frame_pairs(input: &str) -> Result<Vec<VideoFramePair>> {
    if input.is_empty() {
        return Err(Error::unsupported(
            "--video-frames must contain at least one frame or frame pair",
        ));
    }
    let items = split_frame_items(input)?;
    if items.is_empty() {
        return Err(Error::unsupported(
            "--video-frames must contain at least one frame or frame pair",
        ));
    }
    items
        .iter()
        .map(|item| parse_video_frame_pair_item(item))
        .collect()
}

fn split_frame_items(input: &str) -> Result<Vec<String>> {
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut items = Vec::new();
    for (index, ch) in input.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth = depth
                    .checked_sub(1)
                    .ok_or_else(|| Error::unsupported("invalid --video-frames: unmatched `)`"))?;
            }
            ',' if depth == 0 => {
                let item = input[start..index].trim();
                if item.is_empty() {
                    return Err(Error::unsupported("invalid --video-frames: empty item"));
                }
                items.push(item.to_string());
                start = index + 1;
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err(Error::unsupported("invalid --video-frames: unmatched `(`"));
    }
    let item = input[start..].trim();
    if item.is_empty() {
        return Err(Error::unsupported("invalid --video-frames: empty item"));
    }
    items.push(item.to_string());
    Ok(items)
}

fn parse_video_frame_pair_item(item: &str) -> Result<VideoFramePair> {
    if let Some(tuple) = item.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
        let values = tuple
            .split(',')
            .map(str::trim)
            .map(parse_u64_token)
            .collect::<Result<Vec<_>>>()?;
        return match values.as_slice() {
            [reference, distorted] => Ok(VideoFramePair::new(*reference, *distorted)),
            [reference, distorted, label] => {
                Ok(VideoFramePair::with_label(*reference, *distorted, *label))
            }
            _ => Err(Error::unsupported(
                "invalid --video-frames: tuple must be `(REF,DIST)` or `(REF,DIST,LABEL)`",
            )),
        };
    }
    if let Some((reference, distorted)) = item.split_once("->") {
        return Ok(VideoFramePair::new(
            parse_u64_token(reference.trim())?,
            parse_u64_token(distorted.trim())?,
        ));
    }
    if let Some((reference, distorted)) = item.split_once(':') {
        return Ok(VideoFramePair::new(
            parse_u64_token(reference.trim())?,
            parse_u64_token(distorted.trim())?,
        ));
    }
    let frame = parse_u64_token(item)?;
    Ok(VideoFramePair::new(frame, frame))
}

fn parse_u64_token(input: &str) -> Result<u64> {
    if input.is_empty() || !input.chars().all(|c| c.is_ascii_digit()) {
        return Err(Error::unsupported(format!(
            "invalid --video-frames frame index `{input}`"
        )));
    }
    input
        .parse::<u64>()
        .map_err(|_| Error::unsupported(format!("invalid --video-frames frame index `{input}`")))
}

/// Quotes a value for POSIX remote shell usage.
pub fn shell_quote_posix(input: &str) -> String {
    if input.is_empty() {
        return "''".to_string();
    }
    let mut quoted = String::from("'");
    for ch in input.chars() {
        if ch == '\'' {
            quoted.push_str("'\\''");
        } else {
            quoted.push(ch);
        }
    }
    quoted.push('\'');
    quoted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_frame_pairs() {
        assert_eq!(
            parse_video_frame_pairs("0,30,60").unwrap(),
            vec![
                VideoFramePair::new(0, 0),
                VideoFramePair::new(30, 30),
                VideoFramePair::new(60, 60)
            ]
        );
        assert_eq!(
            parse_video_frame_pairs("0:1,30->31,(60,61)").unwrap(),
            vec![
                VideoFramePair::new(0, 1),
                VideoFramePair::new(30, 31),
                VideoFramePair::new(60, 61)
            ]
        );
    }

    #[test]
    fn parses_triple_tuple_with_label_frame() {
        assert_eq!(
            parse_video_frame_pairs("(0,1,0), (2,3,2)").unwrap(),
            vec![
                VideoFramePair::with_label(0, 1, 0),
                VideoFramePair::with_label(2, 3, 2)
            ]
        );
    }

    #[test]
    fn rejects_invalid_frame_pairs() {
        for input in ["", "0:", ":1", "(0,)", "(,1)", "0:-1", "abc", "0, "] {
            assert!(parse_video_frame_pairs(input).is_err(), "{input}");
        }
    }

    #[test]
    fn parses_ssh_uri() {
        let InputSpec::Ssh(input) =
            InputSpec::parse("ssh://alice@example.com:2222/home/alice/a.png").unwrap()
        else {
            panic!("expected ssh input");
        };
        assert_eq!(input.user.as_deref(), Some("alice"));
        assert_eq!(input.host, "example.com");
        assert_eq!(input.port, Some(2222));
        assert_eq!(input.path, "/home/alice/a.png");
    }

    #[test]
    fn parses_ipv6_ssh_uri_and_formats_targets() {
        let InputSpec::Ssh(input) =
            InputSpec::parse("ssh://alice@[2001:db8::1]:2222/home/a.mp4").unwrap()
        else {
            panic!("expected ssh input");
        };
        assert_eq!(input.host, "2001:db8::1");
        assert_eq!(input.port, Some(2222));
        assert_eq!(input.uri(), "ssh://alice@[2001:db8::1]:2222/home/a.mp4");
        assert_eq!(input.ssh_target(), "alice@2001:db8::1");
        assert_eq!(input.scp_target(), "alice@[2001:db8::1]");
    }

    #[test]
    fn rejects_ssh_option_injection_and_invalid_authority() {
        for input in [
            "ssh://-oProxyCommand=evil/tmp/a.png",
            "ssh://-user@example.com/tmp/a.png",
            "ssh://user name@example.com/tmp/a.png",
            "ssh://host:0/tmp/a.png",
            "ssh://[2001:db8::1]junk/tmp/a.png",
        ] {
            assert!(InputSpec::parse(input).is_err(), "{input}");
        }
    }

    #[test]
    fn quotes_posix_shell() {
        assert_eq!(shell_quote_posix("abc'def"), "'abc'\\''def'");
    }
}
