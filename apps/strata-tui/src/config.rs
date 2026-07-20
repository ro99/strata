use std::collections::HashSet;
use std::env;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelType {
    Glm,
    DeepSeek,
}

impl ModelType {
    pub const fn as_arg(self) -> &'static str {
        match self {
            Self::Glm => "glm",
            Self::DeepSeek => "deepseek",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Glm => "GLM-5.2",
            Self::DeepSeek => "DeepSeek V4",
        }
    }

    pub const fn context_limit(self) -> u32 {
        match self {
            Self::Glm => 2_048,
            Self::DeepSeek => 1_048_576,
        }
    }

    pub const fn toggled(self) -> Self {
        match self {
            Self::Glm => Self::DeepSeek,
            Self::DeepSeek => Self::Glm,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RunConfig {
    pub model_path: String,
    pub model_type: ModelType,
    pub devices: String,
    pub context_size: String,
    pub max_new: String,
    pub temperature: String,
    pub vram_fraction: String,
    pub seed: String,
    pub flash_attention: bool,
    pub chat_binary: String,
}

impl Default for RunConfig {
    fn default() -> Self {
        let (model_path, model_type, context_size) = detect_model();
        Self {
            model_path,
            model_type,
            devices: "0,1,2".into(),
            context_size,
            max_new: "512".into(),
            temperature: "0".into(),
            vram_fraction: "0.85".into(),
            seed: "33377335".into(),
            flash_attention: false,
            chat_binary: detect_chat_binary(),
        }
    }
}

impl RunConfig {
    pub fn validate(&self) -> Result<ValidatedConfig, String> {
        if self.model_path.trim().is_empty() {
            return Err("Choose a model directory".into());
        }
        let model = Path::new(self.model_path.trim());
        if !model.is_dir() {
            return Err(format!(
                "Model directory does not exist: {}",
                model.display()
            ));
        }

        let devices = parse_devices(&self.devices)?;
        let context_size = parse_u32("Context size", &self.context_size)?;
        if context_size > self.model_type.context_limit() {
            return Err(format!(
                "{} supports at most {} context tokens",
                self.model_type.label(),
                self.model_type.context_limit()
            ));
        }
        let max_new = parse_u32("Maximum new tokens", &self.max_new)?;
        if max_new > context_size {
            return Err("Maximum new tokens cannot exceed the context size".into());
        }
        let temperature = parse_f64("Temperature", &self.temperature)?;
        if !(0.0..=10.0).contains(&temperature) {
            return Err("Temperature must be between 0 and 10".into());
        }
        let vram_fraction = parse_f64("VRAM fraction", &self.vram_fraction)?;
        if !(0.0 < vram_fraction && vram_fraction <= 0.95) {
            return Err("VRAM fraction must be greater than 0 and at most 0.95".into());
        }
        let seed = self
            .seed
            .trim()
            .parse::<u64>()
            .map_err(|_| "Seed must be an unsigned integer".to_string())?;
        if self.chat_binary.trim().is_empty() {
            return Err("Choose the strata-chat executable".into());
        }
        let binary = self.chat_binary.trim();
        if binary.contains('/') && !Path::new(binary).is_file() {
            return Err(format!("strata-chat executable does not exist: {binary}"));
        }

        Ok(ValidatedConfig {
            model_path: model.to_path_buf(),
            model_type: self.model_type,
            devices: devices
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(","),
            context_size,
            max_new,
            temperature,
            vram_fraction,
            seed,
            flash_attention: self.flash_attention,
            chat_binary: binary.to_string(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct ValidatedConfig {
    pub model_path: PathBuf,
    pub model_type: ModelType,
    pub devices: String,
    pub context_size: u32,
    pub max_new: u32,
    pub temperature: f64,
    pub vram_fraction: f64,
    pub seed: u64,
    pub flash_attention: bool,
    pub chat_binary: String,
}

impl ValidatedConfig {
    pub fn command_args(&self) -> Vec<String> {
        let mut args = vec![
            "--model".into(),
            self.model_path.to_string_lossy().into_owned(),
            "--model-type".into(),
            self.model_type.as_arg().into(),
            "--devices".into(),
            self.devices.clone(),
            "--context-size".into(),
            self.context_size.to_string(),
            "--max-new".into(),
            self.max_new.to_string(),
            "--temperature".into(),
            self.temperature.to_string(),
            "--vram-fraction".into(),
            self.vram_fraction.to_string(),
            "--seed".into(),
            self.seed.to_string(),
            "--protocol".into(),
            "jsonl".into(),
        ];
        if self.flash_attention {
            args.push("--flash-attention".into());
        }
        args
    }

    pub const fn is_exact(&self) -> bool {
        self.temperature == 0.0
    }
}

#[derive(Debug)]
pub struct Cli {
    pub config: RunConfig,
    pub launch_immediately: bool,
}

pub fn parse_cli() -> Result<Cli, String> {
    let mut config = RunConfig::default();
    let mut supplied_model = false;
    let mut supplied_type = false;
    let mut force_setup = false;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        let mut next = || {
            args.next()
                .ok_or_else(|| format!("Missing value after {argument}"))
        };
        match argument.as_str() {
            "--help" | "-h" => return Err(help_text().into()),
            "--setup" => force_setup = true,
            "--flash-attention" => config.flash_attention = true,
            "--model" => {
                config.model_path = next()?;
                supplied_model = true;
            }
            "--model-type" => {
                config.model_type = match next()?.as_str() {
                    "glm" => ModelType::Glm,
                    "deepseek" => ModelType::DeepSeek,
                    value => return Err(format!("Unknown model type: {value}")),
                };
                supplied_type = true;
            }
            "--devices" => config.devices = next()?,
            "--context-size" | "--max-context" => config.context_size = next()?,
            "--max-new" => config.max_new = next()?,
            "--temperature" => config.temperature = next()?,
            "--vram-fraction" => config.vram_fraction = next()?,
            "--seed" => config.seed = next()?,
            "--chat-binary" => config.chat_binary = next()?,
            value => return Err(format!("Unknown argument: {value}\n\n{}", help_text())),
        }
    }
    Ok(Cli {
        config,
        launch_immediately: supplied_model && supplied_type && !force_setup,
    })
}

pub const fn help_text() -> &'static str {
    "strata-tui — a Ratatui cockpit for Strata\n\n\
Usage: strata-tui [OPTIONS]\n\n\
  --model DIR                 model directory\n\
  --model-type glm|deepseek   runtime adapter\n\
  --devices 0,1,2            CUDA device list\n\
  --context-size N            logical context ceiling\n\
  --max-new N                 maximum generated tokens\n\
  --temperature F             0 is exact greedy decoding\n\
  --vram-fraction F           free VRAM cache fraction (max 0.95)\n\
  --seed N                    sampling seed\n\
  --flash-attention           enable the CUDA FlashAttention candidate\n\
  --chat-binary PATH          strata-chat executable\n\
  --setup                     open the launch form even with a full config\n\
  -h, --help                  print this help"
}

fn parse_devices(text: &str) -> Result<Vec<u32>, String> {
    let mut seen = HashSet::new();
    let mut devices = Vec::new();
    for part in text.split(',') {
        let value = part
            .trim()
            .parse::<u32>()
            .map_err(|_| "Devices must be a comma-separated list like 0,1,2".to_string())?;
        if !seen.insert(value) {
            return Err(format!("CUDA device {value} appears more than once"));
        }
        devices.push(value);
    }
    if devices.is_empty() {
        return Err("Choose at least one CUDA device".into());
    }
    Ok(devices)
}

fn parse_u32(label: &str, text: &str) -> Result<u32, String> {
    let value = text
        .trim()
        .parse::<u32>()
        .map_err(|_| format!("{label} must be a positive integer"))?;
    if value == 0 {
        return Err(format!("{label} must be greater than zero"));
    }
    Ok(value)
}

fn parse_f64(label: &str, text: &str) -> Result<f64, String> {
    let value = text
        .trim()
        .parse::<f64>()
        .map_err(|_| format!("{label} must be a number"))?;
    if !value.is_finite() {
        return Err(format!("{label} must be finite"));
    }
    Ok(value)
}

fn detect_model() -> (String, ModelType, String) {
    let deepseek = Path::new("models/DeepSeek-V4-Flash-DSpark");
    if deepseek.is_dir() {
        return (
            deepseek.to_string_lossy().into_owned(),
            ModelType::DeepSeek,
            "8192".into(),
        );
    }
    let glm = Path::new("models/glm52");
    if glm.is_dir() {
        return (
            glm.to_string_lossy().into_owned(),
            ModelType::Glm,
            "2048".into(),
        );
    }
    (String::new(), ModelType::DeepSeek, "8192".into())
}

fn detect_chat_binary() -> String {
    if let Ok(value) = env::var("STRATA_CHAT_BIN")
        && !value.trim().is_empty()
    {
        return value;
    }
    if let Ok(executable) = env::current_exe()
        && let Some(directory) = executable.parent()
    {
        let sibling = directory.join("strata-chat");
        if sibling.is_file() {
            return sibling.to_string_lossy().into_owned();
        }
    }
    let local = Path::new("build/strata-chat");
    if local.is_file() {
        return local.to_string_lossy().into_owned();
    }
    "strata-chat".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn devices_are_normalized_and_duplicates_rejected() {
        assert_eq!(parse_devices("0, 2,1").unwrap(), vec![0, 2, 1]);
        assert!(parse_devices("0,0").unwrap_err().contains("more than once"));
        assert!(parse_devices("0,nope").is_err());
    }

    #[test]
    fn model_context_contract_is_enforced() {
        let config = RunConfig {
            model_path: ".".into(),
            model_type: ModelType::Glm,
            context_size: "2049".into(),
            chat_binary: "strata-chat".into(),
            ..RunConfig::default()
        };
        assert!(config.validate().unwrap_err().contains("at most 2048"));
    }
}
