use serde::Deserialize;

pub const PROTOCOL_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize)]
pub struct Envelope {
    pub protocol: String,
    pub version: u32,
    pub event: String,
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub fatal: bool,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub tokens: u64,
    #[serde(default)]
    pub tok_s: f64,
    #[serde(default)]
    pub load_seconds: f64,
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub prefill_tokens: u64,
    #[serde(default)]
    pub reused_prompt_tokens: u64,
    #[serde(default)]
    pub decode_tokens: u64,
    #[serde(default)]
    pub prefill_seconds: f64,
    #[serde(default)]
    pub prefill_tok_s: f64,
    #[serde(default)]
    pub decode_seconds: f64,
    #[serde(default)]
    pub decode_tok_s: f64,
}

impl Envelope {
    pub fn parse(line: &str) -> Result<Self, String> {
        let event: Self = serde_json::from_str(line)
            .map_err(|error| format!("Invalid runtime protocol record: {error}"))?;
        if event.protocol != "strata-chat" {
            return Err(format!("Unknown runtime protocol: {}", event.protocol));
        }
        if event.version != PROTOCOL_VERSION {
            return Err(format!(
                "Unsupported runtime protocol version {} (expected {PROTOCOL_VERSION})",
                event.version
            ));
        }
        Ok(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_record_round_trips_unicode() {
        let event = Envelope::parse(
            r#"{"protocol":"strata-chat","version":1,"event":"token","text":"snowman ☃","tokens":7,"tok_s":4.2}"#,
        )
        .unwrap();
        assert_eq!(event.event, "token");
        assert_eq!(event.text, "snowman ☃");
        assert_eq!(event.tokens, 7);
    }

    #[test]
    fn incompatible_protocol_is_explicitly_rejected() {
        let error = Envelope::parse(r#"{"protocol":"strata-chat","version":2,"event":"ready"}"#)
            .unwrap_err();
        assert!(error.contains("version 2"));
    }

    #[test]
    fn turn_done_reports_incremental_prefill_work() {
        let event = Envelope::parse(
            r#"{"protocol":"strata-chat","version":1,"event":"turn_done","prompt_tokens":40,"prefill_tokens":12,"reused_prompt_tokens":28}"#,
        )
        .unwrap();
        assert_eq!(event.prompt_tokens, 40);
        assert_eq!(event.prefill_tokens, 12);
        assert_eq!(event.reused_prompt_tokens, 28);
    }
}
