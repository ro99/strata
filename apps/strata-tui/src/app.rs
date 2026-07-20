use crate::config::{Cli, ModelType, RunConfig, ValidatedConfig};
use crate::process::{ProcessEvent, RuntimeProcess};
use crate::protocol::Envelope;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

const MAX_MESSAGES: usize = 128;
const MAX_CHAT_BYTES: usize = 4 * 1024 * 1024;
const MAX_LOG_LINES: usize = 400;
const MAX_PROMPT_HISTORY: usize = 64;
const MAX_INPUT_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Screen {
    Setup,
    Session,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimePhase {
    Offline,
    Launching,
    Loading,
    Ready,
    Prefill,
    Decode,
    Error,
    Exited,
}

impl RuntimePhase {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Offline => "SETUP",
            Self::Launching => "LAUNCHING",
            Self::Loading => "LOADING",
            Self::Ready => "READY",
            Self::Prefill => "PREFILL",
            Self::Decode => "DECODING",
            Self::Error => "ERROR",
            Self::Exited => "EXITED",
        }
    }

    pub const fn is_busy(self) -> bool {
        matches!(
            self,
            Self::Launching | Self::Loading | Self::Prefill | Self::Decode
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Role {
    User,
    Assistant,
    System,
}

#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: Role,
    pub text: String,
    pub complete: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Metrics {
    pub load_seconds: f64,
    pub prompt_tokens: u64,
    pub decode_tokens: u64,
    pub prefill_seconds: f64,
    pub prefill_tok_s: f64,
    pub decode_seconds: f64,
    pub decode_tok_s: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SetupField {
    ModelType,
    ModelPath,
    Devices,
    Context,
    MaxNew,
    Temperature,
    VramFraction,
    Seed,
    FlashAttention,
    ChatBinary,
    Launch,
}

impl SetupField {
    pub const ALL: [Self; 11] = [
        Self::ModelType,
        Self::ModelPath,
        Self::Devices,
        Self::Context,
        Self::MaxNew,
        Self::Temperature,
        Self::VramFraction,
        Self::Seed,
        Self::FlashAttention,
        Self::ChatBinary,
        Self::Launch,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::ModelType => "MODEL",
            Self::ModelPath => "CHECKPOINT",
            Self::Devices => "CUDA DEVICES",
            Self::Context => "CONTEXT",
            Self::MaxNew => "MAX NEW",
            Self::Temperature => "TEMPERATURE",
            Self::VramFraction => "VRAM FRACTION",
            Self::Seed => "SEED",
            Self::FlashAttention => "ATTENTION",
            Self::ChatBinary => "RUNTIME",
            Self::Launch => "",
        }
    }

    const fn is_text(self) -> bool {
        !matches!(self, Self::ModelType | Self::FlashAttention | Self::Launch)
    }
}

pub struct App {
    pub screen: Screen,
    pub phase: RuntimePhase,
    pub setup: RunConfig,
    pub setup_field: usize,
    pub setup_cursor: usize,
    pub config: Option<ValidatedConfig>,
    pub messages: VecDeque<ChatMessage>,
    pub logs: VecDeque<String>,
    pub input: String,
    pub input_cursor: usize,
    pub metrics: Metrics,
    pub speed_history: VecDeque<u64>,
    pub scroll_from_bottom: usize,
    pub show_logs: bool,
    pub show_help: bool,
    pub notice: Option<String>,
    pub tick: u64,
    pub status_message: String,
    pub started_at: Option<Instant>,
    pub quit_armed: bool,
    process: Option<RuntimeProcess>,
    prompt_history: VecDeque<String>,
    history_position: Option<usize>,
    draft_input: String,
}

impl App {
    pub fn new(cli: Cli) -> Self {
        let mut app = Self {
            screen: Screen::Setup,
            phase: RuntimePhase::Offline,
            setup_cursor: cli.config.model_path.len(),
            setup: cli.config,
            setup_field: 0,
            config: None,
            messages: VecDeque::new(),
            logs: VecDeque::new(),
            input: String::new(),
            input_cursor: 0,
            metrics: Metrics::default(),
            speed_history: VecDeque::new(),
            scroll_from_bottom: 0,
            show_logs: false,
            show_help: false,
            notice: None,
            tick: 0,
            status_message: "Configure an exact Strata session".into(),
            started_at: None,
            quit_armed: false,
            process: None,
            prompt_history: VecDeque::new(),
            history_position: None,
            draft_input: String::new(),
        };
        if cli.launch_immediately {
            app.launch();
        }
        app
    }

    #[cfg(test)]
    pub fn for_test() -> Self {
        Self::new(Cli {
            config: RunConfig::default(),
            launch_immediately: false,
        })
    }

    pub fn launch(&mut self) {
        let config = match self.setup.validate() {
            Ok(config) => config,
            Err(error) => {
                self.notice = Some(error);
                return;
            }
        };
        match RuntimeProcess::spawn(&config) {
            Ok(process) => {
                self.process = Some(process);
                self.config = Some(config);
                self.screen = Screen::Session;
                self.phase = RuntimePhase::Launching;
                self.status_message = "Opening runtime protocol".into();
                self.started_at = Some(Instant::now());
                self.messages.clear();
                self.logs.clear();
                self.metrics = Metrics::default();
                self.speed_history.clear();
                self.notice = None;
            }
            Err(error) => self.notice = Some(error),
        }
    }

    pub fn update(&mut self) {
        self.tick = self.tick.wrapping_add(1);
        let (events, exit) = if let Some(process) = self.process.as_mut() {
            (process.drain_events(), process.poll_exit())
        } else {
            (Vec::new(), None)
        };
        for event in events {
            self.handle_process_event(event);
        }
        if let Some(exit) = exit {
            match exit {
                Ok(0) if !matches!(self.phase, RuntimePhase::Error) => {
                    self.phase = RuntimePhase::Exited;
                    self.status_message = "Runtime exited".into();
                }
                Ok(code) => {
                    self.phase = RuntimePhase::Error;
                    self.status_message = format!("Runtime exited with status {code}");
                    self.notice = Some(self.status_message.clone());
                }
                Err(error) => {
                    self.phase = RuntimePhase::Error;
                    self.status_message.clone_from(&error);
                    self.notice = Some(error);
                }
            }
        }
    }

    fn handle_process_event(&mut self, event: ProcessEvent) {
        match event {
            ProcessEvent::Protocol(event) => self.handle_protocol(event),
            ProcessEvent::Log(line) => self.push_log(line),
            ProcessEvent::ProtocolError(error) => {
                self.push_log(format!("[protocol] {error}"));
                self.phase = RuntimePhase::Error;
                self.notice = Some(error);
            }
        }
    }

    fn handle_protocol(&mut self, event: Envelope) {
        match event.event.as_str() {
            "hello" => {
                self.phase = RuntimePhase::Loading;
                self.status_message = "Loading and admitting model weights".into();
            }
            "status" => {
                self.phase = RuntimePhase::Loading;
                self.status_message = event.message;
            }
            "ready" => {
                self.phase = RuntimePhase::Ready;
                self.metrics.load_seconds = event.load_seconds;
                self.status_message = "Model ready — ask anything".into();
                self.notice = None;
            }
            "turn_start" => {
                self.phase = RuntimePhase::Prefill;
                self.status_message = "Processing prompt".into();
            }
            "token" => {
                self.phase = RuntimePhase::Decode;
                self.status_message = "Streaming exact runtime output".into();
                self.metrics.decode_tokens = event.tokens;
                self.metrics.decode_tok_s = event.tok_s;
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let speed_sample = (event.tok_s.clamp(0.0, 1_000_000.0) * 100.0) as u64;
                self.speed_history.push_back(speed_sample);
                while self.speed_history.len() > 72 {
                    self.speed_history.pop_front();
                }
                self.append_assistant_text(&event.text);
            }
            "turn_done" => {
                self.phase = RuntimePhase::Ready;
                self.status_message = "Ready for the next prompt".into();
                self.metrics.prompt_tokens = event.prompt_tokens;
                self.metrics.decode_tokens = event.decode_tokens;
                self.metrics.prefill_seconds = event.prefill_seconds;
                self.metrics.prefill_tok_s = event.prefill_tok_s;
                self.metrics.decode_seconds = event.decode_seconds;
                self.metrics.decode_tok_s = event.decode_tok_s;
                if let Some(message) = self.messages.back_mut() {
                    message.complete = true;
                }
                self.quit_armed = false;
            }
            "error" => {
                self.status_message = event.message.clone();
                self.notice = Some(event.message.clone());
                self.push_message(Role::System, event.message, true);
                if event.fatal {
                    self.phase = RuntimePhase::Error;
                } else if !matches!(self.phase, RuntimePhase::Loading | RuntimePhase::Launching) {
                    self.phase = RuntimePhase::Ready;
                }
            }
            name => self.push_log(format!("[protocol] ignored event {name}")),
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> bool {
        if key.modifiers.contains(KeyModifiers::CONTROL)
            && matches!(key.code, KeyCode::Char('q' | 'c'))
        {
            if self.phase.is_busy() && !self.quit_armed {
                self.quit_armed = true;
                self.notice = Some("Runtime is busy. Press Ctrl+Q again to stop it.".into());
                return false;
            }
            return true;
        }
        if self.show_help {
            if matches!(key.code, KeyCode::Esc | KeyCode::F(1)) {
                self.show_help = false;
            }
            return false;
        }
        if self.show_logs {
            match key.code {
                KeyCode::Esc | KeyCode::Char('l')
                    if key.modifiers.contains(KeyModifiers::CONTROL)
                        || matches!(key.code, KeyCode::Esc) =>
                {
                    self.show_logs = false;
                }
                _ => {}
            }
            return false;
        }
        if matches!(key.code, KeyCode::F(1)) {
            self.show_help = true;
            return false;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('l') {
            self.show_logs = true;
            return false;
        }
        match self.screen {
            Screen::Setup => self.handle_setup_key(key),
            Screen::Session => self.handle_session_key(key),
        }
        false
    }

    pub fn handle_paste(&mut self, text: &str) {
        let remaining = MAX_INPUT_BYTES.saturating_sub(self.input.len());
        if matches!(self.screen, Screen::Session) {
            self.insert_input(&text[..safe_prefix(text, remaining)]);
        } else if self.current_setup_field().is_text() {
            self.insert_setup(&text[..safe_prefix(text, remaining)]);
        }
    }

    pub fn handle_mouse(&mut self, mouse: MouseEvent) {
        if !matches!(self.screen, Screen::Session) {
            return;
        }
        match mouse.kind {
            MouseEventKind::ScrollUp => {
                self.scroll_from_bottom = self.scroll_from_bottom.saturating_add(3);
            }
            MouseEventKind::ScrollDown => {
                self.scroll_from_bottom = self.scroll_from_bottom.saturating_sub(3);
            }
            _ => {}
        }
    }

    fn handle_setup_key(&mut self, key: KeyEvent) {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Enter {
            self.launch();
            return;
        }
        match key.code {
            KeyCode::Tab | KeyCode::Down | KeyCode::Enter => {
                if self.current_setup_field() == SetupField::Launch {
                    self.launch();
                } else {
                    self.move_setup(1);
                }
            }
            KeyCode::BackTab | KeyCode::Up => self.move_setup(-1),
            KeyCode::Left | KeyCode::Right
                if self.current_setup_field() == SetupField::ModelType =>
            {
                self.toggle_model();
            }
            KeyCode::Char(' ') if self.current_setup_field() == SetupField::FlashAttention => {
                self.setup.flash_attention = !self.setup.flash_attention;
            }
            KeyCode::Left => self.setup_cursor_left(),
            KeyCode::Right => self.setup_cursor_right(),
            KeyCode::Home => self.setup_cursor = 0,
            KeyCode::End => self.setup_cursor = self.current_setup_text().map_or(0, str::len),
            KeyCode::Backspace => self.backspace_setup(),
            KeyCode::Delete => self.delete_setup(),
            KeyCode::Char(character)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                self.insert_setup(&character.to_string());
            }
            _ => {}
        }
    }

    fn handle_session_key(&mut self, key: KeyEvent) {
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('u') => {
                    self.input.clear();
                    self.input_cursor = 0;
                    return;
                }
                KeyCode::Char('w') => {
                    self.delete_previous_word();
                    return;
                }
                _ => {}
            }
        }
        match key.code {
            KeyCode::Enter if key.modifiers.contains(KeyModifiers::SHIFT) => {
                self.insert_input("\n");
            }
            KeyCode::Enter => self.submit_prompt(),
            KeyCode::Char(character)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                self.insert_input(&character.to_string());
            }
            KeyCode::Backspace => self.backspace_input(),
            KeyCode::Delete => self.delete_input(),
            KeyCode::Left => self.input_cursor = previous_boundary(&self.input, self.input_cursor),
            KeyCode::Right => self.input_cursor = next_boundary(&self.input, self.input_cursor),
            KeyCode::Home => self.input_cursor = 0,
            KeyCode::End => self.input_cursor = self.input.len(),
            KeyCode::Up if !self.input.contains('\n') => self.history_previous(),
            KeyCode::Down if !self.input.contains('\n') => self.history_next(),
            KeyCode::PageUp => self.scroll_from_bottom = self.scroll_from_bottom.saturating_add(12),
            KeyCode::PageDown => {
                self.scroll_from_bottom = self.scroll_from_bottom.saturating_sub(12);
            }
            KeyCode::Esc => {
                self.notice = None;
                self.quit_armed = false;
            }
            _ => {}
        }
    }

    fn submit_prompt(&mut self) {
        let prompt = self.input.trim().to_string();
        if prompt.is_empty() {
            return;
        }
        if self.phase != RuntimePhase::Ready {
            self.notice = Some(match self.phase {
                RuntimePhase::Loading | RuntimePhase::Launching => {
                    "The model is still loading".into()
                }
                RuntimePhase::Prefill | RuntimePhase::Decode => {
                    "Wait for the current answer to finish".into()
                }
                _ => "The runtime is not ready".into(),
            });
            return;
        }
        let sent = self
            .process
            .as_mut()
            .ok_or_else(|| "Runtime process is unavailable".to_string())
            .and_then(|process| process.send_prompt(&prompt));
        if let Err(error) = sent {
            self.phase = RuntimePhase::Error;
            self.notice = Some(error);
            return;
        }
        self.push_message(Role::User, prompt.clone(), true);
        self.push_message(Role::Assistant, String::new(), false);
        self.prompt_history.push_back(prompt);
        while self.prompt_history.len() > MAX_PROMPT_HISTORY {
            self.prompt_history.pop_front();
        }
        self.input.clear();
        self.input_cursor = 0;
        self.history_position = None;
        self.draft_input.clear();
        self.phase = RuntimePhase::Prefill;
        self.status_message = "Sending prompt to the runtime".into();
        self.scroll_from_bottom = 0;
        self.metrics.prompt_tokens = 0;
        self.metrics.decode_tokens = 0;
        self.metrics.prefill_seconds = 0.0;
        self.metrics.decode_seconds = 0.0;
        self.speed_history.clear();
    }

    fn push_message(&mut self, role: Role, text: String, complete: bool) {
        self.messages.push_back(ChatMessage {
            role,
            text,
            complete,
        });
        self.trim_messages();
    }

    fn push_log(&mut self, line: String) {
        self.logs.push_back(line);
        while self.logs.len() > MAX_LOG_LINES {
            self.logs.pop_front();
        }
    }

    fn append_assistant_text(&mut self, text: &str) {
        if !matches!(self.messages.back(), Some(message) if message.role == Role::Assistant) {
            self.push_message(Role::Assistant, String::new(), false);
        }
        if let Some(message) = self.messages.back_mut() {
            message.text.push_str(text);
        }
        self.trim_messages();
        if self.scroll_from_bottom == 0 {
            self.scroll_from_bottom = 0;
        }
    }

    fn trim_messages(&mut self) {
        while self.messages.len() > MAX_MESSAGES || self.chat_bytes() > MAX_CHAT_BYTES {
            if self.messages.len() <= 1 {
                if let Some(message) = self.messages.front_mut() {
                    let remove = message.text.len().saturating_sub(MAX_CHAT_BYTES / 2);
                    let boundary = next_boundary(&message.text, remove);
                    message
                        .text
                        .replace_range(..boundary, "[… earlier output elided …]\n");
                }
                break;
            }
            self.messages.pop_front();
        }
    }

    fn chat_bytes(&self) -> usize {
        self.messages.iter().map(|message| message.text.len()).sum()
    }

    pub fn current_setup_field(&self) -> SetupField {
        SetupField::ALL[self.setup_field]
    }

    pub fn setup_value(&self, field: SetupField, cursor: bool) -> String {
        let value = match field {
            SetupField::ModelType => format!("‹  {}  ›", self.setup.model_type.label()),
            SetupField::ModelPath => self.setup.model_path.clone(),
            SetupField::Devices => self.setup.devices.clone(),
            SetupField::Context => self.setup.context_size.clone(),
            SetupField::MaxNew => self.setup.max_new.clone(),
            SetupField::Temperature => self.setup.temperature.clone(),
            SetupField::VramFraction => self.setup.vram_fraction.clone(),
            SetupField::Seed => self.setup.seed.clone(),
            SetupField::FlashAttention => {
                if self.setup.flash_attention {
                    "●  CUDA FlashAttention".into()
                } else {
                    "○  Scalar reference".into()
                }
            }
            SetupField::ChatBinary => self.setup.chat_binary.clone(),
            SetupField::Launch => "◆  LAUNCH STRATA".into(),
        };
        if cursor && field.is_text() {
            insert_cursor(&value, self.setup_cursor)
        } else {
            value
        }
    }

    pub fn input_with_cursor(&self) -> String {
        insert_cursor(&self.input, self.input_cursor)
    }

    pub fn uptime(&self) -> Duration {
        self.started_at
            .map_or(Duration::ZERO, |started| started.elapsed())
    }

    fn move_setup(&mut self, direction: isize) {
        let length = SetupField::ALL.len();
        if direction < 0 {
            self.setup_field = (self.setup_field + length - 1) % length;
        } else {
            self.setup_field = (self.setup_field + 1) % length;
        }
        self.setup_cursor = self.current_setup_text().map_or(0, str::len);
        self.notice = None;
    }

    fn toggle_model(&mut self) {
        self.setup.model_type = self.setup.model_type.toggled();
        let limit = self.setup.model_type.context_limit();
        if self
            .setup
            .context_size
            .parse::<u32>()
            .is_ok_and(|value| value > limit)
        {
            self.setup.context_size = limit.to_string();
        } else if self.setup.model_type == ModelType::DeepSeek && self.setup.context_size == "2048"
        {
            self.setup.context_size = "8192".into();
        }
    }

    fn current_setup_text(&self) -> Option<&str> {
        match self.current_setup_field() {
            SetupField::ModelPath => Some(&self.setup.model_path),
            SetupField::Devices => Some(&self.setup.devices),
            SetupField::Context => Some(&self.setup.context_size),
            SetupField::MaxNew => Some(&self.setup.max_new),
            SetupField::Temperature => Some(&self.setup.temperature),
            SetupField::VramFraction => Some(&self.setup.vram_fraction),
            SetupField::Seed => Some(&self.setup.seed),
            SetupField::ChatBinary => Some(&self.setup.chat_binary),
            _ => None,
        }
    }

    fn current_setup_text_mut(&mut self) -> Option<&mut String> {
        match self.current_setup_field() {
            SetupField::ModelPath => Some(&mut self.setup.model_path),
            SetupField::Devices => Some(&mut self.setup.devices),
            SetupField::Context => Some(&mut self.setup.context_size),
            SetupField::MaxNew => Some(&mut self.setup.max_new),
            SetupField::Temperature => Some(&mut self.setup.temperature),
            SetupField::VramFraction => Some(&mut self.setup.vram_fraction),
            SetupField::Seed => Some(&mut self.setup.seed),
            SetupField::ChatBinary => Some(&mut self.setup.chat_binary),
            _ => None,
        }
    }

    fn insert_setup(&mut self, text: &str) {
        if !self.current_setup_field().is_text() {
            return;
        }
        let cursor = self.setup_cursor;
        if let Some(value) = self.current_setup_text_mut() {
            value.insert_str(cursor, text);
            self.setup_cursor = cursor + text.len();
        }
    }

    fn setup_cursor_left(&mut self) {
        if let Some(value) = self.current_setup_text() {
            self.setup_cursor = previous_boundary(value, self.setup_cursor);
        }
    }

    fn setup_cursor_right(&mut self) {
        if let Some(value) = self.current_setup_text() {
            self.setup_cursor = next_boundary(value, self.setup_cursor);
        }
    }

    fn backspace_setup(&mut self) {
        let cursor = self.setup_cursor;
        if cursor == 0 {
            return;
        }
        if let Some(value) = self.current_setup_text_mut() {
            let previous = previous_boundary(value, cursor);
            value.replace_range(previous..cursor, "");
            self.setup_cursor = previous;
        }
    }

    fn delete_setup(&mut self) {
        let cursor = self.setup_cursor;
        if let Some(value) = self.current_setup_text_mut() {
            let next = next_boundary(value, cursor);
            value.replace_range(cursor..next, "");
        }
    }

    fn insert_input(&mut self, text: &str) {
        if self.input.len() + text.len() > MAX_INPUT_BYTES {
            self.notice = Some("Prompt editor is capped at 1 MiB".into());
            return;
        }
        self.input.insert_str(self.input_cursor, text);
        self.input_cursor += text.len();
        self.history_position = None;
    }

    fn backspace_input(&mut self) {
        if self.input_cursor == 0 {
            return;
        }
        let previous = previous_boundary(&self.input, self.input_cursor);
        self.input.replace_range(previous..self.input_cursor, "");
        self.input_cursor = previous;
    }

    fn delete_input(&mut self) {
        let next = next_boundary(&self.input, self.input_cursor);
        self.input.replace_range(self.input_cursor..next, "");
    }

    fn delete_previous_word(&mut self) {
        while self.input_cursor > 0 {
            let previous = previous_boundary(&self.input, self.input_cursor);
            let character = self.input[previous..self.input_cursor]
                .chars()
                .next()
                .unwrap_or(' ');
            self.input.replace_range(previous..self.input_cursor, "");
            self.input_cursor = previous;
            if !character.is_whitespace() {
                break;
            }
        }
        while self.input_cursor > 0 {
            let previous = previous_boundary(&self.input, self.input_cursor);
            let character = self.input[previous..self.input_cursor]
                .chars()
                .next()
                .unwrap_or(' ');
            if character.is_whitespace() {
                break;
            }
            self.input.replace_range(previous..self.input_cursor, "");
            self.input_cursor = previous;
        }
    }

    fn history_previous(&mut self) {
        if self.prompt_history.is_empty() {
            return;
        }
        let position = match self.history_position {
            None => {
                self.draft_input.clone_from(&self.input);
                self.prompt_history.len() - 1
            }
            Some(position) => position.saturating_sub(1),
        };
        self.history_position = Some(position);
        self.input.clone_from(&self.prompt_history[position]);
        self.input_cursor = self.input.len();
    }

    fn history_next(&mut self) {
        let Some(position) = self.history_position else {
            return;
        };
        if position + 1 < self.prompt_history.len() {
            self.history_position = Some(position + 1);
            self.input.clone_from(&self.prompt_history[position + 1]);
        } else {
            self.history_position = None;
            self.input.clone_from(&self.draft_input);
        }
        self.input_cursor = self.input.len();
    }
}

fn insert_cursor(text: &str, cursor: usize) -> String {
    let cursor = cursor.min(text.len());
    let cursor = if text.is_char_boundary(cursor) {
        cursor
    } else {
        previous_boundary(text, cursor)
    };
    let mut output = String::with_capacity(text.len() + 3);
    output.push_str(&text[..cursor]);
    output.push('▏');
    output.push_str(&text[cursor..]);
    output
}

fn previous_boundary(text: &str, cursor: usize) -> usize {
    let mut position = cursor.min(text.len()).saturating_sub(1);
    while position > 0 && !text.is_char_boundary(position) {
        position -= 1;
    }
    position
}

fn next_boundary(text: &str, cursor: usize) -> usize {
    if cursor >= text.len() {
        return text.len();
    }
    let mut position = cursor + 1;
    while position < text.len() && !text.is_char_boundary(position) {
        position += 1;
    }
    position
}

fn safe_prefix(text: &str, maximum: usize) -> usize {
    let mut length = maximum.min(text.len());
    while length > 0 && !text.is_char_boundary(length) {
        length -= 1;
    }
    length
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unicode_editor_moves_on_character_boundaries() {
        let text = "a☃z";
        assert_eq!(next_boundary(text, 1), 4);
        assert_eq!(previous_boundary(text, 4), 1);
        assert_eq!(insert_cursor(text, 4), "a☃▏z");
    }

    #[test]
    fn protocol_token_updates_streaming_message_and_metrics() {
        let mut app = App::for_test();
        app.screen = Screen::Session;
        app.handle_protocol(Envelope::parse(
            r#"{"protocol":"strata-chat","version":1,"event":"token","text":"hello","tokens":3,"tok_s":2.5}"#,
        ).unwrap());
        assert_eq!(app.phase, RuntimePhase::Decode);
        assert_eq!(app.messages.back().unwrap().text, "hello");
        assert_eq!(app.metrics.decode_tokens, 3);
    }
}
