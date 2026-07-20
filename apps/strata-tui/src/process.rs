use crate::config::ValidatedConfig;
use crate::protocol::Envelope;
use serde_json::json;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

#[derive(Debug)]
pub enum ProcessEvent {
    Protocol(Envelope),
    Log(String),
    ProtocolError(String),
}

pub struct RuntimeProcess {
    child: Child,
    stdin: ChildStdin,
    receiver: Receiver<ProcessEvent>,
    exit_reported: bool,
}

impl RuntimeProcess {
    pub fn spawn(config: &ValidatedConfig) -> Result<Self, String> {
        let mut child = Command::new(&config.chat_binary)
            .args(config.command_args())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| format!("Could not launch {}: {error}", config.chat_binary))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "Could not open the runtime input pipe".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Could not open the runtime event pipe".to_string())?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| "Could not open the runtime log pipe".to_string())?;
        let (sender, receiver) = mpsc::channel();
        spawn_protocol_reader(stdout, sender.clone());
        spawn_log_reader(stderr, sender);
        Ok(Self {
            child,
            stdin,
            receiver,
            exit_reported: false,
        })
    }

    pub fn send_prompt(&mut self, prompt: &str) -> Result<(), String> {
        serde_json::to_writer(
            &mut self.stdin,
            &json!({
                "command": "prompt",
                "text": prompt,
            }),
        )
        .map_err(|error| format!("Could not encode the prompt: {error}"))?;
        self.stdin
            .write_all(b"\n")
            .and_then(|()| self.stdin.flush())
            .map_err(|error| format!("Could not send the prompt: {error}"))
    }

    pub fn drain_events(&mut self) -> Vec<ProcessEvent> {
        self.receiver.try_iter().collect()
    }

    pub fn poll_exit(&mut self) -> Option<Result<i32, String>> {
        if self.exit_reported {
            return None;
        }
        match self.child.try_wait() {
            Ok(Some(status)) => {
                self.exit_reported = true;
                Some(Ok(status.code().unwrap_or(-1)))
            }
            Ok(None) => None,
            Err(error) => {
                self.exit_reported = true;
                Some(Err(format!(
                    "Could not read runtime process status: {error}"
                )))
            }
        }
    }
}

impl Drop for RuntimeProcess {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn spawn_protocol_reader(
    stdout: impl std::io::Read + Send + 'static,
    sender: Sender<ProcessEvent>,
) {
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let event = match line {
                Ok(line) if line.trim().is_empty() => continue,
                Ok(line) => match Envelope::parse(&line) {
                    Ok(event) => ProcessEvent::Protocol(event),
                    Err(error) => ProcessEvent::ProtocolError(error),
                },
                Err(error) => ProcessEvent::ProtocolError(format!(
                    "Could not read the runtime event stream: {error}"
                )),
            };
            if sender.send(event).is_err() {
                break;
            }
        }
    });
}

fn spawn_log_reader(stderr: impl std::io::Read + Send + 'static, sender: Sender<ProcessEvent>) {
    thread::spawn(move || {
        for line in BufReader::new(stderr).lines() {
            match line {
                Ok(line) => {
                    if sender.send(ProcessEvent::Log(line)).is_err() {
                        break;
                    }
                }
                Err(error) => {
                    let _ = sender.send(ProcessEvent::ProtocolError(format!(
                        "Could not read runtime diagnostics: {error}"
                    )));
                    break;
                }
            }
        }
    });
}
