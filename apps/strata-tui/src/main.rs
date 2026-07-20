mod app;
mod config;
mod process;
mod protocol;
mod theme;
mod ui;

use app::App;
use crossterm::cursor::{Hide, Show};
use crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event, KeyEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use std::io::{self, IsTerminal, Stdout};
use std::time::{Duration, Instant};

type Tui = Terminal<CrosstermBackend<Stdout>>;

fn main() {
    let cli = match config::parse_cli() {
        Ok(cli) => cli,
        Err(message) => {
            if message.starts_with("strata-tui") {
                println!("{message}");
                return;
            }
            eprintln!("error: {message}");
            std::process::exit(2);
        }
    };
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        eprintln!("error: strata-tui requires an interactive terminal");
        std::process::exit(2);
    }

    install_panic_hook();
    let mut terminal = match start_terminal() {
        Ok(terminal) => terminal,
        Err(error) => {
            eprintln!("error: could not initialize terminal: {error}");
            std::process::exit(1);
        }
    };
    let result = run(&mut terminal, App::new(cli));
    let restore_result = restore_terminal(&mut terminal);
    if let Err(error) = result.or(restore_result) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run(terminal: &mut Tui, mut app: App) -> io::Result<()> {
    const FRAME: Duration = Duration::from_millis(80);
    let mut last_tick = Instant::now();
    loop {
        terminal.draw(|frame| ui::render(frame, &app))?;
        let timeout = FRAME.saturating_sub(last_tick.elapsed());
        if event::poll(timeout)? {
            match event::read()? {
                Event::Key(key)
                    if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) =>
                {
                    if app.handle_key(key) {
                        return Ok(());
                    }
                }
                Event::Paste(text) => app.handle_paste(&text),
                Event::Mouse(mouse) => app.handle_mouse(mouse),
                Event::Resize(_, _) | Event::FocusGained | Event::FocusLost | Event::Key(_) => {}
            }
        }
        if last_tick.elapsed() >= FRAME {
            app.update();
            last_tick = Instant::now();
        }
    }
}

fn start_terminal() -> io::Result<Tui> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    if let Err(error) = execute!(
        stdout,
        EnterAlternateScreen,
        EnableBracketedPaste,
        EnableMouseCapture,
        Hide
    ) {
        let _ = disable_raw_mode();
        return Err(error);
    }
    let mut terminal = match Terminal::new(CrosstermBackend::new(stdout)) {
        Ok(terminal) => terminal,
        Err(error) => {
            let _ = disable_raw_mode();
            let _ = execute!(
                io::stdout(),
                Show,
                DisableMouseCapture,
                DisableBracketedPaste,
                LeaveAlternateScreen
            );
            return Err(error);
        }
    };
    terminal.clear()?;
    Ok(terminal)
}

fn restore_terminal(terminal: &mut Tui) -> io::Result<()> {
    let raw_result = disable_raw_mode();
    let screen_result = execute!(
        terminal.backend_mut(),
        Show,
        DisableMouseCapture,
        DisableBracketedPaste,
        LeaveAlternateScreen
    );
    let cursor_result = terminal.show_cursor();
    raw_result.and(screen_result).and(cursor_result)
}

fn install_panic_hook() {
    let original = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic| {
        let _ = disable_raw_mode();
        let _ = execute!(
            io::stdout(),
            Show,
            DisableMouseCapture,
            DisableBracketedPaste,
            LeaveAlternateScreen
        );
        original(panic);
    }));
}
