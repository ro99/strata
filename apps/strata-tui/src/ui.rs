use crate::app::{App, Role, RuntimePhase, Screen, SetupField};
use crate::theme;
use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Layout, Margin, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::symbols;
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Block, BorderType, Borders, Cell, Clear, Gauge, Padding, Paragraph, Row, Scrollbar,
    ScrollbarOrientation, ScrollbarState, Sparkline, Table, TableState, Wrap,
};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

const SPINNERS: [&str; 8] = ["◐", "◓", "◑", "◒", "◐", "◓", "◑", "◒"];
const ORBIT: [&str; 6] = [
    "·  ◆  ·",
    " · ◆ · ",
    "  ·◆·  ",
    " · ◆ · ",
    "·  ◆  ·",
    " · ◆ · ",
];

pub fn render(frame: &mut Frame, app: &App) {
    let area = frame.area();
    frame.render_widget(Block::new().style(theme::base()), area);
    if area.width < 48 || area.height < 15 {
        render_too_small(frame, area);
        return;
    }
    match app.screen {
        Screen::Setup => render_setup(frame, area, app),
        Screen::Session => render_session(frame, area, app),
    }
    if app.show_logs {
        render_logs_overlay(frame, area, app);
    } else if app.show_help {
        render_help_overlay(frame, area);
    }
}

fn render_too_small(frame: &mut Frame, area: Rect) {
    let message = Text::from(vec![
        Line::from(Span::styled(
            "◆  STRATA",
            Style::new().fg(theme::VIOLET).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "The cockpit needs a little more room.",
            theme::muted(),
        )),
        Line::from(Span::styled("Minimum: 48 × 15", theme::muted())),
    ]);
    frame.render_widget(
        Paragraph::new(message)
            .alignment(Alignment::Center)
            .block(panel_block(" TERMINAL ")),
        centered(area, 44, 8),
    );
}

#[allow(clippy::too_many_lines)]
fn render_setup(frame: &mut Frame, area: Rect, app: &App) {
    if area.width < 64 || area.height < 22 {
        render_setup_too_small(frame, area);
        return;
    }
    render_ambient_header(frame, area, app.tick);
    let modal_width = area.width.saturating_sub(4).min(92);
    let modal_height = area.height.saturating_sub(3).min(23);
    let modal = centered(area, modal_width, modal_height);
    frame.render_widget(Clear, modal);
    let block = Block::new()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme::VIOLET))
        .style(theme::panel())
        .title(Line::from(Span::styled(
            " ◆  STRATA COCKPIT ",
            Style::new()
                .fg(theme::VIOLET)
                .bg(theme::SURFACE)
                .add_modifier(Modifier::BOLD),
        )))
        .title_alignment(Alignment::Center)
        .padding(Padding::horizontal(2));
    let inner = block.inner(modal);
    frame.render_widget(block, modal);

    let chunks = Layout::vertical([
        Constraint::Length(4),
        Constraint::Min(11),
        Constraint::Length(3),
    ])
    .split(inner);
    let hero = Text::from(vec![
        Line::from(vec![
            Span::styled(
                "OUT-OF-CORE INFERENCE",
                Style::new()
                    .fg(theme::CYAN)
                    .bg(theme::SURFACE)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  /  EXACT BY CONTRACT", theme::muted()),
        ]),
        Line::from(Span::styled(
            "Bring a model larger than VRAM. Strata will do the honest work.",
            theme::muted(),
        )),
        Line::from(""),
    ]);
    frame.render_widget(Paragraph::new(hero), chunks[0]);

    let rows = SetupField::ALL.into_iter().map(|field| {
        let selected = field == app.current_setup_field();
        let label_style = if selected {
            Style::new()
                .fg(theme::CYAN)
                .bg(theme::SURFACE_RAISED)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::new().fg(theme::MUTED).bg(theme::SURFACE)
        };
        let value_style = if field == SetupField::Launch {
            Style::new()
                .fg(theme::BACKGROUND)
                .bg(if selected {
                    theme::GREEN
                } else {
                    theme::VIOLET
                })
                .add_modifier(Modifier::BOLD)
        } else if selected {
            Style::new().fg(theme::TEXT).bg(theme::SURFACE_RAISED)
        } else {
            Style::new().fg(theme::TEXT).bg(theme::SURFACE)
        };
        Row::new([
            Cell::from(field.label()).style(label_style),
            Cell::from(app.setup_value(field, selected)).style(value_style),
        ])
        .height(1)
    });
    let table = Table::new(rows, [Constraint::Length(18), Constraint::Min(24)])
        .column_spacing(2)
        .style(theme::panel())
        .row_highlight_style(Style::new().bg(theme::SURFACE_RAISED))
        .highlight_symbol("  ");
    let mut state = TableState::default();
    state.select(Some(app.setup_field));
    frame.render_stateful_widget(table, chunks[1], &mut state);

    let footer = if let Some(notice) = &app.notice {
        vec![
            Line::from(Span::styled(
                format!("  !  {notice}"),
                Style::new()
                    .fg(theme::RED)
                    .bg(theme::SURFACE)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                "Tab navigate  ·  Ctrl+Enter launch",
                theme::muted(),
            )),
        ]
    } else {
        vec![
            Line::from(Span::styled(
                "Temperature 0 preserves exact greedy decoding; sampling is always explicit.",
                theme::muted(),
            )),
            Line::from(Span::styled(
                "Tab navigate  ·  ←/→ choose  ·  Space toggle  ·  Ctrl+Enter launch",
                theme::muted(),
            )),
        ]
    };
    frame.render_widget(Paragraph::new(footer), chunks[2]);
}

fn render_setup_too_small(frame: &mut Frame, area: Rect) {
    let text = Text::from(vec![
        Line::from(Span::styled(
            "◆  STRATA LAUNCH",
            Style::new().fg(theme::VIOLET).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Resize once to configure the runtime; the chat view is more compact.",
            theme::muted(),
        )),
        Line::from(Span::styled("Launch form minimum: 64 × 22", theme::muted())),
    ]);
    frame.render_widget(
        Paragraph::new(text)
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true })
            .block(panel_block(" SETUP ")),
        centered(area, area.width.min(58), 9),
    );
}

fn render_ambient_header(frame: &mut Frame, area: Rect, tick: u64) {
    let top = Rect::new(area.x, area.y, area.width, 2.min(area.height));
    let orbit = ORBIT[animation_index(tick / 2, ORBIT.len())];
    let line = Line::from(vec![
        Span::styled(
            "  STRATA ",
            Style::new()
                .fg(theme::VIOLET)
                .bg(theme::BACKGROUND)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("◆", Style::new().fg(theme::CYAN).bg(theme::BACKGROUND)),
        Span::styled(
            "  inference beyond VRAM",
            Style::new().fg(theme::MUTED).bg(theme::BACKGROUND),
        ),
        Span::raw("   "),
        Span::styled(orbit, Style::new().fg(theme::FAINT).bg(theme::BACKGROUND)),
    ]);
    frame.render_widget(Paragraph::new(line), top);
}

fn render_session(frame: &mut Frame, area: Rect, app: &App) {
    let composer_height = if area.height >= 28 { 6 } else { 4 };
    let chunks = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(6),
        Constraint::Length(composer_height),
        Constraint::Length(1),
    ])
    .split(area);
    render_header(frame, chunks[0], app);

    if area.width >= 96 {
        let columns = Layout::horizontal([Constraint::Percentage(72), Constraint::Percentage(28)])
            .spacing(1)
            .split(chunks[1]);
        render_chat(frame, columns[0], app);
        render_sidebar(frame, columns[1], app);
    } else {
        render_chat(frame, chunks[1], app);
    }
    render_composer(frame, chunks[2], app);
    render_footer(frame, chunks[3], app);
}

fn render_header(frame: &mut Frame, area: Rect, app: &App) {
    let columns = Layout::horizontal([
        Constraint::Length(20),
        Constraint::Min(12),
        Constraint::Length(if area.width >= 84 { 28 } else { 16 }),
    ])
    .split(area);
    let block = Block::new()
        .borders(Borders::BOTTOM)
        .border_style(Style::new().fg(theme::BORDER))
        .style(theme::base());
    frame.render_widget(block, area);
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("  ◆ ", Style::new().fg(theme::CYAN).bg(theme::BACKGROUND)),
            Span::styled(
                "STRATA",
                Style::new()
                    .fg(theme::VIOLET)
                    .bg(theme::BACKGROUND)
                    .add_modifier(Modifier::BOLD),
            ),
        ])),
        columns[0],
    );
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            truncate(
                &app.status_message,
                usize::from(columns[1].width.saturating_sub(2)),
            ),
            Style::new().fg(theme::MUTED).bg(theme::BACKGROUND),
        ))),
        columns[1],
    );
    let (phase_color, spinner) = phase_style(app.phase, app.tick);
    let phase = if area.width >= 84 {
        format!(
            "{spinner}  {}  {:>6.2} tok/s ",
            app.phase.label(),
            app.metrics.decode_tok_s
        )
    } else {
        format!("{spinner} {} ", app.phase.label())
    };
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            phase,
            Style::new()
                .fg(phase_color)
                .bg(theme::BACKGROUND)
                .add_modifier(Modifier::BOLD),
        )))
        .alignment(Alignment::Right),
        columns[2],
    );
}

fn render_chat(frame: &mut Frame, area: Rect, app: &App) {
    let block = panel_block(" CONVERSATION ").padding(Padding::new(1, 1, 0, 0));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if app.messages.is_empty() {
        render_empty_chat(frame, inner, app);
        return;
    }

    let width = usize::from(inner.width.max(4));
    let lines = conversation_lines(app, width);
    let viewport = usize::from(inner.height);
    let maximum_start = lines.len().saturating_sub(viewport);
    let start = maximum_start.saturating_sub(app.scroll_from_bottom.min(maximum_start));
    let visible = lines
        .into_iter()
        .skip(start)
        .take(viewport)
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(Text::from(visible)).style(theme::panel()),
        inner,
    );

    if maximum_start > 0 {
        let mut scrollbar = ScrollbarState::new(maximum_start + viewport).position(start);
        let widget = Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .symbols(symbols::scrollbar::VERTICAL)
            .track_style(Style::new().fg(theme::FAINT).bg(theme::SURFACE))
            .thumb_style(Style::new().fg(theme::VIOLET).bg(theme::SURFACE));
        frame.render_stateful_widget(widget, area.inner(Margin::new(0, 1)), &mut scrollbar);
    }
}

fn render_empty_chat(frame: &mut Frame, area: Rect, app: &App) {
    let loading = matches!(app.phase, RuntimePhase::Launching | RuntimePhase::Loading);
    let spinner = SPINNERS[animation_index(app.tick, SPINNERS.len())];
    let (title, subtitle, detail) = if loading {
        (
            format!("{spinner}  ADMITTING MODEL"),
            "Exact weights are moving into their declared memory tiers.",
            app.logs
                .back()
                .map_or("Waiting for runtime telemetry", String::as_str),
        )
    } else if app.phase == RuntimePhase::Error {
        (
            "!  RUNTIME NEEDS ATTENTION".into(),
            "No fallback was attempted.",
            app.notice
                .as_deref()
                .unwrap_or("Open Ctrl+L for diagnostics"),
        )
    } else {
        (
            "◆  READY TO THINK".into(),
            "Write a prompt below. Responses stream directly from Strata.",
            "Shift+Enter adds a line · Enter sends",
        )
    };
    let color = if app.phase == RuntimePhase::Error {
        theme::RED
    } else {
        theme::CYAN
    };
    let text = Text::from(vec![
        Line::from(""),
        Line::from(Span::styled(
            ORBIT[animation_index(app.tick / 2, ORBIT.len())],
            Style::new().fg(theme::VIOLET).bg(theme::SURFACE),
        )),
        Line::from(""),
        Line::from(Span::styled(
            title,
            Style::new()
                .fg(color)
                .bg(theme::SURFACE)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(subtitle, theme::muted())),
        Line::from(Span::styled(
            truncate(detail, 72),
            Style::new().fg(theme::FAINT).bg(theme::SURFACE),
        )),
    ]);
    frame.render_widget(
        Paragraph::new(text)
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true }),
        centered(area, area.width.min(76), area.height.min(9)),
    );
}

#[allow(clippy::too_many_lines)]
fn render_sidebar(frame: &mut Frame, area: Rect, app: &App) {
    let block = panel_block(" SESSION ").padding(Padding::horizontal(1));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let chunks = Layout::vertical([
        Constraint::Length(8),
        Constraint::Length(9),
        Constraint::Min(4),
    ])
    .split(inner);
    let config = app.config.as_ref();
    let exact = config.is_some_and(crate::config::ValidatedConfig::is_exact);
    let contract_color = if exact { theme::GREEN } else { theme::AMBER };
    let contract = if exact {
        "● EXACT GREEDY"
    } else {
        "● SEEDED SAMPLE"
    };
    let session = vec![
        Line::from(Span::styled(
            contract,
            Style::new()
                .fg(contract_color)
                .bg(theme::SURFACE)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        metric_line(
            "MODEL",
            config.map_or("—", |value| value.model_type.label()),
        ),
        metric_line(
            "DEVICES",
            config.map_or("—", |value| value.devices.as_str()),
        ),
        metric_line(
            "CONTEXT",
            &config.map_or_else(
                || "—".into(),
                |value| format_number(u64::from(value.context_size)),
            ),
        ),
        metric_line("UPTIME", &format_duration(app.uptime().as_secs())),
    ];
    frame.render_widget(Paragraph::new(session), chunks[0]);

    let metrics = vec![
        Line::from(Span::styled("LAST TURN", theme::title())),
        Line::from(""),
        metric_pair(
            "PREFILL",
            app.metrics.prefill_tokens,
            app.metrics.prefill_tok_s,
        ),
        metric_pair(
            "DECODE",
            app.metrics.decode_tokens,
            app.metrics.decode_tok_s,
        ),
        Line::from(""),
        metric_line("KV REUSE", &format_number(app.metrics.reused_prompt_tokens)),
        metric_line("PREFILL", &format!("{:.2} s", app.metrics.prefill_seconds)),
        metric_line("DECODE", &format!("{:.2} s", app.metrics.decode_seconds)),
        metric_line("LOAD", &format!("{:.2} s", app.metrics.load_seconds)),
    ];
    frame.render_widget(Paragraph::new(metrics), chunks[1]);

    let lower = Layout::vertical([
        Constraint::Length(2),
        Constraint::Length(3),
        Constraint::Min(1),
    ])
    .split(chunks[2]);
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled("LIVE DECODE", theme::title()))),
        lower[0],
    );
    let data = app.speed_history.iter().copied().collect::<Vec<_>>();
    frame.render_widget(
        Sparkline::default()
            .data(&data)
            .bar_set(symbols::bar::NINE_LEVELS)
            .style(Style::new().fg(theme::CYAN).bg(theme::SURFACE)),
        lower[1],
    );
    if let Some(config) = config {
        let used = app
            .metrics
            .prompt_tokens
            .saturating_add(app.metrics.decode_tokens);
        let bounded_used =
            u32::try_from(used.min(u64::from(config.context_size))).unwrap_or(config.context_size);
        let ratio = f64::from(bounded_used) / f64::from(config.context_size);
        let label = format!(
            "{} / {} tokens",
            format_number(used),
            format_number(u64::from(config.context_size))
        );
        frame.render_widget(
            Gauge::default()
                .block(Block::new().title(Span::styled(" CONTEXT ", theme::muted())))
                .gauge_style(Style::new().fg(theme::VIOLET).bg(theme::SURFACE_RAISED))
                .label(Span::styled(label, Style::new().fg(theme::TEXT)))
                .ratio(ratio),
            lower[2],
        );
    }
}

fn render_composer(frame: &mut Frame, area: Rect, app: &App) {
    let active = app.phase == RuntimePhase::Ready;
    let border = if active { theme::CYAN } else { theme::BORDER };
    let title = if active {
        " PROMPT  ·  ENTER TO SEND "
    } else {
        " PROMPT "
    };
    let block = Block::new()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(border))
        .title(Span::styled(
            title,
            Style::new()
                .fg(if active { theme::CYAN } else { theme::MUTED })
                .bg(theme::BACKGROUND)
                .add_modifier(Modifier::BOLD),
        ))
        .style(theme::base())
        .padding(Padding::horizontal(1));
    let placeholder = match app.phase {
        RuntimePhase::Launching | RuntimePhase::Loading => "Model is loading…",
        RuntimePhase::Prefill | RuntimePhase::Decode => "Strata is answering…",
        RuntimePhase::Error | RuntimePhase::Exited => "Runtime is unavailable",
        _ => "Ask Strata anything…",
    };
    let input = if app.input.is_empty() {
        Line::from(Span::styled(
            if active {
                format!("▏{placeholder}")
            } else {
                placeholder.into()
            },
            Style::new().fg(theme::FAINT).bg(theme::BACKGROUND),
        ))
    } else {
        Line::from(Span::styled(
            app.input_with_cursor(),
            Style::new().fg(theme::TEXT).bg(theme::BACKGROUND),
        ))
    };
    frame.render_widget(
        Paragraph::new(input)
            .block(block)
            .wrap(Wrap { trim: false }),
        area,
    );
}

fn render_footer(frame: &mut Frame, area: Rect, app: &App) {
    let content = if let Some(notice) = &app.notice {
        Line::from(vec![
            Span::styled(
                "  !  ",
                Style::new()
                    .fg(theme::RED)
                    .bg(theme::BACKGROUND)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                truncate(notice, usize::from(area.width.saturating_sub(7))),
                Style::new().fg(theme::RED).bg(theme::BACKGROUND),
            ),
        ])
    } else {
        Line::from(vec![
            Span::styled(
                "  Shift+Enter",
                Style::new()
                    .fg(theme::TEXT)
                    .bg(theme::BACKGROUND)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                " newline   ",
                Style::new().fg(theme::MUTED).bg(theme::BACKGROUND),
            ),
            Span::styled(
                "PgUp/PgDn",
                Style::new()
                    .fg(theme::TEXT)
                    .bg(theme::BACKGROUND)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                " scroll   ",
                Style::new().fg(theme::MUTED).bg(theme::BACKGROUND),
            ),
            Span::styled(
                "Ctrl+L",
                Style::new()
                    .fg(theme::TEXT)
                    .bg(theme::BACKGROUND)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                " logs   ",
                Style::new().fg(theme::MUTED).bg(theme::BACKGROUND),
            ),
            Span::styled(
                "F1",
                Style::new()
                    .fg(theme::TEXT)
                    .bg(theme::BACKGROUND)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                " help   ",
                Style::new().fg(theme::MUTED).bg(theme::BACKGROUND),
            ),
            Span::styled(
                "Ctrl+Q",
                Style::new()
                    .fg(theme::TEXT)
                    .bg(theme::BACKGROUND)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" quit", Style::new().fg(theme::MUTED).bg(theme::BACKGROUND)),
        ])
    };
    frame.render_widget(Paragraph::new(content), area);
}

fn render_logs_overlay(frame: &mut Frame, area: Rect, app: &App) {
    let modal = centered(
        area,
        area.width.saturating_sub(8).min(108),
        area.height.saturating_sub(6).min(30),
    );
    frame.render_widget(Clear, modal);
    let block = Block::new()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme::AMBER))
        .style(theme::panel())
        .title(Span::styled(
            " RUNTIME DIAGNOSTICS ",
            Style::new()
                .fg(theme::AMBER)
                .bg(theme::SURFACE)
                .add_modifier(Modifier::BOLD),
        ))
        .title_bottom(Line::from(Span::styled(" Esc close ", theme::muted())).right_aligned())
        .padding(Padding::horizontal(1));
    let inner = block.inner(modal);
    frame.render_widget(block, modal);
    let height = usize::from(inner.height);
    let lines = app
        .logs
        .iter()
        .rev()
        .take(height)
        .rev()
        .map(|line| {
            let style = if line.contains("error") || line.contains("failed") {
                Style::new().fg(theme::RED).bg(theme::SURFACE)
            } else if line.starts_with("[ready]") {
                Style::new().fg(theme::GREEN).bg(theme::SURFACE)
            } else {
                Style::new().fg(theme::MUTED).bg(theme::SURFACE)
            };
            Line::from(Span::styled(
                truncate(line, usize::from(inner.width)),
                style,
            ))
        })
        .collect::<Vec<_>>();
    frame.render_widget(Paragraph::new(lines), inner);
}

fn render_help_overlay(frame: &mut Frame, area: Rect) {
    let modal = centered(
        area,
        area.width.saturating_sub(10).min(74),
        area.height.saturating_sub(6).min(26),
    );
    frame.render_widget(Clear, modal);
    let block = panel_block(" KEYBOARD MAP · F1 ")
        .border_style(Style::new().fg(theme::VIOLET))
        .title_alignment(Alignment::Center)
        .padding(Padding::new(2, 2, 1, 1));
    let lines = vec![
        help_line("Enter", "send prompt / advance setup"),
        help_line("Shift+Enter", "insert a prompt newline"),
        help_line("↑ / ↓", "browse prompt history"),
        help_line("PgUp / PgDn", "scroll the conversation"),
        help_line("Ctrl+W", "delete previous word"),
        help_line("Ctrl+U", "clear the prompt editor"),
        help_line("Ctrl+L", "open runtime diagnostics"),
        help_line("F1", "toggle this keyboard map"),
        help_line("Ctrl+Q", "quit; confirm once if runtime is busy"),
        Line::from(""),
        Line::from(Span::styled(
            "The TUI never changes model precision, routing, top-k, or fallback behavior.",
            Style::new().fg(theme::GREEN).bg(theme::SURFACE),
        )),
    ];
    frame.render_widget(
        Paragraph::new(lines).block(block).wrap(Wrap { trim: true }),
        modal,
    );
}

fn conversation_lines(app: &App, width: usize) -> Vec<Line<'static>> {
    let mut output = Vec::new();
    let content_width = width.saturating_sub(3).max(1);
    for message in &app.messages {
        if !output.is_empty() {
            output.push(Line::from(""));
        }
        let (role, color, marker) = match message.role {
            Role::User => ("YOU", theme::CYAN, "●"),
            Role::Assistant => ("STRATA", theme::VIOLET, "◆"),
            Role::System => ("RUNTIME", theme::RED, "!"),
        };
        output.push(Line::from(vec![
            Span::styled(
                format!("{marker} "),
                Style::new().fg(color).bg(theme::SURFACE),
            ),
            Span::styled(
                role,
                Style::new()
                    .fg(color)
                    .bg(theme::SURFACE)
                    .add_modifier(Modifier::BOLD),
            ),
            if !message.complete && message.role == Role::Assistant {
                Span::styled(
                    format!("  {}", SPINNERS[animation_index(app.tick, SPINNERS.len())]),
                    Style::new().fg(theme::CYAN).bg(theme::SURFACE),
                )
            } else {
                Span::raw("")
            },
        ]));
        let mut code = false;
        for wrapped in wrap_text(&message.text, content_width) {
            let trimmed = wrapped.trim_start();
            if trimmed.starts_with("```") {
                code = !code;
                output.push(Line::from(Span::styled(
                    format!("  {wrapped}"),
                    Style::new().fg(theme::CYAN).bg(theme::SURFACE_RAISED),
                )));
            } else if code {
                output.push(Line::from(Span::styled(
                    format!("  {wrapped}"),
                    Style::new().fg(theme::TEXT).bg(theme::SURFACE_RAISED),
                )));
            } else if trimmed.starts_with('#') {
                output.push(Line::from(Span::styled(
                    format!("  {wrapped}"),
                    Style::new()
                        .fg(theme::TEXT)
                        .bg(theme::SURFACE)
                        .add_modifier(Modifier::BOLD),
                )));
            } else if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
                output.push(Line::from(vec![
                    Span::styled("  • ", Style::new().fg(color).bg(theme::SURFACE)),
                    Span::styled(
                        trimmed[2..].to_string(),
                        Style::new().fg(theme::TEXT).bg(theme::SURFACE),
                    ),
                ]));
            } else {
                output.push(Line::from(Span::styled(
                    format!("  {wrapped}"),
                    Style::new().fg(theme::TEXT).bg(theme::SURFACE),
                )));
            }
        }
    }
    output
}

fn wrap_text(text: &str, width: usize) -> Vec<String> {
    if text.is_empty() {
        return vec![String::new()];
    }
    let mut output = Vec::new();
    for source in text.split('\n') {
        if source.is_empty() {
            output.push(String::new());
            continue;
        }
        let mut line = String::new();
        let mut line_width = 0;
        for word in source.split_inclusive(char::is_whitespace) {
            let word_width = UnicodeWidthStr::width(word);
            if line_width > 0 && line_width + word_width > width {
                output.push(line.trim_end().to_string());
                line.clear();
                line_width = 0;
            }
            if word_width > width {
                for character in word.chars() {
                    let character_width = UnicodeWidthChar::width(character).unwrap_or(0);
                    if line_width > 0 && line_width + character_width > width {
                        output.push(std::mem::take(&mut line));
                        line_width = 0;
                    }
                    line.push(character);
                    line_width += character_width;
                }
            } else {
                line.push_str(word);
                line_width += word_width;
            }
        }
        output.push(line.trim_end().to_string());
    }
    output
}

fn panel_block(title: &str) -> Block<'_> {
    Block::new()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme::BORDER))
        .style(theme::panel())
        .title(Span::styled(title, theme::title()))
}

fn phase_style(phase: RuntimePhase, tick: u64) -> (ratatui::style::Color, &'static str) {
    match phase {
        RuntimePhase::Ready => (theme::GREEN, "●"),
        RuntimePhase::Error => (theme::RED, "!"),
        RuntimePhase::Exited | RuntimePhase::Offline => (theme::MUTED, "○"),
        _ => (theme::CYAN, SPINNERS[animation_index(tick, SPINNERS.len())]),
    }
}

fn metric_line(label: &str, value: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<9}"), theme::muted()),
        Span::styled(
            value.to_string(),
            Style::new().fg(theme::TEXT).bg(theme::SURFACE),
        ),
    ])
}

fn metric_pair(label: &str, tokens: u64, speed: f64) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<9}"), theme::muted()),
        Span::styled(
            format!("{}  ", format_number(tokens)),
            Style::new().fg(theme::TEXT).bg(theme::SURFACE),
        ),
        Span::styled(
            format!("{speed:.2}/s"),
            Style::new().fg(theme::CYAN).bg(theme::SURFACE),
        ),
    ])
}

fn help_line<'a>(key: &'a str, description: &'a str) -> Line<'a> {
    Line::from(vec![
        Span::styled(
            format!("{key:<16}"),
            Style::new()
                .fg(theme::CYAN)
                .bg(theme::SURFACE)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(description, theme::muted()),
    ])
}

fn format_number(value: u64) -> String {
    let digits = value.to_string();
    let mut output = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, character) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            output.push(',');
        }
        output.push(character);
    }
    output
}

fn format_duration(seconds: u64) -> String {
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    let seconds = seconds % 60;
    if hours > 0 {
        format!("{hours:02}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes:02}:{seconds:02}")
    }
}

fn animation_index(tick: u64, length: usize) -> usize {
    let length_u64 = u64::try_from(length).unwrap_or(1);
    usize::try_from(tick % length_u64).unwrap_or(0)
}

fn truncate(text: &str, maximum_width: usize) -> String {
    if UnicodeWidthStr::width(text) <= maximum_width {
        return text.to_string();
    }
    if maximum_width <= 1 {
        return "…".chars().take(maximum_width).collect();
    }
    let mut output = String::new();
    let mut width = 0;
    for character in text.chars() {
        let character_width = UnicodeWidthChar::width(character).unwrap_or(0);
        if width + character_width >= maximum_width {
            break;
        }
        output.push(character);
        width += character_width;
    }
    output.push('…');
    output
}

fn centered(area: Rect, width: u16, height: u16) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    Rect::new(
        area.x + area.width.saturating_sub(width) / 2,
        area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    #[test]
    fn setup_renderer_contains_identity_and_launch_action() {
        let app = App::for_test();
        let backend = TestBackend::new(100, 34);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &app)).unwrap();
        let symbols = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect::<String>();
        assert!(symbols.contains("STRATA COCKPIT"));
        assert!(symbols.contains("LAUNCH STRATA"));
        assert!(symbols.contains("EXACT BY CONTRACT"));
    }

    #[test]
    fn wrapping_preserves_unicode_and_blank_lines() {
        assert_eq!(
            wrap_text("hello ☃ world\n\nnext", 8),
            vec!["hello ☃", "world", "", "next"]
        );
    }
}
