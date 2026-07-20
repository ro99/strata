use ratatui::style::{Color, Modifier, Style};

pub const BACKGROUND: Color = Color::Rgb(8, 11, 17);
pub const SURFACE: Color = Color::Rgb(14, 19, 28);
pub const SURFACE_RAISED: Color = Color::Rgb(20, 27, 39);
pub const BORDER: Color = Color::Rgb(49, 62, 82);
pub const TEXT: Color = Color::Rgb(224, 231, 242);
pub const MUTED: Color = Color::Rgb(119, 135, 158);
pub const FAINT: Color = Color::Rgb(72, 84, 104);
pub const CYAN: Color = Color::Rgb(91, 211, 238);
pub const VIOLET: Color = Color::Rgb(174, 137, 255);
pub const GREEN: Color = Color::Rgb(89, 230, 154);
pub const AMBER: Color = Color::Rgb(247, 194, 91);
pub const RED: Color = Color::Rgb(255, 105, 121);

pub const fn base() -> Style {
    Style::new().fg(TEXT).bg(BACKGROUND)
}

pub const fn panel() -> Style {
    Style::new().fg(TEXT).bg(SURFACE)
}

pub const fn title() -> Style {
    Style::new()
        .fg(VIOLET)
        .bg(SURFACE)
        .add_modifier(Modifier::BOLD)
}

pub const fn muted() -> Style {
    Style::new().fg(MUTED).bg(SURFACE)
}
