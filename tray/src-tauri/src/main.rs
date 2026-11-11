// Live Translate tray: a menu-bar / system-tray icon that opens the translator's control window.
// The window is served by the Hermes live-translate plugin on 127.0.0.1:7865; if nothing is listening there,
// the tray starts it with `hermes translate serve`.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::Duration;

use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::image::Image;
use tauri::{AppHandle, Manager, WebviewUrl, WebviewWindowBuilder, WindowEvent};

#[cfg(target_os = "macos")]
const IDLE: &[u8] = include_bytes!("../tray-icons/idle.png"); // template: macOS tints it for the menu bar
#[cfg(not(target_os = "macos"))]
const IDLE: &[u8] = include_bytes!("../tray-icons/idle_light.png"); // Windows / Linux trays are usually dark
const LIVE: [&[u8]; 6] = [
    include_bytes!("../tray-icons/live0.png"), include_bytes!("../tray-icons/live1.png"),
    include_bytes!("../tray-icons/live2.png"), include_bytes!("../tray-icons/live3.png"),
    include_bytes!("../tray-icons/live4.png"), include_bytes!("../tray-icons/live5.png"),
];

/// Is the translator live (microphone in use)? Reads the local control server's state.
fn is_live() -> bool {
    let Ok(mut s) = TcpStream::connect_timeout(&ADDR.parse().unwrap(), Duration::from_millis(300)) else { return false };
    let _ = s.set_read_timeout(Some(Duration::from_millis(800)));
    if s.write_all(b"GET /api/state HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n").is_err() {
        return false;
    }
    let mut body = String::new();
    let _ = s.read_to_string(&mut body);
    body.contains("\"session\": \"live\"") || body.contains("\"session\": \"starting\"")
}

/// Monochrome template icon while idle (macOS tints it for the menu bar); an ember glow that breathes
/// about once a second while the microphone is live.
fn animate_tray(app: AppHandle) {
    std::thread::spawn(move || {
        let idle = Image::from_bytes(IDLE).expect("idle icon");
        let live: Vec<Image> = LIVE.iter().map(|b| Image::from_bytes(b).expect("live icon")).collect();
        let (mut was_live, mut frame, mut last_check) = (None::<bool>, 0usize, std::time::Instant::now() - Duration::from_secs(5));
        let mut live_now = false;
        loop {
            if last_check.elapsed() >= Duration::from_millis(700) {
                live_now = is_live();
                last_check = std::time::Instant::now();
            }
            if let Some(tray) = app.tray_by_id("main") {
                if live_now {
                    let _ = tray.set_icon(Some(live[frame % live.len()].clone()));
                    let _ = tray.set_icon_as_template(false);
                    let _ = tray.set_tooltip(Some("Live Translate: listening"));
                    frame += 1;
                } else if was_live != Some(false) {
                    let _ = tray.set_icon(Some(idle.clone()));
                    let _ = tray.set_icon_as_template(true);
                    let _ = tray.set_tooltip(Some("Live Translate"));
                }
            }
            was_live = Some(live_now);
            std::thread::sleep(Duration::from_millis(if live_now { 166 } else { 400 }));
        }
    });
}

const ADDR: &str = "127.0.0.1:7865";

struct Server(Mutex<Option<Child>>);

fn server_up() -> bool {
    TcpStream::connect_timeout(&ADDR.parse().unwrap(), Duration::from_millis(300)).is_ok()
}

fn start_server() -> Option<Child> {
    if server_up() {
        return None;
    }
    Command::new("hermes").args(["translate", "serve"]).spawn().ok()
}

fn show(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.show();
        let _ = w.set_focus();
        return;
    }
    let builder = WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
        .title("Live Translate")
        .inner_size(1200.0, 800.0)
        .min_inner_size(440.0, 600.0);
    // The window draws its own theme-coloured controls (the page shows them when it runs inside this app).
    let builder = builder.decorations(false).transparent(true).shadow(true);   // our own rounded corners
    let _ = builder.build();
}

fn main() {
    tauri::Builder::default()
        .manage(Server(Mutex::new(None)))
        .setup(|app| {
            *app.state::<Server>().0.lock().unwrap() = start_server();
            let open = MenuItem::with_id(app, "open", "Open Live Translate", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&open, &quit])?;
            TrayIconBuilder::with_id("main")
                .icon(Image::from_bytes(IDLE)?)
                .icon_as_template(true)
                .tooltip("Live Translate")
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "open" => show(app),
                    "quit" => app.exit(0),
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click { button: MouseButton::Left, button_state: MouseButtonState::Up, .. } = event {
                        show(tray.app_handle());
                    }
                })
                .build(app)?;
            animate_tray(app.handle().clone());
            show(app.handle());
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();          // closing hides to the tray; Quit is in the tray menu
                let _ = window.hide();
            }
        })
        .build(tauri::generate_context!())
        .expect("failed to start Live Translate")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                if let Some(mut child) = app.state::<Server>().0.lock().unwrap().take() {
                    let _ = child.kill();     // stop the server only if this tray started it
                }
            }
        });
}
