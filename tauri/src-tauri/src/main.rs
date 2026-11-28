//! GoodPush as a desktop app: goodpush's own server (tauri/sidecar.py, built by PyInstaller into a bundle inside
//! this app) on 127.0.0.1, its page in a window of this app instead of a browser tab, and a tray icon.
//!
//! Everything the app does stays in goodpush (Python): this shell starts the server, shows its page, sends other
//! sites and downloads where a browser would, and stops the server when it quits. Ride now works as in the
//! installed app: the server starts the ride window from its own bundle, here inside this app (so macOS asks this
//! app for Bluetooth; Info.plist says why). The page's Quit stops the server, and this app with it.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs::File;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::webview::{DownloadEvent, NewWindowResponse};
use tauri::{AppHandle, Manager, RunEvent, Url, WebviewUrl, WebviewWindow, WebviewWindowBuilder, WindowEvent};
use tauri_plugin_opener::OpenerExt;

const MAIN: &str = "main";

/// goodpush's server, a process of its own: where it listens, and how to stop it.
struct Server {
    child: Mutex<Option<Child>>,
    port: u16,
    log: PathBuf,
    stopping: AtomicBool,
}

impl Server {
    /// the installed app's port first (a page's settings are kept per port), then the next ones (launcher.App)
    const FIRST_PORT: u16 = 8765;
    const PORTS: u16 = 10;
    /// the first start after an update compiles the model's kernels: give it time
    const START_TIMEOUT: Duration = Duration::from_secs(240);
    const STOP_GRACE: Duration = Duration::from_secs(5);
    const POLL: Duration = Duration::from_millis(300);

    fn free_port() -> Result<u16, String> {
        (Self::FIRST_PORT..Self::FIRST_PORT + Self::PORTS)
            .find(|p| TcpListener::bind(("127.0.0.1", *p)).is_ok())
            .ok_or_else(|| format!("ports {}-{} are all in use", Self::FIRST_PORT, Self::FIRST_PORT + Self::PORTS - 1))
    }

    /// development (`npm run tauri dev`): the checkout's goodpush through uv
    #[cfg(debug_assertions)]
    fn command(port: u16) -> Command {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
        let mut c = Command::new("uv");
        c.arg("run").arg("--project").arg(&repo).arg("python").arg(repo.join("tauri").join("sidecar.py"));
        c.arg("--tauri-serve").arg(port.to_string());
        c
    }

    /// the built app: the server bundle tauri/build_mac.sh puts inside this one
    #[cfg(not(debug_assertions))]
    fn command(port: u16) -> Command {
        let mut c = Command::new(Self::bundled());
        c.arg("--tauri-serve").arg(port.to_string());
        c
    }

    #[cfg(not(debug_assertions))]
    fn bundled() -> PathBuf {
        let exe = std::env::current_exe().unwrap_or_default();
        let here = exe.parent().unwrap_or(Path::new("."));
        if cfg!(target_os = "macos") {
            here.join("../Helpers/goodpush-server.app/Contents/MacOS/goodpush")
        } else {
            here.join("goodpush-server").join(if cfg!(windows) { "goodpush.exe" } else { "goodpush" })
        }
    }

    fn start(app: &AppHandle) -> Result<Server, String> {
        let port = Self::free_port()?;
        let dir = app.path().app_log_dir().map_err(|e| e.to_string())?;
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let log = dir.join("server.log");
        let out = File::create(&log).map_err(|e| format!("{}: {e}", log.display()))?;
        let err = out.try_clone().map_err(|e| e.to_string())?;
        let mut cmd = Self::command(port);
        // your home folder, not wherever the app was opened from: goodpush looks for a project upward from here
        if let Ok(home) = app.path().home_dir() {
            cmd.current_dir(home);
        }
        let child = cmd
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::from(out))
            .stderr(Stdio::from(err))
            .spawn()
            .map_err(|e| format!("goodpush's server didn't start: {e}"))?;
        Ok(Server { child: Mutex::new(Some(child)), port, log, stopping: AtomicBool::new(false) })
    }

    fn url(&self) -> Url {
        Url::parse(&format!("http://127.0.0.1:{}/", self.port)).expect("a valid url")
    }

    fn listening(&self) -> bool {
        let addr = SocketAddr::from(([127, 0, 0, 1], self.port));
        TcpStream::connect_timeout(&addr, Duration::from_millis(500)).is_ok()
    }

    /// the server ended by itself (the page's Quit, or it failed)
    fn exited(&self) -> bool {
        match self.child.lock().unwrap().as_mut() {
            Some(c) => !matches!(c.try_wait(), Ok(None)),
            None => true,
        }
    }

    /// this app quits: ask the server to stop (SIGTERM: it shuts down as on the page's Quit), then make sure
    fn stop(&self) {
        self.stopping.store(true, Ordering::SeqCst);
        let Some(mut child) = self.child.lock().unwrap().take() else { return };
        #[cfg(unix)]
        unsafe {
            libc::kill(child.id() as libc::pid_t, libc::SIGTERM);
        }
        let until = Instant::now() + Self::STOP_GRACE;
        while Instant::now() < until {
            if !matches!(child.try_wait(), Ok(None)) {
                return;
            }
            thread::sleep(Self::POLL);
        }
        let _ = child.kill();
        let _ = child.wait();
    }

    /// until the server listens, then its page in the window; when it ends by itself, this app ends too
    fn watch(app: AppHandle) {
        let server = app.state::<Server>();
        let started = Instant::now();
        let mut ready = false;
        loop {
            if server.exited() {
                if server.stopping.load(Ordering::SeqCst) {
                    return;
                }
                if ready {
                    app.exit(0);
                } else {
                    Shell::problem(&app, "goodpush's server stopped while starting.", &server.log);
                }
                return;
            }
            if !ready && server.listening() {
                ready = true;
                if let Some(w) = app.get_webview_window(MAIN) {
                    let _ = w.navigate(server.url());
                }
            }
            if !ready && started.elapsed() > Self::START_TIMEOUT {
                Shell::problem(&app, "goodpush's server didn't start in time.", &server.log);
                return;
            }
            thread::sleep(Self::POLL);
        }
    }
}

/// the window, the tray icon and what a browser would do for the page
struct Shell;

impl Shell {
    fn window(app: &AppHandle) -> tauri::Result<WebviewWindow> {
        let opener = app.clone();
        let tabs = app.clone();
        let downloads = app.path().download_dir().unwrap_or_else(|_| PathBuf::from("."));
        WebviewWindowBuilder::new(app, MAIN, WebviewUrl::App("index.html".into()))
            .title("goodpush")
            .inner_size(1280.0, 860.0)
            .min_inner_size(380.0, 560.0)
            // the page's own drop zone (an account export's .zip dropped anywhere), not the window's
            .disable_drag_drop_handler()
            .on_navigation(move |url| {
                if Self::ours(url) {
                    return true;
                }
                let _ = opener.opener().open_url(url.as_str(), None::<&str>);
                false
            })
            // a link that opens a new tab (other sites, the dashboard): in your browser
            .on_new_window(move |url, _features| {
                let _ = tabs.opener().open_url(url.as_str(), None::<&str>);
                NewWindowResponse::Deny
            })
            .on_download(move |_webview, event| {
                if let DownloadEvent::Requested { url, destination } = event {
                    *destination = Self::save_as(&downloads, &url, destination);
                }
                true
            })
            .build()
    }

    /// the app's own pages (and this window's start page) stay in the window; any other site opens in your browser
    fn ours(url: &Url) -> bool {
        match url.scheme() {
            "http" | "https" => matches!(url.host_str(), Some("127.0.0.1") | Some("localhost")),
            "tauri" | "about" | "blob" | "data" | "asset" => true,
            _ => false,
        }
    }

    /// a download goes into Downloads under its own name, numbered when that file is there already
    fn save_as(dir: &Path, url: &Url, suggested: &Path) -> PathBuf {
        let name = suggested
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .or_else(|| url.path_segments().and_then(|mut s| s.next_back()).map(str::to_owned))
            .filter(|n| !n.is_empty())
            .unwrap_or_else(|| "download".into());
        let first = dir.join(&name);
        if !first.exists() {
            return first;
        }
        let p = Path::new(&name);
        let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let ext = p.extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default();
        (1..)
            .map(|i| dir.join(format!("{stem} ({i}){ext}")))
            .find(|c| !c.exists())
            .expect("a free name")
    }

    fn show(app: &AppHandle) {
        if let Some(w) = app.get_webview_window(MAIN) {
            let _ = w.show();
            let _ = w.unminimize();
            let _ = w.set_focus();
        }
    }

    fn tray(app: &AppHandle) -> tauri::Result<()> {
        let open = MenuItem::with_id(app, "open", "Open goodpush", true, None::<&str>)?;
        let quit = MenuItem::with_id(app, "quit", "Quit goodpush", true, None::<&str>)?;
        let menu = Menu::with_items(app, &[&open, &quit])?;
        let mut tray = TrayIconBuilder::with_id("goodpush")
            .tooltip("goodpush")
            .menu(&menu)
            .show_menu_on_left_click(cfg!(target_os = "macos"))
            .on_menu_event(|app, event| match event.id().as_ref() {
                "open" => Self::show(app),
                "quit" => app.exit(0),
                _ => {}
            })
            .on_tray_icon_event(|tray, event| {
                // Windows, Linux: a click on the icon itself opens the window; the menu is on right-click
                if let TrayIconEvent::Click { button: MouseButton::Left, button_state: MouseButtonState::Up, .. } = event {
                    Self::show(tray.app_handle());
                }
            });
        if let Some(icon) = app.default_window_icon() {
            tray = tray.icon(icon.clone());
        }
        tray.build(app)?;
        Ok(())
    }

    /// the start page says what went wrong and where the server's log is
    fn problem(app: &AppHandle, what: &str, log: &Path) {
        if let Some(w) = app.get_webview_window(MAIN) {
            let html = format!(
                "<h1>goodpush didn't start</h1><p>{}</p><p>What it wrote is in<br><code>{}</code></p>",
                Self::escape(what),
                Self::escape(&log.display().to_string())
            );
            let _ = w.eval(&format!("document.getElementById('status').innerHTML = {};", Self::js_string(&html)));
        }
    }

    fn escape(s: &str) -> String {
        s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
    }

    fn js_string(s: &str) -> String {
        format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n"))
    }
}

fn main() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let handle = app.handle().clone();
            Shell::window(&handle)?;
            Shell::tray(&handle)?;
            match Server::start(&handle) {
                Ok(server) => {
                    app.manage(server);
                    thread::spawn(move || Server::watch(handle));
                }
                Err(e) => Shell::problem(&handle, &e, Path::new("(no log)")),
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            // closing the window leaves goodpush running (as closing its browser tab did): the tray icon, the Dock
            // icon or opening the app again shows it; Quit (menu, tray, the page) stops it
            if let WindowEvent::CloseRequested { api, .. } = event {
                if window.label() == MAIN {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("the desktop app couldn't start");
    app.run(|app, event| match event {
        RunEvent::ExitRequested { .. } | RunEvent::Exit => {
            if let Some(server) = app.try_state::<Server>() {
                server.stop();
            }
        }
        #[cfg(target_os = "macos")]
        RunEvent::Reopen { .. } => Shell::show(app),
        _ => {}
    });
}
