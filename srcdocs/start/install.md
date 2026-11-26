# Install

## The app (recommended): no Python needed

Download the installer for your computer from the
[latest release](https://github.com/GoodPush/GoodPush/releases/latest):

| Your computer | File |
|---|---|
| Mac with Apple Silicon (M1 or later) | `goodpush-mac-apple-silicon.dmg` |
| Mac with an Intel processor | `goodpush-mac-intel.dmg` |
| Windows 10 / 11 | `goodpush-windows-setup.exe` |
| Linux (64-bit) | `goodpush-linux.tar.gz` |

**Why your computer warns you.** GoodPush isn't *code-signed* yet: signing means paying Apple and Microsoft every
year to vouch for the developer ([what that would cost](../community/costs.md)). Without it your computer can't
check who made the app, so it warns you the first time you open it. The warning is about the missing signature, not
about anything GoodPush does. To open it anyway:

* **Mac:** open the .dmg, drag GoodPush to Applications and open it. macOS says it can't be verified: click
  **Done**, then go to **System Settings > Privacy & Security**, scroll to "goodpush was blocked" and click
  **Open Anyway**. Only the first time.
* **Windows:** run the setup. If "Windows protected your PC" appears, click **More info**, then **Run anyway**.
* **Linux:** `tar -xzf goodpush-linux.tar.gz && sh goodpush/install.sh` (adds it to your applications menu and
  can set up an ANT+ USB stick).

GoodPush opens in your web browser. An icon in the menu bar (Mac) or system tray (Windows) opens it again or quits
it. Your data lives in

* Mac: `~/Library/Application Support/goodpush/athlete`
* Windows: `%APPDATA%\goodpushthlete`
* Linux: `~/.local/share/goodpush/athlete`

and your keys and sign-ins in GoodPush's own folder on this computer:

* Mac: `~/Library/Application Support/goodpush`
* Windows: `%LOCALAPPDATA%\goodpush`
* Linux: `~/.config/goodpush`

To keep either somewhere else, e.g. on another drive: quit GoodPush, move the folder there, and set the environment
variable `GOODPUSH_HOME` (your data) or `GOODPUSH_CONFIG` (keys and sign-ins) to its new place. Earlier versions kept
the data one folder up and the keys in `~/.goodpush`; GoodPush moves them to the places above once, by itself. **To update**, install the new version over
the old one: your data stays. Next: [get your data in](data.md).

### ANT+ USB stick (for riding on a trainer)

Only needed to [ride with GoodPush's trainer bar](../guide/riding.md#goodpush-s-trainer-bar).

* **Windows:** the stick needs a USB driver once. Tick "Set up an ANT+ USB stick now" in the installer (or later:
  Start menu > **goodpush ANT+ USB setup**). That opens [Zadig](https://zadig.akeo.ie), the standard tool for this,
  with the steps in Notepad next to it. In short: plug the stick in and quit other apps that use it;
  **Options > List All Devices**; pick the stick: its USB ID starts with **0FCF** (only the stick: another
  device's driver must not be changed); if its driver already says WinUSB or libusb0, change nothing; otherwise pick
  **WinUSB** and click **Install Driver**; then unplug the stick and plug it back in.
* **Mac:** nothing to do.
* **Linux:** `install.sh` sets it up.

## From source

For developers, or to run the latest code. You need **Python 3.12 or newer** (3.14 recommended) and the code:

```bash
git clone https://github.com/GoodPush/GoodPush.git
cd GoodPush
```

**Option A: uv** (recommended; it installs Python 3.14 and the exact versions in `uv.lock`):

```bash
uv run --extra all goodpush app                # the first run sets up .venv, then starts the app
```

Use `uv sync --extra all` to set up without starting anything, and `uv run goodpush ...` for any command.

**Option B: conda / Miniforge** (for Intel Macs, where pip has no recent JAX or numba builds):

```bash
conda env create -f environment.yml            # Windows: environment-windows.yml
conda activate goodpush
```

**Option C: plain Python with a virtual environment:**

```bash
python -m venv .venv
.venv\Scripts\activate                         # Windows
source .venv/bin/activate                      # macOS / Linux
pip install -e ".[all]"
```

The extras (`uv sync --extra name`, or `pip install -e ".[name]"`):

| Extra | What it adds |
|---|---|
| `physio` | the physiological model (JAX, NumPyro, numba): the Rider and Model pages, `load_model = "physio"` |
| `garmin` | the older Garmin Connect login (the Personal API needs nothing extra) |
| `ride` | `goodpush ride`, the trainer bar (ANT+ stick, Qt) |
| `sync` | pairing your own computers |
| `all` | all of the above: what the installed app bundles |
| `dev` | `all` plus pytest and the commit checks |
| `docs` | Sphinx, to build this guide |

uv and `-e` (editable) both mean changes to the code take effect without reinstalling. Check it worked with
`goodpush --help`, then start the app with `goodpush app`.

### Your folder from the command line

The app makes its folder by itself. From the command line, any folder can be a GoodPush folder (a *workspace*):

```bash
mkdir my_training
cd my_training
goodpush init
```

```
my_training/
  goodpush.toml          <- your settings (the app's Settings page edits it too)
  data/raw/strava/       <- a Strava export goes here
  data/raw/garmin/       <- a Garmin export goes here
  data/processed/        <- caches and analysis tables (rebuilt when needed)
  reports/               <- plan, dashboard, profile and tables
  notes/                 <- your own notes
  logs/                  <- what each run did (made on the first run; this computer's own)
```

Run every `goodpush` command from inside it, or pass `--root path/to/folder`. The settings worth checking first in
`goodpush.toml`:

| Setting | Why |
|---|---|
| `[athlete] name`, `sex` | the greeting; the power-profile tables (male / female) |
| `timezone` | e.g. `"Europe/London"`; puts rides and nights on the right day |
| `weight_kg` | W/kg. Leave 0 to use the Garmin weight log or the Strava profile weight |
| `max_hr`, `resting_hr`, `lthr` | optional; estimated from your data when left at 0 |
| `[ftp.history]` + `mode = "manual"` | optional; your own FTP test results instead of the automatic estimate |
| `[[events]]` | optional; illnesses, injuries, races, life events (shown on charts, used in the analysis) |

See [Commands](../guide/commands.md) for the rest.
