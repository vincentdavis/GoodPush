## goodpush 0.4.1

Analyse your cycling training and plan the next weeks, on your own computer, from your Strava / Garmin exports
and the Garmin Personal API.

### New in 0.4.1

- **Today, one card for the day:** this morning's verdict with its reasons, how you feel, and today's session with
  its steps. **‹ ›** beside the date (or the arrow keys, or a swipe on a phone) steps through the days: what you rode
  and that morning's numbers, or what's planned.
- **The last 7 mornings** are a row of ovals; hover one for the day and why.
- **Setup n/4** at the top right lists what's left to set up: Garmin, your whole history, the model's full run, a
  ride app.
- **Sync icon (two arrows)** at the top right: whether your data is up to date, **Update now**, and the new Sync page
  with the last update's log. Everything still stays on your computer.
- **Colour-blind-safe colours:** every red / amber / green (verdicts, status chips, the morning ovals, the sync
  state) is now a teal green, amber and crimson that stay apart for red-green and blue-yellow colour blindness.
- **The physiology model's lines** (LT1, CP, MAP, ...) and their legends take the colour of the training zone each
  one falls in.
- **The app's own dialogs** instead of the browser's OK / Cancel boxes, with buttons that say what they do.
- **Fixed:** "Unable to decompress Zstandard data" when the installed app and a copy run from source shared a
  workspace.

### New in 0.4.0

- **A broken power meter no longer throws off your fitness.** One ride with power no rider could hold (say minutes
  at 1000 W at an easy heart rate) used to lift the whole fitted history. Now:
  - the model treats a best power far above your curve as a probable bad reading and sets it aside (the Rider page
    says so when you hover it);
  - a ride whose minutes are far above your usual while your heart rate stayed easy loses its best powers. Its
    time and load still count;
  - you can mark power as wrong yourself: **Power is wrong** on the Ride page, or rides, days and date ranges in
    Settings > Importing (a meter that read high for weeks). Those rides count as rides without power. Power that
    was flagged but is real can be kept with one click;
  - the model's starting point is the median of your first year's 8-week bests, so one glitch can't move it.
- **Physiological model v17.** On the first run after updating, your fit is redone from your rides, about a minute.
  It never builds on the old version's fit, so a fit that had gone wrong doesn't carry over. The ranges need a new
  full run (the Model page shows the command).
- **A new look:** light and dark themes with readable colours, one style for buttons and fields, and a bottom bar on
  phones that reaches every page.
- **Settings in sections** you can open and close, and **Fit the model from scratch** under Data and model.
- Python 3.12 or later is needed to run from source.

### What goodpush does

- **Fitness:** training load, FTP, thresholds (LT1 / LT2 from alpha1), power curve, a physiological model of how
  your fitness changes with training.
- **Plan:** an adaptive plan built around your races, the hours you have and how you're recovering; interval
  progression; sessions in your own watts; Incyclist and .zwo / .fit workouts.
- **Calendar:** drag sessions to any day, copy / paste them, drop a ride on a session to link it (or unlink it).
- **Rides:** power, heart rate and alpha1 in riding time (stops left out), intervals in zone colours, the planned
  session's targets lined up with what you rode.
- **Recovery:** sleep, HRV and resting heart rate; readiness that flags short nights and late bedtimes.
- **Ride on the trainer** (ANT+): ERG workouts with a small on-top overlay, recorded as .fit files.

## Download

| Your computer | File |
|---|---|
| Mac with Apple Silicon (M1 or later) | `goodpush-mac-apple-silicon.dmg` |
| Mac with an Intel processor | `goodpush-mac-intel.dmg` |
| Windows 10 / 11 | `goodpush-windows-setup.exe` |
| Linux (64-bit) | `goodpush-linux.tar.gz` |

Not sure which Mac you have? Apple menu > About This Mac: "Chip: Apple M…" is Apple Silicon, "Processor: Intel" is Intel.

## Why your computer warns you, and how to open it anyway

goodpush is free and isn't **code-signed**. Signing means paying Apple and Microsoft every year to vouch for the
developer; without it, your computer can't check who made the app, so it warns you the first time. The warning is
about the missing signature, not about anything goodpush does. It runs only on your computer and your data never
leaves it.

- **Mac:** open the .dmg and drag goodpush to Applications. Open it: macOS says it can't be verified, click **Done**.
  Then go to **System Settings > Privacy & Security**, scroll down to "goodpush was blocked" and click **Open Anyway**.
  Only the first time.
- **Windows:** run the setup. If "Windows protected your PC" appears, click **More info**, then **Run anyway**.
- **Linux:** unpack it (`tar -xzf goodpush-linux.tar.gz`), then `sh goodpush/install.sh`.

goodpush opens in your web browser, with an icon in the menu bar (Mac) or system tray (Windows) to open it again or quit.

## Licences

goodpush is free software under the GNU AGPL v3.0 or later, with the attribution terms in `NOTICE`. Each download includes `THIRD_PARTY_NOTICES.txt` for the libraries it bundles. The Windows
installer includes Zadig (GPLv3, unmodified) for the ANT+ USB driver; its source for this exact version is the
`libwdi-…-source.tar.gz` file below.
