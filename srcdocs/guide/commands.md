# Commands

Everything the app does can also be run from a terminal. Run commands from inside your GoodPush folder, or add
`--root PATH` (a folder) or `--athlete NAME` (a registered athlete). `goodpush COMMAND --help` lists each command's
options.

| Command | What it does |
|---|---|
| `goodpush app [--lan]` | open the app; `--lan` also for your phone on the same Wi-Fi |
| `goodpush init` | make `goodpush.toml` and the folders in the current folder |
| `goodpush import <zip / folder / fit>...` | take in a Strava or Garmin export (then `goodpush run`) |
| `goodpush inspect` | what data GoodPush can see, and how many days each Garmin metric covers |
| `goodpush run` | take in new ride files, analyse, write the reports (the normal command) |
| `goodpush update` | download everything new (Garmin, Strava, Google Health), then `run` |
| `goodpush ingest [--force]` | only read new or changed ride files |
| `goodpush analyze` | rebuild the analysis and reports from what's already read (fast) |
| `goodpush plan [--date D]` | re-plan (+ the calendar file and watch workouts) |
| `goodpush physio [--nuts ...]` | fit the physiological model; `--nuts` for its ranges, `--since-days N` on the last N days only ([details](plan.md#the-physiological-model)) |
| `goodpush ride [scan] [--free] [--session D] [--pick] [--sim]` | ride a session on the trainer (ANT+) with the trainer bar |
| `goodpush summary` | print the headline numbers |
| `goodpush illness` | list the sickness notes found in your rides and the illnesses built from them |
| `goodpush garmin-key [--test \| --forget]` | save, check or remove your Garmin Personal API key |
| `goodpush sync [strava \| garmin \| google-health] [--since DATE]` | download new data only |
| `goodpush auth strava \| garmin` | one-time sign-in for Strava's API or the older Garmin login |
| `goodpush deauth strava \| garmin [--purge-data]` | disconnect: delete the stored sign-in (optionally the downloaded data) |
| `goodpush doctor` | check the Garmin / Strava setup |
| `goodpush garmin-snippet [--since D]` | a browser script that downloads Garmin data without logging in |
| `goodpush garmin-import` | take in the newest browser download, then rebuild |
| `goodpush incyclist [status \| on \| off \| sync] [--folder P]` | write the plan into Incyclist's workout calendar |
| `goodpush athlete [add NAME FOLDER]` | the athletes on this computer |
| `goodpush peer [list \| join ADDRESS CODE \| add NAME USER@HOST FOLDER \| sync NAME]` | your folder on your other computers |
| `goodpush streams [status \| pack \| check \| prune]` | the store of ride samples (`data/processed/streams.h5`) |

## Outputs

`goodpush run` also writes, in `reports/`:

* `dashboard.html`: interactive charts (it loads its charting library from the internet)
* `profile.md`: the written profile with all tables
* `tables/*.csv`: activities, daily and weekly tables for a spreadsheet or your own analysis
* `results.json`: every number in machine-readable form
