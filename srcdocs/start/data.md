# Getting your data in

GoodPush needs your rides (with power if you have it, heart rate helps) and, for readiness and recovery, your
sleep, HRV and resting heart rate. There are three routes; most people use the first two.

## 1. Connect Garmin (recommended)

Garmin's **Personal API** is Garmin's official, read-only way to get your own data, free for you.

1. In Garmin Connect on the web open **Settings > Personal API Key**
   (<https://connect.garmin.com/app/settings/personalApiToken>), create a key and pick how long it lasts (up to
   365 days). Garmin shows it only once and e-mails you before it expires.
2. In GoodPush, click **Connect Garmin** in the setup list at the top right (or go to **Settings > Garmin connection**) and paste it.

The key is kept in GoodPush's own folder on this computer ([where](../guide/privacy.md)), never in your GoodPush
folder. GoodPush then pulls your last
90 days: rides with their original .fit files, sleep, resting HR, HRV, stress, Body Battery, steps and weight.
After that **Update now** (click the cloud icon at the top right) pulls only what's new. To have the app check by itself while it runs, set
**Settings > Garmin connection > Also check Garmin by itself every ... hours** (`[garmin] auto_sync_hours`; 0, the
default, = only when you press Update now).

From the command line: `goodpush garmin-key` (a hidden prompt; `--test` checks the key, `--forget` removes it), then
`goodpush update`. To update without the app open, schedule `goodpush update` (Windows Task Scheduler, cron).

## 2. Bring in your whole history (an account export)

90 days is enough to start, but thresholds, the model and the plan get much better with years of rides. Ask Garmin
or Strava for your account export, then drop the zip anywhere in the app (still zipped, as you downloaded it).

* **Garmin:** sign in at garmin.com, **Account > Data Management > Export Your Data > Request Data Export**. Garmin
  e-mails a download link, usually within a day or two.
* **Strava:** on strava.com, **Settings > My Account > Download or Delete Your Account > Get Started > Request Your
  Archive**. Strava e-mails a link, usually within a few hours.

The export holds every activity you've ever uploaded, so only the newest one is ever needed; importing a newer one
replaces the older files. Rides that are in both a Garmin and a Strava export count once. Ride notes (sickness,
fatigue) are read from the activity descriptions, so keep typing them in Garmin Connect or Strava as before.

From the command line: `goodpush import ~/Downloads/export_12345678.zip`, then `goodpush run`. Single watch `.fit`
files (sleep, HRV, monitoring) can be imported the same way.

<details>
<summary>Where the files go, if you unzip them yourself</summary>

Strava: `activities.csv` directly in `data/raw/strava/`, with the `activities/` folder next to it. Garmin: the
`DI_CONNECT` folder in `data/raw/garmin/`; leave the zips inside it zipped, GoodPush reads inside them.

| Metric | File in the Garmin export |
|---|---|
| Sleep duration, stages, bedtime; sleep score, sleep stress, respiration (newer watches) | `DI-Connect-Wellness/*_sleepData.json` |
| Resting HR, all-day stress, Body Battery, steps | `DI-Connect-Aggregator/UDSFile_*.json` |
| Nightly HRV and its baseline | `DI-Connect-Uploaded-Files/UploadedFiles_*.zip` (HRV FIT files) and `DI-Connect-Wellness/*_healthStatusData.json` |
| HRV 7-day average, training readiness | `DI-Connect-Metrics/TrainingReadinessDTO_*.json` |
| Weight, Garmin's cycling VO2max and FTP | `DI-Connect-Wellness/*_userBioMetrics.json` |

The first run scans the uploaded-files zips for HRV files (about 30 s) and remembers what it found.
`goodpush inspect` lists what GoodPush can see and how many days each metric covers.
</details>

## 3. Other sources

### Fitbit (through Google Health)

Sleep, resting HR, HRV, breathing rate, SpO2 and skin temperature, through the Google Health API (Fitbit's own API
was switched off in November 2026). Free, and only between Google and your computer, but it needs your own Google
Cloud project:

1. In the **Google Health** app, sign in with a personal Google account (move the Fitbit account to it when asked).
2. At [console.cloud.google.com](https://console.cloud.google.com): a new project, enable the **Google Health
   API**, set up the OAuth consent screen (External, add yourself as a test user), then **Publish app** (left in
   "Testing", Google expires the sign-in every 7 days). Then Credentials > **OAuth client ID**, type **Desktop app**.
3. In GoodPush: **Settings > Fitbit (Google Health)**: paste the client ID and secret, **Connect**, approve.

The first update pulls a year; after that it runs with the Garmin check (`goodpush sync google-health`). With a
Garmin too, Garmin wins wherever it has a value and Fitbit fills the gaps; the two HRVs are never mixed, since their
normal ranges differ.

### Strava's API (needs a Strava subscription)

Since July 2026 Strava only serves its API to applications whose owner has an active Strava subscription. If you
have one: create an application at <https://www.strava.com/settings/api> (any name and website, *Authorization
Callback Domain* `localhost`), run `goodpush auth strava` (it asks for the Client ID and the 40-character Client
Secret and opens the browser to approve; tick "view data about your private activities"), then `goodpush update`.
Without a subscription the sync fails with "Application ... Inactive": use the free export instead.

### Garmin without a Personal API key

* **The Garmin Connect login** (`goodpush sync garmin`, the `[garmin]` extra): the same login as the website, through
  a community package, so a Garmin change can break it now and then, and repeated failed logins get you blocked for
  a while (HTTP 429). `goodpush doctor` checks the setup. After a 429 wait 48-72 hours and try once.
* **A browser script** (no login at all): `goodpush garmin-snippet` writes `garmin_snippet.js`. Open
  connect.garmin.com signed in, press F12 > Console, paste the script and press Enter. It downloads a
  `goodpush_garmin_<start>_<end>.json`; `goodpush garmin-import` takes in the newest one from your Downloads.

### Where downloads and sign-ins go

Downloads go to `data/raw/garmin/api/`, `data/raw/strava/api/` and `data/raw/google_health/`, next to your exports
and never touching them. Keys and sign-ins are kept in GoodPush's own folder on this computer
([where](../guide/privacy.md); `GOODPUSH_CONFIG` moves it), never in your GoodPush folder; your Garmin password is never stored. `goodpush deauth strava|garmin` removes
them.
