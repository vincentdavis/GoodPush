# Troubleshooting

| Problem | Fix |
|---|---|
| The computer won't open the app | It isn't code-signed yet: see [Install](../start/install.md) for the one-time "Open Anyway" / "Run anyway". |
| No data after connecting Garmin | Click the cloud icon at the top right, then **Update now**; the Sync page shows what came in and the last update's log. Check the key in Settings (`goodpush garmin-key --test`); keys expire after the time you picked. |
| Only 90 days of rides | That's the Garmin key's first pull. Bring in the rest with an [account export](../start/data.md#2-bring-in-your-whole-history-an-account-export). |
| The export was dropped but nothing changed | Drop the zip as downloaded (not unzipped). Then Settings > **Rebuild everything**. |
| W/kg looks wrong | Set your weight in Settings (`weight_kg`). |
| FTP looks too low / high | The automatic estimate needs some hard efforts. Do a 20-min test, or enter your tests (`[ftp.history]` with `mode = "manual"`). |
| No ranges on Rider / Model | They need the model's longer run once: see [The physiological model](plan.md#the-physiological-model). |
| The app says it needs a restart | GoodPush was updated while the app ran: quit it (tray / menu-bar icon) and start it again. |
| The trainer bar finds no trainer | Windows: the ANT+ stick needs its [driver](../start/install.md#ant-usb-stick-for-riding-on-a-trainer). Close Zwift / Incyclist / Garmin Express, which may hold the stick. |
| Dashboard charts are blank | `reports/dashboard.html` loads its charting library from the internet; the app's own charts work offline. |
| Garmin `429` (older login) | Garmin is rate-limiting logins. Run `goodpush doctor`, wait 48-72 hours, then try `goodpush auth garmin --force` once. Better: use a Personal API key. |
| Strava `401 Application invalid` | The Client ID / Secret don't match. Copy the Client Secret again (not the access token). |
| Strava `403 ... Inactive` | Strava's API needs an active Strava subscription. Use the export instead. |
| `No goodpush.toml found` (command line) | Run commands inside your GoodPush folder, run `goodpush init` first, or add `--root PATH`. |

Still stuck? [Report it](https://github.com/GoodPush/GoodPush/issues/new?template=bug_report.yml), or in the app
**Beta** (top right) > **Report a problem**.
