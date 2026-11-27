<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/logo-dark.svg">
    <img alt="GoodPush" src="docs/_static/logo-light.svg" width="320">
  </picture>
</h1>

**Free cycling-training analysis and an adaptive training plan that run on your own computer.** Connect your Garmin
(or drop in a Strava or Garmin export) and GoodPush shows you:

* **what you can do now**: FTP / critical power, LT1 / LT2, W', your power-duration curve, rider type, zones
* **how your training has gone**: load (CTL), fatigue (ATL), form (TSB), what came before your best gains
* **how your body is doing**: sleep, HRV and resting heart rate against your normal, illness and overreaching
  warnings
* **what to ride next**: a day-by-day plan for your phases and races that adapts to each morning, with workouts for
  your watch, Incyclist, Zwift, or GoodPush's own trainer overlay (ANT+ ERG)

Everything runs locally. Your data never leaves your computer.

> **Beta: expect bugs.** GoodPush is in beta testing. Things will break, numbers can be off, and settings may
> change between versions. Please [tell us what goes wrong](https://github.com/GoodPush/GoodPush/issues/new/choose):
> every report helps.
>
> **Use at your own risk.** GoodPush is not medical advice or a medical device; whether and how you train is
> entirely up to you. See [DISCLAIMER.md](DISCLAIMER.md).

**Full guide:** <https://goodpush.github.io/GoodPush/>

---

## Install

Download the installer for your computer from the
[latest release](https://github.com/GoodPush/GoodPush/releases/latest):

| Your computer | File |
|---|---|
| Mac with Apple Silicon (M1 or later) | `goodpush-mac-apple-silicon.dmg` |
| Mac with an Intel processor | `goodpush-mac-intel.dmg` |
| Windows 10 / 11 | `goodpush-windows-setup.exe` |
| Linux (64-bit) | `goodpush-linux.tar.gz` |

**Your computer will warn you the first time.** GoodPush isn't *code-signed* yet (signing costs money every year;
see [what it would cost](COSTS.md)), so your computer can't check who made it. The warning is about the missing
signature, not anything GoodPush does. To open it anyway:

* **Mac:** open the .dmg, drag GoodPush to Applications and open it. When macOS says it can't be verified, click
  **Done**, go to **System Settings > Privacy & Security**, scroll to "goodpush was blocked" and click
  **Open Anyway**. Only the first time.
* **Windows:** run the setup. If "Windows protected your PC" appears, click **More info**, then **Run anyway**.
* **Linux:** `tar -xzf goodpush-linux.tar.gz && sh goodpush/install.sh`

GoodPush opens in your web browser. An icon in the menu bar (Mac) or system tray (Windows) opens it again or quits
it. To update, install the new version over the old one: your data stays.

Developers: [install from source](https://goodpush.github.io/GoodPush/start/install.html#from-source).

## First steps

1. **Connect Garmin**: on the **Today** page click **Connect Garmin** (also in the **Setup** list at the top right) and paste a Garmin Personal API key (Garmin
   Connect on the web > Settings > Personal API Key). GoodPush pulls your last 90 days.
2. **Bring in your whole history**: request your account export from Garmin (or Strava) and drop the zip anywhere
   in the app. Your thresholds and plan get much better with years of rides.
3. **Set up your plan**: on **Plan**, add your phases (base, build...) and races, and your hours per weekday in
   **Settings**.
4. **Get your workouts onto your watch and apps**: GoodPush sends your plan to
   [intervals.icu](https://intervals.icu) (free account), and intervals.icu passes it on. In intervals.icu, under
   **Settings > Connections**, connect your Garmin, Wahoo, Zwift or MyWhoosh account and switch on sending planned
   workouts to it. Then paste your intervals.icu API key in GoodPush (**Settings > API keys**) and press
   **Send plan now** under **Settings > intervals.icu**, again whenever the plan changes.

No Garmin? A Strava export works too. Details:
[Getting your data in](https://goodpush.github.io/GoodPush/start/data.html).

## Support intervals.icu

GoodPush gets your plan onto your watch, bike computer and training apps through
[intervals.icu](https://intervals.icu): it sends your workouts there, and intervals.icu passes them on to Garmin,
Wahoo, Zwift, MyWhoosh and more. Those connections are what make a free, local tool like GoodPush practical.
Building them ourselves would mean paid developer programmes and a server we would have to run.

intervals.icu is a small independent project that is free to use and paid for by the riders who support it. If you
use GoodPush with intervals.icu, please **[become an intervals.icu supporter](https://intervals.icu)**. It keeps the
bridge GoodPush stands on standing. GoodPush has no connection to intervals.icu and gets nothing from it; we just
rely on it.

## Help, feedback and contributing

* **Something wrong?** In the app: **Beta** (top right) > **Report a problem**. Or
  [open an issue](https://github.com/GoodPush/GoodPush/issues/new/choose).
* **Do the numbers make sense for you?** If you know your body (lab or ramp tests, race results, how training
  feels), tell us where GoodPush agrees and where it doesn't. This is one of the most useful things you can do:
  [Do the numbers make sense?](https://github.com/GoodPush/GoodPush/issues/new?template=numbers.yml)
* **Want to help?** [CONTRIBUTING.md](CONTRIBUTING.md): testing, feedback, wording, docs and code. No programming
  needed for most of it.
* **What it costs:** GoodPush is free. [COSTS.md](COSTS.md) lists what would cost money (code signing, app stores)
  and what each would change for you.

## Privacy

All processing is on your computer. GoodPush only goes online to download your own data (Garmin, Strava, Google
Health), to send your planned workouts to intervals.icu when you press Send plan, for map tiles when you open a
ride, to sync with your own other computers if you pair them, and when you click a link to GitHub. Details: [Privacy](https://goodpush.github.io/GoodPush/guide/privacy.html).

## License

GoodPush is free software under the [GNU Affero General Public License v3.0 or later](LICENSE) (AGPLv3), without
warranty; see also [DISCLAIMER.md](DISCLAIMER.md). You may use, change and share it, also in paid products, as long
as what you distribute or offer over a network is released under the same licence with its source code.

**Attribution is required.** Under the additional terms in [NOTICE](NOTICE) (AGPLv3 section 7), any product or
service that contains GoodPush or is built on it must show, in its About / Help / legal page and in its
documentation, the credit "Based on GoodPush by Leon Foks, free and open-source software" with a working link to
https://github.com/GoodPush/GoodPush. Modified versions must say they are modified and must not be called GoodPush.
Versions up to v0.1.0 were released under the MIT License.

The installers also bundle Python and the open-source libraries GoodPush uses (numpy, pandas, SciPy, JAX, Qt /
PySide6, libusb and others), each under its own licence; their notices and licence texts are in
`THIRD_PARTY_NOTICES.txt` in every download (generated by `packaging/notices.py`). Qt and libusb (LGPL) are
separate library files inside the app and can be replaced. The Windows installer includes
[Zadig](https://zadig.akeo.ie) (GPLv3, unmodified) to set up the ANT+ USB driver; its licence and notice are in
the app's `tools` folder, and the source of the exact version shipped is attached to each release
(`libwdi-<version>-source.tar.gz`).
