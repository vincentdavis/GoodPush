"""Command line interface:  goodpush <command> [--root PATH]"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _proj(args):
    from .config import load_project
    return load_project(args.root, getattr(args, "athlete", None))


def cmd_athlete(args):
    """The athletes on this machine: list them, add one (their folder; a new folder gets a goodpush.toml), move one's
    folder (athletes.FolderMove) or remove one (their sign-ins here go, their folder stays)."""
    from .athletes import Athletes, FolderMove
    from .config import CONFIG_NAME, init_project
    from .progress import Progress
    reg = Athletes()
    if args.action == "move":
        if not args.name or not args.folder:
            sys.exit("usage: goodpush athlete move NAME FOLDER   (quit the app first if it shows NAME)")
        try:
            FolderMove(reg, args.name, args.folder, Progress.terminal()).run()
        except ValueError as e:
            sys.exit(str(e))
        return
    if args.action == "add":
        if not args.name or not args.folder:
            sys.exit("usage: goodpush athlete add NAME FOLDER [--default] [--credentials]")
        try:
            folder = reg.add(args.name, args.folder, args.default, credentials=False)
        except ValueError as e:
            sys.exit(str(e))
        if not (folder / CONFIG_NAME).exists():
            init_project(folder)
        copied = reg.copy_credentials(args.name) if args.credentials else []
        print(f"{args.name}: {folder}" + (" (default)" if reg.default == args.name else ""))
        if copied:
            print(f"  credentials copied to {reg.secrets(args.name)}: {len(copied)} file(s)")
        return
    if args.action == "remove":
        if not args.name:
            sys.exit("usage: goodpush athlete remove NAME")
        try:
            folder = reg.remove(args.name)
        except KeyError as e:
            sys.exit(e.args[0])
        print(f"{args.name} removed (their sign-ins on this computer too); their folder stays: {folder}")
        return
    if not reg.folders:
        print("No athletes yet: `goodpush athlete add NAME FOLDER` (your current folder: --credentials keeps "
              "this machine's Garmin / Strava sign-in for it)")
    for k, v in sorted(reg.folders.items()):
        print(f"{'*' if k == reg.default else ' '} {k:12s} {v}")


def cmd_synth(args):
    """A synthetic rider: `new` writes their whole history as an athlete folder (synth/ holds the truth);
    `check` analyses it as anyone's (`goodpush run` + the physio model's best fit) and scores it against the truth."""
    from .synth import RiderProfile, SyntheticRider
    from .synth.check import SynthCheck
    if args.action == "check":
        print(SynthCheck.text(SynthCheck(args.folder).run()))
        return
    profile = RiderProfile.load(args.profile)
    if args.truth:
        profile.truth = args.truth
    if args.seed is not None:
        profile.seed = args.seed
    folder = SyntheticRider(profile).write(args.folder).root
    if args.register:
        from .athletes import Athletes
        Athletes().add(args.register, folder)
    print(f"synthetic rider in {folder}: `goodpush synth check {args.folder}` scores goodpush's analysis of them"
          + (f"; `goodpush --athlete {args.register} app` shows them" if args.register else ""))


def cmd_peer(args):
    """Your athlete folder on another machine: join a goodpush app showing a code, add one over ssh, list them, or
    sync with one."""
    import socket
    from .app.tomledit import set_values
    from .config import CONFIG_NAME
    from .launcher import PORT
    from .sync.peer import Devices, FolderSide, HttpSide, PeerCert, Peers, PeerServer, PeerSync
    proj, peers = _proj(args), Peers()
    home = (proj.cfg.get("sync") or {}).get("home")
    if args.action == "join":
        if not (args.name and args.host):
            sys.exit("usage: goodpush peer join ADDRESS CODE   (both shown in Settings > Other computers > Show a code "
                     "on the other machine)")
        if not (proj.root / CONFIG_NAME).exists():
            sys.exit(f"No {CONFIG_NAME} in {proj.root}: `goodpush --root FOLDER init` first, or run this in the athlete folder.")
        try:
            print("Allow this computer on the other one when it asks ...")
            entry = HttpSide.pair(args.name, args.host, socket.gethostname())
        except (ValueError, ConnectionError) as e:
            sys.exit(str(e))
        name = entry.pop("name")
        peers.add(name, **entry)
        there = HttpSide(entry["address"], entry["fingerprint"], entry["token"], name)
        try:                                        # its way back here (when this computer's app runs)
            there.link(PeerCert().fingerprint(), PORT + PeerServer.PORT_OFFSET, Devices().pair(name))
        finally:
            there.close()
        print(f"Paired with {name}: `goodpush peer sync {name}`")
        if not home:
            set_values(proj.root / CONFIG_NAME, "sync", {"home": name})
            print(f"  {name} is home: it downloads from Garmin / Strava")
        return
    if args.action == "add":
        if not (args.name and args.host and args.folder):
            sys.exit("usage: goodpush peer add NAME USER@HOST FOLDER   (FOLDER: the athlete folder there, e.g. "
                     "~/goodpush/athletes/NAME)")
        peers.add(args.name, host=args.host, folder=args.folder)
        print(f"{args.name}: {args.host}:{args.folder}")
        if not home:
            set_values(proj.root / CONFIG_NAME, "sync", {"home": socket.gethostname()})
            print(f"  this machine ({socket.gethostname()}) is now home: it downloads from Garmin / Strava")
        return
    if args.action == "list" or not args.name:
        for k, v in sorted(peers.peers.items()):
            where = v["address"] if v.get("kind") == "app" else f"{v['host']}:{v['folder']} (ssh)"
            named = f"{v['label']} ({k})" if v.get("label") else k
            print(f"  {named:10s} {where}" + (f"   last sync {v['last']}" if v.get("last") else ""))
        if not peers.peers:
            print("No other machines yet: `goodpush peer join ADDRESS CODE` or `goodpush peer add NAME USER@HOST FOLDER`")
        return
    try:
        joined = peers.get(args.name).get("kind") == "app"
        there = peers.side(args.name)
        print(f"Comparing {proj.root} with {args.name} ...")
    except (KeyError, ConnectionError) as e:
        sys.exit(str(e.args[0] if isinstance(e, KeyError) else e))
    try:
        sync = PeerSync(FolderSide(proj.root), there, args.name, PeerSync.home_here_for(proj.cfg, joined))
        try:
            steps = sync.plan("send" if args.send else "get" if args.get else "both")
        except (FileNotFoundError, ConnectionError, PermissionError) as e:
            sys.exit(str(e))
        print(sync.describe(steps))
        if args.dry_run:
            return
        if steps and not args.yes and input("Go ahead? [y/N] ").strip().lower() not in ("y", "yes"):
            return
        out = sync.run(steps, progress=bool(steps))      # nothing to move: still notes that both are in step
        got = set(sync.changed["here"])
        again = sync.hand_over(wait=not (args.send or args.get))
        if again:
            got |= sync.changed["here"]
            out["clashes"] += again["clashes"]
    finally:
        there.close()
    peers.update(args.name, last=time.strftime("%Y-%m-%dT%H:%M"), result="ok")
    for c in out["clashes"]:
        print(f"  changed on both machines, {'this' if sync.home_here else 'the home'} machine's kept: {c}")
    if sync.home_here and {"data", "rows"} & got:
        print("New rides / health data arrived here: `goodpush run` to analyse them.")
    print("Done.")


def cmd_init(args):
    from .config import init_project
    proj = init_project(args.root or ".")
    print(f"Project ready at {proj.root}")
    print(f"  1. Unzip your Strava export into  {proj.path('strava')}")
    print(f"  2. Unzip your Garmin export into  {proj.path('garmin')}")
    print(f"  3. Edit {proj.root / 'goodpush.toml'} (weight, timezone, events)")
    print("  4. Run:  goodpush run")


def cmd_ingest(args):
    from .ingest import ingest
    from .progress import Progress
    ingest(_proj(args), workers=args.workers, force=args.force, progress=Progress.terminal())


def cmd_analyze(args):
    from .analyze import run_analysis
    from .progress import Progress
    from .report import write_reports
    proj = _proj(args)
    progress = Progress.terminal()
    r = run_analysis(proj, progress=progress)
    if proj.cfg.get("plan", {}).get("enabled"):
        from .planning import Planner
        try:
            r["plan"] = Planner(proj).build(r, progress=progress)
        except Exception as e:  # never let the plan break the reports
            progress(f"plan skipped: {e}")
    write_reports(proj, r, progress=progress)


def cmd_run(args):
    t0 = time.time()
    cmd_ingest(args)
    cmd_analyze(args)
    proj = _proj(args)
    print(f"done in {time.time() - t0:.0f}s - see it with `goodpush app` "
          f"(or open {proj.reports / 'dashboard.html'})")


def cmd_inspect(args):
    """Show what data goodpush can see (useful after adding new exports)."""
    from .io.garmin import load_wellness
    from .io.strava import project_activities
    proj = _proj(args)
    try:
        acts = project_activities(proj)
        print(f"Strava: {len(acts)} activities {acts['start_utc'].min():%Y-%m-%d} -> "
              f"{acts['start_utc'].max():%Y-%m-%d}")
        print(acts["type"].value_counts().to_string())
    except FileNotFoundError as e:
        print(e)
    daily, inv = load_wellness(proj.path("garmin"), proj.tz, cache_dir=proj.processed)
    print(f"\nGarmin: {inv['json_files']} json files, {inv['fit_files']} loose fit files")
    for label, info in inv["sources"].items():
        print(f"  {label}: {info}")
    print("\nCoverage per metric:")
    for col, cov in inv.get("coverage", {}).items():
        print(f"  {col:18s} {cov}")


def cmd_illness(args):
    """List sickness notes found in Strava and the illness episodes built from them."""
    import pandas as pd
    from .analyze import run_analysis
    from .models import illness as IL
    proj = _proj(args)
    r = run_analysis(proj, progress=lambda *a: None)
    ill = IL.illness_analysis(r["acts"], r["daily"], None, proj.cfg.get("events"))
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 90)
    print(f"{len(ill['notes'])} ride notes mention sickness:\n")
    for n in ill["notes"].itertuples():
        print(f"  {n.date:%Y-%m-%d}  [{n.matched}]  {n.text[:110]}")
    ep = ill["episodes"]
    print(f"\n{len(ep)} illness episodes:\n")
    if len(ep):
        print(ep[["onset", "comeback_note", "days_off", "rhr_peak_delta", "hrv_min_z", "severity"]]
              .round(1).to_string(index=False))
    print("\nAdd missing ones / fix dates in goodpush.toml:\n"
          '  [[events]]\n  date = "2024-03-01"\n  end = "2024-03-08"\n  type = "illness"\n  note = "flu"')


def cmd_import(args):
    from .importer import import_path
    proj = _proj(args)
    kinds = set()
    for p in args.paths:
        try:
            kinds.add(import_path(proj, p))
        except (ValueError, FileNotFoundError) as e:
            print(f"skipped: {e}")
    if kinds:
        print("Next: `goodpush run` (only new files are parsed).")


def cmd_auth(args):
    if args.service == "strava":
        from .sync import strava
        strava.authorize()
    else:
        from .sync import garmin
        garmin.login(force=args.force)


def cmd_doctor(args):
    """Check the Garmin/Strava connection setup without contacting either service."""
    import sys as _sys
    from .sync import garmin, secrets_dir
    print(f"python {_sys.version.split()[0]}  ({_sys.executable})")
    st = garmin.status()
    ok = lambda b: "ok" if b else "MISSING"
    print("\nGarmin")
    print(f"  garminconnect   {st['garminconnect'] or 'MISSING  -> pip install --upgrade garminconnect curl_cffi'}")
    print(f"  curl_cffi       {ok(st['curl_cffi'])}" +
          ("" if st["curl_cffi"] else "  -> pip install curl_cffi  (browser-like login; avoids most 429s)"))
    print(f"  saved session   {'yes' if st['tokens'] else 'no  -> goodpush auth garmin'}  ({st['token_dir']})")
    if st["blocked_hours_left"]:
        print(f"  rate-limit wait {st['blocked_hours_left']} h left (override: goodpush auth garmin --force)")
    from .sync import garmin_api
    ki = garmin_api.key_info()
    print(f"  personal API    {'key saved ' + str(ki['saved']) if ki['configured'] else 'no key  -> goodpush garmin-key'}")
    if ki.get("last_sync"):
        print(f"  last sync       {ki['last_sync'].get('at')}")
    print("\nStrava")
    print(f"  tokens          {'yes' if (secrets_dir() / 'strava.json').exists() else 'no  -> goodpush auth strava'}")


def cmd_streams(args):
    """Per-ride sample streams in data/processed/streams.h5: status / pack / check / prune / bench."""
    import time as _t
    import numpy as np
    from .io import streamstore as SS
    proj = _proj(args)
    P = proj.processed
    if not SS.available():
        sys.exit("h5py is not installed:  pip install h5py   (until then goodpush uses the .npz files)")
    if args.action == "pack":
        SS.pack(P, compact=args.compact)
    elif args.action == "prune":
        SS.prune(P)
    elif args.action == "check":
        bad = 0
        for group in SS.GROUPS:
            staged = SS._npz_ids(P, group)
            with SS._open(SS.h5_path(P)) as f:
                idx = SS._index(f, group)
                for a, path in staged.items():
                    if a not in idx:
                        print(f"  {group} {a}: not packed yet"); bad += 1; continue
                    o, n, r = idx[a]
                    z = np.load(path)
                    for c, dt in SS.GROUPS[group]["channels"].items():
                        h = f[group][c][o:o + n]
                        if c in z and not np.array_equal(np.asarray(z[c], dt), h, equal_nan=h.dtype.kind == "f"):
                            print(f"  {group} {a}: {c} differs"); bad += 1
            print(f"{group}: {len(staged)} .npz files compared")
        print("all identical" if not bad else f"{bad} problems")
    elif args.action == "bench":
        def t(fn):
            t0 = _t.perf_counter(); r = fn(); return _t.perf_counter() - t0, r
        for group, ch in (("ride", ["power", "heart_rate"]), ("alpha", ["a1", "p", "hr"])):
            staged = SS._npz_ids(P, group)
            dt_h5, (k, n, flat) = t(lambda: SS.read_all(P, group, ch))
            dt_npz, many = t(lambda: [np.load(p)[ch[0]] for p in staged.values()])
            one = k[len(k) // 2] if len(k) else None
            dt_one, _ = t(lambda: SS.read(P, one, group)) if one else (0, None)
            print(f"{group:5s}: all {len(k)} rides ({int(n.sum()):,} samples)  h5 {dt_h5:.2f}s  vs  "
                  f"{len(staged)} .npz files {dt_npz:.2f}s  |  one ride from h5 {dt_one * 1000:.1f} ms")
    p = SS.h5_path(P)
    if p.exists():
        with SS._open(p) as f:
            for group in SS.GROUPS:
                if group in f:
                    g = f[group]
                    print(f"{group}: {g["ids"].shape[0]} {"nights" if group == "hrv" else "rides"}, {int(g['length'][:].sum()):,} samples")
            print(f"dead space {SS._dead_fraction(f):.0%}")
        print(f"{p}  {p.stat().st_size / 1e6:.1f} MB;  staged .npz: "
              f"{sum(len(SS._npz_ids(P, g)) for g in SS.GROUPS)}")
    else:
        print("no streams.h5 yet - `goodpush streams pack` builds it from the .npz files")


def cmd_garmin_key(args):
    """Store (or test / forget) your Garmin Personal API key in this computer's machine folder."""
    from .sync import garmin_api as G
    if args.forget:
        G.forget_key()
        print("Garmin Personal API key removed from this computer.")
        return
    if not args.test:
        import getpass
        print("Paste the key from Garmin Connect > Settings > Personal API Key")
        print("(it is not shown while you paste; press Enter after).")
        key = getpass.getpass("Garmin Personal API key: ")
        p = G.save_key(key)
        print(f"Saved to {p} (outside the project folder).")
    G.access_token(force=True)
    print("Garmin accepted the key - `goodpush update` (or the app) now pulls your data automatically.")


def cmd_incyclist(args):
    """Plan -> Incyclist's workout calendar: status / on / off / sync (writes it now)."""
    from .app.tomledit import set_values
    from .sync import incyclist as I
    proj = _proj(args)
    if args.folder is not None:
        if args.folder and not (Path(args.folder).expanduser() / "workouts").is_dir():
            sys.exit(f"No Incyclist data in {args.folder} (expected a 'workouts' folder there)")
        set_values(proj.root / "goodpush.toml", "incyclist", {"folder": args.folder})
        proj = _proj(args)
    if args.action in ("on", "off"):
        set_values(proj.root / "goodpush.toml", "incyclist", {"sync": args.action == "on"})
        proj = _proj(args)
    if args.action in ("on", "sync"):
        from .planning import Planner
        if not proj.cfg["incyclist"].get("sync"):
            sys.exit("Incyclist sync is off - `goodpush incyclist on` first")
        Planner(proj).build()
    st = I.status(proj.cfg)
    print(f"Incyclist data folder: {st['folder']} ({'found' if st['found'] else 'NOT FOUND'})")
    print(f"sync after every plan update: {'on' if st['sync'] else 'off'}; goodpush workouts in its calendar: "
          f"{st['entries']}" + (f" ({st['days'][0]} .. {st['days'][-1]})" if st["days"] else ""))
    if st["running"]:
        print("Incyclist is open: restart it to load the new calendar.")


def cmd_garmin_snippet(args):
    """Write the browser script that downloads Garmin data without a login."""
    from .sync import garmin
    proj = _proj(args)
    js = garmin.browser_snippet(proj.path("garmin"), since=args.since, strava_dir=proj.path("strava"))
    out = proj.root / "garmin_snippet.js"
    out.write_text(js, encoding="utf-8")
    start = js.split('const START = "', 1)[1][:10]
    act = js.split('const ACT_START = "', 1)[1][:10]
    copied = _to_clipboard(js)
    print(f"Script for sleep/HRV from {start} and activities from {act} " +
          ("copied to the clipboard" if copied else f"saved in {out}"))
    print("1. Open https://connect.garmin.com in Chrome (signed in).")
    print("2. Press F12, click Console, " + ("paste (Ctrl+V)" if copied else f"paste the contents of {out.name}") +
          " and press Enter. (First time: Chrome asks you to type 'allow pasting'.)")
    print("3. When the file has downloaded:  goodpush garmin-import")


def _to_clipboard(text: str) -> bool:
    import platform
    import subprocess
    cmd = {"Windows": ["clip"], "Darwin": ["pbcopy"]}.get(platform.system())
    if not cmd:
        return False
    try:
        enc = "utf-16" if cmd == ["clip"] else "utf-8"   # clip.exe wants UTF-16 for non-ASCII
        subprocess.run(cmd, input=text.encode(enc), check=True)
        return True
    except Exception:
        return False


def _rider_kg(proj, default: float = 75.0) -> float:
    """Your weight now: [athlete] weight_kg, else the latest from the weight log (daily table), else `default`."""
    from .io import store
    if proj.athlete.get("weight_kg"):
        return float(proj.athlete["weight_kg"])
    daily = store.load(proj.processed, "daily")
    w = daily["weight"].dropna() if daily is not None and "weight" in daily else ()
    return float(w.iloc[-1]) if len(w) else default


def _downloads_dir(proj) -> Path:
    """[paths] downloads in goodpush.toml (when that folder exists: a setting from another computer
    falls through), else the system's Downloads folder (wherever it was moved to; platformdirs)."""
    import os

    import platformdirs
    cfg = proj.cfg.get("paths", {}).get("downloads")
    if cfg and Path(os.path.expandvars(os.path.expanduser(cfg))).is_dir():
        return Path(os.path.expandvars(os.path.expanduser(cfg)))
    return Path(platformdirs.user_downloads_dir())


def cmd_garmin_import(args):
    """Import the newest browser download (Downloads/goodpush_garmin_*.json), then rebuild."""
    from .sync import garmin
    proj = _proj(args)
    if args.file:
        f = Path(args.file)
    else:
        dl = _downloads_dir(proj)
        found = sorted(dl.glob("goodpush_garmin_*.json"), key=lambda p: p.stat().st_mtime)
        if not found:
            sys.exit(f"No goodpush_garmin_*.json in {dl} - run `goodpush garmin-snippet` first, "
                     "pass the file path, or set `downloads = ...` under [paths] in goodpush.toml.")
        f = found[-1]
    print(f"Importing {f}")
    garmin.import_browser(proj.path("garmin"), f)
    args.workers, args.force = getattr(args, "workers", None), False
    cmd_run(args)          # parse the new ride files too, not only re-analyse


def cmd_deauth(args):
    import shutil
    proj = _proj(args)
    if args.service == "strava":
        from .sync import strava
        strava.deauthorize()
        d = proj.path("strava") / "api"
    else:
        from .sync import garmin
        garmin.logout()
        d = proj.path("garmin") / "api"
    if args.purge_data and d.exists():
        shutil.rmtree(d)
        print(f"Deleted downloaded data in {d}")
    elif d.exists():
        print(f"Downloaded data kept in {d} (add --purge-data to delete it).")


def cmd_sync(args):
    proj = _proj(args)
    which = ["strava", "garmin", "google-health"] if args.service == "all" else [args.service]
    for svc in which:
        try:
            if svc == "strava":
                from .sync import secrets_dir, strava
                if args.service == "all" and not (secrets_dir() / "strava.json").exists():
                    continue          # Strava not connected - Garmin covers activities now
                strava.sync(proj.path("strava"), since=args.since)
            elif svc == "google-health":        # Fitbit; connect it once in the app (Settings)
                from .sync.google_health import GoogleHealth
                if GoogleHealth.connected():
                    GoogleHealth().sync(proj.path("google_health"), since=args.since)
                elif args.service != "all":
                    print("google-health: not connected yet - connect it in `goodpush app` > Settings")
            else:
                from .sync import garmin_api
                if garmin_api.has_key():
                    garmin_api.sync(proj.path("garmin"), proj.path("strava"), since=args.since)
                elif args.service == "all":
                    print("garmin: no Personal API key yet - run `goodpush garmin-key`")
                    continue
                else:
                    from .sync import garmin
                    garmin.sync(proj.path("garmin"), since=args.since)
        except (Exception, SystemExit) as e:
            print(f"{svc} sync failed: {e}")
            if args.service != "all":
                raise SystemExit(1)


def cmd_update(args):
    """Pull new Strava + Garmin data, then ingest, analyse and rebuild the reports."""
    args.service = "all"
    cmd_sync(args)
    cmd_run(args)


def cmd_plan(args):
    """Rebuild the adaptive plan (and the dashboard) from the latest data."""
    import pandas as pd
    from .analyze import run_analysis
    from .planning import Planner
    from .report import plan_out, write_reports
    proj = _proj(args)
    try:
        r = run_analysis(proj, progress=lambda *a: None)
    except FileNotFoundError:                 # no rides yet: a plan from nothing, no reports
        r = None
    today = pd.Timestamp(args.date) if args.date else None
    out = Planner(proj).build(r, today=today)
    if r is not None:
        r["plan"] = out
        write_reports(proj, r, progress=lambda *a: None)
    print()
    print(plan_out.markdown(out).split("## Where")[0].strip())
    print(f"\nFull plan: {proj.reports / 'plan' / 'plan.md'}")
    print(f"Watch workouts: {proj.reports / 'plan' / 'workouts'}  (copy to GARMIN/NewFiles on the watch)")


def cmd_app(args):
    """Start the goodpush app (pages + buttons) in your browser."""
    from .app.server import serve
    serve(_proj(args), port=args.port, lan=args.lan, open_browser=not args.no_browser)


def cmd_physio(args):
    """Fit the physiological model (models/physio.py, physio/) and print today's capacities; in a terminal the output
    also goes to a log on this computer (Project.logs, progress.Tee)."""
    import time

    import numpy as np

    from .models.physio import PhysioRun
    from .progress import Progress, Tee
    proj = _proj(args)
    run = PhysioRun(proj, args.since_days)
    kind = ("physio-backtest" if args.backtest else "physio-coverage" if args.coverage else "physio-refresh"
            if args.refresh else "physio-nuts" if args.nuts else "physio")
    with Tee.into(proj.logs / f"{kind}_{time.strftime('%Y-%m-%d_%H%M')}.log"):
        if not run.analysed():                             # a new model version (or no fit yet): its inputs from
            from .analyze import run_analysis              # its own states first, never the old fit's (reset)
            run.same_code()
            print("The analysis isn't made from this model's own daily states yet: analysing first ...")
            run_analysis(proj, progress=Progress.terminal())
        if args.backtest or args.coverage:
            run.check(args.coverage, args.folds, args.every)
            return
        try:
            out = (run.refresh(Progress.terminal()) if args.refresh else
                   run.fit(samples=args.samples, maxiter=args.maxiter, progress=Progress.terminal(), nuts=args.nuts,
                           fresh=args.fresh, target_ess=args.ess, max_minutes=args.max_minutes, warmup=args.warmup))
        except RuntimeError as e:                          # another NUTS run holds the lock
            print(e)
            sys.exit(1)
        except KeyboardInterrupt:
            if args.refresh and (run.processed / run.REFRESH_CHECKPOINT).exists():
                print("Stopped. The refresh is saved up to its last checkpoint: run the same command today to go on.")
                sys.exit(1)
            if args.nuts and (run.processed / run.CHECKPOINT).exists():
                print("Stopped. The NUTS run is saved up to its last checkpoint: run the same command to go on.")
                sys.exit(1)
            raise
        if out is None:                                    # --refresh without stored draws / short of its targets (said why)
            sys.exit(1)
        model, res, draws, st = out
        last = st.iloc[-1]
        rng = {c: f" ({last[c + '_p05']:.0f}-{last[c + '_p95']:.0f})" if c + "_p05" in st else ""
               for c in ("cp", "lt1", "map")}
        print(f"{st.index[-1].date()}: CP {last.cp:.0f} W{rng['cp']}, LT1 {last.lt1:.0f} W{rng['lt1']}, "
              f"MAP (5 min) {last['map']:.0f} W{rng['map']}, CP/MAP {last.f_t:.2f}, W' {last.w_prime / 1000:.1f} kJ, "
              f"Pmax {last.pmax:.0f} W, 1-h {last.ftp:.0f} W, form {last.form:.2f}")
        t = [300, 1200, 3600]
        p = run.predict(model, res.z, args.ahead, t)
        some = draws[run.spaced(len(draws), min(len(draws), run.MAX_TRAJECTORIES))]   # a band needs no more
        band = np.percentile(run.predict(model, some, args.ahead, t), [5, 95], axis=0) if len(draws) else None
        print(f"all-out in {args.ahead} day(s), resting until then: "
              + ", ".join(f"{s // 60} min {w:.0f} W" + (f" ({band[0, i]:.0f}-{band[1, i]:.0f})" if band is not None else "")
                          for i, (s, w) in enumerate(zip(t, p))))


DEVICES_FILE = "ride_devices.json"     # the machine folder (Folders): the trainer / sensors `goodpush ride` uses
SAVED = "Saved "                       # `goodpush ride`'s line naming the ride it saved (the app reads it)


def cmd_ride(args):
    """Ride today's session on the trainer with the always-on-top overlay; afterwards ingest + re-plan."""
    import tempfile
    from functools import partial
    from .app.tomledit import set_values
    from .ride.recorder import RideRecorder
    from .ride.sensors import ROLES, SensorHub
    from .ride.session import FREE_FTP, FREE_STEP_W, RideSession, WorkoutRun
    from .sync import read_secret, write_secret
    proj = _proj(args)
    cfg = proj.cfg["ride"]
    folder = proj.path("rides")
    devices = read_secret(DEVICES_FILE)                   # per machine, never in goodpush.toml
    hub = SensorHub({r: devices.get(r, "") for r in ROLES})
    if args.sim:
        from .ride.sim import SimLink
        link = SimLink(hub)
        hub.choose({"trainer": "sim:fec:1", "hr": "sim:hr:2"})
        folder = Path(tempfile.gettempdir()) / "goodpush_sim"          # never into your data
    else:
        import importlib.util
        from .ride.ant import AntLink
        from .ride.links import RideLinks
        links = {"ant": AntLink(hub, rider_kg=_rider_kg(proj), bike_kg=float(cfg.get("bike_kg", 9)))}
        if importlib.util.find_spec("bleak"):
            from .ride.ble import BleLink
            links["ble"] = BleLink(hub)
        link = RideLinks(hub, links)
    if args.action == "scan":
        print(f"Listening for ANT+ and Bluetooth devices for {args.seconds:.0f} s (pedal, wear the strap) ...")
        try:
            found = link.scan(args.seconds, on_found=lambda d: print(f"  {d.label:32s} {d.key}"))
        except RuntimeError as e:
            sys.exit(str(e))
        print(f"{len(found)} found. `goodpush ride --pick` to choose them." if found else "Nothing heard.")
        return
    for p in RideRecorder.recover(folder):
        print(f"Saved a ride left unsaved (crash?): {p.name}")
    plan = proj.reports / "plan" / "plan.json"
    run = None
    if not args.free and plan.exists():
        res = proj.reports / "results.json"
        ath = (json.loads(res.read_text(encoding="utf-8")).get("athlete") or {}) if res.exists() else {}
        max_hr = float(proj.cfg["athlete"].get("max_hr") or ath.get("max_hr") or 0)    # heart-rate steps (LSCT)
        run = WorkoutRun.from_plan(plan, args.session, resistance=cfg["resistance"], max_hr=max_hr)
        if run is None and args.session:
            sys.exit(f"no session with steps on {args.session} in the plan")
    if run is None:
        if not args.free:
            print("No session to ride today: free ride.")
        run = WorkoutRun("Free ride", [], resistance=cfg["resistance"])
    ftp = (json.loads(plan.read_text(encoding="utf-8")).get("ftp") or 0) if plan.exists() else 0
    if ftp:
        run.free_w = round(FREE_FTP * ftp / FREE_STEP_W) * FREE_STEP_W
    ride = RideSession(link, hub, run, folder)
    save = (lambda values: None) if args.sim else partial(set_values, proj.root / "goodpush.toml", "ride")
    remember = (lambda choice: None) if args.sim else partial(write_secret, DEVICES_FILE)
    from .ride.overlay import RideWindow
    try:
        path, _ = RideWindow(ride, cfg, ftp, save, remember).run(pick=args.pick)
    except RuntimeError as e:
        sys.exit(str(e))
    if path is None:
        print("Nothing saved.")
        return
    print(f"{SAVED}{path}")
    if args.sim or args.save_only:
        return
    from .ingest import add_ride
    from .progress import Progress
    add_ride(proj, run.key, ride.recorder.id, Progress.terminal())
    cmd_analyze(args)


def cmd_summary(args):
    proj = _proj(args)
    p = proj.reports / "results.json"
    if not p.exists():
        sys.exit("no results yet - run `goodpush run`")
    res = json.loads(p.read_text(encoding="utf-8"))
    pw, ld = res["power"], res["load"]
    print(f"FTP now {pw['ftp_now']} W ({pw['wkg_now']} W/kg); career best {pw['ftp_peak']} W "
          f"on {pw['ftp_peak_date']}; type: {pw['phenotype']}")
    print(f"CTL {ld['ctl_now']}  ATL {ld['atl_now']}  TSB {ld['tsb_now']}")


def main(argv=None):
    from .folders import Folders
    Folders.migrate()                            # the folders from before, once (only a goodpush you start moves them)
    ap = argparse.ArgumentParser(prog="goodpush", description=__doc__)
    ap.add_argument("--root", help="project folder (default: search upward for goodpush.toml)")
    ap.add_argument("--athlete", help="whose folder and credentials (`goodpush athlete list`; default: the folder "
                                      "you're in, else this machine's default athlete)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="create goodpush.toml and folders").set_defaults(fn=cmd_init)
    sp = sub.add_parser("athlete", help="the athletes on this machine: list, add NAME FOLDER, move NAME FOLDER, "
                                        "remove NAME")
    sp.add_argument("action", nargs="?", choices=("list", "add", "move", "remove"), default="list")
    sp.add_argument("name", nargs="?")
    sp.add_argument("folder", nargs="?", help="their folder (add: created with a goodpush.toml if new; move: "
                                              "a new or empty one, or where you moved it yourself)")
    sp.add_argument("--default", action="store_true", help="the athlete used when you don't say")
    sp.add_argument("--credentials", action="store_true",
                    help="copy this machine's current sign-ins (Garmin, Strava, ...) to this athlete")
    sp.set_defaults(fn=cmd_athlete)
    sp = sub.add_parser("synth", help="a synthetic rider with a known physiology: new FOLDER (their data), "
                                      "check FOLDER (goodpush's analysis against the truth)")
    sp.add_argument("action", choices=("new", "check"))
    sp.add_argument("folder", help="the synthetic rider's athlete folder (new: must not have a goodpush.toml)")
    sp.add_argument("--profile", help="new: a TOML profile (synth/profile.py RiderProfile; default: all defaults)")
    sp.add_argument("--truth", choices=("model", "simple"), help="new: the physio model run forward, or an "
                                                                 "independent generator (default: the profile's)")
    sp.add_argument("--seed", type=int, help="new: the random seed (default: the profile's)")
    sp.add_argument("--register", metavar="NAME", help="new: add them as an athlete on this machine (`goodpush "
                                                       "--athlete NAME app`)")
    sp.set_defaults(fn=cmd_synth)
    sp = sub.add_parser("peer", help="your folder on another machine: list, join ADDRESS CODE (a goodpush app showing "
                                     "a code), add NAME USER@HOST FOLDER (ssh), sync NAME")
    sp.add_argument("action", nargs="?", choices=("list", "join", "add", "sync"), default="list")
    sp.add_argument("name", nargs="?", help="join: the other machine's ADDRESS; add / sync: the machine's name")
    sp.add_argument("host", nargs="?", help="join: the CODE it shows; add: USER@HOST (or a Host from ~/.ssh/config)")
    sp.add_argument("folder", nargs="?", help="add: the athlete folder there, e.g. ~/goodpush/athletes/NAME")
    sp.add_argument("--dry-run", action="store_true", help="sync: only show what would move")
    sp.add_argument("--send", action="store_true", help="sync: only files from here to there (your decisions still merge)")
    sp.add_argument("--get", action="store_true", help="sync: only files from there to here (your decisions still merge)")
    sp.add_argument("--yes", action="store_true", help="sync: don't ask before going ahead")
    sp.set_defaults(fn=cmd_peer)
    for name, fn, hlp in (("ingest", cmd_ingest, "parse new activity files"),
                          ("run", cmd_run, "ingest + analyze + report")):
        sp = sub.add_parser(name, help=hlp)
        sp.add_argument("--workers", type=int, default=None)
        sp.add_argument("--force", action="store_true", help="re-parse every file")
        sp.set_defaults(fn=fn)
    sub.add_parser("analyze", help="rebuild analysis + reports from cached data") \
        .set_defaults(fn=cmd_analyze)
    sp = sub.add_parser("inspect", help="show available data")
    sp.set_defaults(fn=cmd_inspect)
    sp = sub.add_parser("app", help="open the goodpush app (Today, Plan, Fitness, Recovery)")
    sp.add_argument("--lan", action="store_true", help="also reachable from your phone on the same Wi-Fi")
    sp.add_argument("--port", type=int, default=8765)
    sp.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    sp.set_defaults(fn=cmd_app)
    sp = sub.add_parser("plan", help="adaptive training plan for the next days (+ .ics and watch workouts)")
    sp.add_argument("--date", help="plan as if today were YYYY-MM-DD")
    sp.set_defaults(fn=cmd_plan)
    sp = sub.add_parser("physio", help="fit the physiological model (CP, LT1, MAP, W', Pmax, durability, fatigue)")
    sp.add_argument("--samples", type=int, default=0,
                    help="with --nuts: also draw this many posterior samples a chain (5-95 %% ranges)")
    sp.add_argument("--nuts", action="store_true",
                    help="draw the --samples by NUTS, the only way to ranges (about half an hour; saved as it goes: "
                         "stop with Ctrl-C, run the same command again to go on)")
    sp.add_argument("--warmup", type=int, default=400,
                    help="NUTS: warm-up steps a chain before the draws (step size + mass; default 400)")
    sp.add_argument("--fresh", action="store_true",
                    help="start from the priors, not the last fit (and drop a NUTS run's checkpoint)")
    sp.add_argument("--ess", type=float, default=None,
                    help="NUTS: after --samples draws, more until the smallest effective sample size reaches this "
                         "(and R-hat <= 1.01); goes on from the last run's checkpoint or saved draws")
    sp.add_argument("--max-minutes", type=float, default=None, help="NUTS with --ess: stop drawing after this long")
    sp.add_argument("--refresh", action="store_true",
                    help="today's ranges by a short NUTS run from the last run's draws (minutes; needs the draws "
                         "of a full --nuts run of this model)")
    sp.add_argument("--since-days", type=int, default=None, metavar="DAYS",
                    help="fit on about the last DAYS days only (365 = a year; from the start of that quarter); "
                         "default: Settings > Physiological model ([physio] since_days), 0 = all your data")
    sp.add_argument("--maxiter", type=int, default=15000)
    sp.add_argument("--ahead", type=int, default=2, help="predict all-out power this many days ahead")
    sp.add_argument("--backtest", action="store_true",
                    help="check the forecast: the model fitted up to each of --folds cut-offs, its near-max days of "
                         "the 6 weeks after each scored (the Model page's forecast error; an hour or so)")
    sp.add_argument("--coverage", action="store_true",
                    help="check the ranges: a NUTS run per cut-off, is a 90 %% range right 90 %% of the time (hours; "
                         "resumable)")
    sp.add_argument("--folds", type=int, default=None, help="--backtest / --coverage cut-offs (default 8 / 24)")
    sp.add_argument("--every", type=int, default=None, help="days between cut-offs (default 91 / 30)")
    sp.set_defaults(fn=cmd_physio)
    sp = sub.add_parser("ride", help="ride today's session on the trainer (ANT+ / Bluetooth) with a small always-on-top overlay")
    sp.add_argument("action", nargs="?", choices=["go", "scan"], default="go",
                    help="scan = list the ANT+ and Bluetooth devices around")
    sp.add_argument("--session", help="the plan's session to ride, e.g. 2026-10-27 (default: today's)")
    sp.add_argument("--free", action="store_true", help="free ride: no workout, resistance mode")
    sp.add_argument("--pick", action="store_true", help="choose the devices again")
    sp.add_argument("--sim", action="store_true",
                    help="a pretend trainer + HR strap: try it without hardware (saved to a temp folder only)")
    sp.add_argument("--seconds", type=float, default=10, help="how long `scan` listens")
    sp.add_argument("--save-only", action="store_true",
                    help="save the ride and stop (the app that started it adds it to your rides and re-plans)")
    sp.set_defaults(fn=cmd_ride)
    sub.add_parser("summary", help="print headline numbers").set_defaults(fn=cmd_summary)
    sub.add_parser("illness", help="list sickness notes and illness episodes") \
        .set_defaults(fn=cmd_illness)
    sp = sub.add_parser("import", help="add a downloaded Strava/Garmin export (.zip or folder)")
    sp.add_argument("paths", nargs="+", help="export .zip file(s), unzipped folder(s) or .fit files")
    sp.set_defaults(fn=cmd_import)
    sp = sub.add_parser("auth", help="one-time login to Strava or Garmin Connect")
    sp.add_argument("service", choices=["strava", "garmin"])
    sp.add_argument("--force", action="store_true",
                    help="Garmin: try even if Garmin rate-limited a recent login")
    sp.set_defaults(fn=cmd_auth)
    sp = sub.add_parser("streams", help="ride sample streams in one HDF5 file (status/pack/check/prune/bench)")
    sp.add_argument("action", nargs="?", default="status", choices=["status", "pack", "check", "prune", "bench"])
    sp.add_argument("--compact", action="store_true", help="pack: rewrite the file without dead space")
    sp.set_defaults(fn=cmd_streams)
    sp = sub.add_parser("garmin-key", help="save your Garmin Personal API key (makes updates automatic)")
    sp.add_argument("--test", action="store_true", help="only check the saved key")
    sp.add_argument("--forget", action="store_true", help="remove the saved key from this computer")
    sp.set_defaults(fn=cmd_garmin_key)
    sp = sub.add_parser("incyclist", help="put the plan into Incyclist's workout calendar (status/on/off/sync)")
    sp.add_argument("action", nargs="?", default="status", choices=["status", "on", "off", "sync"])
    sp.add_argument("--folder", help="Incyclist's data folder ('' = the usual place)")
    sp.set_defaults(fn=cmd_incyclist)
    sp = sub.add_parser("garmin-snippet", help="browser script that downloads Garmin data without logging in")
    sp.add_argument("--since", help="YYYY-MM-DD; default = 2 days before your newest data")
    sp.set_defaults(fn=cmd_garmin_snippet)
    sp = sub.add_parser("garmin-import", help="import the newest browser download, then rebuild reports + plan")
    sp.add_argument("file", nargs="?", help="default: newest goodpush_garmin_*.json in Downloads")
    sp.set_defaults(fn=cmd_garmin_import)
    sub.add_parser("doctor", help="check the Garmin/Strava connection setup") \
        .set_defaults(fn=cmd_doctor)
    sp = sub.add_parser("deauth", help="disconnect Strava or Garmin and delete stored tokens")
    sp.add_argument("service", choices=["strava", "garmin"])
    sp.add_argument("--purge-data", action="store_true",
                    help="also delete the data downloaded through the API")
    sp.set_defaults(fn=cmd_deauth)
    sp = sub.add_parser("sync", help="download new data from Strava, Garmin and / or Google Health (Fitbit)")
    sp.add_argument("service", nargs="?", default="all", choices=["all", "strava", "garmin", "google-health"])
    sp.add_argument("--since", help="YYYY-MM-DD; default = day after your newest data")
    sp.set_defaults(fn=cmd_sync)
    sp = sub.add_parser("update", help="sync everything, then ingest + analyze + report")
    sp.add_argument("--since", help="YYYY-MM-DD; default = day after your newest data")
    sp.add_argument("--workers", type=int, default=None)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(fn=cmd_update)
    args = ap.parse_args(argv)
    from .progress import Console
    Console.no_quick_edit()                      # a click in the window must not freeze goodpush
    args.fn(args)


if __name__ == "__main__":
    main()
