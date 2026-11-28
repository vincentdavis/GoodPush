// Help & feedback (HelpPage): the beta notice, GitHub's issue forms (version and kind of computer filled in), the guide.
import {Page} from "./page.js";
import {S} from "../api.js";
import {esc} from "../util.js";

const REPO = "https://github.com/GoodPush/GoodPush";      // privacy: ok (the project's public home, not a rider)
export const GUIDE = "https://goodpush.github.io/GoodPush/";     // privacy: ok (the project's guide)

class HelpPage extends Page {
  // a GitHub issue form (.github/ISSUE_TEMPLATE/<form>.yml) with what the app knows filled in: its version and the
  // kind of computer, nothing else; GitHub shows it all before anything is posted
  issueUrl(form) {
    const q = new URLSearchParams({template: `${form}.yml`}), v = S.about?.version;
    if (v) q.set("version", v);
    if (form === "bug_report") q.set("computer", this.computer());
    return `${REPO}/issues/new?${q}`;
  }

  computer() {   // the operating system from the browser, and how GoodPush was installed
    const ua = navigator.userAgent;
    const os = /Android/.test(ua) ? "Android" : /iPhone|iPad/.test(ua) ? "iPhone / iPad" : /Windows/.test(ua) ? "Windows"
      : /Mac OS X/.test(ua) ? "macOS" : /Linux/.test(ua) ? "Linux" : "";
    return [os, S.about?.installed ? "installed app" : "from source"].filter(Boolean).join(", ");
  }

  action(form, label, primary, text) {
    return `<div class="helprow"><a class="btn${primary ? " primary" : ""}" href="${esc(this.issueUrl(form))}" target="_blank" rel="noopener">${esc(label)}</a>
      <span class="small">${text}</span></div>`;
  }

  html() {
    const v = S.about?.version;
    return `<h1>Help &amp; feedback</h1>
  <p class="sub">GoodPush${v ? " " + esc(v) : ""} · beta</p>
  <div class="card guide">
    <h3 style="margin-top:0">GoodPush is in beta: expect bugs</h3>
    <p class="small" style="margin:0">Things will break, numbers can be off, and settings may change between versions. Keep your Garmin or Strava account export as your backup. Every report helps.</p>
  </div>
  <h2>Tell us</h2>
  <div class="card helpacts">
    ${this.action("bug_report", "Report a problem", true, "Something broke, an error appeared, or a page doesn't do what it says.")}
    ${this.action("numbers", "Do the numbers make sense?", false, "Know your body? Compare your numbers on <a href=\"#rider\">Rider</a> with a lab or ramp test, a race, or how sessions feel. Matches count too: the model has been tested on very few people.")}
    ${this.action("idea", "Idea or question", false, "Something you'd like it to do, or a sentence that confused you.")}
    <p class="small muted" style="margin:4px 0 0">These open GitHub in your browser (a free account is needed). Only your GoodPush version and kind of computer are filled in, and nothing is posted until you press Submit there. Issues are public: never attach ride files or exports, and check screenshots for your name, maps and health numbers.</p>
    <p class="small muted" style="margin:4px 0 0">Reporting a problem? <a href="#settings/diagnostics">Settings → Diagnostics</a> has <b>Copy for a bug report</b> (versions and counts, no folders or names) to paste into it, and the logs.</p>
  </div>
  <h2>Learn more</h2>
  <div class="card small">
    <ul class="helplinks">
      <li><a href="${GUIDE}" target="_blank" rel="noopener">The guide</a>: install, getting your data in, every page, troubleshooting</li>
      <li><a href="${GUIDE}model/overview.html" target="_blank" rel="noopener">How the model works</a>: the physiology model behind Rider, Model and the plan, and its limits</li>
      <li><a href="${GUIDE}community/contributing.html" target="_blank" rel="noopener">Contributing</a>: testing, feedback, wording and code; most of it needs no programming</li>
      <li><a href="${GUIDE}community/costs.html" target="_blank" rel="noopener">What it costs</a>: GoodPush is free; what would cost money (code signing, app stores)</li>
      <li><a href="${REPO}/blob/main/DISCLAIMER.md" target="_blank" rel="noopener">Disclaimer</a>: not medical advice; training is your decision</li>
      <li><a href="#sources">Sources</a>: where GoodPush's methods come from</li>
    </ul>
  </div>
  <h2>Licence</h2>
  <div class="card small" id="licence">
    <p style="margin:0">GoodPush © 2026 Leon Foks. Free software under the <a href="${REPO}/blob/main/LICENSE" target="_blank" rel="noopener">GNU AGPL v3.0 or later</a>, with no warranty. You can use, change and share it; works built on it must keep the same licence, publish their source and credit GoodPush with a link to <a href="${REPO}" target="_blank" rel="noopener">${REPO}</a> (<a href="${REPO}/blob/main/NOTICE" target="_blank" rel="noopener">NOTICE</a>). Source code: <a href="${REPO}" target="_blank" rel="noopener">GitHub</a>.</p> <!-- privacy: ok (the author's copyright and attribution notice) -->
  </div>`;
  }
}

export const helpPage = new HelpPage();
