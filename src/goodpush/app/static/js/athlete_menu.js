// The menu under the app's name in the top bar (AthleteMenu): who the app shows, switch to another athlete on this
// computer, add one, move one's folder, remove one. The server (AthleteSwitch) changes the folder it serves; the page
// then reloads whole (a move: a job, then the page reloads its data).
import {api, S, toast} from "./api.js";
import {$, esc, LOCAL} from "./util.js";
import {ask} from "./controls.js";

class AthleteMenu {
  open = false;

  // the button's label: the app's name and, with more than one athlete, whose data it shows
  label() {
    const a = S?.athletes, who = a && a.athletes.length > 1 ? a.current || a.name : "";
    $("#brandwho").textContent = who;
    $("#brandwho").hidden = !who;
  }

  html() {
    const a = S.athletes, list = a.athletes.map((x) => `<div class="athrow"><button type="button" class="athitem${x.name === a.current ? " on" : ""}"
      data-athlete="${esc(x.name)}" ${x.name === a.current || x.missing || !LOCAL ? "disabled" : ""} title="${esc(x.folder)}">
      <span class="athcheck">${x.name === a.current ? "✓" : ""}</span><span>${esc(x.name)}</span>
      ${x.missing ? '<span class="small muted">folder missing</span>' : x.name === a.default ? '<span class="small muted">default</span>' : ""}</button>
      ${LOCAL ? `<button type="button" class="athdel" data-folder="${esc(x.name)}" title="${x.missing ? `Where is ${esc(x.name)}'s folder now?` : `Move ${esc(x.name)}'s folder (now ${esc(x.folder)})`}" aria-label="${esc(x.name)}'s folder">📁</button>` : ""}
      ${LOCAL && x.name !== a.current ? `<button type="button" class="athdel" data-remove="${esc(x.name)}" title="Remove ${esc(x.name)} from this computer" aria-label="Remove ${esc(x.name)}">×</button>` : ""}</div>`).join("");
    if (!LOCAL) return `<div class="athhead">Athlete</div>${list}<div class="small muted athnote">Switch or add athletes on the computer running goodpush.</div>`;
    return `<div class="athhead">Athlete</div>
    ${list || `<div class="small muted athnote">Only this folder so far: ${esc(a.name || "unnamed")}.</div>`}
    <div class="athadd" id="athmove" hidden></div>
    <details class="athadd"><summary>+ Add athlete</summary>
      <label class="small">Name <input type="text" id="athname" placeholder="e.g. alex" autocomplete="off" spellcheck="false"></label>
      <label class="small">Folder <input type="text" id="athfolder" placeholder="${esc(this.folderFor("name"))}" spellcheck="false"></label>
      <div class="small muted">A new folder gets its own settings; the athlete signs in to Garmin etc. under Settings.</div>
      <button type="button" class="btn sm primary" id="athaddbtn">Add and switch</button>
    </details>`;
  }

  // where a new athlete's folder goes when you leave Folder empty (next to the default athlete's)
  folderFor(name) {
    const p = S.athletes.parent;
    return p + (p.includes("\\") ? "\\" : "/") + name;
  }

  toggle(show = !this.open) {
    if (!S?.athletes) return;
    const m = $("#athmenu");
    this.open = show;
    if (show) { m.innerHTML = this.html(); this.wireMenu(); }
    m.hidden = !show;
    $("#brandbtn").setAttribute("aria-expanded", String(show));
  }

  wireMenu() {
    document.querySelectorAll("#athmenu [data-athlete]").forEach((b) => b.addEventListener("click", () => this.send({action: "athlete_switch", name: b.dataset.athlete})));
    document.querySelectorAll("#athmenu [data-remove]").forEach((b) => b.addEventListener("click", () => this.remove(b.dataset.remove)));
    document.querySelectorAll("#athmenu [data-folder]").forEach((b) => b.addEventListener("click", () => this.folderForm(b.dataset.folder)));
    $("#athname")?.addEventListener("input", () => { $("#athfolder").placeholder = this.folderFor($("#athname").value.trim().toLowerCase() || "name"); });
    $("#athaddbtn")?.addEventListener("click", () => this.send({action: "athlete_add", name: $("#athname").value, folder: $("#athfolder").value}));
  }

  // give an athlete another folder: moved there (a job: a rename, or on another drive copied, checked, the old one
  // deleted), or, when it's gone from its place, only pointed to where you moved it (athletes.FolderMove)
  folderForm(name) {
    const x = S.athletes.athletes.find((a) => a.name === name), box = $("#athmove");
    box.hidden = false;
    box.innerHTML = `<div class="small"><b>${x.missing ? `Where is ${esc(name)}'s folder now?` : `Move ${esc(name)}'s folder`}</b></div>
      <div class="small muted">${x.missing ? `It isn't at ${esc(x.folder)} any more: give the folder it's in now (the one with their goodpush.toml).`
        : `Now in ${esc(x.folder)}. Pick a new or empty folder; on another drive it's copied and checked, then the old one is deleted.`}</div>
      <label class="small">Folder <input type="text" id="athmovepath" value="${esc(x.missing ? "" : x.folder)}" spellcheck="false"></label>
      <button type="button" class="btn sm primary" id="athmovebtn">${x.missing ? "Use this folder" : "Move"}</button>`;
    $("#athmovebtn").addEventListener("click", () => this.move(name));
    $("#athmovepath").focus();
  }

  async move(name) {
    const folder = $("#athmovepath").value.trim();
    if (folder && await api.action({action: "athlete_move", name, folder}, `Moving ${name}'s folder…`)) this.toggle(false);
  }

  // take an athlete off this computer: their sign-ins here go, their folder stays (the menu redraws, no reload)
  async remove(name) {
    const x = S.athletes.athletes.find((a) => a.name === name);
    if (!await ask.confirm({title: `Remove ${name} from this computer?`, yes: "Remove", danger: true,
      text: `Their sign-ins here (Garmin etc.) are deleted. Their folder stays:\n${x?.folder || ""}`})) return;
    const j = await api.query({action: "athlete_remove", name});
    if (!j) return;
    S.athletes = j.athletes;
    this.label();
    this.toggle(true);
    toast.show(`${name} removed`);
  }

  async send(body) {
    const j = await api.query(body);
    if (!j) return;
    toast.show("Switching…");
    location.reload();                 // every page and cache starts again with the other athlete's data
  }

  wire() {
    $("#brandbtn").addEventListener("click", (e) => { e.stopPropagation(); this.toggle(); });
    $("#athmenu").addEventListener("click", (e) => e.stopPropagation());
    document.addEventListener("click", () => { if (this.open) this.toggle(false); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && this.open) this.toggle(false); });
  }
}
export const athleteMenu = new AthleteMenu();
