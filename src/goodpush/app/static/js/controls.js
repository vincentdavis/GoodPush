// Small controls every page reuses: look here before writing one by hand.
import {esc} from "./util.js";

// a row of buttons with one on (a view or range choice): a click marks it, then onPick(button); multi: each button
// turns on and off by itself (the last one on stays on), onPick(button) after each change
export class Segmented {
  constructor(selector, onPick, multi = false) {
    this.buttons = [...document.querySelectorAll(`${selector} button`)];
    this.multi = multi;
    this.buttons.forEach((b) => b.addEventListener("click", () => { if (this.multi ? this.toggle(b) : this.mark(b)) onPick(b); }));
  }

  mark(b) { this.buttons.forEach((x) => x.classList.toggle("on", x === b)); return true; }

  toggle(b) {                    // false: it was the last one on
    if (b.classList.contains("on") && this.buttons.filter((x) => x.classList.contains("on")).length === 1) return false;
    b.classList.toggle("on");
    return true;
  }
}

// a setting remembered in this browser (localStorage), still working where storage is off; booleans are kept as
// "1" / "0", null removes it
export class Pref {
  constructor(key, fallback) {
    this.key = key;
    let v = null;
    try { v = localStorage.getItem(key); } catch (e) { /* storage off */ }
    this.value = v === null ? fallback : typeof fallback === "boolean" ? v !== "0" : v;
  }

  set(v) {
    this.value = v;
    try {
      if (v === null) localStorage.removeItem(this.key);
      else localStorage.setItem(this.key, typeof v === "boolean" ? (v ? "1" : "0") : v);
    } catch (e) { /* storage off */ }
  }
}

// server data fetched on first use, once (asked again while it's on its way: that one answers); ready() runs when
// it's back, e.g. a redraw of the page showing it. forget() = fetch it again next time (a job changed it)
export class Lazy {
  constructor(url, ready) {
    this.url = url;              // or a function: the address can depend on a choice (the model rung)
    this.ready = ready;
    this.data = null;
    this.loading = false;
  }

  load() {
    if (this.loading || this.data) return;
    this.loading = true;
    fetch(typeof this.url === "function" ? this.url() : this.url).then((r) => r.json())
      .then((j) => { this.data = j; this.loading = false; this.ready(); })
      .catch(() => { this.loading = false; });
  }

  forget() { this.data = null; }
}

// The app's own questions, in place of the browser's confirm() / prompt(): a title, the question in plain words
// and buttons that say what they do. Each one is a <dialog> shown modal (the page behind it can't be clicked, Tab
// stays inside), removed once answered; Esc, Cancel or a click outside it is "no". Keys pressed in it stay in it,
// so a modal behind it (the builder) or a page's shortcuts (the calendar's) don't see them.
export class Ask {
  n = 0;

  // yes or no: true for yes. danger: yes is red and the focus starts on no (deleting, replacing, stopping)
  async confirm({title = "", text = "", yes = "OK", no = "Cancel", danger = false}) {
    return await this.open({title, text, yes, no, danger}) !== null;
  }

  // a line of text: what was typed, or null for no. input: the field's attributes, e.g. {type: "number", min: 1}
  prompt({title = "", text = "", label = "", value = "", yes = "OK", no = "Cancel", input = {}}) {
    return this.open({title, text, yes, no, field: {label, value, input}});
  }

  open({title, text, yes, no, danger = false, field = null}) {
    const id = `ask${++this.n}`, d = document.createElement("dialog");
    const attrs = Object.entries(field?.input || {}).map(([k, v]) => ` ${k}="${esc(v)}"`).join("");
    d.className = "ask";
    d.setAttribute("aria-labelledby", title ? `${id}t` : `${id}b`);
    if (title && text) d.setAttribute("aria-describedby", `${id}b`);
    d.innerHTML = `<form method="dialog" class="ask-box">
      ${title ? `<h3 id="${id}t">${esc(title)}</h3>` : ""}
      ${text ? `<div class="ask-text" id="${id}b">${this.paragraphs(text)}</div>` : ""}
      ${field ? `<label class="f">${esc(field.label)}<input name="answer" value="${esc(field.value)}"${attrs}></label>` : ""}
      <div class="ask-btns"><button type="button" class="btn" data-no>${esc(no)}</button>
        <button class="btn ${danger ? "danger" : "primary"}" data-yes>${esc(yes)}</button></div></form>`;
    const input = d.querySelector("input");
    return new Promise((resolve) => {
      let answer = null, down = null;
      d.querySelector("form").addEventListener("submit", (e) => { e.preventDefault(); answer = input ? input.value : true; d.close(); });
      d.querySelector("[data-no]").addEventListener("click", () => d.close());
      d.addEventListener("pointerdown", (e) => { down = e.target; });
      d.addEventListener("click", (e) => { if (e.target === d && down === d) d.close(); });   // on the backdrop
      d.addEventListener("keydown", (e) => e.stopPropagation());
      d.addEventListener("close", () => { d.remove(); resolve(answer); });
      document.body.appendChild(d);
      d.showModal();
      (input || d.querySelector(danger ? "[data-no]" : "[data-yes]")).focus();
      input?.select();
    });
  }

  // plain text as paragraphs: a blank line starts a new one, a single line break stays a line break
  paragraphs(text) {
    return String(text).split(/\n\s*\n/).map((p) => `<p>${esc(p.trim()).replace(/\n/g, "<br>")}</p>`).join("");
  }
}
export const ask = new Ask();
