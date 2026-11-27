// Account exports (Strava / Garmin .zip) dropped anywhere on the app, or chosen in the setup checklist: sent to
// /api/upload (progress in the job chip), then imported by a job (ExportDrop).
import {jobs, toast} from "./api.js";
import {$, esc} from "./util.js";

class ExportDrop {
  dropDepth = 0;

  uploadExport(file) {
    if (!/\.zip$/i.test(file.name)) { toast.show("Drop the .zip from Garmin or Strava, still zipped"); return; }
    const chip = $("#jobchip"), xhr = new XMLHttpRequest();
    chip.hidden = false;
    chip.innerHTML = `<span class="spin"></span>Uploading ${esc(file.name)}…`;
    xhr.open("POST", "/api/upload");
    xhr.setRequestHeader("X-GoodPush", "1");
    xhr.setRequestHeader("X-Filename", file.name);
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) chip.innerHTML = `<span class="spin"></span>Uploading ${esc(file.name)}… ${Math.round(e.loaded / e.total * 100)}%`; };
    xhr.onload = () => {
      let j = {};
      try { j = JSON.parse(xhr.responseText); } catch (err) { /* not JSON */ }
      if (xhr.status >= 400 || j.error) { chip.hidden = true; toast.show(j.error || "Upload failed"); return; }
      toast.show(`${j.kind === "garmin" ? "Garmin" : "Strava"} export received – importing your history (a few minutes)…`);
      jobs.pollJob();
    };
    xhr.onerror = () => { chip.hidden = true; toast.show("Upload failed"); };
    xhr.send(file);
  }

  hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");

  wire() {
    window.addEventListener("dragenter", (e) => { if (this.hasFiles(e)) { this.dropDepth++; document.body.classList.add("dropping"); } });
    window.addEventListener("dragleave", (e) => { if (this.hasFiles(e) && --this.dropDepth <= 0) { this.dropDepth = 0; document.body.classList.remove("dropping"); } });
    window.addEventListener("dragover", (e) => { if (this.hasFiles(e)) e.preventDefault(); });
    window.addEventListener("drop", (e) => {
      if (!e.dataTransfer?.files?.length) return;
      e.preventDefault();
      this.dropDepth = 0; document.body.classList.remove("dropping");
      this.uploadExport(e.dataTransfer.files[0]);
    });
  }
}
export const exportDrop = new ExportDrop();
exportDrop.wire();
