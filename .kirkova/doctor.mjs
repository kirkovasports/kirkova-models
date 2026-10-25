#!/usr/bin/env node
// Kirkova doctor: the harness check that runs everywhere the rules must hold: by hand, in the git hooks, and in CI (the backstop nobody can skip).
// Installed into each repo as .kirkova/doctor.mjs by kirkova-agent-kit's sync; do not edit it by hand (it checks its own fingerprint).
//
//   node .kirkova/doctor.mjs                 full check: structure + every tracked file
//   node .kirkova/doctor.mjs --ci            same, for CI (does not require local git configuration)
//   node .kirkova/doctor.mjs --changed       only files changed against HEAD
//   node .kirkova/doctor.mjs --hook pre-commit | pre-push   (called by .githooks/*)

import { execFileSync, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readFileSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { breakGlass, scanFiles } from "./policy.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const flag = (f) => args.includes(f);
const hookName = args[args.indexOf("--hook") + 1];

const git = (...a) => {
  try {
    return execFileSync("git", a, { cwd: root, encoding: "utf8", timeout: 20000, stdio: ["ignore", "pipe", "ignore"] }).trim();
  } catch {
    return "";
  }
};
const root = (() => {
  try {
    return execFileSync("git", ["rev-parse", "--show-toplevel"], { cwd: process.cwd(), encoding: "utf8" }).trim();
  } catch {
    return dirname(here);
  }
})();
const sha = (p) => createHash("sha256").update(readFileSync(p)).digest("hex").slice(0, 12);

const problems = [];
const notes = [];
const fail = (m) => problems.push(m);

let info = null;
try {
  info = JSON.parse(readFileSync(join(root, ".kirkova", "repo.json"), "utf8"));
} catch {
  fail(".kirkova/repo.json is missing or not valid JSON (run /kirkova:sync-kit)");
}
const repo = info?.name || "";
const ignore = [".kirkova/", ...(info?.scan_ignore || [])];
const scannable = (files) => files.filter((f) => !ignore.some((p) => f.startsWith(p)) && !/(^|\/)(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|bun\.lockb?|poetry\.lock)$/.test(f) && !/\.(png|jpe?g|gif|webp|ico|mp4|webm|mp3|wav|pdf|ttf|otf|woff2?|zip|gz|tgz|pem\.pub)$/i.test(f));

function structure() {
  const need = [".kirkova/repo.json", ".kirkova/policy.mjs", ".kirkova/doctor.mjs", ".kirkova/VERSION", "AGENTS.md", "CLAUDE.md", ".claude/settings.json", ".githooks/pre-commit", ".githooks/pre-push", ".github/pull_request_template.md"];
  for (const f of need) if (!existsSync(join(root, f))) fail(`missing harness file: ${f} (run /kirkova:sync-kit)`);
  if (existsSync(join(root, "AGENTS.md"))) {
    const t = readFileSync(join(root, "AGENTS.md"), "utf8");
    if (!t.includes("<!-- kirkova-kit:begin") || !t.includes("<!-- kirkova-kit:end")) fail("AGENTS.md lost its managed block (kirkova-kit markers); run /kirkova:sync-kit");
  }
  if (existsSync(join(root, "CLAUDE.md")) && !readFileSync(join(root, "CLAUDE.md"), "utf8").includes("@AGENTS.md")) fail("CLAUDE.md must import AGENTS.md (a line with @AGENTS.md) so Claude Code and Codex read the same rules");
  try {
    const s = JSON.parse(readFileSync(join(root, ".claude", "settings.json"), "utf8"));
    if (s.enabledPlugins?.["kirkova@kirkova"] !== true) fail(".claude/settings.json must enable the kirkova plugin (enabledPlugins[\"kirkova@kirkova\"])");
    if (!s.extraKnownMarketplaces?.kirkova) fail(".claude/settings.json must declare the kirkova marketplace (extraKnownMarketplaces.kirkova)");
    const deny = s.permissions?.deny || [];
    for (const r of ["Read(./.env)", "Bash(git push --force:*)", "Bash(git commit --no-verify:*)"]) if (!deny.includes(r)) fail(`.claude/settings.json is missing the deny rule ${r}`);
  } catch {
    if (existsSync(join(root, ".claude", "settings.json"))) fail(".claude/settings.json is not valid JSON");
  }
  try {
    const v = JSON.parse(readFileSync(join(root, ".kirkova", "VERSION"), "utf8"));
    if (existsSync(join(root, ".kirkova/policy.mjs")) && sha(join(root, ".kirkova/policy.mjs")) !== v.policy) fail(".kirkova/policy.mjs was edited by hand (fingerprint differs). Harness files come from kirkova-agent-kit: run /kirkova:sync-kit");
    const self = fileURLToPath(import.meta.url);
    if (sha(self) !== v.doctor) fail(".kirkova/doctor.mjs was edited by hand (fingerprint differs). Run /kirkova:sync-kit");
  } catch {
    if (existsSync(join(root, ".kirkova", "VERSION"))) fail(".kirkova/VERSION is not valid JSON");
  }
  if (!flag("--ci") && !flag("--hook") && git("config", "--get", "core.hooksPath") !== ".githooks") fail("git hooks are not switched on in this clone: run `git config core.hooksPath .githooks` (the Claude Code plugin does it for you at session start)");
}

function contentChecks(files) {
  files = scannable(files);
  for (const f of scanFiles(root, files, repo)) fail(`${f.file}: ${f.message}`);
  for (const f of files) {
    if (/(^|\/)\.env(\.[\w.-]+)?$/.test(f) && !/\.(example|sample|template)$/.test(f)) fail(`${f}: an .env file is tracked. Remove it from git (git rm --cached) and rotate anything it held.`);
    try {
      const size = statSync(join(root, f)).size;
      if (size > 25 * 1024 * 1024) fail(`${f}: ${(size / 1048576).toFixed(0)} MB is too large for git (GitHub rejects 100 MB). Keep big media out of the repo.`);
    } catch { /* deleted */ }
  }
}

function pythonLint(files) {
  const py = files.filter((f) => f.endsWith(".py"));
  if (!py.length || !(info?.verify || []).some((c) => /ruff/.test(c))) return;
  for (const a of [["check", ...py], ["format", "--check", ...py]]) {
    let ran = false;
    for (const cmd of [["ruff"], ["python", "-m", "ruff"], ["python3", "-m", "ruff"]]) {
      const r = spawnSync(cmd[0], [...cmd.slice(1), ...a], { cwd: root, encoding: "utf8", timeout: 90000 });
      if (r.error) continue;
      ran = true;
      if (r.status !== 0) fail(`ruff ${a[0]} failed:\n${(r.stdout || r.stderr).split("\n").slice(0, 25).join("\n")}\n(fix with: ruff check --fix ${py.length > 3 ? "." : py.join(" ")} && ruff format ${py.length > 3 ? "." : py.join(" ")})`);
      break;
    }
    if (!ran) notes.push("ruff is not installed here, so Python lint was skipped (pip install -e \".[dev]\"). CI will run it.");
  }
}

function prePush() {
  const lines = readFileSync(0, "utf8").split("\n").filter(Boolean);
  const zero = /^0+$/;
  for (const l of lines) {
    const [localRef, localSha, remoteRef, remoteSha] = l.split(" ");
    if (/^refs\/heads\/(main|master)$/.test(remoteRef) && !breakGlass()) fail(`push to ${remoteRef} is blocked: work on a branch and open a pull request (/kirkova:pr)`);
    if (!zero.test(remoteSha) && !zero.test(localSha)) {
      const ok = spawnSync("git", ["merge-base", "--is-ancestor", remoteSha, localSha], { cwd: root }).status === 0;
      if (!ok && !breakGlass()) fail(`push of ${localRef} would rewrite history on ${remoteRef} (not a fast-forward): force-pushes are blocked`);
    }
  }
}

if (flag("--hook")) {
  if (hookName === "pre-commit") {
    const staged = git("diff", "--cached", "--name-only", "--diff-filter=ACMR").split("\n").filter(Boolean);
    contentChecks(staged);
    pythonLint(staged);
  } else if (hookName === "pre-push") {
    prePush();
  }
} else if (flag("--changed")) {
  const changed = [...git("diff", "--name-only", "HEAD").split("\n"), ...git("ls-files", "--others", "--exclude-standard").split("\n")].filter(Boolean);
  contentChecks(changed);
  pythonLint(changed);
} else {
  structure();
  contentChecks(git("ls-files").split("\n").filter(Boolean));
}

for (const n of notes) console.log(`note: ${n}`);
if (problems.length) {
  console.error(`\nKirkova doctor: ${problems.length} problem(s)\n`);
  for (const p of problems) console.error(`  x ${p}`);
  console.error("\nThese checks also run in CI, so a bypass only moves the failure. Fix the cause (docs: Agent toolkit).");
  process.exit(1);
}
console.log(`Kirkova doctor: ok (${repo || "no repo name"}${flag("--hook") ? `, ${hookName}` : ""})`);
