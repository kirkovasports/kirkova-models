// The Kirkova policy engine: ONE set of rules, used by the Claude Code hooks, the git hooks and the CI check, so the three can never disagree.
// No dependencies (Node 18+). Pure functions plus a few small helpers; nothing here talks to the network.
//
// A rule says: when this command / this file write happens, DENY it (or ASK the human first) and tell the agent what to do instead.
// Break-glass: a human may export KIRKOVA_BREAK_GLASS=1 in their own shell BEFORE starting Claude Code; deny rules then become asks (never allows),
// and secrets are still refused. An agent cannot set it: the hook reads the environment of the Claude Code process, not the commands it runs.

import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { basename, join, relative, resolve, sep } from "node:path";

export const POLICY_VERSION = "1";
export const SIBLINGS = ["platform", "mobile", "infra", "marketing", "docs", "models", "api-docs", "agent-kit"];

// ── rule catalogue (also rendered into the docs: keep `summary` and `fix` human) ─────────────────────────────────────────────────────────────────
export const RULES = {
  "no-git-on-servers": { level: "deny", scope: "bash", summary: "No git commands on a server (over ssh).", fix: "Production is the source of truth, not a git checkout. Read the real file (`ssh host cat path`), diff your change against it, then use /kirkova:deploy." },
  "no-force-push": { level: "deny", scope: "bash", summary: "No force-push (--force, -f, --force-with-lease, +refspec).", fix: "Push a new commit instead. If history really must change, ask a human to do it." },
  "no-push-to-main": { level: "deny", scope: "bash", summary: "No pushing straight to main/master.", fix: "Work on a branch and open a pull request with /kirkova:pr. CI runs on the PR." },
  "no-verify": { level: "deny", scope: "bash", summary: "No bypassing git hooks (--no-verify, core.hooksPath, HUSKY=0).", fix: "Fix what the hook reports. The same checks run in CI, so skipping them only moves the failure." },
  "no-remote-destruction": { level: "deny", scope: "bash", summary: "No docker prune/rm/down/volume removal over ssh.", fix: "Containers and volumes on servers are removed only by a human, with a backup. Use the zero-downtime swap scripts for replacements." },
  "ask-remote-write": { level: "ask", scope: "bash", summary: "Uploading to a server (scp/rsync) needs a human yes.", fix: "Before uploading: read the file currently on the server, diff against it, and take a .bak copy (see /kirkova:deploy)." },
  "ask-remote-compose": { level: "ask", scope: "bash", summary: "`docker compose up/restart` over ssh needs a human yes.", fix: "Recreating a service can drop traffic: use the swap scripts for api/web/worker, never `up -d` on caddy, cloudflared or postgres." },
  "ask-env-files": { level: "ask", scope: "bash", summary: "Reading .env files (locally or on a server) needs a human yes.", fix: "Print variable NAMES only (`grep -o '^[A-Z_]*=' .env`), never values." },
  "ask-destructive-git": { level: "ask", scope: "bash", summary: "Commands that throw away uncommitted work need a human yes (reset --hard, checkout ., restore ., clean -f, stash drop, branch -D).", fix: "Uncommitted work has been lost this way before. Commit or stash first." },
  "ask-pipe-to-shell": { level: "ask", scope: "bash", summary: "curl | sh needs a human yes.", fix: "Download, read, then run." },
  "no-secrets-in-files": { level: "deny", scope: "write", summary: "No secrets (keys, tokens, webhook URLs, passwords) written into files.", fix: "Secrets live in the server's .env or a password manager. Use a placeholder like <TOKEN> in code and docs." },
  "no-env-files": { level: "deny", scope: "write", summary: "No creating or editing .env files (except .env.example).", fix: "Document variable names in settings docs; put real values on the server only." },
  "no-edit-generated": { level: "deny", scope: "write", summary: "No hand edits to generated files (docs reference pages, native iOS/Android folders).", fix: "Change the source and regenerate (docs: tools/generate_reference.py; mobile: app config and config plugins, never ios/ or android/)." },
  "migrations-append-only": { level: "deny", scope: "write", summary: "Existing database migrations are never edited; add a new one.", fix: "Run /kirkova:new-migration. Production has already applied the old revision, so editing it changes nothing there and breaks fresh databases." },
  "workflows-self-hosted": { level: "deny", scope: "write", summary: "GitHub Actions run only on our self-hosted runner (runs-on: [self-hosted, kirkova-ci]); no pull_request_target.", fix: "Use `runs-on: [self-hosted, kirkova-ci]`. Hosted runners are never used, and pull_request_target would run untrusted code with secrets." },
  "no-cross-repo-reach": { level: "deny", scope: "write", summary: "Repos are independent: no paths or dependencies that reach into a sibling checkout.", fix: "Share through the API contract (OpenAPI), a published package, or copy the value and add a parity test that skips when the other repo is absent." },
  "ask-runtime-config": { level: "ask", scope: "write", summary: "Editing dependency, Docker or proxy files needs a human yes (they mirror production).", fix: "pyproject/package.json/Dockerfile/docker-compose.yml/Caddyfile changes need a deploy review. Dependency upgrades are deliberately deferred." },
  "ask-harness-files": { level: "ask", scope: "write", summary: "Editing the harness itself (.kirkova/, .githooks/, .claude/settings.json) needs a human yes.", fix: "Harness changes go through the kirkova-agent-kit repo and /kirkova:sync-kit, so every repo gets the same rules." },
};

// ── small helpers ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
export const breakGlass = () => process.env.KIRKOVA_BREAK_GLASS === "1";

export function repoInfo(projectDir) {
  try {
    return JSON.parse(readFileSync(join(projectDir, ".kirkova", "repo.json"), "utf8"));
  } catch {
    return null;
  }
}

export function currentBranch(cwd) {
  try {
    return execFileSync("git", ["rev-parse", "--abbrev-ref", "HEAD"], { cwd, encoding: "utf8", timeout: 4000, stdio: ["ignore", "pipe", "ignore"] }).trim();
  } catch {
    return "";
  }
}

/** Remove heredoc bodies so text inside a commit message or a script is not mistaken for commands. */
export function stripHeredocs(cmd) {
  return cmd.replace(/<<-?\s*(['"]?)([A-Za-z_][\w-]*)\1[^\n]*\n[\s\S]*?\n[ \t]*\2[ \t]*(?=\n|$)/g, "<<heredoc");
}

/** Split a shell command line into segments (on && || ; | newline outside quotes) of tokens (quotes removed). Good enough for policy checks. */
export function parseCommand(raw) {
  const cmd = stripHeredocs(raw);
  const segments = [];
  let tokens = [], cur = "", q = null, started = false;
  const pushTok = () => { if (started) tokens.push(cur); cur = ""; started = false; };
  const pushSeg = () => { pushTok(); if (tokens.length) segments.push(tokens); tokens = []; };
  for (let i = 0; i < cmd.length; i++) {
    const c = cmd[i];
    if (q) {
      if (c === q) q = null;
      else if (c === "\\" && q === '"' && i + 1 < cmd.length) { cur += cmd[++i]; }
      else cur += c;
      continue;
    }
    if (c === "'" || c === '"') { q = c; started = true; continue; }
    if (c === "\\" && i + 1 < cmd.length) { cur += cmd[++i]; started = true; continue; }
    if (c === "&" && cmd[i + 1] === "&") { pushSeg(); i++; continue; }
    if (c === "|") { pushSeg(); if (cmd[i + 1] === "|") i++; continue; }
    if (c === ";" || c === "\n") { pushSeg(); continue; }
    if (c === " " || c === "\t") { pushTok(); continue; }
    cur += c; started = true;
  }
  pushSeg();
  return segments;
}

const WRAPPERS = new Set(["sudo", "time", "command", "nohup", "env", "exec", "nice"]);
function head(tokens) {
  let i = 0;
  while (i < tokens.length && (/^[A-Za-z_][A-Za-z0-9_]*=/.test(tokens[i]) || WRAPPERS.has(tokens[i]))) i++;
  return { name: tokens[i] || "", args: tokens.slice(i + 1), rest: tokens.slice(i) };
}

function gitSub(args) {
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === "-C" || a === "-c" || a === "--git-dir" || a === "--work-tree" || a === "--namespace") { i++; continue; }
    if (a.startsWith("-")) continue;
    return { sub: a, flags: args.slice(i + 1) };
  }
  return { sub: "", flags: [] };
}

const REMOTE_GIT = /(^|[\s;&|"'`(])git\s+(?:-C\s+\S+\s+|-c\s+\S+\s+)?(pull|fetch|checkout|switch|reset|clone|status|log|diff|show|stash|remote|branch|merge|rebase|restore|add|commit|push|rev-parse|ls-files|worktree|cherry-pick|apply|blame|tag)\b/;
const REMOTE_DESTROY = /\bdocker\s+(?:system\s+prune|volume\s+(?:prune|rm)|image\s+prune|container\s+prune|builder\s+prune|network\s+prune|rm\b|rmi\b|kill\b)|\bdocker\s+compose\b[^|;&]*\b(down|rm|kill)\b|\bdocker-compose\b[^|;&]*\b(down|rm|kill)\b/;
const REMOTE_COMPOSE_UP = /\bdocker\s+compose\b[^|;&]*\b(up|restart|stop|start|recreate)\b|\bdocker-compose\b[^|;&]*\b(up|restart)\b/;
const REMOTE_HOST = /\b[\w.-]+@[\w.-]+\.[a-z]{2,}|\b[\w-]+@\d{1,3}(\.\d{1,3}){3}|\bks6b\b|\bid_ks6b\b|\bid_mailvps\b|\bid_ci_runner\b/;

// ── bash evaluation ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
/** @returns {{rule: string, level: 'deny'|'ask', message: string}[]} */
export function evaluateBash(command, ctx = {}) {
  const hits = [];
  const add = (rule, extra = "") => hits.push({ rule, level: RULES[rule].level, message: `${RULES[rule].summary}${extra ? ` (${extra})` : ""} ${RULES[rule].fix}` });
  const segments = parseCommand(command);
  const branch = ctx.branch ?? "";
  for (const seg of segments) {
    const { name, args, rest } = head(seg);
    const line = rest.join(" ");
    const isSsh = name === "ssh";
    if (isSsh && REMOTE_GIT.test(line)) add("no-git-on-servers");
    if (isSsh && REMOTE_DESTROY.test(line)) add("no-remote-destruction");
    if (isSsh && REMOTE_COMPOSE_UP.test(line)) add("ask-remote-compose");
    if (name === "docker" && /^(system|volume|image|builder)$/.test(args[0] || "") && args[1] === "prune" && /\s-a\b|--all|--volumes/.test(line) && !ctx.remoteOk) add("ask-destructive-git", "docker prune -a");
    if ((name === "scp" || name === "rsync") && rest.some((t) => /^[\w.-]+@[\w.-]+:/.test(t) || /^[\w.-]+:[~/]/.test(t))) {
      const last = rest[rest.length - 1] || "";
      if (/@[\w.-]+:|^[\w.-]+:[~/]/.test(last)) add("ask-remote-write");
    }
    if (["cat", "less", "more", "head", "tail", "grep", "sed", "awk", "cp", "type", "get-content", "bat"].includes(name.toLowerCase()) || isSsh) {
      if (rest.some((t) => /(^|[\/\\])\.env(\.[\w.-]+)?$/.test(t) && !/\.(example|sample|template)$/.test(t)) || (isSsh && /(^|[\s\/'"])\.env(\s|$|['"])/.test(line))) {
        if (!/grep\s+-[a-z]*[oc]/.test(line) || !/\|\s*(sed|cut)/.test(command)) add("ask-env-files");
      }
    }
    if (name === "git") {
      const { sub, flags } = gitSub(args);
      if (sub === "push") {
        const force = flags.some((f) => f === "--force" || f === "-f" || f.startsWith("--force-with-lease") || /^\+\S/.test(f) || (/^-[a-z]*f[a-z]*$/.test(f) && !f.startsWith("--")));
        if (force) add("no-force-push");
        const refs = flags.filter((f) => !f.startsWith("-"));
        const target = refs.length >= 2 ? refs[refs.length - 1] : "";
        const pushesMain = /(^|:)(main|master)$/.test(target) || (refs.length <= 1 && /^(main|master)$/.test(branch)) || (refs.length >= 1 && /^(main|master)$/.test(refs[refs.length - 1]));
        if (pushesMain) add("no-push-to-main");
      }
      if (sub === "commit" && flags.some((f) => f === "--no-verify" || (/^-[a-zA-Z]*n[a-zA-Z]*$/.test(f) && !f.startsWith("--")))) add("no-verify");
      if (sub === "push" && flags.includes("--no-verify")) add("no-verify");
      if (sub === "config" && flags.some((f) => /core\.hookspath/i.test(f))) add("no-verify");
      if (sub === "reset" && flags.includes("--hard")) add("ask-destructive-git", "git reset --hard");
      if ((sub === "checkout" || sub === "restore") && flags.some((f) => f === "." || f === "--" || f === ":/")) add("ask-destructive-git", `git ${sub} .`);
      if (sub === "clean" && flags.some((f) => /^-[a-z]*f/.test(f))) add("ask-destructive-git", "git clean -f");
      if (sub === "stash" && ["drop", "clear"].includes(flags[0])) add("ask-destructive-git", "git stash drop");
      if (sub === "branch" && flags.includes("-D")) add("ask-destructive-git", "git branch -D");
    }
    if (/^(HUSKY|SKIP|PRE_COMMIT_ALLOW_NO_CONFIG)=/.test(seg[0] || "") && /HUSKY=0|SKIP=/.test(seg[0] || "")) add("no-verify");
  }
  if (/\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba)?sh\b/.test(stripHeredocs(command))) add("ask-pipe-to-shell");
  return dedupe(hits);
}

// ── secrets ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
const PLACEHOLDER = /(<[^>]+>|\$\{?[A-Z_]+\}?|xxxx+|\*{4,}|changeme|example|placeholder|your[-_ ]|\.\.\.|redacted|dummy|fake|test[-_]?key)/i;
export const SECRET_PATTERNS = [
  ["private key block", /-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----/],
  ["GitHub token", /\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{40,}\b/],
  ["Stripe key", /\b(?:sk|rk)_live_[A-Za-z0-9]{16,}\b|\bwhsec_[A-Za-z0-9]{20,}\b/],
  ["OpenRouter / OpenAI-style key", /\bsk-(?:or-v1-)?[A-Za-z0-9_-]{32,}\b/],
  ["Slack token", /\bxox[abprs]-[A-Za-z0-9-]{20,}\b/],
  ["AWS access key", /\bAKIA[0-9A-Z]{16}\b/],
  ["Discord webhook URL", /discord(?:app)?\.com\/api\/webhooks\/\d{15,}\/[\w-]{40,}/],
  ["Discord bot token", /\b[MNO][A-Za-z\d_-]{23,25}\.[\w-]{6}\.[\w-]{27,}\b/],
  ["Google API key", /\bAIza[0-9A-Za-z_-]{35}\b/],
  ["JWT", /\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/],
  ["URL with a password", /\b(?:postgres(?:ql)?|mysql|redis|amqp|mongodb(?:\+srv)?):\/\/[^\s:@\/]+:[^\s@\/]{6,}@/],
];
/** Lines like KEY=longvalue where the name says it is secret. */
const ASSIGN = /\b([A-Z][A-Z0-9_]*(?:SECRET|TOKEN|PASSWORD|PASSWD|API_?KEY|PRIVATE_?KEY|WEBHOOK)[A-Z0-9_]*)\s*[=:]\s*['"]?([^\s'"#]{16,})/g;

export function findSecrets(text) {
  const found = [];
  if (!text) return found;
  for (const [label, re] of SECRET_PATTERNS) {
    const m = text.match(re);
    if (m && !PLACEHOLDER.test(m[0])) found.push(label);
  }
  let m;
  ASSIGN.lastIndex = 0;
  while ((m = ASSIGN.exec(text))) {
    const v = m[2];
    if (!PLACEHOLDER.test(v) && /[A-Za-z]/.test(v) && /\d/.test(v) && !/^(true|false|none|null)$/i.test(v) && !/^[a-z_.\/-]+$/.test(v)) found.push(`${m[1]} assignment`);
  }
  return [...new Set(found)];
}

// ── write evaluation ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
const norm = (p) => p.split(sep).join("/");
const FORBIDDEN_RUNS_ON = /runs-on:\s*(?:\[\s*)?['"]?(?:ubuntu|macos|windows)[\w.-]*/i;
const SIBLING_REACH = new RegExp(String.raw`(?:\.\.[\\/])+kirkova-(?:${SIBLINGS.join("|")})\b|file:\.\.[\\/]kirkova-`);

/**
 * @param {{path: string, text?: string, oldText?: string, exists?: boolean}} w  path relative to the project root (or absolute); `text` is the new content or the new_string
 * @param {{repo?: string, projectDir?: string}} ctx
 */
export function evaluateWrite(w, ctx = {}) {
  const hits = [];
  const add = (rule, extra = "") => hits.push({ rule, level: RULES[rule].level, message: `${RULES[rule].summary}${extra ? ` (${extra})` : ""} ${RULES[rule].fix}` });
  const abs = resolve(ctx.projectDir || process.cwd(), w.path);
  const rel = norm(relative(ctx.projectDir || process.cwd(), abs));
  const base = basename(abs);
  const text = w.text ?? "";
  const repo = ctx.repo || "";

  const secrets = findSecrets(text);
  if (secrets.length) add("no-secrets-in-files", secrets.join(", "));
  if (/^\.env(\..+)?$/.test(base) && !/\.(example|sample|template)$/.test(base)) add("no-env-files");

  if (repo === "docs" && /^docs\/reference\//.test(rel)) add("no-edit-generated", rel);
  if (repo === "mobile" && /^(ios|android)\//.test(rel)) add("no-edit-generated", rel);
  if (repo === "platform" && /^alembic\/versions\/[^/]+\.py$/.test(rel) && (w.exists ?? existsSync(abs))) add("migrations-append-only", rel);

  if (/^\.github\/workflows\/[^/]+\.ya?ml$/.test(rel) && (FORBIDDEN_RUNS_ON.test(text) || /\bpull_request_target\b/.test(text))) add("workflows-self-hosted", rel);

  if (!/^(docs\/|README|.*\.md$)/i.test(rel) || /\.(json|ya?ml|toml|ts|tsx|js|mjs|py)$/.test(rel)) {
    if (SIBLING_REACH.test(text)) add("no-cross-repo-reach", rel);
    if (repo === "platform" && /['"`](?:\.\.\/)*mobile\/src\//.test(text) && !/skipif|isfile\(\)|exists\(\)|KIRKOVA_MOBILE_DIR/.test(text)) add("no-cross-repo-reach", "platform code must not read mobile/src");
  }

  if (/^(pyproject\.toml|package\.json|Dockerfile|docker-compose\.ya?ml|Caddyfile|requirements[\w.-]*\.txt|(web|admin|studio|renderer)\/(package\.json|Dockerfile))$/.test(rel)) add("ask-runtime-config", rel);
  if (/^(\.kirkova\/|\.githooks\/|\.claude\/settings\.json$)/.test(rel)) add("ask-harness-files", rel);
  return dedupe(hits);
}

function dedupe(hits) {
  const seen = new Set();
  return hits.filter((h) => (seen.has(h.rule) ? false : seen.add(h.rule)));
}

/** Collapse hits into one decision: deny beats ask. Break-glass turns deny into ask except for secrets. */
export function decide(hits) {
  if (!hits.length) return null;
  const bg = breakGlass();
  const effective = hits.map((h) => (bg && h.level === "deny" && h.rule !== "no-secrets-in-files" ? { ...h, level: "ask", message: `${h.message} [break-glass is on: a human must confirm]` } : h));
  const deny = effective.filter((h) => h.level === "deny");
  const chosen = deny.length ? deny : effective;
  return {
    level: deny.length ? "deny" : "ask",
    reason: `Kirkova policy (${chosen.map((h) => h.rule).join(", ")}):\n` + chosen.map((h) => `- ${h.message}`).join("\n"),
    rules: chosen.map((h) => h.rule),
  };
}

// ── repo scans (used by the git hooks, the stop gate and CI) ──────────────────────────────────────────────────────────────────────────────────────
/** Scan files (project-relative paths) with the write rules. Returns findings that are denials (asks are ignored: they are for interactive use). */
export function scanFiles(projectDir, files, repo) {
  const out = [];
  for (const f of files) {
    const abs = join(projectDir, f);
    let text = "";
    try {
      const buf = readFileSync(abs);
      if (buf.length > 2_000_000 || buf.includes(0)) continue;
      text = buf.toString("utf8");
    } catch {
      continue;
    }
    // generated-file and migration rules are about EDITS, not about the file existing: skip them in scans
    const hits = evaluateWrite({ path: f, text, exists: false }, { repo, projectDir }).filter((h) => h.level === "deny" && !["no-edit-generated", "migrations-append-only"].includes(h.rule));
    for (const h of hits) out.push({ file: f, rule: h.rule, message: h.message });
  }
  return out;
}
