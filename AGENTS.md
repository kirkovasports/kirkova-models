# AGENTS.md (kirkova-models)

Instructions for AI coding agents (Claude Code, Codex) and a quick orientation for humans. Claude Code reads this through CLAUDE.md.

<!-- kirkova-kit:begin v1.0.0 (managed by kirkova-agent-kit: do not edit between the markers) -->
## Kirkova rules (managed block: change it in kirkova-agent-kit, never here)

You are working in **one of eight independent repositories** of Kirkova Sports (the `kirkovasports` GitHub org). Kirkova used to be one monorepo; it is not any more. Read the section for this repo below, then follow these rules. They are enforced by Claude Code hooks, git hooks and CI, so working around them only moves the failure: fix the cause instead.

### Non-negotiable
1. **Production is the source of truth, not git.** Never run git on a server. Never overwrite a file on a server without reading its current content first, diffing against it, and taking a `.bak_YYYYMMDD_*` copy. Deploy the zero-downtime way (`/kirkova:deploy`).
2. **Branch, verify, PR.** Never push to `main`, never force-push, never `--no-verify`. Work on `type/short-name`, run `/kirkova:verify`, ship with `/kirkova:pr`. A human merges.
3. **No secrets in git.** No keys, tokens, webhook URLs, passwords, `.env` files. Use placeholders (`<TOKEN>`). Refer to variables by name; print names, never values.
4. **CI runs only on our self-hosted runner** (`runs-on: [self-hosted, kirkova-ci]`). Never a GitHub-hosted runner; never `pull_request_target`.
5. **Repos are independent.** Do not open, import or copy files from a sibling checkout (`../kirkova-*`). Cross repo boundaries only through contracts: the HTTP API (additive changes only; older apps are in the wild), generated docs, published packages. `/kirkova:cross-repo` plans it.
6. **Migrations are append-only; generated files are never hand-edited.** Add a new migration; regenerate docs references and never edit them.
7. **Mobile builds and Apple review are scarce.** Never start an EAS build or change native config without the human's explicit go-ahead.

### How to work
- Start with `/kirkova:start`. Finish with `/kirkova:verify`. Fix red CI with `/kirkova:fix-ci`. Ask "where does X live?" with `/kirkova:map`.
- Do what the task says, no more. A refactor you noticed is a separate PR.
- Every behavior change has a test. Never delete or skip a failing test to get green: find out whether the test or the code is wrong.
- Update the docs when behavior, settings, schedules or deploy steps change (`/kirkova:docs`).
- Comments explain WHY, with the incident or date when there is one. Be honest in commit messages and PRs about what is not done or not verified.
- Unsure about something that touches production, money, user data or another team's area: stop and ask the human.

### Where things are documented
Developer docs: https://docs.kirkova.cc (GitHub sign-in; the Markdown is in the `kirkova-docs` repo). Rules and tools: the `kirkova` Claude Code plugin from `kirkova-agent-kit` (installed automatically once you trust this folder).
<!-- kirkova-kit:end -->

## This repo: kirkova-models

public: model training and evaluation code, papers, accuracy reports, training history.

## This repo is PUBLIC

Everything committed here is visible to the whole internet, forever (git history included). Before every commit ask: would I be fine seeing this on the front page?

- No secrets, keys, tokens, webhook URLs, customer data, internal hostnames or IPs, server paths, or anything about how production is wired.
- No code or numbers from private repos unless the human has explicitly decided to publish them.
- Be honest and precise: this repo is how we show our work. Corrections are welcome and should be visible, not rewritten.
- Issues are public too: do not paste logs, stack traces with internal paths, or user data into them.
