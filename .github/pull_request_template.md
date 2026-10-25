## What and why
<!-- What changed, and the reason (the incident, the decision, the ticket). One short paragraph. -->

## How I tested it
<!-- Commands and results, e.g. `pytest tests/unit` 1798 passed; what I clicked through. -->

## Risk and rollback
<!-- What could break, who is affected (live games? the app?), and how to undo it. -->

## Checklist
- [ ] CI is green (it runs on our self-hosted runner)
- [ ] API changes are additive (older app versions keep working)
- [ ] Docs updated, or not needed: <!-- /kirkova:docs -->
- [ ] Needs a production deploy? <!-- yes: use /kirkova:deploy after merge; no -->
- [ ] Needs a migration or a new setting? <!-- named here, applied deliberately after a backup -->
- [ ] Other repos involved? <!-- link their PRs; backend first -->
