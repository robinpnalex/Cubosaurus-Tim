# GitHub Automation

This repo uses only free, built-in GitHub Actions — no paid add-ons.

## What's included

| Workflow | File | Trigger | What it does |
|---|---|---|---|
| Regular Tests | `workflows/regular-tests.yml` | Push/PR to `main` | Runs the 20 always-green sanity tests. |
| PR Validation | `workflows/pr-test.yml` | PR opened to `main` | Pulls hidden tests from the `tests` branch and runs the full suite to auto-validate bug fixes. |
| PR Path Labeler | `workflows/pr-labeler.yml` + `pr-labeler.yml` | PR opened/updated | Labels PRs (`rendering`, `encoding`, `scene`, `display`, `camera`, `model`, `engine`, `tests`, `docs`, `dependencies`, `ci`) based on which files changed. |
| Community Labeler | `workflows/community.yml` + `labeler.yml` | Issue/PR opened | Labels issues/PRs (`bug`, `help wanted`, `enhancement`) based on keywords and posts a welcome comment. |
| Dependabot | `dependabot.yml` | Weekly | Opens PRs to bump outdated Python deps and GitHub Action versions. |
| Dependency Review | `workflows/dependency-review.yml` | PR to `main` | Blocks PRs that introduce dependencies with known vulnerabilities. |
| CodeQL | `workflows/codeql.yml` | Push/PR to `main`, weekly | Static security analysis. Results appear under Security → Code scanning. |
| Stale Bot | `workflows/stale.yml` | Daily | Labels issues/PRs `stale` after 60 days, closes after 14 more. |
| Greetings | `workflows/greetings.yml` | First issue/PR from a user | Posts a one-time Cubosaurus-Tim welcome comment. |

## Two-branch testing strategy

- **`main`**: The public challenge branch. `regular-tests.yml` runs only the 20 always-passing tests (`tests/test_regular.py`). The 4 bug-exposing tests are intentionally excluded — failing those is the challenge.
- **`tests`**: A hidden branch containing `tests/test_pr.py` with tests that verify each bug fix. When a PR is opened to `main`, `pr-test.yml` overlays this branch's `tests/` directory and runs everything — auto-reviewing whether the fix is correct.

## One-time setup

1. **Push to GitHub.** All workflows activate as soon as they are on `main` — nothing to install.
2. **Enable code scanning** (for CodeQL): repo → *Settings → Code security* → enable Code scanning. Free on public repos.
3. **Labels**: `actions/labeler` auto-creates labels it needs on first use. No pre-creation required.
4. No secrets needed — everything uses the built-in `secrets.GITHUB_TOKEN`.

## Customizing

- **Path→label mapping**: edit `.github/pr-labeler.yml`.
- **Keyword→label mapping**: edit `.github/labeler.yml`.
- **Stale timing**: edit `.github/workflows/stale.yml`.
- **Welcome messages**: edit `.github/workflows/greetings.yml`.
- **Dependency cadence**: edit `.github/dependabot.yml`.
