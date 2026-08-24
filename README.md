# Green_View_Score

# Git Daily Workflow

This document records the Git commands needed for daily syncing on this project. Come back here whenever you forget.

## Daily Workflow (every time you finish editing and want to upload)

Fixed order, don't skip a step:

```bash
git add .
git commit -m "short description of what changed"
git push
```

- `git add .` — marks all changed files as "ready to commit" (the LF-will-be-replaced-by-CRLF warning is a normal Windows thing, safe to ignore)
- `git commit -m "..."` — packages the marked changes into a "save point", write clearly what you did so you can find it later
- `git push` — actually uploads your local save point to GitHub

## Before You Start Editing (on a new device, or if others are editing too)

Before you start changing code, pull the latest version from GitHub first, to avoid conflicts:

```bash
git pull
```

## Connecting a New Project to an Empty Repo for the First Time

This is the full setup process we used this time, recorded here for reuse when starting new projects:

```bash
git remote add origin your-repo-URL (copy from the green Code button on your GitHub repo page)
git add .
git commit -m "first commit"
git push -u origin master
```

Notes:
- `-u origin master` is only needed on the **very first push** (it links your local branch to the remote branch name). `master` is the name of the local default branch — on some machines the default is `main` instead, so run `git branch` first to check what it's actually called.
- After the first push, daily use just needs plain `git push` — no need for `-u origin master` again.

## Checking Current Status (when unsure)

```bash
git status
```
Tells you which files have changed, which are already committed, and which are committed but not yet pushed.

```bash
git remote -v
```
Confirms which GitHub repo this project is currently connected to.

## GRASS Environment Reminder (not Git-related, but another thing we hit this time)

Every Python script that uses GRASS needs these two lines at the top, otherwise you'll get a `ModuleNotFoundError`:

```python
import sys
sys.path.append(r'C:\ProgramData\anaconda3\envs\myenv_campus\Library\lib\grass85\etc\python')

import grass.script as gs
```
