# Git & GitHub — Complete Reference Guide

## The Big Picture

```
Your Computer (Local)          GitHub (Remote/Cloud)
─────────────────────          ─────────────────────
Working Directory              Repository (origin)
      ↓  git add
Staging Area (Index)
      ↓  git commit
Local Repository    ──── git push ──→  Remote Repo
                    ←─── git pull ───  Remote Repo
```

- **Git**    = tool that runs on YOUR machine, tracks changes locally
- **GitHub** = website that hosts your code in the cloud
- **origin** = nickname for your GitHub repo URL (just a shorthand)
- **main/master** = the default branch name

---

## PART 1 — One-Time Setup (do this once ever)

### Configure your identity
```bash
git config --global user.name "Your Name"
git config --global user.email "you@email.com"
```
Every commit you make will carry this name and email.

### Check your config
```bash
git config --list
```

---

## PART 2 — Starting a New Project (Local → GitHub)

### Step 1: Initialize git in your project folder
```bash
cd path/to/your/project
git init
```
Creates a hidden `.git` folder — this IS your local repository.

### Step 2: Create .gitignore BEFORE staging
```bash
# Create .gitignore file and list files/folders to never track
# Common things to ignore:
# .venv/          → Python virtual environments
# __pycache__/    → compiled Python files
# *.parquet       → large data files
# .env            → secret keys and tokens
# node_modules/   → JS dependencies
```

### Step 3: Stage your files
```bash
git add .                    # stage ALL files (respects .gitignore)
git add filename.py          # stage a specific file only
git add folder/              # stage a specific folder only
```
Staging = telling Git "I want to include THIS in the next commit".

### Step 4: Check what is staged
```bash
git status
```
- **Green** = staged, will be committed
- **Red**   = changed but NOT staged yet
- Always run this before committing to verify no secrets are included.

### Step 5: Commit
```bash
git commit -m "Your message here"
```
A commit = a permanent snapshot of your staged files.
Good message format: `"Add feature X"` / `"Fix bug in Y"` / `"Initial commit: project description"`

### Step 6: Check your commits
```bash
git log --oneline
```
Shows all commits as one line each. Example:
```
9e0a016 Initial commit: LOB price movement prediction
a1b2c3d Add new model architecture
```

### Step 7: Create repo on GitHub
1. Go to github.com → click **"+"** → **"New repository"**
2. Give it a name
3. **Do NOT** check "Add README", "Add .gitignore" — you already have files locally
4. Click **Create repository**
5. Copy the repo URL: `https://github.com/username/repo-name.git`

### Step 8: Connect local repo to GitHub
```bash
git remote add origin https://github.com/username/repo-name.git
```
`origin` = the nickname you give to the GitHub URL. You could call it anything but `origin` is the convention.

### Step 9: Rename branch to main (GitHub default)
```bash
git branch -M main
```
Older Git versions create a branch called `master`. This renames it to `main` to match GitHub's default.

### Step 10: Push to GitHub
```bash
git push -u origin main
```
- `push`     = send your local commits to GitHub
- `-u`       = sets "upstream" — links your local `main` to remote `main`
             After doing this once, future pushes just need `git push`
- `origin`   = which remote to push to
- `main`     = which branch to push

---

## PART 3 — Day-to-Day Workflow (after setup)

```bash
# 1. Make changes to your files

# 2. Check what changed
git status
git diff                     # see exact line-by-line changes

# 3. Stage changes
git add .                    # or specific files

# 4. Commit
git commit -m "Describe what you changed"

# 5. Push to GitHub
git push
```

---

## PART 4 — Pull Before You Push

### Why pull first?
If GitHub has commits you don't have locally (e.g., you pushed from another machine,
or GitHub added a README when you created the repo), pushing will be **rejected**.
You must pull those changes first.

```bash
git pull origin main
```
This = fetch remote changes + merge them into your local branch.

### Pull with unrelated histories
When you init locally AND GitHub created its own first commit (README, license etc.),
Git sees them as two completely separate histories. Normal pull fails. Use:

```bash
git pull origin main --allow-unrelated-histories
```
This forces Git to merge two histories that don't share a common ancestor.

### The safe workflow every time
```bash
git pull origin main          # get latest from GitHub first
# ... make your changes ...
git add .
git commit -m "message"
git push                      # now push is clean, no conflicts
```

---

## PART 5 — When and Why to Use --force

### Normal push (safe, always prefer this)
```bash
git push origin main
```
Only works if your local history is AHEAD of GitHub.
GitHub will reject if remote has commits you don't have.

### Force push (dangerous, use carefully)
```bash
git push origin main --force
```
**What it does:** Overwrites GitHub's history with your local history.
GitHub's version is completely replaced — no questions asked.

### When --force is SAFE to use
- You just created the repo and GitHub only has an auto-generated README
- You rewrote local history (e.g., removed a committed secret) and need to overwrite remote
- You are the only person working on the repo

### When --force is DANGEROUS (never do this)
- Other people have cloned the repo and are working on it
- You are on a shared/team repo — you will destroy their work
- On `main`/`master` branch in a team project

### Safer alternative: --force-with-lease
```bash
git push origin main --force-with-lease
```
Force pushes ONLY if no one else pushed since you last pulled.
Acts as a safety check — fails if remote changed unexpectedly.
**Use this instead of --force when possible.**

---

## PART 6 — Managing Remotes

```bash
git remote -v                                      # see current remote URLs
git remote add origin <url>                        # add a remote named origin
git remote set-url origin <new-url>                # change the remote URL
git remote remove origin                           # remove the remote
git remote rename origin upstream                  # rename a remote
```

### Fix a wrong remote URL
```bash
git remote set-url origin https://github.com/username/correct-repo.git
git remote -v                                      # verify it updated
```

---

## PART 7 — Fixing Mistakes

### Unstage a file (before commit)
```bash
git restore --staged filename.py
```

### Undo last commit but keep the changes
```bash
git reset --soft HEAD~1
```

### Completely wipe local git history and start fresh
```bash
rm -rf .git          # delete entire git history
git init             # start fresh
git add .
git commit -m "Fresh start"
git remote add origin <url>
git push --force
```
Use this when you accidentally committed secrets and need a clean history.

### Remove a file from git tracking (but keep it on disk)
```bash
git rm --cached filename.py       # stop tracking, keep file locally
echo "filename.py" >> .gitignore  # make sure it stays ignored
git commit -m "Remove tracked file"
```

---

## PART 8 — Checking Things

```bash
git status                    # what's staged, unstaged, untracked
git log --oneline             # compact commit history
git log --oneline --graph     # visual branch graph
git diff                      # changes not yet staged
git diff --staged             # changes staged but not committed
git remote -v                 # remote URLs
git branch                    # list branches, current branch marked with *
```

---

## PART 9 — .gitignore Cheatsheet

```gitignore
# Ignore a specific file
secret.json

# Ignore all files with an extension
*.parquet
*.env
*.log

# Ignore a folder
.venv/
node_modules/
__pycache__/

# Ignore folder anywhere in the project (any depth)
**/.venv/
**/__pycache__/

# Ignore everything inside a folder but keep the folder
logs/*
!logs/.gitkeep

# Exception: track this even though the pattern above would ignore it
!important.log
```

---

## PART 10 — This Project's Push Journey (What Happened)

```
Problem 1: git push rejected
  Reason:  GitHub had an auto-created README commit we didn't have locally
  Fix:     git pull origin main --allow-unrelated-histories

Problem 2: API key committed accidentally
  Reason:  .claude-code-router/config.json had a live OpenRouter API key
  Fix:     Added .claude/ and .claude-code-router/ to .gitignore
           Wiped history: rm -rf .git → git init → recommit → force push
  Lesson:  ALWAYS check .gitignore before first commit
           NEVER commit .env, config.json, settings.json with real keys

Problem 3: Wrong remote URL (truncated)
  Reason:  URL was pasted incorrectly, got cut off
  Fix:     git remote set-url origin https://github.com/full/correct-url.git
```

---

## Quick Reference Card

```bash
# FIRST TIME SETUP
git init
git add .
git commit -m "Initial commit"
git remote add origin <github-url>
git branch -M main
git push -u origin main

# DAILY WORKFLOW
git pull                          # always pull first
git add .
git commit -m "what you changed"
git push

# EMERGENCY: committed a secret
rm -rf .git
git init
git add .
git commit -m "Clean history"
git remote add origin <url>
git push --force
# AND: immediately revoke the exposed key/token
```
