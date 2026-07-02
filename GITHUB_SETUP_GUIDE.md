# 📋 Step-by-Step Guide: Add Project to GitHub

## ✅ Pre-Flight Checklist

Your environment is ready:

- ✅ **Git**: Installed (v2.54.0)
- ✅ **Python**: Installed (v3.11.3)
- ✅ **Python Version Note**: Project requires Python ≥3.12, consider upgrading
- ✅ **.gitignore**: Already configured
- ✅ **.env.example**: Already present (for credentials)

---

## 🚀 Setup Instructions

### Step 1: Set Up Git Identity (First Time Only)

```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

**Verify it worked:**

```powershell
git config --global user.name
git config --global user.email
```

---

### Step 2: Initialize Local Git Repository

Navigate to your project folder and initialize:

```powershell
cd c:\Users\dwive\OneDrive\Desktop\nimbus
git init
```

**Verify:**

```powershell
git status
```

You should see: "On branch master" and multiple untracked files.

---

### Step 3: Stage All Files

```powershell
git add .
```

**View staged files:**

```powershell
git status
```

---

### Step 4: Create First Commit

```powershell
git commit -m "Initial commit"
```

**Verify:**

```powershell
git log --oneline
```

---

### Step 5: Create GitHub Repository

1. Go to [github.com](https://github.com) and sign in
2. Click **+** (top right) → **New repository**
3. **Repository name**: `multiagent-automl` (or your preferred name)
4. **Description**: "Multi-agent AutoML pipeline (CSV-first)"
5. **Visibility**: Choose **Public** or **Private**
6. **DO NOT** initialize with README, .gitignore, or license (we already have these)
7. Click **Create repository**

---

### Step 6: Connect Local Repo to GitHub

After creating the repo on GitHub, you'll see instructions. Use these commands:

```powershell
# Add the remote repository
git remote add origin https://github.com/YOUR_USERNAME/multiagent-automl.git

# Rename branch to 'main' (optional but recommended)
git branch -M main

# Push to GitHub
git push -u origin main
```

**Replace `YOUR_USERNAME` with your actual GitHub username!**

---

### Step 7: Verify on GitHub

1. Refresh your GitHub repository page
2. You should see all your files!
3. Check the `.github/` workflows (if any) automatically appear

---

## 🔑 Setting Up Environment Variables Locally

Before running the project:

```powershell
# Copy example file
cp .env.example .env

# Edit .env and add your API keys:
# - GOOGLE_API_KEY from https://aistudio.google.com/apikey
# - GROQ_API_KEY from https://console.groq.com/keys
```

---

## 📦 Install Project Dependencies

```powershell
# Install uv package manager (if not already installed)
pip install uv

# Sync dependencies
uv sync

# Run tests to verify everything works
uv run pytest tests/ -v

# Verify your API keys work
uv run python scripts/verify_providers.py
```

---

## 🚀 Pushing Future Changes

After making changes:

```powershell
# See what changed
git status

# Stage changes
git add .

# Commit with a message
git commit -m "Describe your changes here"

# Push to GitHub
git push
```

---

## 📌 Important Notes

### Never Commit These (Already in .gitignore):

- `.env` - Contains sensitive API keys
- `__pycache__/` - Python cache files
- `.venv/` - Virtual environment
- `data/raw/*` - Raw data files
- `runs/` - Experiment runs
- `TODO.md` - Local notes

### Using SSH (Optional, More Secure)

Instead of HTTPS, you can use SSH:

1. Generate SSH key:

   ```powershell
   ssh-keygen -t ed25519 -C "your.email@example.com"
   ```

2. Add to GitHub:
   - Go to GitHub → Settings → SSH and GPG keys
   - Click **New SSH key**
   - Paste your public key

3. Use SSH remote:
   ```powershell
   git remote remove origin
   git remote add origin git@github.com:YOUR_USERNAME/multiagent-automl.git
   git push -u origin main
   ```

---

## ❌ Troubleshooting

### "fatal: not a git repository"

→ Run `git init` in your project folder first

### "remote: Repository not found"

→ Check your GitHub username and repo name are correct

### "fatal: 'origin' does not appear to be a git repository"

→ Run `git remote add origin https://github.com/YOUR_USERNAME/multiagent-automl.git`

### Authentication fails

→ Use personal access token instead of password:

- GitHub → Settings → Developer settings → Personal access tokens
- Generate new token with `repo` scope
- Use token as password when prompted

---

## ✨ Next Steps After Pushing

1. Add a **License** file (MIT, Apache 2.0, etc.)
2. Set up **branch protection** rules
3. Enable **GitHub Actions** for CI/CD
4. Create **Issues** for your TODO items
5. Add **GitHub Pages** for documentation

---

**Ready to go! Delete this file once you're done.** 🚀
