<<<<<<< HEAD
# Daily Commit Automation

This project automates daily GitHub commits to keep your contribution graph "Dark Green".
## Setup Guide

### 1. Clone the Repo
```
git clone https://github.com/ADAMSmugwe/daily-commit.git
cd daily-commit
### 2. Initialize Local Git (if not already initialized)
```
git remote add origin https://github.com/ADAMSmugwe/daily-commit.git

- Use `git config --global credential.helper cache` to cache credentials.
- For HTTPS, use a Personal Access Token (PAT) as your password.
To test locally without pushing:
```
python mass_commit.py --dry-run
- The workflow `.github/workflows/daily_green.yml` triggers the script daily and allows manual runs.
- All commits modify `activity_log.txt`.
```

git push -u origin main
```


### 5. Dry Run Mode

# Daily Commit Automation


### 1. Clone the Repo

```

cd daily-commit
```

### 2. Initialize Local Git (if not already initialized)

```
```

### 3. Link to Remote GitHub Repo
git branch -M main
git push -u origin main
### 4. Git Credentials

- Use `git config --global credential.helper cache` to cache credentials.
- For HTTPS, use a Personal Access Token (PAT) as your password.
- For SSH, set up your SSH keys and add them to GitHub.

### 5. Dry Run Mode

To test locally without pushing:

```
python mass_commit.py --dry-run
```

### 6. Manual Run

To run the script and push commits:

```
python mass_commit.py
```

### 7. GitHub Actions (Automatic Daily Push)

- The workflow `.github/workflows/daily_green.yml` triggers the script daily and allows manual runs.
- Ensure your repo has `activity_log.txt` and `mass_commit.py` in the root.
- GitHub Actions will automatically push commits every day at 02:00 UTC.

## Notes

- The script generates 45–55 commits daily, each with a unique timestamp and random message.
- All commits modify `activity_log.txt`.
- Dry run mode prevents actual git push for safe testing.

## Troubleshooting

- If you see authentication errors, check your Git credentials.
- If the workflow fails, check the Actions tab for logs.
```

### 6. Manual Run

To run the script and push commits:

```
python mass_commit.py
```

### 7. GitHub Actions

- The workflow `.github/workflows/daily_green.yml` triggers the script daily and allows manual runs.
- Ensure your repo has `activity_log.txt` and `mass_commit.py` in the root.

---

## Notes

- The script generates 45–55 commits daily, each with a unique timestamp and random message.
- All commits modify `activity_log.txt`.
- Dry run mode prevents actual git push for safe testing.

---

## Troubleshooting

- If you see authentication errors, check your Git credentials.
- If the workflow fails, check the Actions tab for logs.
=======
# daily-commit
>>>>>>> origin/main
