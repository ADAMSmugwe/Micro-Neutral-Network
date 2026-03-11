import os
import random
import datetime
import subprocess
import sys

DRY_RUN = "--dry-run" in sys.argv

COMMIT_MESSAGES = [
    "Refactor: update logs",
    "Docs: update documentation",
    "Chore: routine maintenance",
    "Feat: add new log entry",
    "Fix: minor log fix",
    "Style: improve log format",
    "Test: add log test",
    "Build: update log build",
    "CI: update log workflow",
    "Perf: optimize log write"
]

def make_commit(message, timestamp):
    with open("activity_log.txt", "a") as f:
        f.write(f"{timestamp} - {message}\n")
    if not DRY_RUN:
        subprocess.run(["git", "add", "activity_log.txt"])
        subprocess.run(["git", "commit", "-m", message])

def main():
    num_commits = random.randint(45, 55)
    print(f"Generating {num_commits} commits...")
    for i in range(num_commits):
        now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
        msg = random.choice(COMMIT_MESSAGES)
        make_commit(msg, now)
    if not DRY_RUN:
        subprocess.run(["git", "push"])
    else:
        print("Dry run mode: No git push performed.")

if __name__ == "__main__":
    main()
