import os

def log_group_start(title):
    if os.getenv("GITHUB_ACTIONS"):
        print(f"::group::{title}")
    else:
        print(f"\n--- {title} ---")

def log_group_end():
    if os.getenv("GITHUB_ACTIONS"):
        print("::endgroup::")

def log_notice(msg):
    if os.getenv("GITHUB_ACTIONS"):
        print(f"::notice::{msg}")
    else:
        print(f"NOTICE: {msg}")

def log_error(msg):
    if os.getenv("GITHUB_ACTIONS"):
        print(f"::error::{msg}")
    else:
        print(f"ERROR: {msg}")

def log_group_cb(title):
    if title:
        log_group_start(title)
    else:
        log_group_end()
