"""Generate training pairs for bol-cleanup: (noisy dictation -> clean prompt).

Two corruption channels:
  - rule-based "spoken-izer": fillers, stutters, spoken punctuation/tokens,
    homophone swaps, dropped capitalization
  - optional --roundtrip: speak each seed with macOS `say` and transcribe it
    with Parakeet, harvesting the STT engine's REAL error patterns

Plus identity pairs (clean -> clean) so the model learns to leave clean
text alone. Output: mlx-lm chat-format JSONL (train/valid/test).

Usage:
  uv run python training/generate_data.py --out training/data [--roundtrip]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

SYSTEM = "Clean this voice dictation for a coding agent. Output only the cleaned text."

# ---------------------------------------------------------------- seed bank

SEEDS = [
    "Refactor the auth module in auth.py and run the tests, but don't touch login.py.",
    "Add a unit test for the tokenizer that covers unicode and emoji.",
    "Run pytest --verbose and show me the failures, but don't fix anything yet.",
    "Rename get_user to fetch_user everywhere except in the public API.",
    "Delete the build folder, but don't touch the dist folder.",
    "Show me the git diff for src/parser.ts.",
    "Commit everything with the message 'fix pagination bug'.",
    "Create a new branch called feature/voice-input and switch to it.",
    "Revert the last commit, but keep the changes staged.",
    "Add error handling to the fetch call in api.js.",
    "Why is the login test failing? Don't change any code, just explain.",
    "Update requirements.txt and pin numpy to version 1.26.",
    "Run npm install and then npm run dev.",
    "Add a --dry-run flag to the deploy script.",
    "Check if config.yaml has a database section, and add one if it's missing.",
    "Move the helpers from utils.py into a new file called text_utils.py.",
    "Write a docstring for the process_queue function.",
    "Find every TODO comment in the src folder and list them.",
    "Add logging to the retry loop in client.py, but don't change the retry logic.",
    "Make the sidebar responsive below 768 pixels.",
    "Add a dark mode toggle to the settings page.",
    "Fix the off-by-one error in the pagination logic in list_view.py.",
    "Squash the last three commits into one.",
    "Add an index on the email column in the users table.",
    "Write a migration that adds a created_at column to the orders table.",
    "Cache the API response for five minutes using Redis.",
    "Switch the HTTP client from requests to httpx, but keep the retry behavior.",
    "Profile the import time and tell me what's slow.",
    "Add type hints to everything in models.py.",
    "Set up a GitHub Actions workflow that runs the tests on every push.",
    "Mock the S3 client in the upload tests.",
    "Extract the validation logic into its own function and test it separately.",
    "Bump the version to 2.1.0 and update the changelog.",
    "Remove the unused imports in main.py.",
    "Convert the class component in Header.jsx to a function component.",
    "Add a health check endpoint that returns the git commit hash.",
    "Explain what the middleware in app.py does, don't change it.",
    "Replace every print statement in the daemon with proper logging.",
    "Add retries with exponential backoff to the webhook sender.",
    "Make the Dockerfile use a multi-stage build.",
    "Kill the process running on port 8080.",
    "Show me what changed between main and the release branch.",
    "Undo the formatting changes but keep the logic changes.",
    "Run the linter and fix only the errors, not the warnings.",
    "Grep for hardcoded API keys in the src directory.",
    "Add pagination to the /users endpoint with a default page size of 50.",
    "Read database.py and summarize what each function does.",
    "Split the 800 line views.py into smaller modules.",
    "Don't use a regex here, parse it with the ast module instead.",
    "Update the README with install instructions for Windows.",
    "Add a pre-commit hook that runs black and isort.",
    "Store the session token in the keychain instead of a plain text file.",
    "Why does the build fail on Node 20 but pass on Node 18?",
    "Deploy the site to Vercel, production, and give me the URL.",
    "Rate limit the login endpoint to five attempts per minute.",
    "Change the default branch from master to main.",
    "Add a fixture that spins up a temporary Postgres database.",
    "Rewrite this SQL query to use a join instead of a subquery.",
    "Check whether the cron job ran last night and show me its logs.",
    "Turn the bash script deploy.sh into a Python script.",
    "Add French and Spanish translations for the error messages.",
    "The dropdown flickers on Safari, find out why and fix it.",
    "Increase the test coverage of parser.py to at least 90 percent.",
    "Stub out the payment provider so the tests don't hit Stripe.",
    "Load the config from environment variables, falling back to config.toml.",
    "Print the ten largest files in the repository.",
    "Add keyboard navigation to the modal, escape should close it.",
    "Compress the images in the assets folder losslessly.",
    "Downgrade react to 18.2 and see if the crash goes away.",
    "Wrap the database calls in a transaction and roll back on any error.",
    "Don't push yet, run the integration tests first.",
    "Merge main into this branch and resolve the conflicts in favor of main.",
    "Schedule the cleanup job to run every day at 3 am.",
    "Replace the magic numbers in physics.py with named constants.",
    "Check if we're vulnerable to the latest lodash CVE and upgrade if so.",
    "Make the error message tell the user which field was invalid.",
    "Add a debounce of 300 milliseconds to the search input.",
    "Export the report as CSV with a download button.",
    "Run mypy in strict mode and fix the top ten errors.",
    "Track how long each request takes and log anything over 500 milliseconds.",
    "The tests pass locally but fail in CI, figure out what's different.",
    "Use a virtual environment, don't install anything globally.",
    "Copy the retry decorator from utils.py into the new service, don't import it.",
    "Show me every place we catch a bare exception.",
    "Add an environment variable called BOL_API_KEY to the deploy config.",
    "Open a pull request with a summary of these changes.",
    "Generate an OpenAPI spec from the FastAPI app and save it as openapi.json.",
    "What does the -R flag do in this chmod command?",
    "Trim the Docker image, it's 2 gigabytes and it should be under 500 megabytes.",
    "Alphabetize the imports in every file under src/api.",
    "Escape the user input before it goes into the SQL query.",
    "Rewrite the callback chain in uploader.js with async await.",
    "Draw the database schema as a mermaid diagram in the docs.",
    "Enable strict mode in tsconfig.json and fix what breaks.",
    "Move the secrets out of docker-compose.yml into a .env file.",
    "Benchmark json against orjson for our payload sizes.",
    "Add a 404 page that links back to the dashboard.",
    "Only run the expensive test suite when files in core/ change.",
    "Give the CLI a --json flag that switches the output format.",
    "Find out why the websocket disconnects after 60 seconds.",
    "Delete every feature flag that has been fully rolled out.",
    "Update the copyright year in the footer to 2026.",
    "Rebase this branch onto main and force push.",
    "Turn on gzip compression for responses over 1 kilobyte.",
    "Write a script that seeds the database with 100 fake users.",
    "The memory usage grows over time, find the leak in the worker.",
    "Sort the results by created date, newest first.",
    "Don't retry on 4xx errors, only on 5xx and timeouts.",
    "Sync the fork with the upstream repository.",
    "Show the test output without capturing, I want to see the prints.",
    "Change the primary key of the sessions table to a UUID.",
    "Add a tooltip that explains the pricing tiers.",
    "Extract the duplicated fetch logic in these three components into a hook.",
    "Pin the GitHub Action versions to commit hashes.",
    "Verify the webhook signature before processing the payload.",
    "Restart the dev server and tail the logs.",
    "Reduce the bundle size, the main chunk is over a megabyte.",
    "Add an option to export the data as JSON instead of CSV.",
    "Run the migration against staging, not production.",
    "Look at PR number 42 and address the review comments.",
    "Set the cache header to one hour for the static assets.",
    "Handle the case where the config file doesn't exist yet.",
    "Silence the deprecation warning from urllib3 in the test output.",
    "Make control C shut the daemon down cleanly.",
    "Send a Slack message to the deploys channel when the build finishes.",
    "Read the last 50 lines of the error log and tell me what's wrong.",
    "Switch the date formatting from moment to date-fns.",
    "Reject uploads over 10 megabytes with a clear error.",
    "Store timestamps in UTC and convert at display time.",
    "Enforce that every public function in api/ has a docstring.",
    "Check out the commit from before the regression and confirm it works there.",
    "Show me the slowest five database queries from yesterday.",
    "Turn this jupyter notebook into a proper module with tests.",
    "Don't log the request body, it contains passwords.",
    "Bring the dependencies up to date, one major version at a time.",
    "Make the retry count configurable with a default of three.",
    "What would break if we dropped support for Python 3.9?",
    "Combine these two SQL migrations into one.",
    "Add a loading skeleton to the results table.",
    "The search is case sensitive, make it case insensitive.",
    "Tag this commit as v1.4.2 and push the tag.",
    "Hash the passwords with bcrypt instead of md5.",
    "Let the user cancel a running export.",
    "Fix the flaky test in test_scheduler.py, it fails about one run in five.",
    "Print a progress bar while the model downloads.",
    "Return a 429 with a retry-after header when the rate limit hits.",
    "Diff the staging config against production and highlight the differences.",
    "Write the uninstall instructions and add them to the README.",
]

# ------------------------------------------------------------- corruptions

FILLERS = ["um", "uh", "umm", "uhh", "erm"]
LEAD_INS = ["so", "okay so", "hey", "alright", "hey claude", "can you", "please"]
HOMOPHONES = {
    "git": "get",
    "cache": "cash",
    "route": "root",
    "sql": "sequel",
    "suite": "sweet",
    "break": "brake",
    "merge": "murge",
}


def _spoken_symbols(text: str, rng: random.Random) -> str:
    text = re.sub(r"\b([\w-]+)\.(\w{1,5})\b", r"\1 dot \2", text)
    text = text.replace("--", "dash dash ")
    text = text.replace("/", " slash ")
    text = re.sub(r"(\d+)\.(\d+)\.(\d+)", r"\1 dot \2 dot \3", text)
    return text


def _drop_punct(text: str) -> str:
    text = re.sub(r"[.,;:!?'\"]", "", text)
    return text.lower()


def _add_fillers(text: str, rng: random.Random) -> str:
    words = text.split()
    n = max(1, len(words) // rng.randint(6, 12))
    for _ in range(n):
        pos = rng.randrange(len(words))
        words.insert(pos, rng.choice(FILLERS))
    return " ".join(words)


def _stutter(text: str, rng: random.Random) -> str:
    words = text.split()
    for _ in range(rng.randint(1, 2)):
        pos = rng.randrange(len(words))
        words.insert(pos, words[pos])
    return " ".join(words)


def _homophone(text: str, rng: random.Random) -> str:
    for src, dst in HOMOPHONES.items():
        if src in text.lower() and rng.random() < 0.5:
            text = re.sub(rf"\b{src}\b", dst, text, flags=re.IGNORECASE)
    return text


def corrupt(clean: str, rng: random.Random, heavy: bool) -> str:
    noisy = _spoken_symbols(clean, rng)
    noisy = _drop_punct(noisy)
    if rng.random() < 0.85:
        noisy = _add_fillers(noisy, rng)
    if rng.random() < (0.7 if heavy else 0.4):
        noisy = _stutter(noisy, rng)
    if rng.random() < 0.5:
        noisy = _homophone(noisy, rng)
    if rng.random() < 0.6:
        noisy = rng.choice(LEAD_INS) + " " + noisy
    return re.sub(r"\s+", " ", noisy).strip()


# --------------------------------------------------------------- roundtrip

def roundtrip_batch(seeds: list[str], sample_rate: int = 16000) -> list[tuple[str, str]]:
    """Speak each seed with `say`, transcribe with Parakeet: real STT noise."""
    import subprocess
    import sys
    import tempfile

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from bol.config import Config
    from bol.stt.parakeet import ParakeetTranscriber
    import asyncio
    import numpy as np
    import wave

    transcriber = ParakeetTranscriber(Config())
    pairs = []

    async def run():
        await transcriber.warmup()
        for seed in seeds:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                path = Path(f.name)
            try:
                subprocess.run(
                    ["say", "-o", str(path), f"--data-format=LEI16@{sample_rate}", seed],
                    check=True, capture_output=True,
                )
                with wave.open(str(path), "rb") as wf:
                    audio = np.frombuffer(
                        wf.readframes(wf.getnframes()), dtype=np.int16
                    ).astype(np.float32) / 32768.0
                text = await transcriber.transcribe(audio, sample_rate)
                if text and len(text) > 10:
                    pairs.append((text, seed))
            finally:
                path.unlink(missing_ok=True)
    asyncio.run(run())
    return pairs


# -------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="training/data")
    ap.add_argument("--variants", type=int, default=6)
    ap.add_argument("--roundtrip", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pairs: list[tuple[str, str]] = []

    for clean in SEEDS:
        for i in range(args.variants):
            pairs.append((corrupt(clean, rng, heavy=i % 3 == 0), clean))
        # Identity pairs: clean input must come back untouched.
        if rng.random() < 0.9:
            pairs.append((clean, clean))

    if args.roundtrip:
        print("roundtrip: speaking and transcribing seeds…")
        rt = roundtrip_batch(SEEDS)
        print(f"roundtrip: {len(rt)} pairs")
        pairs.extend(rt)

    rng.shuffle(pairs)
    n = len(pairs)
    splits = {
        "train": pairs[: int(n * 0.92)],
        "valid": pairs[int(n * 0.92): int(n * 0.96)],
        "test": pairs[int(n * 0.96):],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(out / f"{name}.jsonl", "w") as f:
            for noisy, clean in rows:
                f.write(json.dumps({
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": noisy},
                        {"role": "assistant", "content": clean},
                    ]
                }) + "\n")
        print(f"{name}: {len(rows)}")


if __name__ == "__main__":
    main()
