"""
Day-1 exit check: run this after filling in .env to confirm every provider
you've configured actually works, before writing a single line of agent logic.

    uv run python scripts/verify_providers.py
"""

import os
import sys

sys.path.insert(0, "src")

from automl_agents.llm_client import ping  # noqa: E402

PROVIDERS_TO_CHECK = ["gemini", "groq"]  # add "ollama" if you're using it


def main() -> None:
    results = {}
    for provider in PROVIDERS_TO_CHECK:
        key_env = {"gemini": "GOOGLE_API_KEY", "groq": "GROQ_API_KEY"}.get(provider)
        if key_env and not os.getenv(key_env):
            results[provider] = f"SKIPPED ({key_env} not set)"
            continue
        try:
            reply = ping(provider=provider)
            results[provider] = f"OK -> {reply!r}"
        except Exception as e:  # noqa: BLE001 -- deliberately broad for a diagnostic script
            results[provider] = f"FAILED -> {e}"

    print("\nProvider connectivity check:")
    for provider, outcome in results.items():
        print(f"  {provider:8s}: {outcome}")

    if any("FAILED" in v for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
