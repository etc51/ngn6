# reference

## Purpose
Imported reference implementations and legacy source material used for comparison or migration.

## Contents
- `ngn6_signal_source/` - legacy Node.js signal watcher and signal engine reference.

## Rules
- Treat this folder as reference code unless a task explicitly says to update the legacy implementation.
- Do not wire runtime Python bot behavior directly to reference code without tests.
- Keep imports and migration notes explicit so future work can separate active code from historical reference.

## Quick Checks
- Active Python bot code lives in `ngn6_bot/`, not here.
- Legacy bridge entry point is `reference/ngn6_signal_source/bridge/compute_signal.js`.
