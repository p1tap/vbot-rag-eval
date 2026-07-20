# Adversarial cases

This partition contains 12 AI-authored/AI-reviewed observed-failure regression
probes in `vbot-ai-observed-failures-12.jsonl`. Each case is derived from an
action failure on the visible 100-case development run and records its source
case ID in machine-readable tags and review notes.

These cases are not independent holdout questions, do not increase the
human-reviewed count, and cannot support a production-traffic claim. They test
whether known action failures recur under edited wording. The same AI agent
authored and source-verified every case; claims, actions, and evidence IDs were
inherited from already approved source cases and mechanically checked.

Generated fuzz variants remain separate and do not count as independent cases
unless their wording, expected action, claims, and evidence are verified.
