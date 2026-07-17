# Release cases

The V2 release set contains 50 cases in
`vbot-ai-reviewed-release-50.jsonl`: 42 answerable cases and eight
non-answer cases. The same AI agent authored and source-verified them before
candidate evaluation. They add no human-review count and are not a
production-traffic sample.

Release cases are authored before inspecting candidate-system output, sealed
from routine development, and frozen under an explicit dataset version. The
solo AI-only path must disclose its lack of independent human review and use
`model_proposed_ai_verified` provenance. Do not copy visible V1 cases here or
move observed-failure-derived cases into this partition.
