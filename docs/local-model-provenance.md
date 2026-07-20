# Local model provenance

The offline fallback uses the Ollama model `qwen3.5-rag-32k:latest`. The local
tag is derived from `qwen3.5:9b`, with a 32,768-token context window and the
deterministic RAG sampling configuration recorded by the generated Modelfile.

At the time of the V2 full-suite run, `ollama show` resolved the derived model
to this immutable local weights blob:

```text
sha256-dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c
```

The derived Ollama image ID was `a2845456d7ad`. Reports retain the requested
model name, endpoint kind, request options, input hash, token counts, and
latency. The full public runner additionally freezes the entire generation
profile in its run-identity hash.

`qwen3.5-9b-local-diagnostic` is intentionally excluded from the cross-family
judge bakeoff roster. It may be run against the 175 frozen human calibration
labels to measure its agreement, or used as a fail-closed operational
diagnostic when paid providers are unavailable. Because it belongs to the same
model family as the local generator, its verdicts are not independent truth
and cannot promote a release on their own.
