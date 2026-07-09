"""One-shot authoring helper for the FROZEN golden set.

The questions, answers, and relevance judgments below are hand-written from
the corpus (not model-generated — the generator must not grade its own
homework). This script only expands readable section aliases to the exact
heading_keys the index uses and validates that every referenced section
actually exists, then writes evals/golden.jsonl once. Re-run only to
regenerate after a deliberate golden-set edit; the committed jsonl is the
artifact the harness reads.

Run: PYTHONNOUSERSITE=1 conda run -n ft python scripts/_author_golden.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent

# readable alias -> exact heading_key (validated against the live index below)
A = {
    "gw_intro":  "gateway-readme::vbot-model-gateway",
    "gw_today":  "gateway-readme::what-it-does-today",
    "gw_run":    "gateway-readme::run",
    "gw_today2": "gateway-readme::what-it-does-today-cont",
    "gw_find":   "gateway-readme::measured-findings-2026-07-08-k6-through-the-gateway-openrouter",
    "gw_deploy": "gateway-readme::deploy-vps-runbook-done-2026-07-08-on-do-sgp1-2gb",
    "gw_k8s":    "gateway-readme::kubernetes-current-production-shape-migrated-2026-07-08",
    "gw_gitops": "gateway-readme::gitops-argo-cd-live-since-2026-07-08",
    "gw_road":   "gateway-readme::roadmap",
    "bp_intro":  "backbone-plan::ft-lab-backbone-fine-tune-plan",
    "bp_why":    "backbone-plan::why-this-target-and-not-generic-rp-style",
    "bp_lanes":  "backbone-plan::behavior-families-dataset-lanes",
    "bp_data":   "backbone-plan::dataset-spec",
    "bp_train":  "backbone-plan::training",
    "bp_eval":   "backbone-plan::evaluation-the-credibility-section",
    "bp_ship":   "backbone-plan::ship-list",
    "bf_sweep":  "backbone-findings::sweep-findings-2026-07-09-1-5b-fixed-eff-batch-16",
    "bf_dpo":    "backbone-findings::dpo-findings-2026-07-09-run-20260709-210645-dpo-1-5b-sft-merged-fresh-lora",
}

# (id, question, relevant_aliases, reference_answer). Empty aliases = must abstain.
GOLD = [
    ("q01", "What is the gateway's failover ladder, in order?", ["gw_intro"],
     "Retry the same deployment, then cooldown (circuit breaker), then model-level fallback."),
    ("q02", "Why is a hot-standby model a necessity rather than a nice-to-have for the RP primary?", ["gw_intro"],
     "Because the primary RP model (Euryale 70B) is single-host on OpenRouter, so a standby model is the only second rung available."),
    ("q03", "Does the model lab use the gateway's fallback, and why or why not?", ["gw_intro"],
     "No. The lab stays on direct OpenRouter with no fallback, because silent model swapping would poison blind ratings."),
    ("q04", "At what URL does the gateway expose its OpenAI-compatible endpoint?", ["gw_today"],
     "http://127.0.0.1:4000/v1"),
    ("q05", "After how many failures is a deployment pulled from rotation, and for how long?", ["gw_today"],
     "A deployment failing more than 3 times per minute is pulled for 30 seconds."),
    ("q06", "What is the purpose of the failover-demo model?", ["gw_today"],
     "It has intentionally broken credentials so the smoke test can prove fallback fires end-to-end rather than trusting config."),
    ("q07", "How do clients authenticate, and do they ever see the OpenRouter key?", ["gw_today"],
     "Clients authenticate with a master key and never see the OpenRouter key."),
    ("q08", "What backing store lets cooldown/circuit-breaker state survive multiple replicas?", ["gw_today"],
     "Redis."),
    ("q09", "What p95 latency and user-visible error rate were measured at 10 concurrent chatters?", ["gw_find"],
     "p95 of 10.2s and 0.00% user-visible errors (at max_tokens 400)."),
    ("q10", "What causes the empty-content HTTP 200 responses?", ["gw_find"],
     "Hermes-4 stochastically reasoning-spirals, consuming the whole completion budget as reasoning_content and returning HTTP 200 with empty content and finish_reason length; all empties were on Nebius."),
    ("q11", "Is the reasoning-disable parameter honored by Nebius?", ["gw_find"],
     "No; in a direct A/B, 5 of 16 calls still reasoned even with reasoning:{enabled:false} set."),
    ("q12", "What client contract reduces the empty-response rate to zero failed turns?", ["gw_find"],
     "Treat HTTP 200 with empty content as retryable once; this converts a 2.34% raw empty rate to 0.00% failed turns."),
    ("q13", "Which host serves Euryale, and at what precision?", ["gw_find"],
     "NextBit, at bf16."),
    ("q14", "Why doesn't same-model host failover exist for these RP 70Bs?", ["gw_find"],
     "Each RP-tuned 70B is single-host on OpenRouter, so the model-level fallback chain itself provides host diversity (3 models = 3 infrastructures)."),
    ("q15", "How is per-key spend tracked and capped?", ["gw_today2"],
     "Postgres-backed virtual keys minted via /key/generate with hard max_budget caps; each key has its own spend ledger and can't exceed its cap."),
    ("q16", "What does the k6 load profile simulate?", ["gw_today2", "gw_run"],
     "10 concurrent chatters with real think-time, with thresholds on p95 latency and error rate."),
    ("q17", "What OS-user gotcha affects the Prometheus scrape-token file on deploy?", ["gw_deploy"],
     "The Prometheus container runs as uid 65534 (nobody), so the token file must be chown 65534:65534 and chmod 400."),
    ("q18", "How is the repo shipped to the droplet without secrets riding along?", ["gw_deploy"],
     "git archive master piped over ssh and extracted to /opt/model-gateway (committed files only); fresh production secrets are piped separately via stdin."),
    ("q19", "How are the gateway's ports kept off the public internet?", ["gw_deploy", "gw_k8s"],
     "Everything binds 127.0.0.1 and is reached via SSH tunnel; the service ports are unreachable from the internet."),
    ("q20", "What orchestrator runs the production gateway and how is it installed?", ["gw_k8s"],
     "k3s, installed via the get.k3s.io script with server --disable traefik."),
    ("q21", "What NodePorts expose the gateway API and Grafana, and are they externally reachable?", ["gw_k8s"],
     "API on :30400 and Grafana on :30301; they are droplet-local and the DigitalOcean firewall blocks them externally."),
    ("q22", "How was the compose-to-Kubernetes migration done without downtime?", ["gw_k8s"],
     "Verify-then-cut: k8s was brought up beside compose, the full ladder passed on NodePorts, then docker compose was shut down."),
    ("q23", "In the GitOps setup, what triggers a deploy?", ["gw_gitops"],
     "Merging to master; Argo CD watches k8s/ and keeps the cluster identical to git."),
    ("q24", "How fast did Argo CD self-heal a manual cluster change?", ["gw_gitops"],
     "About 10 seconds (grafana scaled to 0 was reverted by selfHeal)."),
    ("q25", "What credential does Argo CD use to read the repo?", ["gw_gitops"],
     "A dedicated read-only deploy key (least privilege), never account credentials."),
    ("q26", "What two headless-install gotchas did Argo CD require?", ["gw_gitops"],
     "The ApplicationSet CRD needs a --server-side apply, and the default AppProject must be created manually."),
    ("q27", "What behavior family does the Backbone fine-tune target?", ["bp_intro"],
     "Calibrated resistance to social pressure — the anti-sycophancy behavior family."),
    ("q28", "What is lane C and why is it non-negotiable?", ["bp_lanes"],
     "The calibration control lane (~25%): when the user's correction is actually correct the model updates gracefully; without it the fine-tune would train a contrarian, sycophancy's mirror image."),
    ("q29", "How large is the training dataset and when does pressure arrive in a conversation?", ["bp_data"],
     "About 1,500–2,000 multi-turn conversations; pressure arrives at turn 2 or later, never turn 1."),
    ("q30", "What base model and quantization does the fine-tune use, and does it fit the GPU?", ["bp_train"],
     "Qwen2.5-7B-Instruct with QLoRA 4-bit (bnb nf4); it fits in ~10–11GB of the RTX 5080's 16GB."),
    ("q31", "Why must every ft-env command set PYTHONNOUSERSITE=1?", ["bp_train"],
     "A broken half-installed boto3 in user-site leaks into conda envs and crashes accelerate's import chain (and pip under-installs deps it thinks user-site provides)."),
    ("q32", "At what epoch did validation loss bottom in the 5-epoch overfit probe?", ["bf_sweep"],
     "Epoch 2 (1.496), rising to 1.557 by epoch 5 while train loss kept falling to 1.32."),
    ("q33", "Which LoRA rank had the best eval loss, and why wasn't it deployed?", ["bf_sweep"],
     "r64 had the best eval loss (1.473) but lost a full point of calibration (7.17 vs r16's 8.17), so r16 stays deployed."),
    ("q34", "How many preference pairs did the DPO run use, and where did chosen/rejected come from?", ["bf_dpo"],
     "620 contrast-filtered pairs: chosen were judge>=7 teacher holds, rejected were anti-steered caves scored <=4."),
    ("q35", "Why was the DPO model not shipped despite reward accuracy reaching 1.0?", ["bf_dpo"],
     "The frozen suite showed anti-sycophancy -1.33 and general -0.5; it held positions via fabricated authority and confident wrong math — posture without grounding."),
    ("q36", "What GGUF quantization format is planned for the ollama export?", ["bp_ship"],
     "Q4_K_M."),
    ("q37", "How much did the DigitalOcean droplet cost per month and in which region?", ["gw_road", "gw_deploy"],
     "$12/month, in the SGP1 region (2GB droplet)."),
    ("q38", "What self-hosted model became rp-selfhosted, and how did its gate PR go?", ["gw_road"],
     "An in-cluster Ollama qwen2.5:0.5b (CPU); its gate PR was first REJECTED on CI-topology drift the gate exposed, then fixed to PROMOTE and deployed."),

    # --- abstention lane: in-domain but NOT in the corpus (must refuse) ---
    ("u01", "What is the Grafana admin password?", [], "ABSTAIN"),
    ("u02", "How many daily active users does the gateway serve?", [], "ABSTAIN"),
    ("u03", "What vector database does the fine-tuning lab use?", [], "ABSTAIN"),
    ("u04", "What is the accuracy of the RoBERTa emotion classifier?", [], "ABSTAIN"),
    ("u05", "What uptime SLA is promised to gateway customers?", [], "ABSTAIN"),
    ("u06", "How many GPU-hours did the DPO training run take?", [], "ABSTAIN"),
    ("u07", "What is the context window of the Euryale model?", [], "ABSTAIN"),
    ("u08", "Which cloud region hosts the backup droplet?", [], "ABSTAIN"),
    ("u09", "What embedding model does the gateway use for semantic caching?", [], "ABSTAIN"),
    ("u10", "How many engineers are on the Vbot team?", [], "ABSTAIN"),
    ("u11", "What price per token do gateway clients pay?", [], "ABSTAIN"),
]


def main():
    index_keys = set()
    for line in (ROOT / "index" / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
        index_keys.add(json.loads(line)["heading_key"])

    bad = sorted({A[a] for _, _, aliases, _ in GOLD for a in aliases} - index_keys)
    if bad:
        print("ERROR — golden references sections not in the index:")
        for b in bad:
            print("  ", b)
        sys.exit(1)

    out = []
    for qid, q, aliases, ans in GOLD:
        out.append({
            "id": qid, "question": q, "answerable": bool(aliases),
            "relevant": [A[a] for a in aliases], "answer": ans,
        })
    (ROOT / "evals" / "golden.jsonl").write_text(
        "\n".join(json.dumps(o, ensure_ascii=False) for o in out) + "\n", encoding="utf-8")
    n_ans = sum(o["answerable"] for o in out)
    print(f"wrote {len(out)} golden items ({n_ans} answerable, {len(out) - n_ans} abstain) — all sections validated")


if __name__ == "__main__":
    main()
