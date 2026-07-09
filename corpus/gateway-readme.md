<!-- source: p1tap/vbot-model-gateway README (snapshot for RAG corpus; public-safe, no secrets) -->

# Vbot Model Gateway

Multi-provider LLM gateway for the Vbot platform, built on [LiteLLM proxy].
Implements the failover ladder from the provider-reliability research
(doc 09) per the ADR in doc 15: **retry same deployment → cooldown
(circuit breaker) → model-level fallback** — because our primary RP model
(Euryale 70B) is single-host on OpenRouter, so a hot-standby model IS the
second rung, not a nice-to-have.

**Scope note:** the model lab stays on direct OpenRouter with NO fallback —
silent model swapping would poison blind ratings. The gateway is a product
concern, not a lab concern.

## What it does today

- OpenAI-compatible endpoint at `http://127.0.0.1:4000/v1` fronting
  OpenRouter (later: direct providers + self-hosted vLLM behind the same
  interface).
- `rp-primary` (Euryale 70B) with automatic fallback to `rp-fallback`
  (Hermes-4 70B) after retries are exhausted.
- Cooldowns: a deployment failing >3×/min is pulled from rotation for 30s
  (Redis-backed, so state survives multi-replica later).
- `failover-demo` model: intentionally broken credentials, exists so the
  smoke test can PROVE fallback fires end-to-end instead of trusting config.
- Auth via master key — clients never see the OpenRouter key.

## Run

```sh
cp .env.example .env    # fill in OPENROUTER_API_KEY, keys, postgres password
docker compose up -d    # litellm + redis + postgres
node scripts/smoke.mjs        # readiness + primary + forced-failover checks
node scripts/spend-check.mjs  # mints a budget-capped virtual key, proves
                              # per-key spend lands in Postgres
```

Monitoring (auto-provisioned, no clicking through setup wizards):

```sh
# one-time: give Prometheus the scrape token (gitignored)
grep '^LITELLM_MASTER_KEY=' .env | cut -d= -f2 | tr -d '\r\n' > monitoring/.litellm_token
docker compose up -d
# Grafana  → http://127.0.0.1:3001  (admin / vbotlab or $GRAFANA_PASSWORD)
# Prometheus → http://127.0.0.1:9090
```

The **Vbot Model Gateway** dashboard ships provisioned: requests/min by
model, LLM latency p50/p95, failed requests/min, tokens/min in/out,
deployment state (cooldown visibility — you can watch the circuit breaker
trip), total spend, in-flight requests, success rate. LiteLLM's
`prometheus` callback verified working on the standard image (the
"enterprise-only" doc note did not bite on this version).

Load test (10 concurrent chatters, ~3 min, < $0.05 in tokens):

```sh
docker run --rm -i --network=model-gateway_default \
  -e GATEWAY_URL=http://litellm:4000 -e LITELLM_MASTER_KEY=$LITELLM_MASTER_KEY \
  grafana/k6 run - < scripts/load-test.js
```

Call it like any OpenAI endpoint:

```sh
curl http://127.0.0.1:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"rp-primary","messages":[{"role":"user","content":"hi"}]}'
```

## What it does today (cont.)

- **Per-key spend tracking**: Postgres-backed virtual keys via
  `/key/generate` with hard budget caps (`max_budget`) — every client key
  has its own spend ledger and can never exceed its cap. This is the
  billing/abuse primitive for any future product tier.
- **k6 load profile** (`scripts/load-test.js`): 10 concurrent chatters,
  real think-time, thresholds on p95 latency and error rate.

## Measured findings (2026-07-08, k6 through the gateway → OpenRouter)

- **10 concurrent chatters, 0.00% user-visible errors, p95 10.2s** at
  realistic RP budgets (max_tokens 400) — with the client contract below.
- **Empty-content 200s are real**: Hermes-4 stochastically reasoning-spirals
  (~2–10% of calls depending on window/budget), consuming the entire
  completion budget as `reasoning_content` and returning HTTP 200 with
  empty `content`, `finish_reason: length`. Found by tagging every k6
  response with OpenRouter's `provider` field — all empties were Nebius.
- **The reasoning-disable param is not honored by Nebius** (direct A/B:
  5/16 calls still reasoned with `reasoning: {enabled: false}` set). Sent
  anyway as a courtesy for compliant hosts; never relied upon.
- **Client contract: HTTP 200 + empty content = retryable, once.** This
  converts a 2.34% raw empty rate into 0.00% failed turns. Status-code-only
  error handling ships blank messages to users.
- **One-model-one-host**: every RP-tuned 70B checked is single-host on
  OpenRouter (Euryale=NextBit bf16, Hermes-4=Nebius fp8,
  Hermes-3=DeepInfra fp8). Same-model host failover doesn't exist in this
  class — the model-level fallback chain IS the host diversity
  (3 models = 3 independent infrastructures).

## Deploy (VPS runbook — done 2026-07-08 on DO SGP1, 2GB)

```sh
# 0. droplet: Docker marketplace image, SSH key auth
# 1. swap (2GB box + 5 containers = want the safety net)
ssh root@<ip> "fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile && echo '/swapfile none swap sw 0 0' >> /etc/fstab"
# 2. ship the repo (committed files only — no secrets ride along)
git archive master | ssh root@<ip> "mkdir -p /opt/model-gateway && tar -x -C /opt/model-gateway"
# 3. FRESH production secrets (never promote dev keys), piped via stdin
#    -> /opt/model-gateway/.env  (umask 077)
# 4. prometheus token: derive from .env, then — Linux gotcha — the
#    prometheus container runs as uid 65534 (nobody):
chown 65534:65534 monitoring/.litellm_token && chmod 400 monitoring/.litellm_token
# 5. docker compose up -d, wait for /health/readiness
# 6. verify: smoke.mjs + spend-check.mjs via
docker run --rm --network=host -v /opt/model-gateway:/app -w /app node:22-alpine node scripts/smoke.mjs
# 7. outside-in sweep: ports 4000/9090/3001 must be UNREACHABLE from the
#    internet (everything binds 127.0.0.1; access via SSH tunnel):
ssh -L 3002:127.0.0.1:3001 root@<ip>   # → Grafana at http://127.0.0.1:3002
```

## Kubernetes (current production shape — migrated 2026-07-08)

The stack runs on **k3s** on the droplet; compose remains for local dev.
Manifests in `k8s/` (kustomize), self-contained under `k8s/config/`.

```sh
# one-time on the node: k3s + secrets (never in git)
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="server --disable traefik" sh -
kubectl -n gateway create secret generic gateway-env --from-env-file=.env
kubectl -n gateway create secret generic prom-token --from-file=token=monitoring/.litellm_token
kubectl -n gateway create secret generic grafana-admin --from-literal=admin-password=<pw>

# deploy / update everything
kubectl apply -k k8s/

# access (NodePorts are droplet-local; DO firewall blocks them externally)
# gateway API:  :30400      grafana: :30301
ssh -L 3002:127.0.0.1:30301 root@<ip>   # → http://127.0.0.1:3002
```

Migration was verify-then-cut: k8s side brought up beside compose, full
ladder (smoke incl. forced failover, spend tracking, scrape, dashboard)
passed on NodePorts, then `docker compose down`. Zero downtime.

## GitOps (Argo CD — live since 2026-07-08)

**Merge to `master` = deploy.** Argo CD (core/headless install) watches
`k8s/` and keeps the cluster identical to git:

- `apps/gateway.yaml` — the Application: automated sync + selfHeal + prune
- Repo access: dedicated **read-only deploy key** (least privilege), never
  account credentials
- **Proven live:** manual cluster sabotage (grafana scaled to 0) reverted
  by selfHeal in **10s**; a git-only label change arrived on-cluster in
  **197s** (one poll cycle) with no human action
- Operational consequence: no more ad-hoc kubectl edits — Argo reverts
  them. Change git or it didn't happen.
- Headless-install gotchas (in the runbook): ApplicationSet CRD needs
  `--server-side` apply; the `default` AppProject must be created manually

## Roadmap

- [x] Postgres for per-key spend tracking / virtual keys
- [x] k6 load test — run, thresholds passed, findings recorded above
- [x] Prometheus + Grafana, provisioned-as-code (datasource + dashboard
      in `monitoring/`), verified scraping live
- [x] VPS deploy — LIVE on DigitalOcean SGP1 (2GB, $12/mo, credit-funded):
      all smoke + spend checks pass in cloud, Prometheus scraping, ports
      internet-invisible (localhost-bound, SSH-tunnel access)
- [ ] Public Grafana (read-only anonymous view) — pending decision
- [x] Eval-gated promotion CI (model-gate workflow): PR -> candidate
      gateway in CI -> gauntlet (reachability, empty-content probe,
      latency, cost guard, LLM-judge) -> PROMOTE/NEEDS_REVIEW/REJECT exit
      code -> merge -> Argo deploys. First real promotion shipped: PR #1
      (Hermes-4 -> Hermes-3 fallback), gate returned NEEDS_REVIEW on a
      peak-hour NextBit latency flag (environmental), human override
      documented on the PR, merged, GitOps-deployed, verified serving.
- [x] Hash-suffixed ConfigMaps: config-only merges auto-roll pods (proven live)
- [ ] Swap `rp-primary` to the lab verdict winner when Phase 1/2 concludes
- [x] Repo public (full-history secret audit first) · public read-only
      Grafana: http://188.166.230.62:30301 (anonymous Viewer, writes 403)
- [x] Self-hosted leg: in-cluster Ollama (qwen2.5:0.5b, CPU) as rp-selfhosted
      behind the same gateway — merged via gate PR #2 (first run REJECTED on
      CI-topology drift the gate itself exposed; fixed, PROMOTE, deployed)
- [ ] Writeup/blog-post polish pass

[LiteLLM proxy]: https://docs.litellm.ai/docs/proxy/reliability
