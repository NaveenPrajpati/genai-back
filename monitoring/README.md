# RAG Observability (Prometheus + Grafana + Alertmanager)

Ops metrics for the AIEngineer API — the layer LangSmith isn't. LangSmith traces
one request for debugging; this answers *"what's my error rate / cost / latency
right now, and page me when it's bad."*

## What you get

Five SLOs, on one Grafana dashboard, with email alerts:

| Metric | Source | Dashboard | Alert |
|---|---|---|---|
| **Latency per stage** | `rag_stage_latency_seconds` histogram (embed → cache → retrieve → rerank → gate → stream → persist → evaluate) | "Latency per stage (p95)" | p95 end-to-end > 10s |
| **Error rate** | `rag_requests_total{outcome="error"}` | "Error rate" tile + trend | > 5% for 5m (critical) |
| **Cache hit rate** | `rag_cache_lookups_total{result}` | "Cache hit rate" tile + trend | < 10% for 30m |
| **Refusal rate** | `rag_requests_total{outcome="refused"}` | "Refusal rate" tile + trend | > 35% for 15m |
| **Cost per query** | `llm_cost_usd_total` / requests (token usage × pricing) | "Cost per query" tile + trend | > $0.02 for 15m |

Plus: request-outcome breakdown, per-model spend, token throughput, p50/p95/p99,
and an `APIDown` alert when the scrape fails.

## How it fits together

```
 API (:8000, single gunicorn worker)
   └── GET /metrics   ← prometheus_client, in-process registry
          ▲ scrape (15s)
 Prometheus (:9090, localhost-only)  ── evaluates alerts.yml ──▶ Alertmanager (:9093) ──▶ email
   └── datasource for
 Grafana (:3000)  ── auto-provisions the "RAG Observability" dashboard
```

Instrumentation lives in the app: [`app/core/metrics.py`](../app/core/metrics.py)
(metric defs, pricing table, cost callback), wired into the shared LLM clients in
[`app/core/llm.py`](../app/core/llm.py) and the RAG routes in
[`app/routers/rag.py`](../app/routers/rag.py). The `/metrics` endpoint is in
[`app/main.py`](../app/main.py).

Because the API runs **one** gunicorn worker on purpose, a plain in-process
registry is correct — no `PROMETHEUS_MULTIPROC_DIR` needed. If you ever scale to
multiple workers, switch to multiprocess mode first.

## Bring it up

```bash
cd monitoring
cp .env.example .env                                        # set GRAFANA_ADMIN_PASSWORD
cp alertmanager/alertmanager.yml.example alertmanager/alertmanager.yml   # fill SMTP + recipient
docker compose up -d
```

Open Grafana at `http://<host>:3000` → dashboard **RAG Observability**.

The API must be reachable at `host.docker.internal:8000` from the Prometheus
container (handled via `extra_hosts` on Linux/EC2). If your API listens on a
different port, edit `prometheus/prometheus.yml`.

## Security

- `/metrics` exposes internal counters. Set `METRICS_TOKEN=<secret>` on the API,
  then uncomment the `authorization` block in `prometheus/prometheus.yml` with the
  same value. Also restrict :8000 to the monitoring host via the EC2 security group.
- Prometheus (:9090) and Alertmanager (:9093) bind to `127.0.0.1` only — reach
  them over an SSH tunnel. Grafana (:3000) is the one UI to expose; front it with
  TLS / a reverse proxy and change the admin password.

## Pricing

Cost is estimated from `app/core/metrics.py`'s pricing table (USD per 1M tokens).
**Keep it current with OpenAI's list**, or override without a code change:

```bash
export LLM_PRICING_JSON='{"gpt-4o":[2.5,10.0],"gpt-4o-mini":[0.15,0.6]}'
```

Embedding tokens (~$0.02/1M) are intentionally not priced — noise next to
generation. Chat models (`llm` / `fast_llm`) are captured on every call site.

## Tuning alerts

Thresholds in `prometheus/alerts.yml` are starting points. After ~a week of real
traffic, set them from your own baselines (e.g. p95 latency, a realistic cost
ceiling). Ratio alerts stay quiet with no traffic (0/0 = NaN); the cache/cost/
refusal alerts add an explicit volume guard so a trickle of requests can't trip
them. Reload rules after editing: `docker compose kill -s SIGHUP prometheus`.
