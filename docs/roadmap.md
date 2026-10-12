# Technical design and next milestones

## Current implementation

**Task:** Simulated e-commerce customer requests maximum valid refund.

**Visible to agent:** Natural language goal, current simulated date, five declared tools, and only the tool responses it has requested.

**Hidden from agent:** Scenario family, the expected resolution, expected refund amount, and internal ledger used to verify reward. There is no direct state mutation outside the tool interface in the HTTP integration.

**Transition:** Each tool invocation advances one step and returns an observation. `issue_refund` mutates only episode-local ledger after policy, amount, and idempotency validation. `finish` ends the episode. Reaching the limit without finishing truncates the episode.

**Reward:** Sparse (+1 correct completion, 0 benign failure, -1 unsafe mutation attempt), with hard checks for ledger state and inspection evidence. Refund receipt retries with the same idempotency key do not double pay.

## Real challenges worth tackling

### Milestone 2 — LLM agent benchmark

- Implement a generic adapter for OpenAI-compatible tool-call messages: convert `TOOL_SCHEMAS` to model tools, send task prompt, route function calls to `env.step`, append tool responses to conversation history, and stop when done.
- Write **actual** model messages and action traces to JSONL. Redact sensitive content and maintain request/correlation IDs.
- Measure success by scenario family, unsafe-refund rate, mean steps, and model tokens/cost. Compare to naive and scripted baselines.
- Run independent seed ranges for training and eval; add a third holdout set.

### Milestone 3 — Hard environments

- Add return inspection failures, partial captures, duplicate webhooks, pickup failures and inventory hold/unhold semantics.
- Add configurable policy revisions and orders that cross date boundaries.
- Introduce noisy/incomplete observations and tool timeouts, while keeping simulator replay deterministic via seeded fault injection.
- Add invariant/property-based tests: total refunded never exceeds merchandise amount; invalid actions never mutate state; same action/key yields same receipt; same seed/action sequence yields same trace.
- Probe reward hacking (e.g. close without reading tools, guess amounts, edit order id, replay payment).

### Milestone 4 — LLM training, not just toy tabular RL

- Use a maintained trainer such as NeMo Gym + NeMo RL or another compatible GRPO stack. Their interfaces evolve: read current docs before wiring an adapter.
- For each prompt/task, sample several rollouts from the current model, compute episode-level rewards, train policy weights using relative advantages (GRPO), and validate on held-out tasks.
- Keep model/tool API credentials out of the environment. The environment runs CPU-only; the trainer/inference service handles GPU.
- Track before/after held-out success, unsafe-refund rate, cost and performance by case category. Compare with SFT-only baselines and ensure model actually improves.
- 1,000 seed variants of the same six families is NOT sufficient evidence of generalization; add diverse hidden policies and human-reviewed tasks.

### Milestone 5 — Service hardening

- Add authentication, TLS, tenant isolation and authorization, request/body caps, structured logging, OpenTelemetry traces and Prometheus metrics.
- Move episodes and tool event journal into Postgres, lock one episode row per tool call; enforce CAS/serializable semantics for monetary side effects.
- Ensure cleanup/persistence across worker restarts and use request IDs to deduplicate replays. Apply concurrency/stress and chaos tests.
- Sandbox untrusted tools. Never connect real payment APIs to a training environment.

## Why Python for this project?

The user's production background may be TypeScript, but most LLM RL trainers, Gym adapters and ML tooling are Python-native. Core environment logic is simple typed Python without dependencies. The service boundary is HTTP, so a Node.js agent is a perfectly valid client.
