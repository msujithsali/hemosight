# Contributing to HemoSight

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

> **SCREENING AID ONLY — NOT A DIAGNOSTIC DEVICE.** No contribution may
> introduce disease-diagnosis language or any claim of clinical accuracy that
> is not backed by a logged MLflow run against a held-out split.

## Ground rules
1. **No fabricated numbers.** Any metric in code, README, or the model card
   must come from an MLflow run with a fixed seed and a dataset hash. If a run
   has not been executed, write `PENDING` plus the exact `make` target.
2. **Provenance tags are mandatory.** Every metric is `[BOOTSTRAP]` (public
   dataset, off-domain) or `[TARGET]` (real PHC samples). Never blend them.
3. **Determinism.** Call `seed_everything()` at the top of every entrypoint.
4. **The inference path never touches the network** except sanctioned
   federated weight-delta exchange. `no_egress()` enforces this at runtime.

## Workflow
- `make bootstrap` to install dev deps + pre-commit.
- `make lint` (ruff + mypy strict) and `make test` (pytest, 80% coverage gate
  on `ml/` and `api/`) must pass before a PR.
- Commits are authored as **M Sujith Sali, ISE Dept, VTU Karnataka**.
