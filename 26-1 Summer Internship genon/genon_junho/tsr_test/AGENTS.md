# AGENTS

## Memory Loading Rule
- At the start of every session and before each new task, read `docs/CODEX_MEMORY.md` and `docs/SESSION_HANDOFF.md`.
- If present, also read local-only files: `docs/CODEX_MEMORY.local.md` and `docs/SESSION_HANDOFF.local.md`.
- Use `docs/CODEX_MEMORY.md` as the source of current goals, decisions, open issues, and next actions.
- Use `docs/SESSION_HANDOFF.md` as the source of latest progress, pending TODOs, and blockers.
- Keep sensitive/private data in local-only files, not in git-tracked memory files.
- For evaluation/serving work, also read `train/vlm/docs/EVAL_TEST_SESSION_SUMMARY_20260220.md`.

## Memory Update Rule
- After completing a meaningful change, update `docs/CODEX_MEMORY.md` and `docs/SESSION_HANDOFF.md`.
- Update local-only files when a change includes personal notes, credentials, internal hosts, or other sensitive details.
- Keep both files concise, factual, and aligned with the current repository state.

## Session Snapshot (2026-02-20)
- Scope: vLLM serving + local 126-sample evaluation pipeline stabilization.
- Detailed record: `train/vlm/docs/EVAL_TEST_SESSION_SUMMARY_20260220.md`.

### Key Reusable Facts
- Training server: `ssh -p 2222 root@192.168.75.174`
- Serving server: `ssh -p 2220 root@192.168.75.173`
- External serving endpoint: `http://192.168.75.173:8010` (container maps `8010:8000`)
- Active evaluated model alias: `e6ckpt600`
- Evaluation dataset (fixed 126 samples):
  - `train/vlm/data/processed/from_server/eval_used_data/eval_used_data_20260213_141329/eval_localized.jsonl`

### Latest Verified Evaluation Output
- Run output dir:
  - `train/vlm/eval_results/local_api_e6ckpt600_126_20260220_101300`
- Metrics (126 samples):
  - `avg_teds=0.5025`
  - `avg_teds_structure=0.6472`
  - `avg_span_f1=0.3872`
  - `avg_attribute_accuracy=0.5025`
  - `avg_inference_time=6.4458s/sample`

### Operational Notes
- Local eval should use project venv Python:
  - `test/test_model/.venv/bin/python`
- If missing dependency appears:
  - `uv pip install --python test/test_model/.venv/bin/python editdistance`
- For `test/test_model/test_table.py`, model can be auto-discovered from `/v1/models` when profile model is empty.
