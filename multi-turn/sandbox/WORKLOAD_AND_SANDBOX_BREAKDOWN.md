# Workload and Sandbox Breakdown (for security review)

## 1. What was running on the instance

- **Primary workload**: LLM reinforcement learning (RL) training — GRPO/PPO-style policy optimization using the **verl** framework.
- **Entry point**: A shell script (`train_perturb.sh` in the repo root) that sets environment variables and launches:
  ```bash
  python -m recipe.simpletir.main_simpletir --config-name simpletir_trainer ...
  ```
- **Stack**: Python → Hydra config → **Ray** (distributed execution) → **verl** (actor/rollout/ref workers, FSDP, vLLM for rollout). Training uses WANDB for logging and loads data from parquet files.

---

## 2. How the sandbox is supposed to work

The repo includes a **code-execution sandbox** under `mismatch_all_perturb_agent_new/sandbox/`. It is used to run **untrusted Python code** (model-generated or from datasets) for **reward scoring** (e.g. math/coding tasks).

### 2.1 Components

| Component | Role |
|----------|-----|
| **sandbox_api.py** | FastAPI app. Exposes `POST /faas/sandbox/`; receives JSON `{ "code": "<python source>", "stdin": "...", "language": "python", "run_timeout": <float> }`. |
| **local_sandbox.py** | Client used by the training pipeline. Reads `SANDBOX_ENDPOINT` (e.g. `http://127.0.0.1:12345/faas/sandbox/`), sends code snippets via HTTP POST. |
| **train_perturb.sh** | Sets `export SANDBOX_ENDPOINT=http://127.0.0.1:12345/faas/sandbox/` so the reward path uses this local sandbox when the env is set. |

### 2.2 Execution flow (intended)

1. **Sandbox service** is started separately (e.g. on the same instance):
   ```bash
   cd sandbox && nohup uvicorn sandbox_api:app --host 127.0.0.1 --port 12345 --workers 4 &
   ```
2. **Training** runs; Ray workers call the reward manager (e.g. `math_verify_with_exec.py`, or code that uses `recipe/simpletir/utils/reward_score/code.py` / `recipe/simpletir/agent_utils.py`).
3. When a **code solution** must be scored, the reward manager sends that **code string** (model output or from data) to the sandbox via `parallel_sandbox(...)` → HTTP POST to `SANDBOX_ENDPOINT`.
4. **sandbox_api.py** (see `_run_in_firejail` in `sandbox_api.py`):
   - Creates a **temporary directory** under `/dev/shm` with prefix `fj_` (e.g. `/dev/shm/fj_37b6yarr`).
   - Writes the **received `code`** to a file named **`main.py`** in that directory.
   - Runs: `firejail --private=<workdir> --net=none --timeout=... -- python3 main.py` (with resource limits).
5. After the run, the sandbox deletes the temp dir. The **content of `main.py`** is exactly the **code string from the HTTP request** — i.e. **model-generated or dataset-provided code**, not application source from this repo.

### 2.3 Important point for the incident

- **`/dev/shm/fj_<random>/main.py`** is **not** part of the application’s source tree. It is **user/model-provided code** written by the sandbox for that single execution.
- So if the sandbox was used for scoring, the **contents of `main.py`** (including any `ssh ...` or other commands) would be whatever was in the **`code`** field of the request — e.g. from the **LLM’s output** or from **training/eval data**. The application itself does not write or inject an SSH command into that file.

---

## 3. Summary for security / SIRT

| Question | Answer |
|----------|--------|
| What is the main workload? | LLM RL training (GRPO/PPO via verl + Ray). Launched by `train_perturb.sh` → `recipe.simpletir.main_simpletir`. |
| What creates `/dev/shm/fj_*` and `main.py`? | The **sandbox service** (`sandbox/sandbox_api.py`). It creates a temp dir under `/dev/shm` with prefix `fj_`, writes the **request body’s `code`** to `main.py`, and runs it under Firejail. |
| Whose code is in `main.py`? | The **code string sent in the POST request** — i.e. **model-generated or dataset code** used for reward scoring, not this repo’s application code. |
| How could an SSH command appear? | If the **`code`** submitted to the sandbox (from model output or data) contained that SSH invocation, it would be written to `main.py` and executed inside the sandbox. Firejail is configured with `--net=none`; the SSH attempt might still spawn a process visible to the host/VPC flow depending on isolation. |

---

## 4. References in repo

- Sandbox API and Firejail flow: `sandbox/sandbox_api.py` (e.g. `_run_in_firejail`, `workdir = Path(tempfile.mkdtemp(prefix="fj_", dir="/dev/shm"))`, `src = workdir / "main.py"`, `src.write_text(code)`).
- Client and env: `sandbox/local_sandbox.py` (uses `SANDBOX_ENDPOINT`), `train_perturb.sh` (sets `SANDBOX_ENDPOINT`).
- Callers: `recipe/simpletir/workers/reward_manager/math_verify_with_exec.py`, `recipe/simpletir/utils/reward_score/code.py`, `recipe/simpletir/agent_utils.py` (they call `parallel_sandbox` when `SANDBOX_ENDPOINT` is set).
