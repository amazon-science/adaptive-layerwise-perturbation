"""
Minimal Firejail-based sandbox HTTP API.
Dependencies:
    pip install "fastapi[all]" uvicorn
System prerequisites:
    * firejail installed (sudo apt install firejail)
    * a non-login user called `sandboxer` (sudo useradd -r -M -s /usr/sbin/nologin sandboxer)
    * a Firejail profile at /etc/firejail/sandbox.profile such as:
        # /etc/firejail/sandbox.profile
        env none                    # start with a clean env
        env keep PATH
        env keep LANG
        env keep PYTHONIOENCODING
        net none                    # disable networking
        private                     # private filesystem rooted at --private dir
        rlimit as 512M              # memory cap
        rlimit cpu 3                # cpu-time cap
        rlimit nproc 50             # process count cap
        caps.drop all               # drop all capabilities
        seccomp
        whitelist /usr/bin/python3
        include /etc/firejail/whitelist-common.inc

Run the service with:
    uvicorn sandbox_api:app --host 0.0.0.0 --port 8000 --workers 4

The API is compatible with the `RunCodeRequest` / `RunResult` schema used by
ByteIntl Seed-Sandbox. A simple curl test:
    curl -X POST http://127.0.0.1:8000/faas/sandbox/ \
         -H 'Content-Type: application/json' \
         -d '{"code":"print(2+2)","language":"python","compile_timeout":1,"run_timeout":3}'
"""

import asyncio
import os
import shutil
import signal
import subprocess
import tempfile
from datetime import datetime
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------- Pydantic models ----------------

class RunStatus(str, Enum):
    """Execution outcome."""
    success = "success"
    timeout = "timeout"
    runtime_error = "runtime_error"


class RunCodeRequest(BaseModel):
    """Incoming JSON body from the client."""

    code: str
    stdin: str = ""
    language: str = "python"
    compile_timeout: float = 1.0  # kept for sdk compatibility, unused here
    run_timeout: float = 3.0


class RunResult(BaseModel):
    """JSON response back to the client."""

    status: RunStatus
    run_result: dict
    created_at: datetime


# ---------------- Core runner ----------------

async def _run_in_firejail(code: str, timeout: float, stdin_data: str = "") -> dict:
    """Execute *code* inside a fresh Firejail sandbox and return stdout/stderr.
    
    Enhanced with multiple layers of protection against runaway processes:
    1. Firejail's built-in resource limits and timeout
    2. Process group management for proper cleanup
    3. Multi-stage termination (SIGTERM → SIGKILL)
    4. Asyncio timeout as final safeguard
    """

    # 1) Write user code to a tmpfs directory → zero-copy, fast cleanup
    workdir = Path(tempfile.mkdtemp(prefix="fj_", dir="/dev/shm"))
    src = workdir / "main.py"
    src.write_text(code)

    # 2) Build Firejail command line with enhanced protections
    # Add --timeout as a hard kill timer (wall-clock time)
    hard_timeout = int(timeout) + 5  # Give extra buffer for graceful shutdown
    cmd = [
        "firejail",
        "--quiet",
        "--profile=/etc/firejail/default.profile",
        f"--private={workdir}",
        "--net=none",              # disable network
        f"--timeout=00:00:{hard_timeout}",  # HARD timeout: firejail will kill tree after this
        # Resource limits
        "--rlimit-as=2048m",       # 2GB max RAM per sandbox
        f"--rlimit-cpu={int(timeout) + 2}",  # CPU time limit
        "--rlimit-nproc=256",      # Max processes inside sandbox
        "--",
        "python3",
        src.name,
    ]

    # 3) Strip environment to stay below Firejail's MAX_ENVS=256 limit
    whitelist = ("PATH", "LANG", "LC_ALL", "PYTHONIOENCODING", "TERM")
    clean_env = {k: os.environ[k] for k in whitelist if k in os.environ}

    # 4) Launch subprocess with process group for proper cleanup
    # start_new_session=True ensures we can kill the entire process tree
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=workdir,
        env=clean_env,
        start_new_session=True,  # Create new process group
    )

    try:
        input_bytes = (stdin_data + "\n").encode() if len(stdin_data) > 0 else None
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=input_bytes),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        # Multi-stage cleanup to ensure complete process termination
        await _cleanup_process_tree(proc)
        shutil.rmtree(workdir, ignore_errors=True)
        return {
            "status": RunStatus.timeout,
            "stdout": "",
            "stderr": "Timeout\n",
        }
    finally:
        # Extra safety: ensure process is cleaned up even if communicate() succeeds
        # but process is still running (shouldn't happen, but defensive programming)
        if proc.returncode is None:
            try:
                await _cleanup_process_tree(proc)
            except Exception:
                pass

    status = RunStatus.success if proc.returncode == 0 else RunStatus.runtime_error

    # 5) Clean up tmpfs directory
    shutil.rmtree(workdir, ignore_errors=True)

    return {
        "status": status,
        "stdout": stdout.decode(),
        "stderr": stderr.decode(),
    }


async def _cleanup_process_tree(proc: asyncio.subprocess.Process) -> None:
    """
    Aggressively cleanup a process and its entire tree.
    Multi-stage approach:
    1. SIGTERM to process group (graceful)
    2. Wait briefly
    3. SIGKILL to process group (force)
    4. SIGKILL to process itself (fallback)
    """
    if proc.returncode is not None:
        # Already terminated
        return
    
    try:
        # Stage 1: Try graceful termination of entire process group
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            # Process or group doesn't exist, try process directly
            try:
                proc.terminate()
            except ProcessLookupError:
                return
        
        # Stage 2: Wait briefly for graceful shutdown (0.5 seconds)
        try:
            await asyncio.wait_for(proc.wait(), timeout=0.5)
            return  # Success!
        except asyncio.TimeoutError:
            pass  # Continue to force kill
        
        # Stage 3: Force kill the entire process group
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        
        # Stage 4: Force kill the process itself as final fallback
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        
        # Wait for cleanup with timeout
        try:
            await asyncio.wait_for(proc.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            # Process is really stuck, but we've done all we can
            pass
            
    except Exception as e:
        # Ultimate fallback: just try to kill the process
        try:
            proc.kill()
            await asyncio.wait_for(proc.wait(), timeout=1.0)
        except Exception:
            pass  # Nothing more we can do


# ---------------- FastAPI wiring ----------------

app = FastAPI()
POOL = asyncio.Semaphore(20)  # 20 per worker * 8 workers = 160 total concurrent sandboxes


@app.post("/faas/sandbox/", response_model=RunResult)
async def run_code(req: RunCodeRequest):
    """HTTP endpoint: compatible with the Seed-Sandbox client SDK."""

    if req.language != "python":
        raise HTTPException(400, "Only Python is supported in this minimal demo.")

    async with POOL:
        result = await _run_in_firejail(req.code, req.run_timeout, req.stdin)

    return RunResult(status=result["status"], run_result=result, created_at=datetime.utcnow())
