#!/usr/bin/env python3
"""Local web dashboard for the autonomous Mac-to-GPU training loop."""

from __future__ import annotations

import fcntl
import json
import secrets
import socket
import subprocess
import sys
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from wildrobot.agents import remote_training_loop as remote


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTONOMOUS_SCRIPT = Path(__file__).with_name("autonomous_training_loop.py")
STATE_PATH = remote.LOCAL_JOB_ROOT / "autonomous_state.json"
LOCK_PATH = remote.LOCAL_JOB_ROOT / ".autonomous.lock"
SUPERVISOR_LOG_PATH = remote.LOCAL_JOB_ROOT / "web-supervisor.log"
MAX_REQUEST_BYTES = 64 * 1024


class WebActionError(RuntimeError):
    """A user-facing web action error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_cli(args: list[str], *, timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(AUTONOMOUS_SCRIPT), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )


def _command_output(result: subprocess.CompletedProcess[str]) -> str:
    output = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    return output or f"Command exited with status {result.returncode}."


def _load_state_optional() -> dict[str, Any] | None:
    if not STATE_PATH.is_file():
        return None
    try:
        return remote._read_json(STATE_PATH)
    except remote.TrainingLoopError:
        return None


def _mac_supervisor_running() -> bool:
    if not LOCK_PATH.is_file():
        return False
    with LOCK_PATH.open("r") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def _gpu_service_status(state: dict[str, Any] | None) -> dict[str, Any]:
    host = str((state or {}).get("host") or remote.DEFAULT_GPU_HOST)
    user = str((state or {}).get("user") or remote.DEFAULT_GPU_USER)
    port = (state or {}).get("port")
    target = f"{user}@{host}"
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=3",
    ]
    if port is not None:
        command.extend(["-p", str(int(port))])
    command.extend(
        [target, "systemctl --user is-active wildrobot-training-gpu.service"]
    )
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=6,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "host": host,
            "target": target,
            "reachable": False,
            "service": "unknown",
            "error": str(exc),
        }

    service = result.stdout.strip() or "unknown"
    reachable = result.returncode != 255
    return {
        "host": host,
        "target": target,
        "reachable": reachable,
        "service": service if reachable else "unreachable",
        "error": None if result.returncode == 0 else _command_output(result),
    }


def _tail_log(path: Path, *, lines: int) -> str:
    if not path.is_file():
        return ""
    data = path.read_bytes()[-256 * 1024 :]
    return "\n".join(data.decode("utf-8", errors="replace").splitlines()[-lines:])


def _text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, (str, int)):
        raise WebActionError(f"{key} must be text.")
    return str(value).strip()


def _integer(
    payload: dict[str, Any], key: str, default: int, *, minimum: int
) -> int:
    value = payload.get(key, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise WebActionError(f"{key} must be an integer.") from exc
    if parsed < minimum:
        raise WebActionError(f"{key} must be at least {minimum}.")
    return parsed


def _build_start_args(payload: dict[str, Any]) -> list[str]:
    config = _text(payload, "config")
    if not config:
        raise WebActionError("config is required.")

    mode = _text(payload, "start_mode") or "none"
    source = _text(payload, "source")
    args = ["start", "--config", config]
    if mode == "adopt_completed":
        args.append("--adopt-completed")
        if source:
            args.append(source)
    elif mode in {"init_policy", "resume"}:
        if not source:
            raise WebActionError(f"source is required for {mode}.")
        args.extend([f"--{mode.replace('_', '-')}", source])
    elif mode != "none":
        raise WebActionError(f"Unsupported start_mode: {mode}")

    training_git_sha = _text(payload, "training_git_sha")
    if training_git_sha:
        if mode != "adopt_completed":
            raise WebActionError(
                "training_git_sha is valid only when adopting a completed run."
            )
        args.extend(["--training-git-sha", training_git_sha])

    for key, option in (
        ("branch", "--branch"),
        ("gpu_host", "--host"),
        ("gpu_user", "--user"),
        ("remote_repo", "--remote-repo"),
    ):
        value = _text(payload, key)
        if value:
            args.extend([option, value])

    port = _text(payload, "gpu_port")
    if port:
        port_number = _integer(payload, "gpu_port", 22, minimum=1)
        if port_number > 65535:
            raise WebActionError("gpu_port must be at most 65535.")
        args.extend(["--port", str(port_number)])

    args.extend(
        [
            "--max-cycles",
            str(_integer(payload, "max_cycles", 20, minimum=1)),
        ]
    )

    standing_checkpoint = _text(payload, "standing_checkpoint")
    standing_config = _text(payload, "standing_config")
    if bool(standing_checkpoint) != bool(standing_config):
        raise WebActionError(
            "standing_checkpoint and standing_config must be provided together."
        )
    if standing_checkpoint:
        args.extend(["--standing-checkpoint", standing_checkpoint])
        args.extend(["--standing-config", standing_config])
    if bool(payload.get("new_run", False)):
        args.append("--new-run")
    return args


class TrainingLoopWebController:
    def __init__(self) -> None:
        self._action_lock = threading.Lock()

    def status(self, *, last: int = 5) -> dict[str, Any]:
        if last < 1 or last > 100:
            raise WebActionError("last must be between 1 and 100.")
        state = _load_state_optional()
        try:
            result = _run_cli(
                ["status", "--last", str(last), "--json"], timeout_s=15
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            loop: dict[str, Any] = {
                "status": "unavailable",
                "error": str(exc),
                "recent_cycles": [],
            }
        else:
            if result.returncode == 0:
                try:
                    loop = json.loads(result.stdout)
                except json.JSONDecodeError:
                    loop = {
                        "status": "unavailable",
                        "error": "Status command returned invalid JSON.",
                        "recent_cycles": [],
                    }
            else:
                loop = {
                    "status": "not_started" if state is None else "unavailable",
                    "error": _command_output(result),
                    "recent_cycles": [],
                }

        defaults = {
            "config": str((state or {}).get("active_config") or ""),
            "branch": str((state or {}).get("branch") or "main"),
            "gpu_host": str((state or {}).get("host") or remote.DEFAULT_GPU_HOST),
            "gpu_user": str((state or {}).get("user") or remote.DEFAULT_GPU_USER),
            "gpu_port": (state or {}).get("port"),
            "remote_repo": str(
                (state or {}).get("remote_repo") or remote.DEFAULT_REMOTE_REPO
            ),
            "max_cycles": int((state or {}).get("max_cycles") or 20),
            "standing_checkpoint": str(
                (state or {}).get("standing_checkpoint") or ""
            ),
            "standing_config": str((state or {}).get("standing_config") or ""),
        }
        return {
            "timestamp": _utc_now(),
            "loop": loop,
            "mac": {
                "host": socket.gethostname(),
                "online": True,
                "supervisor_running": bool(loop.get("mac_supervisor_running")),
            },
            "gpu": {
                **_gpu_service_status(state),
                "job_status": loop.get("gpu_job_status", "unknown"),
            },
            "start_defaults": defaults,
            "supervisor_log": _tail_log(SUPERVISOR_LOG_PATH, lines=120),
        }

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        args = _build_start_args(payload)
        with self._action_lock:
            try:
                result = _run_cli(args, timeout_s=120)
            except subprocess.TimeoutExpired as exc:
                raise WebActionError(
                    "Start timed out. The durable state may still be resumable with Run."
                ) from exc
            if result.returncode:
                raise WebActionError(_command_output(result))
        return {"ok": True, "message": _command_output(result)}

    def run(self) -> dict[str, Any]:
        with self._action_lock:
            if not STATE_PATH.is_file():
                raise WebActionError("No loop state exists. Use Start first.")
            if _mac_supervisor_running():
                return {"ok": True, "message": "Mac supervisor is already running."}
            SUPERVISOR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with SUPERVISOR_LOG_PATH.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [sys.executable, str(AUTONOMOUS_SCRIPT), "run"],
                    cwd=REPO_ROOT,
                    text=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        return {
            "ok": True,
            "message": f"Mac supervisor started with PID {process.pid}.",
        }

    def stop(self) -> dict[str, Any]:
        with self._action_lock:
            try:
                result = _run_cli(
                    ["stop", "--reason", "paused from training-loop web UI"],
                    timeout_s=30,
                )
            except subprocess.TimeoutExpired as exc:
                raise WebActionError("Stop request timed out.") from exc
            if result.returncode:
                raise WebActionError(_command_output(result))
        return {"ok": True, "message": _command_output(result)}


_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>WildRobot Training Loop</title>
  <style>
    :root { color-scheme: dark; --bg:#0b1118; --panel:#121b26; --line:#26384a;
      --text:#e8f0f7; --muted:#93a7b8; --good:#4fd1a1; --warn:#f7c96b;
      --bad:#ff7a86; --accent:#67b7ff; }
    * { box-sizing: border-box; }
    body { margin:0; background:linear-gradient(145deg,#081019,#101a24); color:var(--text);
      font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; }
    main { max-width:1180px; margin:0 auto; padding:28px 20px 60px; }
    h1 { margin:0 0 4px; font-size:25px; } h2 { font-size:16px; margin:0 0 14px; }
    .muted { color:var(--muted); } .grid { display:grid; gap:14px;
      grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); margin:22px 0; }
    .panel { background:rgba(18,27,38,.94); border:1px solid var(--line);
      border-radius:12px; padding:16px; box-shadow:0 12px 30px rgba(0,0,0,.18); }
    .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }
    .value { font-size:18px; margin-top:5px; overflow-wrap:anywhere; }
    .good { color:var(--good); } .warn { color:var(--warn); } .bad { color:var(--bad); }
    .actions { display:flex; flex-wrap:wrap; gap:10px; margin:14px 0; }
    button { border:1px solid var(--line); border-radius:8px; padding:9px 14px;
      background:#1a2a3a; color:var(--text); cursor:pointer; font:inherit; }
    button.primary { background:#145c8f; border-color:#267fba; }
    button.stop { background:#672733; border-color:#a54250; }
    button:hover { filter:brightness(1.13); } button:disabled { opacity:.55; cursor:wait; }
    details { margin:16px 0; } summary { cursor:pointer; color:var(--accent); }
    form { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; margin-top:14px; }
    label { color:var(--muted); display:flex; flex-direction:column; gap:5px; }
    label.wide { grid-column:1/-1; } input,select { width:100%; background:#0b141d;
      color:var(--text); border:1px solid var(--line); border-radius:7px; padding:8px; font:inherit; }
    .check { flex-direction:row; align-items:center; } .check input { width:auto; }
    table { width:100%; border-collapse:collapse; font-size:12px; }
    th,td { border-bottom:1px solid var(--line); padding:9px 7px; text-align:left; vertical-align:top; }
    th { color:var(--muted); } td { overflow-wrap:anywhere; }
    pre { background:#080d12; border:1px solid var(--line); padding:12px; border-radius:8px;
      max-height:330px; overflow:auto; white-space:pre-wrap; font-size:12px; }
    #notice { min-height:22px; margin:8px 0; } @media(max-width:700px){form{grid-template-columns:1fr}}
  </style>
</head>
<body><main>
  <h1>WildRobot Training Loop</h1>
  <div class="muted">Mac supervisor ↔ GPU worker · refreshes every 10 seconds</div>
  <div class="grid">
    <section class="panel"><div class="label">Loop</div><div id="loopStatus" class="value">Loading…</div>
      <div id="loopDetail" class="muted"></div></section>
    <section class="panel"><div class="label">Mac</div><div id="macStatus" class="value">Loading…</div>
      <div id="macDetail" class="muted"></div></section>
    <section class="panel"><div class="label">GPU Linux</div><div id="gpuStatus" class="value">Loading…</div>
      <div id="gpuDetail" class="muted"></div></section>
    <section class="panel"><div class="label">Stability champion</div><div id="championStatus" class="value">Loading…</div>
      <div id="championDetail" class="muted"></div></section>
  </div>

  <section class="panel">
    <h2>Actions</h2>
    <div class="actions">
      <button id="runButton" class="primary">Run / Resume</button>
      <button id="stopButton" class="stop">Stop / Pause</button>
      <button id="refreshButton">Refresh</button>
    </div>
    <div id="notice" class="muted"></div>
    <details>
      <summary>Start a new loop</summary>
      <form id="startForm">
        <label class="wide">Training config<input name="config" required></label>
        <label>Start mode<select name="start_mode">
          <option value="adopt_completed">Adopt completed run</option>
          <option value="init_policy">Initialize from policy</option>
          <option value="resume">Resume checkpoint</option>
          <option value="none">Train from scratch</option>
        </select></label>
        <label>Run name or checkpoint<input name="source" placeholder="offline-run-… or checkpoint.pkl"></label>
        <label class="wide">Training Git SHA<input name="training_git_sha" placeholder="Required for an older adopted run"></label>
        <label>Maximum cycles<input name="max_cycles" type="number" min="1" value="20"></label>
        <label>GPU host<input name="gpu_host"></label><label>GPU user<input name="gpu_user"></label>
        <label>GPU SSH port<input name="gpu_port" type="number" min="1" max="65535"></label>
        <label>Git branch<input name="branch" value="main"></label>
        <label class="wide">Remote repository<input name="remote_repo"></label>
        <label class="wide">Standing checkpoint<input name="standing_checkpoint"></label>
        <label class="wide">Standing training config<input name="standing_config"></label>
        <label class="check"><input name="new_run" type="checkbox" checked> Replace a stopped/paused loop</label>
        <div><button class="primary" type="submit">Start</button></div>
      </form>
    </details>
  </section>

  <section class="panel" style="margin-top:14px"><h2>Recent cycles</h2>
    <div style="overflow:auto"><table><thead><tr><th>Cycle</th><th>Status</th><th>Config</th><th>Result</th><th>Family</th><th>Next change</th></tr></thead>
      <tbody id="history"></tbody></table></div></section>
  <section class="panel" style="margin-top:14px"><h2>Mac supervisor log</h2><pre id="log">No log yet.</pre></section>
</main>
<script>
const token=__TOKEN__;
const $=id=>document.getElementById(id);
let defaultsApplied=false;
function text(el,value){el.textContent=value==null?'—':String(value)}
function tone(el,value){el.className='value '+(value==='active'||value==='ready'||value===true?'good':value==='stopped_error'||value==='unreachable'||value===false?'bad':'warn')}
async function request(path,options={}){
  const response=await fetch(path,{...options,headers:{'Content-Type':'application/json','X-WildRobot-Token':token,...(options.headers||{})}});
  const data=await response.json(); if(!response.ok) throw new Error(data.error||`HTTP ${response.status}`); return data;
}
function applyDefaults(values){if(defaultsApplied)return; defaultsApplied=true; const form=$('startForm');
  for(const [key,value] of Object.entries(values||{})){const field=form.elements.namedItem(key); if(field&&value!==null&&value!=='')field.value=value;}}
function render(data){const loop=data.loop||{},mac=data.mac||{},gpu=data.gpu||{};
  text($('loopStatus'),loop.status||'unknown'); tone($('loopStatus'),loop.status);
  text($('loopDetail'),`${loop.stage||'—'} on ${loop.stage_machine||'—'} · cycle ${loop.cycle??'—'}/${loop.max_cycles??'—'}`);
  text($('macStatus'),mac.supervisor_running?'supervisor running':'supervisor stopped'); tone($('macStatus'),mac.supervisor_running);
  text($('macDetail'),mac.host||'—'); text($('gpuStatus'),`${gpu.service||'unknown'} · job ${gpu.job_status||'unknown'}`); tone($('gpuStatus'),gpu.reachable&&gpu.service==='active');
  const champion=loop.champion||{},metrics=champion.metrics||{}; text($('championStatus'),champion.checkpoint_path?champion.checkpoint_path.split('/').pop():'not established');
  text($('championDetail'),champion.checkpoint_path?`falls ${metrics.walking_fall_env_count??'—'} · saturation ${metrics.walking_stable_max_actuator_torque_sat_frac==null?'—':(100*metrics.walking_stable_max_actuator_torque_sat_frac).toFixed(2)+'%'}`:'waiting for a comparable screen');
  text($('gpuDetail'),gpu.error||gpu.target||'—'); const body=$('history'); body.replaceChildren();
  for(const item of loop.recent_cycles||[]){const tr=document.createElement('tr'); for(const value of [item.cycle,item.status,item.config,item.training_result,item.experiment_family||'—',item.next_patch||'—']){const td=document.createElement('td'); td.textContent=value??'—'; tr.appendChild(td)} body.appendChild(tr)}
  text($('log'),data.supervisor_log||'No log yet.'); applyDefaults(data.start_defaults);
}
async function refresh(){try{render(await request('/api/status?last=10'))}catch(error){text($('notice'),error.message)}}
async function action(path,payload={}){for(const button of document.querySelectorAll('button'))button.disabled=true;
  try{const result=await request(path,{method:'POST',body:JSON.stringify(payload)}); text($('notice'),result.message); await refresh()}
  catch(error){text($('notice'),error.message)} finally{for(const button of document.querySelectorAll('button'))button.disabled=false}}
$('runButton').onclick=()=>action('/api/run'); $('stopButton').onclick=()=>action('/api/stop'); $('refreshButton').onclick=refresh;
$('startForm').onsubmit=event=>{event.preventDefault(); const form=new FormData(event.target); const payload=Object.fromEntries(form.entries()); payload.new_run=form.has('new_run'); action('/api/start',payload)};
async function poll(){await refresh(); setTimeout(poll,10000)} poll();
</script></body></html>"""


def _make_handler(controller: TrainingLoopWebController, token: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status.value)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'unsafe-inline'; "
                "script-src 'unsafe-inline'",
            )
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            self._send(
                status,
                (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = _HTML.replace("__TOKEN__", json.dumps(token)).encode("utf-8")
                self._send(HTTPStatus.OK, body, "text/html; charset=utf-8")
                return
            if parsed.path == "/api/status":
                try:
                    query = parse_qs(parsed.query)
                    last = int(query.get("last", ["5"])[0])
                    self._json(HTTPStatus.OK, controller.status(last=last))
                except (ValueError, WebActionError) as exc:
                    self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return
            self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found."})

        def do_POST(self) -> None:
            if self.headers.get("X-WildRobot-Token") != token:
                self._json(
                    HTTPStatus.FORBIDDEN,
                    {"ok": False, "error": "Invalid request token."},
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > MAX_REQUEST_BYTES:
                    raise WebActionError("Request body is too large.")
                payload = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(payload, dict):
                    raise WebActionError("Request body must be a JSON object.")
                action = {
                    "/api/start": lambda: controller.start(payload),
                    "/api/run": controller.run,
                    "/api/stop": controller.stop,
                }.get(urlparse(self.path).path)
                if action is None:
                    self._json(
                        HTTPStatus.NOT_FOUND,
                        {"ok": False, "error": "Not found."},
                    )
                    return
                self._json(HTTPStatus.OK, action())
            except (json.JSONDecodeError, ValueError, WebActionError) as exc:
                self._json(HTTPStatus.CONFLICT, {"ok": False, "error": str(exc)})
            except OSError as exc:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})

    return Handler


def serve(*, host: str, port: int) -> int:
    if not 1 <= port <= 65535:
        raise WebActionError("port must be between 1 and 65535.")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            "WARNING: the training-loop UI has no user authentication; expose it "
            "only on a trusted network.",
            flush=True,
        )
    token = secrets.token_urlsafe(32)
    server = ThreadingHTTPServer(
        (host, port), _make_handler(TrainingLoopWebController(), token)
    )
    print(f"WildRobot training-loop UI: http://{host}:{port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nTraining-loop UI stopped.", flush=True)
    finally:
        server.server_close()
    return 0
