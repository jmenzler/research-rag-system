"""MinerU PDF parser — pipeline + VLM fallback.

Pipeline mode handles standard PDFs with a parseable text layer.
VLM fallback handles PDFs where the text layer is corrupt (U+FFFD glyphs from
broken font encoding tables); reads the rendered page image instead.

MinerU lives in an isolated venv (`.venv-mineru/`) because `mineru[pipeline]`
requires `transformers<5.0.0` while the main venv's `mlx-vlm` needs `>=5.1.0`.

The VLM API server is started lazily on first VLM call and reused across all
subsequent calls in the same process; cleaned up at exit via atexit.

Formula recognition operates per formula rather than per page. Math-heavy
documents can be expensive to parse. Any U+FFFD glyph triggers the fallback.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Bin paths — MinerU lives in an isolated venv (transformers conflict with main)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MINERU_BIN = _REPO_ROOT / ".venv-mineru" / "bin" / "mineru"
_MINERU_API_BIN = _REPO_ROOT / ".venv-mineru" / "bin" / "mineru-api"
_MINERU_VLLM_BIN = _REPO_ROOT / ".venv-mineru" / "bin" / "mineru-vllm-server"

_VENV_BIN = _REPO_ROOT / ".venv-mineru" / "bin"

_BASE_ENV: dict[str, str] = {**os.environ}
if sys.platform == "darwin":
    # Homebrew Python 3.12 links a newer libexpat than /usr/lib/libexpat.1.dylib;
    # mineru's pypdf import fails without this.
    _BASE_ENV["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/expat/lib"

# Two VLM server topologies, per MinerU docs (extension_modules.md):
#   Linux + vllm: mineru-vllm-server (OpenAI-compatible) + client `-b vlm-http-client -u <url>`
#   macOS:        mineru-api (mineru's own FastAPI)      + client `-b vlm-auto-engine --api-url <url>`  # noqa: E501
# The two flag forms (`-u` vs `--api-url`) target different protocols.
_USE_VLLM = sys.platform != "darwin" and _MINERU_VLLM_BIN.exists()

_VLM_ENV: dict[str, str] = {**_BASE_ENV}
if sys.platform == "darwin":
    # MPS path: force batch_size=1 to avoid OOM with the transformers engine.
    _VLM_ENV["MINERU_VIRTUAL_VRAM_SIZE"] = "4"
elif _USE_VLLM:
    # vllm uses flashinfer for attention, which JIT-compiles CUDA kernels at
    # startup via ninja + nvcc. Prepend venv bin so ninja is on PATH; set
    # CUDA_HOME for Ubuntu's distro nvcc (apt nvidia-cuda-toolkit → /usr/bin).
    _VLM_ENV["PATH"] = f"{_VENV_BIN}:{_BASE_ENV.get('PATH', '')}"
    _VLM_ENV.setdefault("CUDA_HOME", "/usr")
    # vllm sleep mode requires DEV mode flag — lets us POST /sleep?level=1
    # to offload weights to CPU and free VRAM between VLM calls (so the
    # pipeline backend doesn't compete with vllm for the 8GB on RTX 2060).
    _VLM_ENV["VLLM_SERVER_DEV_MODE"] = "1"


# ---------------------------------------------------------------------------
# Lazy VLM server (singleton per process)
# ---------------------------------------------------------------------------

_vlm_proc: subprocess.Popen[bytes] | None = None
_vlm_port: int | None = None
_vlm_lock = threading.Lock()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_vlm_api(url: str, timeout: float = 600.0) -> None:
    """Poll /health until 200. vllm load+compile can take 1–3 min on first start."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2):
                return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"VLM API at {url} did not become healthy within {timeout}s")


def _shutdown_vlm_server() -> None:
    """Kill the vllm server process *group* — vllm's EngineCore worker is a
    separate process that survives plain SIGTERM and orphans to init,
    accumulating across runs and pinning GPU memory."""
    global _vlm_proc
    with _vlm_lock:
        if _vlm_proc is None or _vlm_proc.poll() is not None:
            _vlm_proc = None
            return
        try:
            pgid = os.getpgid(_vlm_proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            _vlm_proc.terminate()
        try:
            _vlm_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                pgid = os.getpgid(_vlm_proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                _vlm_proc.kill()
        _vlm_proc = None


def _ensure_vlm_server() -> str:
    """Start the VLM API server if not running; return its base URL.

    On Linux/CUDA (vllm installed): mineru-vllm-server, OpenAI-compatible.
    Elsewhere:                     mineru-api with mineru's protocol.
    The `parse_vlm_only` client uses `-u` vs `--api-url` to match.
    """
    global _vlm_proc, _vlm_port
    with _vlm_lock:
        if _vlm_proc is not None and _vlm_proc.poll() is None:
            return f"http://127.0.0.1:{_vlm_port}"
        port = _free_port()
        if _USE_VLLM:
            cmd = [str(_MINERU_VLLM_BIN), "--port", str(port), "--enable-sleep-mode"]
            logger.info("Starting MinerU vLLM server on port %d (sleep-mode)…", port)
        else:
            cmd = [str(_MINERU_API_BIN), "--enable-vlm-preload", "true", "--port", str(port)]
            logger.info("Starting MinerU transformers/mlx VLM server on port %d…", port)
        log_dir = _REPO_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"vlm_server_{int(time.time())}_{port}.log"
        log_handle = log_path.open("w")
        logger.info("VLM server stdout/stderr → %s", log_path)
        proc = subprocess.Popen(
            cmd, env=_VLM_ENV, stdout=log_handle, stderr=subprocess.STDOUT,
            # New session so the engine workers vllm spawns are in our group
            # and killpg in _shutdown_vlm_server can take them all down.
            start_new_session=True,
        )
        _vlm_proc = proc
        _vlm_port = port
        atexit.register(_shutdown_vlm_server)
    url = f"http://127.0.0.1:{port}"
    _wait_for_vlm_api(url)
    vlm_wake()
    logger.info("VLM server ready at %s", url)
    return url


# ---------------------------------------------------------------------------
# Backend invocations
# ---------------------------------------------------------------------------


def _persist_out_dir(pdf: Path, backend: str) -> Path:
    """Wipe and recreate the per-backend MinerU output dir next to the PDF.

    Layout produced by MinerU:
        <pdf.parent>/mineru_<backend>/<pdf.stem>/{auto|vlm}/{*.md, images/, ...}
    """
    out_dir = pdf.parent / f"mineru_{backend}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    return out_dir


def _multipart_post(url: str, fields: dict[str, str], file_field: str,
                    file_path: Path, timeout: float = 600.0) -> dict[str, Any]:
    """POST multipart/form-data with one file attachment. Returns parsed JSON body."""
    boundary = uuid.uuid4().hex
    body = bytearray()

    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(f"{value}\r\n".encode())

    filename = file_path.name
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        f'Content-Disposition: form-data; name="{file_field}"; '
        f'filename="{filename}"\r\n'.encode()
    )
    body.extend(b"Content-Type: application/pdf\r\n\r\n")
    body.extend(file_path.read_bytes())
    body.extend(f"\r\n--{boundary}--\r\n".encode())

    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    req = urllib.request.Request(url, data=bytes(body), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())  # type: ignore[no-any-return]


def _write_content_list_sidecar(path: Path, content_list: Any) -> None:  # noqa: ANN401
    """Persist MinerU's structured content_list beside the source PDF.

    `content_list.json` holds typed elements (text with text_level, equation,
    table with full HTML body + caption, list, header, page_footnote, page_number,
    chart) plus bbox + page_idx. The markdown loses tables, page numbers, and
    element typing; the JSON is what the structural chunker needs.

    The daemon returns `content_list` already serialized as a JSON string, so
    re-encoding via json.dumps would double-wrap it. Write strings verbatim,
    serialize only when given a list/dict.
    """
    if not content_list:
        return
    sidecar = path.parent / "content_list.json"
    try:
        if isinstance(content_list, str):
            sidecar.write_text(content_list)
        else:
            sidecar.write_text(json.dumps(content_list, ensure_ascii=False))
    except Exception as e:
        logger.warning("failed to write content_list sidecar for %s: %s", path, e)


def parse_pipeline_only(path: Path) -> str:
    """Run only the pipeline backend; raise on missing output. No VLM fallback.

    Always goes through ``mineru_daemon`` (and therefore through the gpu-broker
    daemon's claim). The previous CLI-subprocess fallback bypassed the broker
    entirely and OOM'd against the bare GPU whenever the reranker was loaded;
    callers MUST invoke ``mineru_daemon.start_pipeline_daemon()`` first.

    Side effect: writes ``path.parent / "content_list.json"`` (typed structural
    output used by the structural chunker).
    """
    from src.pdf_parsers import mineru_daemon  # noqa: PLC0415

    daemon_url = mineru_daemon.pipeline_daemon_url()
    if daemon_url is None:
        raise RuntimeError(
            "MinerU pipeline daemon not running. Callers must invoke "
            "mineru_daemon.start_pipeline_daemon() before parse_pipeline_only(). "
            "All GPU work is gated by the gpu-broker via the daemon-level claim."
        )

    resp = _multipart_post(
        f"{daemon_url}/file_parse",
        fields={
            "backend": "pipeline",
            "lang_list": "en",
            "return_md": "true",
            "return_content_list": "true",
        },
        file_field="files",
        file_path=path,
    )
    results: dict[str, Any] = resp.get("results", {})
    if not results:
        raise RuntimeError(f"MinerU daemon returned no results for {path}")
    first_result: dict[str, Any] = next(iter(results.values()))
    md_content: str = first_result.get("md_content", "")
    if not md_content:
        raise RuntimeError(f"MinerU daemon returned empty markdown for {path}")
    _write_content_list_sidecar(path, first_result.get("content_list"))
    return md_content


def _vllm_post(path: str) -> bool:
    """POST a no-body request to the running vllm server. Returns success bool."""
    if _vlm_port is None:
        return False
    url = f"http://127.0.0.1:{_vlm_port}{path}"
    try:
        req = urllib.request.Request(url, method="POST", data=b"")
        with urllib.request.urlopen(req, timeout=60) as resp:
            status: int = resp.status
            return 200 <= status < 300
    except Exception as e:
        logger.warning("vllm POST %s failed: %s", path, e)
        return False


def vlm_sleep() -> None:
    """Offload vllm weights to CPU and discard KV cache; frees ~6 GB VRAM.

    Use between VLM calls when the pipeline backend needs the GPU. Wake before
    the next inference call. No-op on macOS / when vllm isn't the engine.
    """
    if _USE_VLLM and _vlm_proc is not None and _vlm_proc.poll() is None:
        if _vllm_post("/sleep?level=1"):
            logger.info("vllm asleep")


def vlm_wake() -> None:
    """Reload vllm weights from CPU back to GPU. ~few seconds, much faster
    than a cold start. No-op on macOS / when vllm isn't the engine."""
    if _USE_VLLM and _vlm_proc is not None and _vlm_proc.poll() is None:
        if _vllm_post("/wake_up"):
            logger.info("vllm awake")


def parse_vlm_only(path: Path) -> str:
    """Run only the VLM backend (slow; routed through the lazy singleton server).

    Side effect: writes MinerU output to `path.parent / "mineru_vlm/"`.
    On Linux/CUDA/vllm: wakes the engine before inference, sleeps it after,
    so the pipeline backend has full VRAM during pipeline-only PDFs.
    """
    server_url = _ensure_vlm_server()
    vlm_wake()
    out_dir = _persist_out_dir(path, "vlm")
    # Per MinerU docs: `-u` targets OpenAI-compatible servers (vllm/lmdeploy),
    # `--api-url` targets mineru-api's protocol. They are not interchangeable.
    if _USE_VLLM:
        url_flag = ["-u", server_url, "-b", "vlm-http-client"]
    else:
        url_flag = ["--api-url", server_url, "-b", "vlm-auto-engine"]
    try:
        subprocess.run(
            [str(_MINERU_BIN), "-p", str(path.resolve()), "-o", str(out_dir),
             *url_flag, "-l", "en"],
            env=_VLM_ENV,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    finally:
        vlm_sleep()
    md_files = sorted(out_dir.rglob("*.md"))
    if not md_files:
        raise RuntimeError(f"MinerU VLM produced no output for {path}")
    cl_files = sorted(out_dir.rglob("*_content_list.json"))
    if cl_files:
        try:
            (path.parent / "content_list.json").write_text(cl_files[0].read_text())
        except Exception as e:
            logger.warning("failed to copy content_list sidecar for %s: %s", path, e)
    return md_files[0].read_text()


def has_broken_fonts(text: str) -> bool:
    """Detect corrupt font encoding in pipeline output (U+FFFD replacement glyphs).

    Currently fires on any single occurrence — known to over-trigger on math-heavy
    docs where pipeline already captured most formulas correctly. See KB.
    """
    return "�" in text


def parse_pdf(path: Path) -> str:
    """Default PDF parser: pipeline → VLM fallback if pipeline output looks corrupt."""
    text = parse_pipeline_only(path)
    if has_broken_fonts(text):
        logger.warning("Broken font encoding in %s — retrying with VLM", path.name)
        return parse_vlm_only(path)
    return text


__all__ = [
    "parse_pdf",
    "parse_pipeline_only",
    "parse_vlm_only",
    "has_broken_fonts",
]
