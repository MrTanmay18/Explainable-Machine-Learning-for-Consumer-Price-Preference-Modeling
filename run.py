"""Start the FastAPI backend and the Streamlit frontend together (no Docker).

    python run.py

API:      http://127.0.0.1:8000  (docs at /docs)
Frontend: http://127.0.0.1:8501
Press Ctrl+C to stop both.
"""
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API_HOST, API_PORT, UI_PORT = "127.0.0.1", int(os.getenv("API_PORT", 8000)), int(os.getenv("UI_PORT", 8501))


def wait_for_api(timeout: int = 90) -> bool:
    for _ in range(timeout):
        try:
            urllib.request.urlopen(f"http://{API_HOST}:{API_PORT}/health", timeout=1)
            return True
        except Exception:
            time.sleep(1)
    return False


def main() -> None:
    env = {**os.environ, "API_URL": f"http://{API_HOST}:{API_PORT}", "PYTHONPATH": str(ROOT)}
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", API_HOST, "--port", str(API_PORT)],
        cwd=ROOT, env=env,
    )
    ui = None
    try:
        if not wait_for_api():
            print("API did not become healthy in time - check the log above.")
            return
        ui = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "streamlit_app/app.py", "--server.port", str(UI_PORT)],
            cwd=ROOT, env=env,
        )
        ui.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in (ui, api):
            if p and p.poll() is None:
                p.terminate()


if __name__ == "__main__":
    main()
