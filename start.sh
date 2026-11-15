#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

pick_python() {
  if [[ -n "${PYTHON_BIN:-}" ]] && command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "${PYTHON_BIN}"
    return 0
  fi
  for candidate in \
    "${HOME}/miniconda3/envs/study_env/bin/python" \
    "${HOME}/anaconda3/envs/study_env/bin/python" \
    "${ROOT_DIR}/.venv/bin/python" \
    "$(command -v python3 || true)"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

ensure_node() {
  if command -v npm >/dev/null 2>&1; then
    return 0
  fi
  export NVM_DIR="${NVM_DIR:-${HOME}/.nvm}"
  if [[ -s "${NVM_DIR}/nvm.sh" ]]; then
    # shellcheck disable=SC1091
    source "${NVM_DIR}/nvm.sh"
  fi
  command -v npm >/dev/null 2>&1
}

PYTHON_EXE="$(pick_python || true)"
if [[ -z "${PYTHON_EXE}" ]]; then
  echo "Error: Could not find a Python interpreter. Set PYTHON_BIN=/path/to/python." >&2
  exit 1
fi

if ! ensure_node; then
  echo "Error: npm was not found on PATH (nor via \$NVM_DIR/nvm.sh). Please install Node.js/npm." >&2
  exit 1
fi

echo "Using Python: ${PYTHON_EXE}"
echo "Using npm:    $(command -v npm)"

if ! "${PYTHON_EXE}" -c "import importlib.util as u; mods=['fastapi','uvicorn','PIL','websockets']; assert all(u.find_spec(m) for m in mods) and (u.find_spec('python_multipart') or u.find_spec('multipart'))" >/dev/null 2>&1; then
  echo "Installing backend dependencies from backend/requirements.txt..."
  "${PYTHON_EXE}" -m pip install -r "${ROOT_DIR}/backend/requirements.txt"
fi

if [[ ! -d "${ROOT_DIR}/frontend/node_modules" ]]; then
  echo "Installing frontend dependencies in frontend/..."
  (cd "${ROOT_DIR}/frontend" && npm install)
fi

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  echo ""
  echo "Stopping backend and frontend..."
  if [[ -n "${FRONTEND_PID}" ]] && kill -0 "${FRONTEND_PID}" >/dev/null 2>&1; then
    kill "${FRONTEND_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${BACKEND_PID}" ]] && kill -0 "${BACKEND_PID}" >/dev/null 2>&1; then
    kill "${BACKEND_PID}" >/dev/null 2>&1 || true
  fi
  wait >/dev/null 2>&1 || true
}

trap cleanup INT TERM EXIT

cd "${ROOT_DIR}"
echo "Starting FastAPI backend at http://localhost:${BACKEND_PORT} ..."
"${PYTHON_EXE}" -m uvicorn backend.main:app --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" &
BACKEND_PID=$!

echo "Starting Next.js frontend at http://localhost:${FRONTEND_PORT} ..."
(
  cd "${ROOT_DIR}/frontend"
  NEXT_PUBLIC_API_URL="${NEXT_PUBLIC_API_URL:-http://localhost:${BACKEND_PORT}}" \
    npm run dev -- --port "${FRONTEND_PORT}"
) &
FRONTEND_PID=$!

echo "Both servers launched. Press Ctrl+C to stop."
wait -n "${BACKEND_PID}" "${FRONTEND_PID}"
