#!/usr/bin/env bash
set -u

prefix="[devcontainer jupyter]"
workspace_dir="/workspaces/vibespin"
venv_jupyter="${workspace_dir}/.venv/bin/jupyter"
log_file="/tmp/jupyter.log"
port="8888"
wait_seconds="240"

if [[ ! -d "$workspace_dir" ]]; then
  echo "$prefix Skipping startup: workspace dir not found at $workspace_dir."
  exit 0
fi

cd "$workspace_dir"

# Rebuild flow can invoke postStart before postCreate has finished creating .venv.
for ((i=0; i<wait_seconds; i++)); do
  if [[ -x "$venv_jupyter" ]]; then
    break
  fi
  sleep 1
done

if [[ ! -x "$venv_jupyter" ]]; then
  echo "$prefix Skipping startup after ${wait_seconds}s: $venv_jupyter is missing or not executable."
  exit 0
fi

if pgrep -f "jupyter-lab.*--port=${port}" >/dev/null; then
  echo "$prefix JupyterLab already running on port ${port}."
  exit 0
fi

if command -v setsid >/dev/null 2>&1; then
  setsid "$venv_jupyter" lab --ip=0.0.0.0 --port="$port" --no-browser >"$log_file" 2>&1 < /dev/null &
else
  nohup "$venv_jupyter" lab --ip=0.0.0.0 --port="$port" --no-browser >"$log_file" 2>&1 < /dev/null &
fi

echo "$prefix Started JupyterLab on port ${port}; logs: ${log_file}"
