#!/usr/bin/env bash
set -u

prefix="[devcontainer ssh]"

candidate_sockets=(
  "${SSH_AUTH_SOCK:-}"
  "$HOME/.gnupg/S.gpg-agent.ssh"
  "$HOME/.gnupg/S.gpg-agent"
)

resolved_socket=""
for candidate in "${candidate_sockets[@]}"; do
  if [[ -n "$candidate" && -S "$candidate" ]]; then
    resolved_socket="$candidate"
    break
  fi
done

if [[ -n "$resolved_socket" ]]; then
  export SSH_AUTH_SOCK="$resolved_socket"
fi

if [[ -z "${SSH_AUTH_SOCK:-}" ]]; then
  echo "$prefix SSH_AUTH_SOCK is not set."
  echo "$prefix Host agent forwarding may be unavailable."
  echo "$prefix Rebuild container after confirming host SSH agent is running."
  exit 0
fi

if [[ ! -S "$SSH_AUTH_SOCK" ]]; then
  if [[ -f "$SSH_AUTH_SOCK" ]]; then
    echo "$prefix No usable forwarded SSH agent socket was found."
    echo "$prefix Dev Containers may still have forwarded only a GPG agent without SSH support."
    exit 0
  fi

  echo "$prefix SSH_AUTH_SOCK path does not point to a socket: $SSH_AUTH_SOCK"
  echo "$prefix Check your devcontainer mounts configuration."
  exit 0
fi

ssh-add -l >/dev/null 2>&1
status=$?

if [[ $status -eq 0 ]]; then
  echo "$prefix SSH agent is reachable via $SSH_AUTH_SOCK and has identities loaded."
  exit 0
fi

if [[ $status -eq 1 ]]; then
  echo "$prefix SSH agent is reachable via $SSH_AUTH_SOCK but has no identities loaded."
  echo "$prefix On host: ssh-add ~/.ssh/id_ed25519"
  exit 0
fi

echo "$prefix Unable to query SSH agent (ssh-add exit code: $status)."
echo "$prefix Ensure the host agent is running and forwarded into the container."
exit 0
