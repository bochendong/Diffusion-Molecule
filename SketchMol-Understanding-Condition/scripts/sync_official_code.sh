#!/usr/bin/env bash
# Clone or fast-forward official baseline repositories declared in the registry.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

REGISTRY="${DM_BENCHMARK_REGISTRY:-SketchMol-Understanding-Condition/configs/official_benchmark_registry.json}"
CODE_ROOT="${DM_OFFICIAL_CODE_ROOT:-Research/Molecule Generation/OfficialBaselines}"
PYTHON_BIN="${DM_REGISTRY_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
DRY_RUN="${DRY_RUN:-0}"
METHOD_FILTER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --method|--benchmark)
      METHOD_FILTER="${2:-}"
      shift 2
      ;;
    --registry)
      REGISTRY="${2:-}"
      shift 2
      ;;
    --code-root)
      CODE_ROOT="${2:-}"
      shift 2
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$REGISTRY" ]]; then
  echo "ERROR: registry not found: $REGISTRY" >&2
  exit 2
fi

echo "Official code sync"
echo "  registry=$REGISTRY"
echo "  code_root=$CODE_ROOT"
echo "  method_filter=${METHOD_FILTER:-all}"
echo "  dry_run=$DRY_RUN"

records="$("$PYTHON_BIN" - <<'PY' "$REGISTRY" "$CODE_ROOT" "$METHOD_FILTER"
import json
import sys
from pathlib import Path

registry = Path(sys.argv[1])
code_root = Path(sys.argv[2])
method_filter = {item.strip().lower() for item in sys.argv[3].split(",") if item.strip()}
payload = json.loads(registry.read_text(encoding="utf-8"))
for item in payload.get("benchmarks", []):
    official = item.get("official_code") or {}
    repo_url = official.get("repo_url")
    if not repo_url:
        continue
    keys = {
        str(item.get("id", "")).lower(),
        str(item.get("method_id", "")).lower(),
        str(item.get("display_name", "")).lower(),
    }
    if method_filter and not (keys & method_filter):
        continue
    local_path = official.get("local_path")
    if not local_path:
        local_path = str(code_root / str(item.get("method_id", item.get("id"))).replace("/", "_") / "repo")
    print("\t".join([
        str(item.get("method_id", item.get("id"))),
        str(item.get("id", "")),
        str(repo_url),
        str(local_path),
        str(official.get("expected_entrypoint") or ""),
        str(official.get("verified_head") or ""),
    ]))
PY
)"

if [[ -z "$records" ]]; then
  echo "No official repositories matched the registry/filter."
  exit 0
fi

while IFS=$'\t' read -r method_id benchmark_id repo_url local_path expected_entry verified_head; do
  [[ -z "$repo_url" ]] && continue
  echo
  echo "== $method_id ($benchmark_id) =="
  echo "  repo=$repo_url"
  echo "  path=$local_path"
  echo "  expected_entry=${expected_entry:-none}"
  echo "  registry_verified_head=${verified_head:-none}"

  if [[ "$DRY_RUN" == "1" ]]; then
    if [[ -d "$local_path/.git" ]]; then
      echo "  would: git -C '$local_path' pull --ff-only"
    elif [[ -e "$local_path" ]]; then
      echo "  would fail: path exists but is not a git checkout"
    else
      echo "  would: git clone '$repo_url' '$local_path'"
    fi
    continue
  fi

  if [[ -d "$local_path/.git" ]]; then
    if [[ -n "$(git -C "$local_path" status --porcelain)" ]]; then
      echo "ERROR: official checkout has local changes; refusing to pull: $local_path" >&2
      exit 2
    fi
    git -C "$local_path" pull --ff-only
  elif [[ -e "$local_path" ]]; then
    echo "ERROR: path exists but is not a git checkout: $local_path" >&2
    exit 2
  else
    mkdir -p "$(dirname "$local_path")"
    git clone "$repo_url" "$local_path"
  fi

  if [[ -n "$expected_entry" && ! -e "$local_path/$expected_entry" ]]; then
    echo "WARN: expected official entrypoint is missing: $local_path/$expected_entry" >&2
  fi
  current_head="$(git -C "$local_path" rev-parse HEAD 2>/dev/null || true)"
  echo "  current_head=${current_head:-unknown}"
done <<< "$records"

echo
echo "Official code sync complete."
