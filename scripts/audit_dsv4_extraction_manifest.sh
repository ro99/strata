#!/usr/bin/env bash
# Traceability gate for the rank-local TP2 decode landing.
#
# Every path in the experiment delta main...a31ac58 must carry exactly one
# disposition in the manifest's complete classification index. An unclassified
# path blocks the landing.
#
# This checks classification coverage only. It is deliberately not a licence to
# add production code: an unclassified path is resolved by classifying it, and
# only by porting it if the landing plan already called for it.
#
# Usage: scripts/audit_dsv4_extraction_manifest.sh [base] [experiment-head]
set -o pipefail

base="${1:-main}"
head_ref="${2:-a31ac58}"
manifest="docs/dsv4-rank-local-extraction-manifest.md"

if [[ ! -f "${manifest}" ]]; then
    printf 'audit: manifest %s not found\n' "${manifest}" >&2
    exit 2
fi
if ! git rev-parse --verify --quiet "${head_ref}^{commit}" > /dev/null; then
    printf 'audit: experiment head %s is not a commit in this repository\n' "${head_ref}" >&2
    exit 2
fi

delta="$(git diff --name-only "${base}...${head_ref}")"
if [[ -z "${delta}" ]]; then
    printf 'audit: %s...%s is empty; refusing to report a vacuous pass\n' \
        "${base}" "${head_ref}" >&2
    exit 2
fi

# Paths are also named in the prose tables above, so the index section is
# sliced out first: only it is authoritative for a disposition.
index="$(awk '/^## Complete classification index/{found=1} found' "${manifest}")"
if [[ -z "${index}" ]]; then
    printf 'audit: manifest has no "## Complete classification index" section\n' >&2
    exit 2
fi

total=0
unclassified=0
declare -A seen=()

while IFS= read -r path; do
    [[ -z "${path}" ]] && continue
    total=$((total + 1))
    # Index rows are: | `path` | disposition | note |
    row="$(printf '%s' "${index}" | grep -F "| \`${path}\` |" | head -1)"
    if [[ -z "${row}" ]]; then
        printf 'UNCLASSIFIED  %s\n' "${path}"
        unclassified=$((unclassified + 1))
        continue
    fi
    disposition="$(printf '%s' "${row}" | awk -F'|' '{gsub(/^ +| +$/,"",$3); print $3}')"
    case "${disposition}" in
        ported|documentation|excluded|deferred)
            seen["${disposition}"]=$(( ${seen["${disposition}"]:-0} + 1 ))
            ;;
        *)
            printf 'BAD DISPOSITION  %s -> %s\n' "${path}" "${disposition}"
            unclassified=$((unclassified + 1))
            ;;
    esac
done <<< "${delta}"

printf '\n%s...%s\n' "${base}" "${head_ref}"
printf 'total paths        %d\n' "${total}"
for key in ported documentation excluded deferred; do
    printf '%-18s %d\n' "${key}" "${seen[${key}]:-0}"
done

if (( unclassified != 0 )); then
    printf '\naudit: FAIL — %d path(s) unclassified; landing is blocked\n' "${unclassified}" >&2
    exit 1
fi

printf '\naudit: PASS — every path classified\n'
