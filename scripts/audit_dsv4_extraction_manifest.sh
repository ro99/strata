#!/usr/bin/env bash
# Traceability gate for the rank-local TP2 decode landing. Two dimensions:
#
#   1. File coverage. Every path in the experiment delta main...a31ac58 must
#      carry exactly one disposition in the manifest's classification index.
#   2. Capability coverage. All thirteen program capabilities must be accounted
#      for in BOTH the manifest ledger and the canonical architecture document.
#      Exclusion from the production binary does not permit omission of a
#      capability's invariants, evidence, failure lessons, or reuse guidance.
#
# Either dimension failing blocks the landing.
#
# This checks coverage only. It is deliberately not a licence to add production
# code: an unclassified path or capability is resolved by classifying or
# documenting it, and only by porting it if the landing plan already called
# for it.
#
# Usage: scripts/audit_dsv4_extraction_manifest.sh [base] [experiment-head]
set -o pipefail

base="${1:-main}"
head_ref="${2:-a31ac58}"
manifest="docs/dsv4-rank-local-extraction-manifest.md"
architecture="docs/dsv4-rank-local-architecture.md"
capability_count=13

if [[ ! -f "${manifest}" ]]; then
    printf 'audit: manifest %s not found\n' "${manifest}" >&2
    exit 2
fi
if [[ ! -f "${architecture}" ]]; then
    printf 'audit: architecture document %s not found\n' "${architecture}" >&2
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

# Capability coverage. Each capability needs a ledger entry carrying its
# disposition, invariants, evidence, failure lesson and reuse guidance, and a
# named location in the architecture document.
printf '\ncapabilities\n'
missing_capabilities=0
for index in $(seq -w 1 "${capability_count}"); do
    id="CAP-${index}"
    ledger_row="$(grep -c "^### ${id} " "${manifest}")"
    index_row="$(grep -c "^| ${id} |" "${manifest}")"
    arch_row="$(grep -c "^| ${id} |" "${architecture}")"
    detail=""
    if (( ledger_row == 0 )); then detail+=" no-ledger-entry"; fi
    if (( index_row == 0 )); then detail+=" not-in-manifest-index"; fi
    if (( arch_row == 0 )); then detail+=" not-in-architecture"; fi
    if [[ -n "${detail}" ]]; then
        printf 'MISSING  %s:%s\n' "${id}" "${detail}"
        missing_capabilities=$((missing_capabilities + 1))
        continue
    fi
    # A ledger entry that states no evidence or no reuse guidance is a
    # placeholder, not preservation.
    section="$(awk -v id="### ${id} " \
        'index($0, id)==1 {found=1; next} found && /^### /{exit} found' \
        "${manifest}")"
    for required in "**Invariants" "**Evidence" "**Failure lesson" "**Reuse guidance"; do
        if ! printf '%s' "${section}" | grep -qF "${required}"; then
            printf 'INCOMPLETE  %s: missing %s\n' "${id}" "${required}"
            missing_capabilities=$((missing_capabilities + 1))
        fi
    done
done
printf 'accounted         %d/%d\n' \
    $(( capability_count - missing_capabilities > 0
        ? capability_count - missing_capabilities : 0 )) "${capability_count}"

if (( unclassified != 0 || missing_capabilities != 0 )); then
    printf '\naudit: FAIL — %d unclassified path(s), %d capability gap(s); landing is blocked\n' \
        "${unclassified}" "${missing_capabilities}" >&2
    exit 1
fi

printf '\naudit: PASS — every path classified, all %d capabilities accounted for\n' \
    "${capability_count}"
