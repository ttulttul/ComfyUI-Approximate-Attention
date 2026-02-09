#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TEX_FILE="${REPO_ROOT}/docs/flux2ttr_v2_paper.tex"
OUT_DIR="${REPO_ROOT}/docs"
ENGINE="auto"
CLEAN=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Build the Flux2TTR v2 paper PDF.

Usage:
  scripts/build_paper.sh [options]

Options:
  --tex PATH        Path to the .tex source file.
  --outdir PATH     Output directory for PDF and build artifacts.
  --engine NAME     Build engine: auto (default), latexmk, or pdflatex.
  --clean           Remove intermediate files after a successful build.
  --dry-run         Print build commands without executing them.
  -h, --help        Show this help message.
EOF
}

fail() {
  echo "error: $*" >&2
  exit 1
}

require_cmd() {
  local cmd="$1"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  command -v "${cmd}" >/dev/null 2>&1 || fail "missing required command: ${cmd}"
}

run_cmd() {
  echo "+ $*"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tex)
      [[ $# -ge 2 ]] || fail "--tex requires a path"
      TEX_FILE="$2"
      shift 2
      ;;
    --outdir)
      [[ $# -ge 2 ]] || fail "--outdir requires a path"
      OUT_DIR="$2"
      shift 2
      ;;
    --engine)
      [[ $# -ge 2 ]] || fail "--engine requires a value"
      ENGINE="$2"
      shift 2
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown argument: $1 (use --help for usage)"
      ;;
  esac
done

[[ -f "${TEX_FILE}" ]] || fail "TeX file not found: ${TEX_FILE}"
mkdir -p "${OUT_DIR}"

if [[ "${ENGINE}" == "auto" ]]; then
  if command -v latexmk >/dev/null 2>&1; then
    ENGINE="latexmk"
  else
    ENGINE="pdflatex"
  fi
fi

BASE_NAME="$(basename "${TEX_FILE}" .tex)"

case "${ENGINE}" in
  latexmk)
    require_cmd latexmk
    run_cmd latexmk \
      -pdf \
      -interaction=nonstopmode \
      -halt-on-error \
      -file-line-error \
      -output-directory="${OUT_DIR}" \
      "${TEX_FILE}"
    if [[ "${CLEAN}" -eq 1 ]]; then
      run_cmd latexmk -c -output-directory="${OUT_DIR}" "${TEX_FILE}"
    fi
    ;;
  pdflatex)
    require_cmd pdflatex
    run_cmd pdflatex \
      -interaction=nonstopmode \
      -halt-on-error \
      -file-line-error \
      -output-directory="${OUT_DIR}" \
      "${TEX_FILE}"
    run_cmd pdflatex \
      -interaction=nonstopmode \
      -halt-on-error \
      -file-line-error \
      -output-directory="${OUT_DIR}" \
      "${TEX_FILE}"
    if [[ "${CLEAN}" -eq 1 ]]; then
      run_cmd rm -f \
        "${OUT_DIR}/${BASE_NAME}.aux" \
        "${OUT_DIR}/${BASE_NAME}.fdb_latexmk" \
        "${OUT_DIR}/${BASE_NAME}.fls" \
        "${OUT_DIR}/${BASE_NAME}.log" \
        "${OUT_DIR}/${BASE_NAME}.out" \
        "${OUT_DIR}/${BASE_NAME}.toc"
    fi
    ;;
  *)
    fail "--engine must be one of: auto, latexmk, pdflatex"
    ;;
esac

if [[ "${DRY_RUN}" -eq 0 ]]; then
  PDF_PATH="${OUT_DIR}/${BASE_NAME}.pdf"
  [[ -f "${PDF_PATH}" ]] || fail "build completed but PDF is missing: ${PDF_PATH}"
  echo "Built ${PDF_PATH}"
fi
