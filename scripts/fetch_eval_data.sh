#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Downloads and verifies the BEIR evaluation datasets listed in eval/datasets.yaml.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${root}/eval/datasets.yaml"
data_dir="${FOOTNOTE_EVAL_DATA_DIR:-${HOME}/.cache/rayld/footnote/datasets}"

if [[ $# -gt 0 ]]; then
  names=("$@")
else
  names=()
  while IFS= read -r line; do names+=("${line}"); done < <(yq -r '.datasets | keys | .[]' "${manifest}")
fi

mkdir -p "${data_dir}"
for name in "${names[@]}"; do
  url="$(yq -r ".datasets[\"${name}\"].url // \"\"" "${manifest}")"
  want="$(yq -r ".datasets[\"${name}\"].sha256 // \"\"" "${manifest}")"
  if [[ -z "${url}" || -z "${want}" ]]; then
    echo "fetch_eval_data: unknown dataset '${name}' (see eval/datasets.yaml)" >&2
    exit 2
  fi
  zip="${data_dir}/${name}.zip"
  verified=0
  if [[ -f "${zip}" ]]; then
    got="$(shasum -a 256 "${zip}" | cut -d' ' -f1)"
    [[ "${got}" == "${want}" ]] && verified=1
  fi
  if [[ "${verified}" -eq 0 ]]; then
    echo "downloading ${name}"
    curl -fL --retry 3 -o "${zip}.part" "${url}"
    got="$(shasum -a 256 "${zip}.part" | cut -d' ' -f1)"
    if [[ "${got}" != "${want}" ]]; then
      rm -f "${zip}.part"
      echo "fetch_eval_data: ${name}: sha256 mismatch (got ${got}, want ${want})" >&2
      exit 1
    fi
    mv "${zip}.part" "${zip}"
    rm -rf "${data_dir:?}/${name}"
  fi
  if [[ ! -f "${data_dir}/${name}/corpus.jsonl" ]]; then
    unzip -q -o "${zip}" -d "${data_dir}"
  fi
  echo "ok ${name} ${data_dir}/${name}"
done
