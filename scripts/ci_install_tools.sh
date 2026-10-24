#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Installs the pinned CLI tools used by the lint, security and bootstrap targets on a CI runner.
set -euo pipefail

HADOLINT_VERSION=2.15.1
GITLEAKS_VERSION=8.30.1
TRIVY_VERSION=0.76.0
KUBECONFORM_VERSION=0.8.0
SHELLCHECK_VERSION=0.11.0
BIN="${HOME}/.local/bin"
mkdir -p "${BIN}"
echo "${BIN}" >>"${GITHUB_PATH:-/dev/null}"
export PATH="${BIN}:${PATH}"

case "$(uname -m)" in
  x86_64) goarch=amd64 harch=x86_64 gitleaks_arch=x64 trivy_arch=64bit ;;
  aarch64 | arm64) goarch=arm64 harch=arm64 gitleaks_arch=arm64 trivy_arch=ARM64 ;;
  *) echo "unsupported architecture $(uname -m)" >&2; exit 1 ;;
esac
gh="https://github.com"

curl -sSfL -o "${BIN}/hadolint" \
  "${gh}/hadolint/hadolint/releases/download/v${HADOLINT_VERSION}/hadolint-Linux-${harch}"
chmod +x "${BIN}/hadolint"

curl -sSfL "${gh}/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_${gitleaks_arch}.tar.gz" \
  | tar -xz -C "${BIN}" gitleaks

curl -sSfL "${gh}/aquasecurity/trivy/releases/download/v${TRIVY_VERSION}/trivy_${TRIVY_VERSION}_Linux-${trivy_arch}.tar.gz" \
  | tar -xz -C "${BIN}" trivy

curl -sSfL "${gh}/yannh/kubeconform/releases/download/v${KUBECONFORM_VERSION}/kubeconform-linux-${goarch}.tar.gz" \
  | tar -xz -C "${BIN}" kubeconform

curl -sSfL "${gh}/koalaman/shellcheck/releases/download/v${SHELLCHECK_VERSION}/shellcheck-v${SHELLCHECK_VERSION}.linux.${harch/arm64/aarch64}.tar.xz" \
  | tar -xJ -C /tmp
install "/tmp/shellcheck-v${SHELLCHECK_VERSION}/shellcheck" "${BIN}/"

for tool in hadolint gitleaks trivy kubeconform shellcheck; do
  command -v "${tool}"
done
