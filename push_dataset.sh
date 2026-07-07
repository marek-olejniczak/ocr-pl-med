#!/usr/bin/env bash
#
# push_dataset.sh — wrzuca aktualny stan dataset/ na DagsHub jednym poleceniem.
#
# Co robi:
#   1. dvc add dataset   → przelicza wskaznik dataset.dvc
#   2. dvc push          → wysyla pliki (bloby) do storage DagsHub
#   3. commit dataset.dvc lokalnie (na biezacej galezi)
#   4. publikuje wskaznik na galezi 'dvc' DagsHub — jednym czystym commitem
#      (przez tymczasowy worktree), zeby ominac duze pliki w historii gita.
#
# Uzycie:
#   ./push_dataset.sh "opis partii"
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

MSG="${1:-Update dataset}"
DAGSHUB_REMOTE="dagshub"
DAGSHUB_BRANCH="dvc"

# Aktywuj venv (dvc jest w nim zainstalowany)
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "BLAD: brak .venv — uruchom: python3 -m venv .venv && pip install -r requirements.txt" >&2
  exit 1
fi

echo "==> [1/4] dvc add dataset"
dvc add dataset

echo "==> [2/4] dvc push (wysylka danych do DagsHub)"
dvc push --jobs 2

echo "==> [3/4] commit wskaznika lokalnie"
git add dataset.dvc
if git diff --cached --quiet; then
  echo "    (brak zmian wskaznika — nic do commita)"
else
  git commit -m "$MSG"
fi

echo "==> [4/4] publikacja wskaznika na DagsHub ($DAGSHUB_REMOTE/$DAGSHUB_BRANCH)"
git fetch "$DAGSHUB_REMOTE" "$DAGSHUB_BRANCH:refs/remotes/$DAGSHUB_REMOTE/$DAGSHUB_BRANCH"

WT="$(mktemp -d)"
cleanup() { git worktree remove "$WT" --force >/dev/null 2>&1 || true; }
trap cleanup EXIT

git worktree add -f "$WT" "$DAGSHUB_REMOTE/$DAGSHUB_BRANCH" >/dev/null
cp dataset.dvc "$WT/dataset.dvc"
git -C "$WT" add dataset.dvc
if git -C "$WT" diff --cached --quiet; then
  echo "    (DagsHub juz ma aktualny wskaznik)"
else
  git -C "$WT" \
    -c user.email="labeling@local" -c user.name="labeling-tool" \
    commit -m "$MSG" >/dev/null
  git -C "$WT" push "$DAGSHUB_REMOTE" "HEAD:$DAGSHUB_BRANCH"
fi

echo
echo "GOTOWE ✅  Dane + wskaznik sa na DagsHub (galaz '$DAGSHUB_BRANCH')."
echo "Podejrzyj: https://dagshub.com/JakubGorniak-git/inzynierka/src/dvc"
