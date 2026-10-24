#!/usr/bin/env bash
set -euo pipefail
version=${1:?Explicit version required}
expected_current=${2:?Expected current release required}
[[ "$version" =~ ^[a-zA-Z0-9_-]+$ ]]
[[ "$expected_current" =~ ^[a-zA-Z0-9_-]+$ ]]
base=/opt/ghana-corpus-review
previous="$base/releases/$expected_current"
release="$base/releases/$version"
stage="$base/stage-$version"
test "$(readlink -f "$base/current")" = "$previous"
test -f "$base/.env"
test -f "$stage/server.cjs"
test ! -e "$release"
test ! -e "$stage/tmp/corpus-review"
test ! -e "$base/next-$version"
ledger=tmp/corpus-releases/twi-stage-v1-20260911/ledger.sqlite3
test "$(sha256sum "$stage/$ledger" | cut -d' ' -f1)" = "$(sha256sum "$previous/$ledger" | cut -d' ' -f1)"

# Old browser tabs may still request their versioned bundle during the switch.
for asset in "$previous"/public/app-*.js "$previous"/public/app-*.css; do
  test ! -f "$asset" || cp -an "$asset" "$stage/public/"
done
mv "$stage" "$release"
chown -R root:root "$release"
ln -s "$base/reviews" "$release/tmp/corpus-review"
ln -s "$release" "$base/next-$version"
rollback() {
  if test "$(readlink -f "$base/current")" = "$release"; then
    ln -s "$previous" "$base/rollback-$version"
    mv -Tf "$base/rollback-$version" "$base/current"
    systemctl restart ghana-corpus-review
  fi
}
trap rollback ERR
mv -Tf "$base/next-$version" "$base/current"
systemctl restart ghana-corpus-review
for attempt in $(seq 1 20); do
  if curl -fsS http://127.0.0.1:13121/healthz >/dev/null; then break; fi
  sleep 1
done
curl -fsS http://127.0.0.1:13121/healthz >/dev/null
test "$(readlink -f "$release/tmp/corpus-review")" = "$base/reviews"
trap - ERR
printf '%s\n' 'Review service updated. Existing reviews, credentials, Caddy and public chat unchanged.'
