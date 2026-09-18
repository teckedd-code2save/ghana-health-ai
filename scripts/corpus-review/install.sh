#!/usr/bin/env bash
set -euo pipefail
base=/opt/ghana-corpus-review
release="$base/releases/v1"
stage=/opt/ghana-corpus-review-stage-v1
site=/etc/caddy/sites/55-ghana-health-ai.caddy
expected=8b94bc9629f944e481162a6adb037aa7ec1749b1a1a06752190f7c066daea605
test "$(sha256sum "$site" | cut -d' ' -f1)" = "$expected"
test -f "$stage/server.cjs"
test ! -e "$release"
test ! -e "$base/current"
if ss -ltn | grep -q '127.0.0.1:13121 '; then
  printf '%s\n' 'Review port is occupied; no changes applied.'
  exit 1
fi
install -d "$base/releases"
mv "$stage" "$release"
if ! id gha-corpus-review >/dev/null 2>&1; then
  useradd --system --home "$base" --shell /usr/sbin/nologin gha-corpus-review
fi
install -d -o gha-corpus-review -g gha-corpus-review -m 0700 "$base/reviews"
if test -d "$release/tmp/corpus-review"; then
  cp -an "$release/tmp/corpus-review/." "$base/reviews/"
  mv "$release/tmp/corpus-review" "$release/imported-review-journal"
fi
chown -R gha-corpus-review:gha-corpus-review "$base/reviews"
ln -s "$base/reviews" "$release/tmp/corpus-review"
ln -s "$release" "$base/current"
python3 - "$base/.env" <<'PY'
import os,secrets,sys
fd=os.open(sys.argv[1],os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as handle:
    handle.write('NODE_ENV=production\nPORT=13121\nCORPUS_RELEASE=twi-stage-v1-20260911\n')
    handle.write('REVIEW_ORIGIN=https://ghanahealth.serendepify.com\n')
    handle.write('REVIEW_PASSWORD='+secrets.token_urlsafe(24)+'\n')
PY
install -m 0644 "$release/deploy/ghana-corpus-review.service" /etc/systemd/system/ghana-corpus-review.service
systemctl daemon-reload
systemctl enable --now ghana-corpus-review
for attempt in $(seq 1 20); do
  if curl -fsS http://127.0.0.1:13121/healthz >/dev/null; then break; fi
  sleep 1
done
curl -fsS http://127.0.0.1:13121/healthz >/dev/null
backup="$site.before-corpus-$(date +%s).bak"
cp -p "$site" "$backup"
install -m 0644 "$release/deploy/site.caddy" "$site"
if ! caddy validate --config /etc/caddy/Caddyfile; then
  cp -p "$backup" "$site"
  exit 1
fi
systemctl reload caddy
printf '%s\n' 'Private corpus review is installed; public app upstream unchanged.'
