#!/usr/bin/env bash
# 1枚レポート PNG を Chrome ヘッドレスで生成する
# 使い方:
#   bash build_summary.sh             # 両バージョン
#   bash build_summary.sh plain       # 標準版のみ
#   bash build_summary.sh illustrated # イラスト版のみ
set -euo pipefail
cd "$(dirname "$0")"

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

render() {
  local name="$1"
  local html="$(pwd)/${name}.html"
  local out="$(pwd)/${name}.png"
  if [[ ! -f "$html" ]]; then
    echo "skip: ${html} not found"
    return
  fi
  "$CHROME" \
    --headless=new \
    --disable-gpu \
    --hide-scrollbars \
    --force-device-scale-factor=2 \
    --window-size=1920,1280 \
    --default-background-color=00000000 \
    --screenshot="$out" \
    "file://$html" \
    2>&1 | grep -v "DevTools" || true
  echo "→ ${out}"
}

case "${1:-all}" in
  plain)        render summary_report ;;
  illustrated)  render summary_report_illustrated ;;
  all|*)        render summary_report; render summary_report_illustrated ;;
esac
