#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────
# 動画→記事化パイプラインの自動セットアップ
#
# 前提: 同じフォルダに  x_video_to_article.py  と  transcribe.yml  を置く
# 実行: bash setup.sh            (リポジトリ名はデフォルト video-to-article)
#       bash setup.sh 好きな名前   (名前を指定したいとき)
#
# これは Claude Code に「setup.sh を実行して」と言えばそのまま走ります。
# ──────────────────────────────────────────────────────────
set -euo pipefail

REPO_NAME="${1:-video-to-article}"

echo "▶ 1. GitHub CLI を確認"
command -v gh >/dev/null || { echo "✗ gh が無い → https://cli.github.com で入れて"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "✗ 未ログイン → 先に  gh auth login  を実行して"; exit 1; }

echo "▶ 2. ファイルを配置"
[ -f x_video_to_article.py ] || { echo "✗ x_video_to_article.py が同じフォルダに無い"; exit 1; }
mkdir -p .github/workflows
if [ -f transcribe.yml ]; then mv -f transcribe.yml .github/workflows/transcribe.yml; fi
[ -f .github/workflows/transcribe.yml ] || { echo "✗ transcribe.yml が見つからない"; exit 1; }

echo "▶ 3. git 初期化＆コミット"
git init -q
git add .
git commit -qm "init: video-to-article pipeline" || true

echo "▶ 4. 公開リポジトリを作って push"
gh repo create "$REPO_NAME" --public --source=. --push

echo "▶ 5. Groq の APIキーを secret に登録"
echo "   console.groq.com で取得したキーを貼り付けて Enter(入力は画面に出ません):"
gh secret set GROQ_API_KEY

echo ""
echo "✅ 完了！"
echo "   次:  GitHubの Actions タブ → video-to-article → Run workflow → 動画URLを貼る"
echo "   数分後に  outputs/◯◯_article.md  に記事ができます。"
