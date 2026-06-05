#!/usr/bin/env python3
"""
X / YouTube 等の長尺動画 → mp3 → 文字起こし(Groq Whisper) → 記事化(markdown)

「②動画→テキスト→記事」だけを担当する新ブロック。
既存の LINE→Notion パイプラインからは process(url) を import して呼ぶ想定。

────────────────────────────────────────────────────────
セットアップ
  pip install yt-dlp groq
  # ffmpeg も必要:  mac → brew install ffmpeg / ubuntu → apt install ffmpeg
  export GROQ_API_KEY="自分のキー"          # https://console.groq.com で無料取得
  # (任意) X のログイン必須動画用:
  export COOKIES_FROM_BROWSER="chrome"      # chrome / safari / firefox など

使い方(単体)
  python x_video_to_article.py "https://x.com/..../status/...."
  python x_video_to_article.py "URL" --lang ja        # 記事を日本語で(既定)
  python x_video_to_article.py "URL" --lang original  # 元言語のまま

出力
  outputs/<title>_transcript.txt   … 文字起こし全文
  outputs/<title>_article.md       … 記事(markdown)

LINE→Notion へ組み込むとき
  from x_video_to_article import process
  result = process(url)            # {"title","transcript","article"} を返す
  # result["article"] を既存の Notion 追加コードに渡すだけ
────────────────────────────────────────────────────────

無料の前提:
  - 文字起こしは Groq Whisper 無料枠(1日2,000リクエスト/モデルは large-v3-turbo)
  - 記事化も Groq の Llama 無料枠を使用 → 新ブロックは丸ごと無料で回る
  - 既存パイプラインで Claude 等を使って記事化したい場合は write_article() の
    中身を既存関数の呼び出しに差し替えるだけでOK

プライバシー注意:
  Groq 無料枠はプライバシーSLA対象外。公開教材動画なら問題ないが、
  非公開音声を流すなら transcribe() をローカル faster-whisper に置き換えること
  (外にデータを出さない・無料)。
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile

from groq import Groq

# ── 設定（必要なら触る場所はここだけ） ─────────────────────
SEGMENT_SECONDS = 600          # 音声を何秒ごとに分割するか(10分)
SAMPLE_RATE = 16000            # Whisper最適: 16kHz
AUDIO_BITRATE = "48k"          # 音声品質(speechは48kで十分・サイズ最小)
WHISPER_MODEL = "whisper-large-v3-turbo"
NOTES_MODEL = "llama-3.1-8b-instant"      # map: 各チャンクの要点メモ(速い)
ARTICLE_MODEL = "llama-3.3-70b-versatile"  # reduce: 最終記事(高品質)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
# ──────────────────────────────────────────────────────────

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def _groq_retry(fn, tries=4, base_wait=8):
    """無料枠の一時的な混雑(429等)に備えた素朴なリトライ。確実性アップ用。"""
    import time
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(base_wait * (i + 1))


def _run(cmd):
    """サブプロセス実行。失敗したら標準エラーを見せて終了。"""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"コマンド失敗: {' '.join(cmd)}\n{proc.stderr[-2000:]}")
    return proc.stdout


def _sanitize(name):
    name = re.sub(r"[^\w\s\-一-龯ぁ-んァ-ン]", "", name).strip()
    name = re.sub(r"\s+", "_", name)
    return (name or "video")[:80]


def get_title(url):
    try:
        out = _run(["yt-dlp", "--skip-download", "--print", "%(title)s", url])
        return out.strip().splitlines()[0] if out.strip() else "video"
    except Exception:
        return "video"


def download_audio(url, workdir):
    """動画から音声を mp3 で取得。X のログイン必須動画用に cookie 対応。"""
    out_path = os.path.join(workdir, "raw.mp3")
    cmd = ["yt-dlp", "-x", "--audio-format", "mp3", "-o", out_path]
    cookies_file = os.environ.get("COOKIES_FILE")           # CI/サーバ用(cookies.txt)
    browser = os.environ.get("COOKIES_FROM_BROWSER")        # 手元PC用(ブラウザ直読み)
    if cookies_file and os.path.exists(cookies_file):
        cmd += ["--cookies", cookies_file]
    elif browser:
        cmd += ["--cookies-from-browser", browser]
    cmd.append(url)
    _run(cmd)
    if not os.path.exists(out_path):
        raise RuntimeError("音声の取得に失敗しました(ログイン必須動画なら cookies を設定)")
    return out_path


def segment_audio(mp3_path, workdir):
    """16kHzモノラルに再エンコードしつつ、10分ごとに分割。
    → 各チャンクは数MBに収まり Groq の25MB上限を自然にクリア。"""
    pattern = os.path.join(workdir, "seg_%03d.mp3")
    _run([
        "ffmpeg", "-y", "-i", mp3_path,
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-b:a", AUDIO_BITRATE,
        "-f", "segment", "-segment_time", str(SEGMENT_SECONDS),
        pattern,
    ])
    return sorted(glob.glob(os.path.join(workdir, "seg_*.mp3")))


def transcribe(chunk_path):
    """1チャンクを Groq Whisper で文字起こし。言語は自動判定。"""
    def _call():
        with open(chunk_path, "rb") as f:
            return client.audio.transcriptions.create(
                file=(os.path.basename(chunk_path), f.read()),
                model=WHISPER_MODEL,
                response_format="text",
            )
    result = _groq_retry(_call)
    return result if isinstance(result, str) else getattr(result, "text", str(result))


def make_notes(chunk_text, lang):
    """map: 1チャンク分の要点を箇条書きメモに(コマンド/コードは原文保持)。"""
    lang_line = "日本語で" if lang == "ja" else "元の言語のまま"
    prompt = (
        f"以下は長尺動画の文字起こしの一部です。{lang_line}、重要な論点・手順・具体例を"
        "箇条書きで簡潔に抽出してください。コマンド・コード・固有名詞・数値はそのまま"
        "正確に残すこと。創作・補完はしないでください。\n\n"
        f"----\n{chunk_text}\n----"
    )
    resp = _groq_retry(lambda: client.chat.completions.create(
        model=NOTES_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    ))
    return resp.choices[0].message.content.strip()


def write_article(notes_list, title, lang):
    """reduce: 全チャンクのメモを統合して最終記事(markdown)に。

    既存の Claude ベース記事化を使いたい場合は、この関数の中身を
    既存関数の呼び出しに差し替えるだけでよい。
    """
    lang_line = "日本語で" if lang == "ja" else "動画の元言語で"
    joined = "\n\n".join(f"## メモ {i+1}\n{n}" for i, n in enumerate(notes_list))
    prompt = (
        f"あなたは技術記事の編集者です。次の各メモは1本の動画『{title}』を順に要約した"
        f"ものです。これらを統合し、{lang_line}、読みやすい長文記事(markdown)に再構成して"
        "ください。要件:\n"
        "- 見出し(##)で章立てし、論理的な流れにする\n"
        "- 手順・コマンド・コードは ``` で囲んで正確に残す\n"
        "- 重複は統合し、メモにない情報は足さない(創作禁止)\n"
        "- 冒頭に3〜5行の要約(TL;DR)を置く\n\n"
        f"{joined}"
    )
    resp = _groq_retry(lambda: client.chat.completions.create(
        model=ARTICLE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    ))
    return resp.choices[0].message.content.strip()


def process(url, lang="ja"):
    """新ブロックの本体。URL → {title, transcript, article} を返す。"""
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("環境変数 GROQ_API_KEY が未設定です")

    title = get_title(url)
    with tempfile.TemporaryDirectory() as workdir:
        print(f"[1/4] 音声ダウンロード: {title}", file=sys.stderr)
        mp3 = download_audio(url, workdir)

        print("[2/4] 16kHzモノラル化＋10分分割", file=sys.stderr)
        chunks = segment_audio(mp3, workdir)

        print(f"[3/4] 文字起こし({len(chunks)}チャンク)", file=sys.stderr)
        parts = []
        for i, c in enumerate(chunks):
            print(f"      - {i+1}/{len(chunks)}", file=sys.stderr)
            parts.append(transcribe(c))
        transcript = "\n".join(p.strip() for p in parts if p.strip())

        print("[4/4] 記事化(map→reduce)", file=sys.stderr)
        notes = [make_notes(p, lang) for p in parts if p.strip()]
        article = write_article(notes, title, lang)

    return {"title": title, "transcript": transcript, "article": article}


def main():
    ap = argparse.ArgumentParser(description="動画→mp3→文字起こし→記事化")
    ap.add_argument("url", help="X / YouTube 等の動画URL")
    ap.add_argument("--lang", choices=["ja", "original"], default="ja",
                    help="記事の言語(既定: ja)")
    args = ap.parse_args()

    result = process(args.url, lang=args.lang)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = _sanitize(result["title"])
    t_path = os.path.join(OUTPUT_DIR, f"{base}_transcript.txt")
    a_path = os.path.join(OUTPUT_DIR, f"{base}_article.md")
    with open(t_path, "w", encoding="utf-8") as f:
        f.write(result["transcript"])
    with open(a_path, "w", encoding="utf-8") as f:
        f.write(result["article"])

    print(f"\n✅ 完了\n  文字起こし: {t_path}\n  記事:       {a_path}")


if __name__ == "__main__":
    main()
