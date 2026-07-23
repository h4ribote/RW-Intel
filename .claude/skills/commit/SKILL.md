---
name: commit
description: Inspect uncommitted changes, craft an appropriate commit message, and create the commit. Use when the user asks to "commit", "コミットして", "コミットを作成", or similar.
allowed-tools: Bash
---

# Commit

未コミットの変更を読み取り、適切なコミットメッセージを考えてコミットを作成する。

## 手順

### 1. 現状の確認

以下のコマンドを並列で実行する:

- `git status` — 変更・未追跡ファイルを把握する（`-uall` フラグは使わない）
- `git diff` — ステージ済みと未ステージの差分の両方を確認する
- `git log --oneline -10` — このリポジトリのコミットメッセージのスタイルを把握する

変更が一切ない場合（`git status` がクリーンな場合）は、コミットせず「コミットすべき変更がありません」と報告して終了する。

### 2. コミットメッセージの作成

**フォーマット:**

```
<簡潔な要約 (英語の命令形、1行、70文字以内)>

- <変更1: 何を / なぜ / どうしたか>
- <変更2: 何を / なぜ / どうしたか>
- ...

Co-Authored-By: Claude <noreply@anthropic.com>
```

トレーラーは `Co-Authored-By: Claude <noreply@anthropic.com>` のほか、モデル名を添えた `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` の形も使ってよい（どちらもアドレスは `noreply@anthropic.com`）。

**ルール:**

- **1行目**: 動詞で始まる英語の命令形短文（`Add` / `Update` / `Fix` / `Refactor` / `Rewrite` / `Test` / `Docs` など）。変更全体を一言で表す
- **本文**: 個々の変更点を箇条書きにする。各項目は1行で「何を / なぜ / どうしたか」が伝わるように書く。冗長な説明や `What:` / `Why:` ラベルは付けない
- **【厳守】本文は箇条書きだけで構成する**: 要約行の後に地の文（散文の段落）を置かない。背景や理由を述べたい場合も、その内容を該当する箇条書きの項目に織り込む。git log は本文を整形済みテキストとして表示するため、段落は折り返しの見た目に依存して読みづらくなるうえ、変更点との対応が追えなくなる
- **十分に要約する**: 箇条書きは6項目以内、1項目180文字以内を目安にする。7項目以上になる、または1項目が長くなる場合は、要約が足りないか、変更が複数の論理的まとまりに分かれている合図なので、記述を圧縮するか、ユーザーに確認のうえコミットを分割する
- **できる限り ASCII の文字を使用する**: コミットメッセージ本文は可能な限り ASCII 文字で書く。日本語の説明が必要な場合を除き、記号・矢印・約物などは ASCII で代替する（例: `→` の代わりに `->`、`“ ”` の代わりに `"`、`…` の代わりに `...`、`—` の代わりに `-`）。ASCII 以外の文字は環境によって文字化けや表示崩れを起こすため避ける
- **箇条書きを途中で改行しない**: 見た目を整えるために `- テキスト{改行}{空白2文字}テキスト` のように1項目を複数行へ折り返さない。各バレットは長くても物理的に1行で書く。git log や GitHub は本文を整形済みテキストとして表示するため、折り返しのインデントがそのまま余分な空白として残り読みづらくなる。長すぎる場合は項目を分割する
- 変更が1つだけ・自明な場合は本文を省略してもよい
- 機密情報（`.env`、認証情報、APIキーなど）が含まれる場合は**コミットせず**ユーザーに警告する
- **チルダ `~` を使わない**: GitHub / VS Code などは GitHub-flavored Markdown でメッセージを表示するため、`~` で囲んだ範囲が打ち消し線として描画されてしまう。「約」の意味では `~67x` の代わりに `roughly 67x` / `about 67x` / `approx. 67x` を、範囲は `5-10` のようにハイフンで表記する
- **【厳守】GitHub Issue/PR と誤認される番号参照を書かない**: `#1` / `#2` / `#42` のような `#<数字>` 表記は GitHub が Issue/PR 番号への自動リンクとして解釈し、無関係な Issue/PR を指してしまう。課題・所見を番号で参照してはならず、変更内容そのものを言葉で記述する（例: `(#8 を解消)` ではなく `(security/leisure を欠くストレス式の矛盾を解消)`）。`good.food#drink` のように識別子の一部として `#` を含むトークンは可（`#` の直後が数字でなければ誤リンクしない）
- **【厳守】実質ローカルにのみ存在するファイル・知識を参照しない**: コミットメッセージは git 履歴だけを見る第三者にとって自己完結していなければならない。ローカル作業物（`tmp/` 配下のメモ・ロードマップ・スクラッチ計画書など、リポジトリにコミットされていない、または共有されない成果物）の採番・行番号・見出し、チャット限定の符牒、外部に存在しないチケット番号を参照しない。指したい事項はメッセージ本文に内容を直接書き下す。リポジトリにコミット済みで誰でも辿れるパスや、ドキュメントの章番号（例 `doc 12 12.9`）の参照は許可する

**良い例**:

```
Implement Market Maker WebSocket and worker functionality

- Replace apscheduler with websockets in requirements
- Add WebSocket connection for admin interface to manage market maker (MM) state
- Create MM worker to handle real-time inventory updates and order management
- Introduce activity logging for MM operations in the admin UI
- Update HTML structure to include worker status and activity log
- Modify Docker setup to include MM worker service
- Enhance MM logic to support spread configuration and order placement
```

**悪い例:**

- `Update files` — 何を / なぜが不明
- 本文に diff の内容をそのまま並べる — 「なぜ」が抜ける
- 1行に複数の論点を詰め込む — 読みづらい
- `- テキスト{改行}{空白2文字}テキスト` と1項目を複数行に折り返す — 余分な空白が残り読みづらい
- 要約行の後に地の文の段落を置いて経緯を語る — 箇条書きの形式から外れ、変更点との対応が追えない。段落の内容は該当する項目に織り込む
- 箇条書きが7項目以上ある、または1項目が数百文字に及ぶ — 要約になっておらず、変更を分割すべき合図でもある
- `(#8 を解消)` のような `#<数字>` 参照 — GitHub が無関係な Issue/PR へ自動リンクし、参照先 (例 `tmp/` のローカル採番) も第三者には辿れない。所見の内容を言葉で書く

### 3. スタイル自動検査

コミット実行前に、作成したメッセージをチェッカースクリプト (`check-commit-style.ps1`、このスキルと同じディレクトリ) で検査する:

1. メッセージ全文 (Co-Authored-By 行まで含む) を一時ファイルに UTF-8 で保存する
2. チェッカーを実行する:

```bash
MSG="$(mktemp)"
cat > "$MSG" <<'MSGEOF'
<作成したコミットメッセージ全文>
MSGEOF
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$HOME/.claude/skills/commit/check-commit-style.ps1")" -Path "$(cygpath -w "$MSG")"
```

Windows 以外 (macOS / Linux) や PowerShell が使えない環境では、同等の Python 版 (`check-commit-style.py`) を使う:

```bash
python3 "$HOME/.claude/skills/commit/check-commit-style.py" --path "$MSG"
```

3. `error` が1件でも報告されたらメッセージを修正し、error が 0 件になるまで再検査する
4. `warning` は内容を確認し、ルール上問題ない場合 (例: 必要な日本語説明) のみそのまま進む
5. `info` は要約の質に関する助言であり、コミットは妨げない。ただし `bullet-count` / `bullet-length` が出た場合は、項目を統合・圧縮できないか、あるいは変更を分割すべきでないかを一度検討する
6. スクリプトが実行できない環境では、セクション2のルールに沿って手動で確認する

検知対象: 非 ASCII 記号 / 不可視文字 / チルダ / `#<数字>` / 箇条書きの折り返し / 本文の地の文 / 1行目の形式 / ローカル作業物への参照 / トレーラー欠落 / 要約不足 (項目数・項目長) など (詳細はスクリプト冒頭のコメント)。コミット済みメッセージの事後検査は引数なし実行 (`check-commit-style.ps1` = HEAD を検査) でも可能。

### 4. ステージングとコミット

以下を並列に実行する:

- 関連する未追跡ファイルをステージング（`git add -A` や `git add .` は避け、ファイルを明示的に指定する）
- HEREDOC でコミットメッセージを渡してコミット作成

```bash
git commit -m "$(cat <<'EOF'
Add /commit slash command for auto-generated commit messages

- Add .claude/commands/commit.md describing the workflow for inspecting diffs and drafting messages
- Document the preferred summary + bullet point format with a concrete good example

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

コミット完了後、`git status` を実行して成功を確認する。

### 5. プリコミットフック失敗時の対応

- `--no-verify` で**スキップしない**
- 根本原因を修正し、再ステージングして**新しいコミット**を作成する（フック失敗時はコミットが作成されていないため、`--amend` は前回のコミットを書き換えてしまう）

### 6. 完了報告

作成したコミットのハッシュとタイトルを報告する。プッシュはユーザーから明示的に指示されない限り行わない。

## 注意事項

- **明示的に指示されない限りプッシュしない**
- **明示的に指示されない限り `--amend` を使わない**
- **`--no-verify` でフックをスキップしない**
- **git config を変更しない**
- 変更が複数の論理的なまとまりに分かれている場合、ユーザーに分割コミットの可否を確認する
