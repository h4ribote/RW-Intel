# 文書の構成

内容を三つに分けている。**分ける軸は「何によって変わるか」である。**

```
docs/
  game/     Rusted Warfare の仕様。ゲームが更新されない限り変わらない
  system/   現状の実装が実際に何をするか。コードが変われば変わる
  record/   進捗と開発の記録。測るたびに増え、測り直せば書き換わる
```

**`game/` は Rusted Warfare そのものの性質**であり、このプロジェクトの都合とは無関係に成り立つ。**`system/` は RW-Intel が何でできているかの仕様書**であり、コードと一対一で対応する——ここに書いてあって実装されていないもの、実装されていてここに無いものは、どちらも不整合として直す対象である。**`record/` は何を決め、何を測り、何を測り直して取り下げたかの記録**であり、`system/` の各節が「なぜこの形なのか」を言うときの根拠がここにある。

**`system/` は学習した方策の強さについて何も主張しない。** それは `record/` の仕事である。逆に `record/` は形式や定数を定義しない。**同じ数字が両方に出ることはあるが、定義するのは常に `system/` の側である。**

## `game/` — ゲームの仕様と内部構造

Rusted Warfare 1.15 build #28 を逆アセンブルし、実行時に検証した結果である。クラス名とフィールド名は難読化されているため、各項目には **実行時確認 / 逆アセンブル確認 / 推定** の別を付けてある。このうち [game/07-multiplayer.md](game/07-multiplayer.md) は大半が逆アセンブル確認であり、実行時に確かめてあるのは二つのプロセスを一つのセッションに入れて同期が保たれることまでである。

| 文書 | 内容 |
| --- | --- |
| [01-internals.md](game/01-internals.md) | 起動経路、ゲームループの構造、時間の進み方、クラス対応表 |
| [02-launch.md](game/02-launch.md) | コマンドライン引数、ヘッドレスの可否、速度と忠実度の制御 |
| [03-observation.md](game/03-observation.md) | 読み取れる状態のフィールド対応表。ユニット、種別、プレイヤー、マップ、視界 |
| [04-actions.md](game/04-actions.md) | 命令の発行方法。種別、対象指定、生産とアップグレード |
| [05-match-control.md](game/05-match-control.md) | 試合の開始、設定、終了検出、リセット、決定性 |
| [06-content.md](game/06-content.md) | ユニット種別の一覧と価格、マップの資源配置と出撃地点 |
| [07-multiplayer.md](game/07-multiplayer.md) | 対戦セッションの開設と参加、ロックステップ、同期検査、パケット形式。**接続と同期検査は実行時確認、残りは逆アセンブル確認** |
| [08-builtin-ai.md](game/08-builtin-ai.md) | 内蔵 AI の構造。思考の周期、発行する命令、難易度、打ち回しを学習に使えるか。**全て逆アセンブル確認** |

## `system/` — 実装の仕様書

| 文書 | 内容 | 実装 |
| --- | --- | --- |
| [01-architecture.md](system/01-architecture.md) | 層の構成、契約、部隊、制約が締め出すもの、人間の介入、学習環境の作り分け | 全体 |
| [02-interface.md](system/02-interface.md) | 観測と行動の外部化。通信、フレーム形式、周期、領域の切り出し | `agent/`、`rwintel/wire` |
| [03-script-policy.md](system/03-script-policy.md) | 契約の項目と値域、部隊管理、五層のスクリプト方策、スクリプト乱入者 | `rwintel/control/policy/`、`rwintel/control/intruder.py` |
| [04-learning.md](system/04-learning.md) | 符号化、行動空間、報酬、交戦アリーナ、構築作戦アリーナ、推論の集約、軌跡の収集、最適化、模倣 | `rwintel/learn` |
| [05-evaluation.md](system/05-evaluation.md) | 採点、必要エピソード数、対にした比較、記録するもの | `rwintel/eval` |
| [06-runtime.md](system/06-runtime.md) | ゲームの複製、並列実行の構成、計測エージェント、macOS でのコンテナ構成 | `tools/` |

指揮系統の外から命令するものは二つの文書に分かれる。`rwintel/control/intervention.py`(唯一の介入経路)と `rwintel/control/console.py`(人間が実際に打つ行入力)が [system/01-architecture.md](system/01-architecture.md) の[人間の介入](system/01-architecture.md#人間の介入)、`rwintel/control/intruder.py`(設計の頻度で干渉するスクリプト乱入者)が [system/03-script-policy.md](system/03-script-policy.md) の[乱入への頑健性](system/03-script-policy.md#乱入への頑健性)である。`rwintel/control/pairing.py` は 2 プロセスを 1 つのロックステップ試合に入れるもので、セッションと同期検査そのものは [game/07-multiplayer.md](game/07-multiplayer.md)、それで確かめた生成命令の同期安全性は [system/01-architecture.md](system/01-architecture.md) の[学習環境の作り分け](system/01-architecture.md#学習環境の作り分け)にある。`rwintel/data` が読むマップとユニット定義は [game/06-content.md](game/06-content.md) である。

## `record/` — 進捗と開発の記録

| 文書 | 内容 |
| --- | --- |
| [01-approach.md](record/01-approach.md) | 接続方式の選定とその根拠 |
| [02-throughput.md](record/02-throughput.md) | 速度と並列数の実測値。学習のサンプル収集予算 |
| [03-tactics.md](record/03-tactics.md) | 戦術層と交戦アリーナ。測定の床、動かせる幅、訓練の相手が既定で自分自身だったこと、作戦アリーナで戦術層を測る道が構造で閉じていること、学習実行、アリーナの八つの欠陥、取り下げた読み |
| [04-operations.md](record/04-operations.md) | 作戦層と構築作戦アリーナ。フル試合で分解できないこと、場を組んだ経緯、梯子を上回ったこと、集中が効くこと、報酬の二つの欠陥とバッチの census、凍結交互学習の一巡と、梯子も集中も越えたこと、そしてその計器が系統的に正へ傾いていること |
| [05-strategy.md](record/05-strategy.md) | 戦略層。この層だけが試合の結果を受け取ること、構築した場が作れないこと、手書きの鎖がどこから始まるか |
| [06-open-questions.md](record/06-open-questions.md) | 作業の残り、分かっていないこと、次に試すこと |

**記録は「何が正しかったか」だけでなく「何を誤って読み、なぜそう読めたか」を残す。** この計画では、測定そのものの欠陥が測定の結論を繰り返し書き換えている。取り下げた読みを消してしまうと、同じ誤りを二度目に防ぐものが無くなる。

## 読む順序

**初めて読む場合は [record/01-approach.md](record/01-approach.md) から入るとよい。** なぜゲームのプロセスに寄生する方式を選んだのかが分かり、そこから `game/` の各文書が何のためにあるかが見える。

**実装に手を付ける場合は [game/01-internals.md](game/01-internals.md) と [system/06-runtime.md](system/06-runtime.md) を先に読む。** 前者がゲームの構造、後者が実際に動かす手順である。そのうえで [system/02-interface.md](system/02-interface.md) が最初に書くものを決めている。

**モデルの設計に関わる場合は [record/02-throughput.md](record/02-throughput.md) を先に読む。** ここに書かれたサンプル収集の予算が、[system/01-architecture.md](system/01-architecture.md) の選択肢をほとんど決めている。骨格は 01 にあり、具体の形式と値は 02 から 05 にある。

**学習の数字を読む場合は [record/03-tactics.md](record/03-tactics.md) の「測定の床は ±0.02 で、これがすべての読み方を縛る」から入る。** そこを読まずにこの計画の成績を読むと、床の上の揺らぎを改善と読むことになる。
