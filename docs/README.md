# 文書の構成

内容を二つに分けている。**`game/` は Rusted Warfare そのものの性質**であり、このプロジェクトの都合とは無関係に成り立つ。**`project/` は RW-Intel の判断と設計**であり、こちらは方針が変われば変わる。

```
docs/
  game/      Rusted Warfare の仕様と内部構造(解析の結果)
  project/   RW-Intel の方針、実行基盤、モデル設計
```

実装が対応する文書は次のとおりである。`agent/` と `rwintel/wire` が [project/05-interface.md](project/05-interface.md)、`rwintel/data` が [game/06-content.md](game/06-content.md)、`rwintel/control/policy/` が [project/06-script-policy.md](project/06-script-policy.md)、`rwintel/eval` が [project/07-evaluation.md](project/07-evaluation.md)、`rwintel/learn` が [project/08-learning.md](project/08-learning.md) である。指揮系統の外から命令するものは二つの文書に分かれる。`rwintel/control/intervention.py`(唯一の介入経路)と `rwintel/control/console.py`(人間が実際に打つ行入力)が [project/04-model-design.md](project/04-model-design.md) の[人間の介入](project/04-model-design.md#人間の介入)、`rwintel/control/intruder.py`(設計の頻度で干渉するスクリプト乱入者)が [project/06-script-policy.md](project/06-script-policy.md) の[乱入への頑健性](project/06-script-policy.md#乱入への頑健性)である。`rwintel/control/pairing.py` は 2 プロセスを 1 つのロックステップ試合に入れるもので、セッションと同期検査そのものは [game/07-multiplayer.md](game/07-multiplayer.md)、それで確かめた生成命令の同期安全性は [project/04-model-design.md](project/04-model-design.md) の[学習環境の作り分け](project/04-model-design.md#学習環境の作り分け)にある。

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

## `project/` — このプロジェクトの設計

| 文書 | 内容 |
| --- | --- |
| [01-approach.md](project/01-approach.md) | 接続方式の選定とその根拠 |
| [02-runtime.md](project/02-runtime.md) | ゲームの複製、並列実行の構成、計測エージェント |
| [03-throughput.md](project/03-throughput.md) | 速度と並列数の実測値。学習のサンプル収集予算 |
| [04-model-design.md](project/04-model-design.md) | 機械学習モデルの骨格。層の構成、制約、報酬、人間の介入。**具体の形式と値は 05 から 08 にある** |
| [05-interface.md](project/05-interface.md) | 観測と行動の外部化。通信、形式、周期、領域の切り出し。**実装済み** |
| [06-script-policy.md](project/06-script-policy.md) | 契約の項目と値域、部隊管理、スクリプト方策、スクリプト乱入者。**実装済み** |
| [07-evaluation.md](project/07-evaluation.md) | 方策の比較手順と必要なエピソード数。**実装済み** |
| [08-learning.md](project/08-learning.md) | 符号化、行動空間、報酬、交戦アリーナ、推論の集約、軌跡の収集、最適化。**実装済み。学習は未実行** |

## 読む順序

初めて読む場合は [project/01-approach.md](project/01-approach.md) から入るとよい。なぜゲームのプロセスに寄生する方式を選んだのかが分かり、そこから `game/` の各文書が何のためにあるかが見える。

実装に手を付ける場合は [game/01-internals.md](game/01-internals.md) と [project/02-runtime.md](project/02-runtime.md) を先に読む。前者がゲームの構造、後者が実際に動かす手順である。そのうえで [project/05-interface.md](project/05-interface.md) が最初に書くものを決めている。

モデルの設計に関わる場合は [project/03-throughput.md](project/03-throughput.md) を先に読む。ここに書かれたサンプル収集の予算が、[project/04-model-design.md](project/04-model-design.md) の選択肢をほとんど決めている。骨格は 04 にあり、具体の形式と値は 05 から 08 にある。
