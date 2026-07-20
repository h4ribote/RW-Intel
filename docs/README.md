# 文書の構成

内容を二つに分けている。**`game/` は Rusted Warfare そのものの性質**であり、このプロジェクトの都合とは無関係に成り立つ。**`project/` は RW-Intel の判断と設計**であり、こちらは方針が変われば変わる。

```
docs/
  game/      Rusted Warfare の仕様と内部構造(解析の結果)
  project/   RW-Intel の方針、実行基盤、モデル設計
```

## `game/` — ゲームの仕様と内部構造

Rusted Warfare 1.15 build #28 を逆アセンブルし、実行時に検証した結果である。クラス名とフィールド名は難読化されているため、各項目には **実行時確認 / 逆アセンブル確認 / 推定** の別を付けてある。

| 文書 | 内容 |
| --- | --- |
| [01-internals.md](game/01-internals.md) | 起動経路、ゲームループの構造、時間の進み方、クラス対応表 |
| [02-launch.md](game/02-launch.md) | コマンドライン引数、ヘッドレスの可否、速度と忠実度の制御 |
| [03-observation.md](game/03-observation.md) | 読み取れる状態のフィールド対応表。ユニット、種別、プレイヤー、マップ、視界 |
| [04-actions.md](game/04-actions.md) | 命令の発行方法。種別、対象指定、生産とアップグレード |
| [05-match-control.md](game/05-match-control.md) | 試合の開始、設定、終了検出、リセット、決定性 |

## `project/` — このプロジェクトの設計

| 文書 | 内容 |
| --- | --- |
| [01-approach.md](project/01-approach.md) | 接続方式の選定とその根拠 |
| [02-runtime.md](project/02-runtime.md) | ゲームの複製、並列実行の構成、計測エージェント |
| [03-throughput.md](project/03-throughput.md) | 速度と並列数の実測値。学習のサンプル収集予算 |
| [04-model-design.md](project/04-model-design.md) | 機械学習モデルの方針。**未実装** |

## 読む順序

初めて読む場合は [project/01-approach.md](project/01-approach.md) から入るとよい。なぜゲームのプロセスに寄生する方式を選んだのかが分かり、そこから `game/` の各文書が何のためにあるかが見える。

実装に手を付ける場合は [game/01-internals.md](game/01-internals.md) と [project/02-runtime.md](project/02-runtime.md) を先に読む。前者がゲームの構造、後者が実際に動かす手順である。

モデルの設計に関わる場合は [project/03-throughput.md](project/03-throughput.md) を先に読む。ここに書かれたサンプル収集の予算が、[project/04-model-design.md](project/04-model-design.md) の選択肢をほとんど決めている。
