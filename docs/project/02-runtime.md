# 実行基盤

学習環境としてゲームを走らせるための構成と道具を記録する。ゲーム側の起動引数や速度制御の仕組みそのものは [../game/02-launch.md](../game/02-launch.md) にある。

本基盤は二つの実行系を持つ。ネイティブに走らせる Windows 系(`tools/windows`)と、amd64 Linux コンテナの中でゲームを走らせて制御プロセスだけをホストにネイティブで置く macOS/Linux 系(`tools/macos`)である。まず Windows 系を、次に macOS/Linux 系を記す。制御プロセス、方策、通信形式、学習と評価のコードは共通で、違うのはゲームプロセスをどう起動して並列化するかだけである。

## ゲームの複製

インストール先をそのまま使わず、`local/rw` に複製して使う。32bit 版 JVM とログ類は不要である。

```powershell
robocopy "<ゲームのインストール先>" local\rw /E /XD jvm cache /XF "hs_err_pid*.log" lastrun.log crashes.txt preferences.ini
```

複製する理由は二つある。実行するとゲームは作業ディレクトリへ設定やセーブを書き戻すため、元のインストールを汚さないこと。そして並列実行のための構成をそこに組み立てるためである。

`local/` はバージョン管理の対象外である。ゲーム本体を含むためである。

## 並列実行のためのディレクトリ構成

ゲームは作業ディレクトリに対して `preferences.ini` を書き、`saves`、`cache`、`replays` を使う。したがって同時実行するプロセスには別々の作業ディレクトリが要る。

インストール全体を複製すると 1 インスタンスあたり 340MB かかる。代わりに、読み取りしかされないものはマスター複製を参照させる。

- `assets`、`font`、`res`、`mods` は NTFS のジャンクションでマスターを指す
- インストール直下のネイティブ DLL 23 個(55MB)はハードリンクで置く
- `saves`、`cache`、`replays` はインスタンス毎の実体を持つ

結果としてインスタンスの実ディスク消費はほぼゼロになる。

```mermaid
flowchart TD
    master["local/rw<br/>マスター複製 340MB"]
    subgraph inst["local/instances/NN"]
        j["assets, font, res, mods<br/>ジャンクション"]
        h["*.dll<br/>ハードリンク"]
        o["saves, cache, replays<br/>preferences.ini<br/>インスタンス固有"]
    end
    master -.->|参照| j
    master -.->|参照| h
```

DLL を省略できない理由がある。`-Djava.library.path` で `rocketConnector64.dll` 自体は見つかるが、その従属 DLL は Windows のローダが標準の探索順で解決し、そこに含まれるのは作業ディレクトリである。DLL を置かないと次で起動に失敗する。

```
java.lang.UnsatisfiedLinkError: rocketConnector64.dll: Can't find dependent libraries
```

ディレクトリはファイル単位のジャンクションを作れないためハードリンクを使う。ハードリンクは同一ボリューム内でのみ作成でき、管理者権限は不要である。

## 道具

| 道具 | 用途 |
| --- | --- |
| `tools/windows/New-RwInstance.ps1` | インスタンス用ディレクトリを作成する |
| `agent/build.ps1` | 制御エージェントをビルドする |
| `tools/windows/Start-RwAgents.ps1` | 制御プロセスへ接続するインスタンスを起動する |
| `tools/probe-agent/build.ps1` | 計測エージェントをビルドする |
| `tools/windows/Start-RwProbe.ps1` | 指定数のインスタンスを起動し、速度を集計する |
| `tools/windows/Measure-MatchOutcomes.ps1` | 同一の対戦を多数のエピソード回し、勝敗と長さの分布を報告する |
| `tools/Show-MapRegions.py` | マップを領域に切り出し、数と大きさを報告する |
| `tools/Show-UnitCatalog.py` | 定義ファイル由来のユニット種別を一覧し、価格が戦闘力を代理するかを検査する |

```powershell
.\tools\probe-agent\build.ps1
.\tools\windows\New-RwInstance.ps1 -Count 8
.\tools\windows\Start-RwProbe.ps1 -Count 8 -Speed 10 -Seconds 90
```

Python の二つはゲームを起動せずに動く。読むのはエンジンが読むのと同じファイルであり、追加の依存はない。共通の読み取りは `rwintel/data` にある。

## 二つのエージェント

**`agent/` が本体、`tools/probe-agent/` が計測用**である。役割が違うので統合しない。

| | `agent/` | `tools/probe-agent/` |
| --- | --- | --- |
| 目的 | 制御プロセスの指示で観測と行動を運ぶ | ゲームへの介入が成立することを確かめる |
| 相手 | 制御プロセス | ログ |
| 使う場面 | 学習と評価 | 解析、性能計測、文書に載せた実測の再現 |

計測エージェントは文書中の実測値を再現する手段でもあるため、本体が育っても残す。

## 制御プロセスとの起動順序

**制御プロセスを先に起動する。** エージェントは接続できるまで待ち、接続してから初めてエピソードが始まる。試合の設定は制御プロセス側にあり、エージェントは自分では何も始めない。

```powershell
python -m rwintel.control --instances 2 --episodes 2 --map Lake --max-seconds 300
.\tools\windows\Start-RwAgents.ps1 -Count 2 -Speed 10
```

制御プロセスは待ち受けポートを排他で確保する。**同じポートで二重に起動すると、Windows では後から起動した側も待ち受けに成功してしまい**、どちらが接続を受け取るかが不定になる。古いプロセスが生き残ったまま新しいコードを試していたことに気づかない、という形で現れる。

## 計測エージェント

`tools/probe-agent/RwProbeAgent.java` は `-javaagent` としてゲームプロセスに入り、**リフレクションのみで動作する**。バイトコード改変は行わない。現時点で次を行える。

| 機能 | 内容 |
| --- | --- |
| 速度制御 | 倍率 `H` を設定し、フレーム数とゲーム内時間から実効速度を報告する |
| 状態のダンプ | ゲームオブジェクトとプレイヤーの全フィールドを実値付きで出力する |
| 種別の一覧 | 登録済みの全ユニット種別を価格と技術レベル付きで出力する |
| 観測の計測 | 観測をゲームスレッド上で組み立て、その費用を報告する |
| 命令の発行 | ユニットに移動を命じ、追従したかを報告する |
| ユニットの生成 | システム命令でユニットを作り、実際に現れたかを報告する |
| 試合の進行 | スキルミッシュを開始し、勝敗を検出し、次のエピソードへ進む |
| 対戦の顔ぶれ | 対戦者を指定した数だけ残し、残りを観戦者に移す |

これは学習用の本体ではなく、**ゲームへの介入が成立することを実地で確かめるための道具**である。ここで確かめた経路が、そのまま観測と行動の実装の土台になる。

エージェントが依拠する三つの経路は次のとおりで、いずれも実行時に検証済みである。

- 入口は `Main.m`(静的な自己参照)と `l.B()`(エンジンのシングルトン)
- ゲームへの介入は `game.i.k` へ `Runnable` を投入する。命令の発行にもマップの読み込みにもこれが必要である
- 命令は `l.cf.b(player)` で取得したコマンドにフィールドを埋めることで発行される

詳細はそれぞれ [../game/01-internals.md](../game/01-internals.md)、[../game/04-actions.md](../game/04-actions.md)、[../game/05-match-control.md](../game/05-match-control.md) にある。

## macOS(および amd64 Linux)での実行

Apple Silicon の macOS ではゲームをネイティブに走らせられない。ゲームが同梱する LWJGL 2.9.3 の macOS ネイティブは x86_64 専用であり、その表示モード列挙のネイティブは現行 macOS でフォールトする。そこで、ゲーム本体は amd64 Linux ディストリビューションを Docker コンテナに入れて Rosetta で駆動し、制御プロセスはホストにネイティブで置く。同じイメージは実機の x86-64 Linux ホストではエミュレーションなしで走る。

Linux ディストリビューションは独自の JVM(`jvm-linux`、JRE)と独自のネイティブ(`liblwjgl64.so`、`librocketConnector.so`、libRocket 一式)を同梱するので、イメージが供給するのはそれらがリンクする先だけである。仮想ディスプレイ(Xvfb)、ソフトウェア OpenGL ラスタライザ(Mesa)、LWJGL が要求する X クライアントライブラリ、libRocket が要求する freetype と libstdc++ である。エンジンが開くウィンドウは 10x10 なので、ソフトウェアラスタライズの費用は測るに値しない。

### ゲームの複製とディレクトリ構成

macOS 系が使うゲーム本体は `local/RustedWarfare_Linux`、すなわち `jvm-linux` と `.so` ネイティブを持つ Linux ディストリビューションである。これは `local/rw`(JVM とネイティブを剥いだ macOS 複製)とは別物で、コンテナはネイティブなしでは動かない。

Windows 系のようなインスタンス用ディレクトリをホストに作る手順はない。コンテナモデルでは、ゲーム本体を読み取り専用でマウントし、インスタンスごとの作業ディレクトリはコンテナ内で起動時に作る(`tools/macos/rw-run.sh`)。作業ディレクトリは `assets`、`font`、`res`、`mods` とネイティブへのシンボリックリンク、および `saves`、`cache`、`replays` の実体だけからなる。したがってインスタンスの実ディスク消費はほぼゼロで、Windows 系のジャンクションとハードリンクが果たす役割をシンボリックリンクが果たす。

### 道具

| 道具 | 用途 |
| --- | --- |
| `tools/macos/build-image.sh` | ゲームを走らせる amd64 Linux ランタイムイメージ(`rw-linux:latest`)をビルドする |
| `agent/build.sh` | 制御エージェントをビルドする。JDK 9 以上が要る(Linux 版は JRE のみ、macOS 版は同梱なし)。Java 8 バイトコードに落とすのでゲームの Java 8 JVM で読める |
| `tools/probe-agent/build.sh` | 計測エージェントをビルドする。同様に JDK が要る |
| `tools/macos/start-probe.sh` | 指定数のインスタンスをコンテナで起動し、速度を集計する |
| `tools/macos/start-agents.sh` | 制御プロセスへ接続するインスタンスをコンテナで起動する |
| `tools/macos/learn-run.sh` | ホストの制御・学習コマンドとゲームコンテナのライフサイクルを結ぶ |
| `tools/macos/measure-match-outcomes.sh` | 同一の対戦を多数のエピソード回し、勝敗と長さの分布を報告する |
| `tools/macos/start-paired-match.sh` | 二つのゲームを一つのロックステップ試合に入れる |

Python の二つ(`tools/Show-MapRegions.py`、`tools/Show-UnitCatalog.py`)はゲームを起動せずに動くので、どちらの実行系でも同じである。読むのはエンジンが読むのと同じファイルで、追加の依存はない。

### 起動順序とホスト・コンテナの結線

Windows 系と同じく制御プロセスを先に起動する。ただしコンテナはホストを別ホストとして見るので、制御プロセスは `127.0.0.1` ではなく `0.0.0.0` で待ち受けさせ、コンテナ側は `host.docker.internal` でホストへ達する。

```bash
tools/macos/build-image.sh
agent/build.sh
python -m rwintel.control --host 0.0.0.0 --instances 2 --episodes 2 --map Lake --max-seconds 300
tools/macos/start-agents.sh -Count 2 -Speed 10
```

学習と評価の実行では、ホストの制御プロセスとコンテナのゲームは同時に生きていなければならない。Windows では二つのコンソールを人が並べるが、macOS では一方がホストプロセス、もう一方がコンテナで、両者の寿命を結ぶものがない。`learn-run.sh` がこれを結ぶ。ホストのコマンドを `--` の後にそのまま与えると、それを起動し、コンテナをその相手として立ち上げ、ホストのコマンドがエピソードを終えた瞬間にコンテナを止める。取り残したゲームが次の実行とコアを取り合うことがない。

```bash
tools/macos/learn-run.sh --count 8 --speed 10 -- \
    python -m rwintel.learn tactics --host 0.0.0.0 --instances 8 --episodes 40 \
        --load local/tactics-bc.pt --warmup 5 --save local/tactics.pt
```

### 待ち受けポートの排他はプラットフォームで向きが逆である

制御プロセスは待ち受けポートを、そのプラットフォームで安全な方の指定で確保する。二つのプラットフォームは逆の指定を要る。Windows では `SO_REUSEADDR` が既に待ち受けているポートへの二重 bind を許してしまい、どちらが接続を受け取るかが不定になるので、`SO_EXCLUSIVEADDRUSE` でそれを禁じる。macOS と Linux が使う BSD ソケットでは、`SO_REUSEADDR` は稼働中のリスナーからポートを奪うことを許さず(それには `SO_REUSEPORT` が要り、設定していない)、既に閉じた制御プロセスが `TIME_WAIT` に残したポートへの bind だけを通す。連続して実行を回すには、この bind が成功しなければならない。設定しないと、数秒あけた二つの実行が `Address already in use` で弾かれる。

### イメージの存在確認は名前で問うと amd64 単一プラットフォームで誤る

`docker image inspect <名前>` は名前をホストのプラットフォームのマニフェストに照らして解決するので、arm64 ホストでは amd64 のイメージを「無い」と報告する。`docker run` はそれでも見つけて走らせる。`tools/macos/_common.sh` の存在確認は `docker images -q` で、タグの実体を直接読んで存在すれば id を、なければ何も返さない。これがこの確認が本来問うている存在の問いである。

## 再現性のための注意

- **mod は無効化する。** 導入済み mod はユニット定義を書き換えるため、`-nomods` を付けないと条件が揃わない。実測環境には利用者が導入した mod が存在した。
- **計測は他の負荷がない状態で行う。** 本作業中、背景で解析処理を走らせたまま取得した計測値は 4 分の 1 以下に歪んだ。
- **最初の 30 秒程度は実行時コンパイルの影響で値が変動する。** フレームレートが 77 から 226 まで上昇する例を観測した。集計には初期のサンプルを含めない。
- **同一の設定でも試合そのものは再現しない。** これは環境の作り方ではなくゲームの性質である。[../game/05-match-control.md](../game/05-match-control.md) を参照する。

実測値は [03-throughput.md](03-throughput.md) にある。
