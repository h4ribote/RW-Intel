# 起動と速度制御

ゲームを起動する方法と、シミュレーションを実時間より速く回す仕組み、実時間から切り離す仕組みを記す。ここに書くのはゲーム側の性質である。これを使って実際に学習環境を組む手順は [../project/02-runtime.md](../project/02-runtime.md) にある。

## 起動

Linux 版はゲームに同梱された JVM `jvm-linux` を使う。これは **Java 8(1.8.0_131)の JRE** であり、`java` と `libinstrument.so` を含むので `-javaagent` は使えるが、`javac` も `jar` も `javap` も含まない。エージェントのビルドと逆アセンブルには別途 JDK が要る。ビルドは `--release 8` で行い、Java 8 のクラスファイルと API に揃える。

`game-lib.jar` はプラットフォームに依らない。Linux 版の jar は、解析の対象にした Steam 版と MD5 が一致する([../project/01-approach.md](../project/01-approach.md))。

最小の起動コマンドは次のとおりである。インストール先で実行する。

```bash
LD_LIBRARY_PATH=. jvm-linux/bin/java -Xmx800M -Dfile.encoding=UTF-8 -Djava.library.path=. -cp "game-lib.jar:libs/*" com.corrodinggames.rts.java.Main -nodisplay -nosound -nomusic -nomods
```

クラスパスの区切りは `:` である。`LD_LIBRARY_PATH` が要るのは、`librocketConnector.so` が依存する `libRocketCore.so.1` などを動的リンカが探すためで、`java.library.path` はそれに効かない。公式の起動スクリプト `rustedWarfareLinux.sh` も同じ指定をしている。インストール先以外のディレクトリから起動する場合は、両方にインストール先を渡す。

同梱 JRE の AWT は X のライブラリ `libXtst.so.6` を要求し、無ければ LWJGL の初期化で `UnsatisfiedLinkError` になって起動しない。

## コマンドライン引数

`Main` の引数解析部から抽出した一覧である。公式のヘルプやドキュメントには記載がない。未知の引数を渡すとゲームは終了するため、綴りは正確である必要がある。

**動作を実行時に確認したもの**

| 引数 | 動作 |
| --- | --- |
| `-nodisplay` | 表示モードの問い合わせを飛ばし、ウィンドウを 10x10 で作る。終了はせず、シミュレーションは完全に動作する |
| `-nosound` `-nomusic` | 音声を無効化する。`NullSoundFactory` に差し替わる |
| `-nomods` | 導入済み mod を読み込まない |
| `-width <n>` `-height <n>` | ウィンドウ寸法を上書きする |
| `-nobackground` | メニュー背景で動く戦闘を止める。オブジェクト数がゼロのまま推移する |
| `-sandbox` | 起動直後に固定マップのサンドボックスを開始する。詳細は [05-match-control.md](05-match-control.md) |

**引数として存在するが動作未確認のもの**

`-debug` `-debugscript` `-log` `-nologfile` `-lang` `-logcolor` `-canvasgl` `-replay_debug` `-nopreferipv4` `-noresources` `-safemode` `-extrasafemode` `-disable_vbos` `-disable_atlas` `-force_vbos` `-allowsoftwarerender` `-fullscreen` `-printunits` `-outputunitimages` `-oldreplays` `-teamshaders` `-noteamshaders` `-devdebug` `-postprocessing` `-nopostprocessing` `-disabletextureread` `-steam` `+connect_lobby`

このうち `-noresources` はエンジンの静的フィールド `l.aB` を立てることまで逆アセンブルで確認した。`-printunits` と `-outputunitimages` はバッチ処理的な用途を示唆するが未確認である。公式の起動スクリプトは `-nologfile` を先頭に渡されない限り出力を `lastrun.log` へ書き出す。

**メニュー背景の戦闘は本物のシミュレーションである。** 起動しただけの状態でも数百個のオブジェクトが動く実戦が進んでおり、性能計測の負荷として使える。純粋な待機状態を測りたい場合は `-nobackground` で止める。

## ヘッドレスの可否

**完全なヘッドレスは不可能である。** `-nodisplay` を付けても OpenGL コンテキストの生成は回避されず、10x10 のウィンドウが実際に作られる。起動ログに次が残る。

```
--- ERROR: Skipping display mode call
INFO:TargetDisplayMode: 10 x 10 x 0 @0Hz
INFO:Starting display 10x10
```

したがって表示装置の無い環境では仮想ディスプレイが要る。Xvfb と Mesa のソフトウェア描画で動作し、速度倍率とフレームレートの挙動は表示装置がある場合と変わらない。

**ウィンドウが 10x10 でも描画は安くない。** 費用の大半は塗る画素ではなく、ユニットごとの描画呼び出し、毎フレームのクリア、画面の入れ替えにかかり、スキルミッシュではシミュレーションそのものより重い。描画は描画層で止められ、止めてもシミュレーションは変わらない([01-internals.md](01-internals.md) の描画の経路)。

マップの読み込みはこの OpenGL コンテキストを必要とする。外部スレッドから試合を開始しようとすると失敗する。詳細は [05-match-control.md](05-match-control.md) を参照する。

## 速度制御

[01-internals.md](01-internals.md) のとおり、エンジンは delta 駆動で、delta は実経過時間に比例する。つまみは 2 つあり、直交している。

### 速度: `game.i.H`

`float` 型のフィールドで初期値は 1.0 である。delta に無条件で掛かるため、**ゲーム内時間はきっかり実時間の `H` 倍で進む**。フレームレートとは無関係であり、CPU に余裕がある限り倍率どおりの速度が出る。

エンジンのシングルトンは試合をまたいで作り直されない。ただし試合の開始処理が `H` に触れるかは未確認であるため、エージェントは値を定期的に確認して必要なら設定し直す。

### 忠実度: フレームレート

1 ステップが担うゲーム内時間は次で決まる。

```
1ステップのゲーム内時間(ミリ秒) = 1000 * H / fps
```

フレームレートの上限は設定 `highRefreshRate` が真なら 300、偽なら 120 である。`java.u.a()` が `setTargetFrameRate` でこの定数を書き、そのメソッドが `render()` から**毎フレーム**呼ばれるため、別スレッドから一度書き換えても次のフレームで戻される(逆アセンブル確認)。ただしコンテナが上限を読むのは `Display.sync` の直前であり、その直前の `GameContainer.GL.flush` の位置で書き換えれば毎フレーム効く([01-internals.md](01-internals.md) のループ構造、実行時確認)。

300fps を上限とすると、`H = 10` で 1 ステップ平均 33 ミリ秒、すなわち通常の 30fps プレイと同じ粗さになる。Rusted Warfare は元来モバイルで 30fps 動作する delta 駆動設計であり、この範囲なら挙動は通常のプレイと変わらないと考えてよい。`H` をさらに上げるか、CPU が足りずフレームレートが落ちるとステップが粗くなり、投射体の当たり判定や経路追従の挙動が変わりうる。

実際の 1 ステップは平均の周りで割れる。経過時間が整数ミリ秒で渡るため、300fps・倍率 10 では 30 ミリ秒と 40 ミリ秒が混ざり、上限を外すと 0 ミリ秒のフレームが大半になる([01-internals.md](01-internals.md) の時間の進み方)。

### 固定ステップ

経過時間 `java.u.t` と倍率 `H` を毎フレーム update と render の間で書き、上限を -1 にすると、**全フレームがちょうど同じゲーム内時間だけ進み、実時間から切り離される**(実行時確認)。1 ステップの粗さは書いた値で決まり、1 秒あたりに進むゲーム内時間は CPU が許す限り上がる。粗さは負荷に左右されなくなり、速さは CPU の余力をそのまま使う。

実時間から切り離しても、試合の中身は変わらない。経路探索は別スレッドで解くが、結果が効く時刻はゲーム内時間の期限で決まる([01-internals.md](01-internals.md) の経路探索)。内蔵 AI 同士の対戦を多数回したときの長さ、終局のユニット数、残存価値の差の分布は、ゲーム自身の時計で描画した場合と区別できない(実行時確認)。同じ乱数種の試合が再現しないことは変わらない([05-match-control.md](05-match-control.md))。

**この delta 駆動という性質は、速度を上げられる理由であると同時に、試合が再現しない理由の一つでもある。** 固定ステップにしても再現はしない。詳細は [05-match-control.md](05-match-control.md) を参照する。
