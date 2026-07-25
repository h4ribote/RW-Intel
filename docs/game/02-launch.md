# 起動と速度制御

ゲームを起動する方法と、シミュレーションを実時間より速く回す仕組みを記録する。ここに書くのはゲーム側の性質である。これを使って実際に学習環境を組む手順は [../system/06-runtime.md](../system/06-runtime.md) にある。

## 起動

ゲームに同梱された JVM を使う。`jvm64` は **OpenJDK 13 の完全な JDK** であり、`javac`、`javap`、`jar`、`instrument.dll` を含む。したがって JDK を別途導入する必要はなく、エージェントのビルドから実行まで同梱 JVM だけで完結する。同じ JVM でビルドと実行を行えばクラスファイルのバージョン不一致も起こらない。

最小の起動コマンドは次のとおりである。

```
jvm64\bin\java.exe -Xmx800M -Dfile.encoding=UTF-8 -Djava.library.path=. -cp "game-lib.jar;libs/*" com.corrodinggames.rts.java.Main -nodisplay -nosound -nomusic -nomods
```

## コマンドライン引数

`Main` の引数解析部から抽出した一覧である。公式のヘルプやドキュメントには記載がない。未知の引数を渡すとゲームは終了するため、綴りは正確である必要がある。

**動作を実測で確認したもの**

| 引数 | 動作 |
| --- | --- |
| `-nodisplay` | 表示モードの問い合わせを飛ばし、ウィンドウを 10x10 で作る。終了はせず、シミュレーションは完全に動作する |
| `-nosound` `-nomusic` | 音声を無効化する。`NullSoundFactory` に差し替わる |
| `-nomods` | 導入済み mod を読み込まない |
| `-width <n>` `-height <n>` | ウィンドウ寸法を上書きする |
| `-nobackground` | メニュー背景で動く戦闘を止める。オブジェクト数がゼロのまま推移することで確認した |
| `-sandbox` | 起動直後に固定マップのサンドボックスを開始する。詳細は [05-match-control.md](05-match-control.md) |

**引数として存在するが動作未確認のもの**

`-debug` `-debugscript` `-log` `-nologfile` `-lang` `-logcolor` `-canvasgl` `-replay_debug` `-nopreferipv4` `-noresources` `-safemode` `-extrasafemode` `-disable_vbos` `-disable_atlas` `-force_vbos` `-allowsoftwarerender` `-fullscreen` `-printunits` `-outputunitimages` `-oldreplays` `-teamshaders` `-noteamshaders` `-devdebug` `-postprocessing` `-nopostprocessing` `-disabletextureread` `-steam` `+connect_lobby`

このうち `-noresources` はエンジンの静的フィールド `l.aB` を立てることまで逆アセンブルで確認した。`-printunits` と `-outputunitimages` はバッチ処理的な用途を示唆するが未検証である。

**メニュー背景の戦闘は本物のシミュレーションである。** 起動しただけの状態でもオブジェクトが 130 から 890 個まで増える実戦が動いており、性能計測の負荷として使える。逆に純粋な待機状態を測りたい場合は `-nobackground` で止める。

## ヘッドレスの可否

**完全なヘッドレスは不可能である。** `-nodisplay` を付けても OpenGL コンテキストの生成は回避されず、10x10 のウィンドウが実際に作られる。起動ログに次が残る。

```
--- ERROR: Skipping display mode call
INFO:TargetDisplayMode: 10 x 10 x 0 @0Hz
INFO:Starting display 10x10
```

実用上は不可視に近く問題にならないが、GPU や表示装置のない環境で動かすには仮想ディスプレイが要る。この制約は学習基盤をクラウドへ持ち出す際に効いてくる。

なお、マップの読み込みはこの OpenGL コンテキストを必要とする。外部スレッドから試合を開始しようとすると失敗する。詳細は [05-match-control.md](05-match-control.md) を参照する。

## 速度制御

[01-internals.md](01-internals.md) のとおり、エンジンは delta 駆動で、delta は実経過時間に比例する。つまみは 2 つあり、直交している。

### 速度: `game.i.H`

`float` 型のフィールドで初期値は 1.0 である。delta に無条件で掛かるため、**ゲーム内時間はきっかり実時間の `H` 倍で進む**。フレームレートとは無関係であり、8 倍、10 倍、30 倍のいずれでも誤差なく一致することを実測した。

エンジンのシングルトンは試合をまたいで作り直されない。ただし試合の開始処理が `H` に触れるかは未確認であるため、値を定期的に確認して必要なら設定し直す作りにしておくのが安全である。

### 忠実度: フレームレート

1 ステップが担うゲーム内時間は次で決まる。

```
1ステップのゲーム内時間(ミリ秒) = 1000 * H / fps
```

フレームレートは 300 に固定されている。`java.u.a()` が `setTargetFrameRate(300)` を呼び、そのメソッドが `render()` から**毎フレーム**呼ばれるため、リフレクションで上書きしても次のフレームで戻される。上限を外すにはバイトコード改変が必要である。

したがって 300fps を上限とすると、`H = 10` で 1 ステップ 33 ミリ秒、すなわち通常の 30fps プレイと同じ粗さになる。Rusted Warfare は元来モバイルで 30fps 動作する delta 駆動設計であり、この範囲なら挙動は通常のプレイと変わらないと考えてよい。`H` をさらに上げるとステップが粗くなり、投射体の当たり判定や経路追従の挙動が変わりうる。

**この delta 駆動という性質は、速度を上げられる理由であると同時に、試合が再現しない理由でもある。** 詳細は [05-match-control.md](05-match-control.md) を参照する。
