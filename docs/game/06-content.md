# ユニット種別とマップの内容

ゲームが持つ内容データのうち、指揮の判断に必要なものを記録する。ユニットの価格と性能、マップの資源配置と出撃地点である。

観測と行動の対応表([03-observation.md](03-observation.md)、[04-actions.md](04-actions.md))が「どう読むか」であるのに対し、こちらは「何があるか」である。

## 内容データは二箇所にあり、片方だけでは足りない

**ユニット定義はディスク上の `.ini` とゲームコードの両方に存在する。** `assets/units` 以下の定義ファイルはエンジンのパーサ([03-observation.md](03-observation.md) の `custom.ag`)が読むものであり、ゲームを起動せずに解析できる。しかし `commandCenter`、`builder`、`landFactory`、`airFactory`、`seaFactory` のような中核の種別には定義ファイルが存在せず、enum `game.units.ar` の実装としてコードの中にしかない。

したがって**完全な一覧は実行中のプロセスからしか取れない**。エンジンは登録済みの全種別を静的な `ar.ae`(`ArrayList`)に保持しており、ここから列挙できる。**実行時確認**である。

`-nomods` で起動したときの内訳は次のとおりである。

| 項目 | 数 |
| --- | --- |
| レジストリ `ar.ae` の登録数 | 173 |
| 実効の種別数 | 156 |
| 定義ファイルが組み込みを置き換えている数 | 18 |

### 名前が二つあり、用途が違う

**`ar.a(String)` が受け付ける名前と、種別自身が `as.v()` で返す名前は一致しないことがある。** 定義ファイルが `overrideAndReplace` で組み込みの種別を置き換えると、引くときの名前は組み込みのもの、返る名前は定義ファイルのものになる。

```
spawn: resolved 'mammothTank' to com.corrodinggames.rts.game.units.custom.l reporting name 'c_mammothTank'
```

実際に生成されたユニットは `c_mammothTank` を名乗る。**生産とアップグレードの特殊アクション識別子 `"u_" + 内部名`([04-actions.md](04-actions.md))は `as.v()` の側で組み立てる必要がある。** 引くときの名前で組み立てると一致しない。

置き換えが起きている 18 種別では、価格などの値も定義ファイルの側が有効になる。例えば `helicopter` は enum の実装が 650、定義ファイルが 700 で、実効は 700 である。**レジストリの要素をそのまま読むのではなく、名前で引き直した結果を読む。**

### 種別から取れる値

`game.units.as` インタフェース経由で取る。難読化されたユニット種別は非公開の無名クラスであるため、実装クラスからは直接取得できない([04-actions.md](04-actions.md))。

| 項目 | メソッド | 確認 |
| --- | --- | --- |
| 内部名 | `as.v()` | 実行時確認 |
| 表示名(翻訳済み) | `as.e()` | 実行時確認 |
| クレジット価格 | `as.c()` | 実行時確認 |
| 技術レベル | `as.g()` | 実行時確認 |
| 建物か | `as.j()` | 実行時確認 |
| 建設を行えるユニットか | `as.l()` | 実行時確認 |
| 建造速度(1 フレームあたりの進捗) | `as.D()` | 実行時確認 |
| 移動タイプ | `as.o()` | 実行時確認 |

**`as.l()` は「建設可能か」ではなく「建設を行えるか」である。** 実行時に真を返したのは `builder` と `builderShip` だけであった。

移動タイプ `game.units.ao` の定数は順に `NONE`、`LAND`、`BUILDING`、`AIR`、`WATER`、`HOVER`、`OVER_CLIFF`、`OVER_CLIFF_WATER` である。建物は `NONE` を返す。

数値ステータス(最大体力、装甲、視界、速度、射程)は `as` には出ておらず、定義ファイル由来の種別なら `custom.l.cL` にある([03-observation.md](03-observation.md))。定義ファイルを持たない種別についてはコードの中にあり、実体の `am.cv` などから読むことになる。

### 実装クラスから取れる値

`as` に出ていなくても、定義ファイル由来の種別の実装クラス `com.corrodinggames.rts.game.units.custom.l` からは直接取れる。パーサ `custom.ag` がどの定義キーをどのフィールドへ書くかで確定する。いずれも**逆アセンブル確認**である。

| 項目 | 定義キー | メンバ | 型 |
| --- | --- | --- | --- |
| 最大攻撃距離(ワールド単位) | `maxAttackRange` | `custom.l.cL.i` | `float` |
| 対空可否 | `canAttackFlyingUnits` | `custom.l.eq` | `LogicBoolean` |
| 対地可否 | `canAttackLandUnits` | `custom.l.er` | `LogicBoolean` |
| 資源地点の上にしか置けないか | `placeOnlyOnResPool` | `custom.l.aJ` | `boolean` |

`custom.l.cL` は型のベース値である数値ステータス `custom.as` であり、その `i` が最大攻撃距離にあたる([03-observation.md](03-observation.md))。

**`LogicBoolean` は難読化されていない。** `LogicBoolean.isStaticTrue(x)` と `LogicBoolean.isStaticFalse(x)` が定数の場合を実体なしで決着させ、残りは武装したユニットを渡す `x.read(unit)` で決まる。

**採掘施設を種別から見分けられるのは `aJ` である。** `assets/units/extractor/extractor_common.ini` が `placeOnlyOnResPool: true` を持つ。資源地点の隣にたまたま建っているだけの建物と区別できるのはこの値であり、位置ではない。

定義ファイルを持たない種別にはこれらのいずれも存在しないが、それで正しい。該当するのはすべて建物か建設機であり、いずれも攻撃しないからである。**したがって射程と対空可否の種別表は起動時に一度で完成する。実体を見るまで待つ必要はない。**

## 主要なユニット

1 対 1 の陸戦で使うものである。価格は実行時のレジストリから引き直した実効値、名前は左が引くときの名前、右が名乗る名前である。

### 建物

| 引く名前 | 名乗る名前 | 価格 | 技術 | 建造速度 |
| --- | --- | --- | --- | --- |
| `turret` | `c_turret_t1` | 500 | 1 | 0.0008 |
| `antiAirTurret` | `c_antiAirTurret` | 600 | 1 | 0.0008 |
| `extractor` | `extractorT1` | 700 | 1 | 0.0010 |
| `landFactory` | `landFactory` | 700 | 1 | 0.0010 |
| `airFactory` | `airFactory` | 1000 | 1 | 0.0010 |
| `seaFactory` | `seaFactory` | 1000 | 1 | 0.0007 |
| `laserDefence` | `laserDefence` | 1200 | 1 | 0.0010 |
| `commandCenter` | `commandCenter` | 3000 | 1 | 0.0005 |

建造速度は 1 フレームあたりの進捗であり、エンジンの基準 60fps で `1 / (速度 × 60)` 秒かかる。`extractor` なら約 17 秒、`landFactory` なら約 17 秒、`commandCenter` なら約 33 秒である。

### 可動ユニット

| 引く名前 | 名乗る名前 | 価格 | 技術 | 移動 |
| --- | --- | --- | --- | --- |
| `tank` | `c_tank` | 350 | 1 | LAND |
| `hoverTank` | `hoverTank` | 450 | 1 | HOVER |
| `builder` | `builder` | 500 | 1 | LAND |
| `airShip` | `c_interceptor` | 600 | 1 | AIR |
| `hovercraft` | `hovercraft` | 600 | 1 | HOVER |
| `helicopter` | `c_helicopter` | 700 | 1 | AIR |
| `megaTank` | `megaTank` | 800 | 1 | LAND |
| `artillery` | `c_artillery` | 900 | 1 | LAND |
| `heavyTank` | `heavyTank` | 800 | 2 | LAND |
| `laserTank` | `laserTank` | 1300 | 2 | LAND |
| `mammothTank` | `c_mammothTank` | 3900 | 3 | LAND |
| `experimentalTank` | `experimentalTank` | 14000 | 3 | LAND |

### 距離の単位

**タイルは 20 ワールド単位である。** 定義ファイルの中に `${core.fogOfWarSightRange * 20 - 40}` という式が残っており、視界がタイル、射程がワールド単位であることと換算率の両方が確認できる。

| 項目 | 値 |
| --- | --- |
| 視界の既定値 | 15 タイル = 300 ワールド単位 |
| 視界の最大(前哨基地) | 44 タイル = 880 ワールド単位 |
| 武装可動ユニットの射程の中央値 | 200 ワールド単位 |
| 同 最大 | 400 ワールド単位 |
| 戦車の射程 | 130 ワールド単位 |

領域の大きさを決めるときの尺度になる([../system/02-interface.md](../system/02-interface.md))。

### 価格と戦闘力の関係

定義ファイルを持つ武装可動ユニットについて、価格と戦闘を決める量の対数相関を取った。

| 対 | 相関 |
| --- | --- |
| 価格 と 最大体力 | 0.89 |
| 価格 と 最大体力 × 毎秒ダメージ | 0.80 |
| 価格 と 毎秒ダメージ | 0.47 |
| 価格 と 射程 | 0.43 |

**価格は耐久をよく代理する。** この事実を軍事価値の定義に使う判断は [../system/03-script-policy.md](../system/03-script-policy.md) にある。

## マップ

組み込みの対戦マップは `assets/maps/skirmish` に 48 個ある。形式は Tiled の TMX で、エンジンが読むのと同じファイルである。

### 資源地点と出撃地点はタイルで表される

| 対象 | 表現 |
| --- | --- |
| 資源地点 | `res_pool` 属性を持つタイル。エンジンはこれをタイルオブジェクト `game.b.g` の `boolean` フィールド `i` として保持する |
| 出撃地点 | units タイルセットの `unit` 属性が `commandCenter` のタイル。`team` 属性で番号が付く |
| 樹木など | 同じ units タイルセットの他の `unit` 属性 |

**資源地点はタイルのフラグのままであり、ゲームオブジェクトにはならない。** したがってその位置は実行中のプロセスからは取れず、このマップファイルから読むしかない([03-observation.md](03-observation.md))。制御プロセスが読んだ位置をゲーム側へ渡し、ゲーム側が採掘施設との突き合わせで占有を数える。

外部タイルセットの参照はマップからの相対ではなく `assets/tilesets` からの相対で書かれている。またタイルセットの画像が地図の確保した ID 数より多くのタイルを持つことがあり、**次のタイルセットの `firstgid` で範囲を打ち切らないと、地面のタイルがユニットとして読み出される**。

### 実測: 48 マップの内容

出撃地点の数はファイル名の `[pN]` と一致した。

| マップ | 寸法(タイル) | 出撃地点 | 資源地点 |
| --- | --- | --- | --- |
| Beach landing (2p) | 200x160 | 2 | 20 |
| Big Island (2p) | 180x180 | 2 | 14 |
| Dire_Straight (2p) | 110x110 | 2 | 6 |
| Fire Bridge (2p) | 120x120 | 2 | 8 |
| Hills (2p) | 130x130 | 2 | 20 |
| Ice Island (2p) | 145x145 | 2 | 7 |
| Lake (2p) | 130x130 | 2 | 9 |
| Small_Island (2p) | 110x110 | 2 | 4 |
| Two_cold_sides (2p) | 135x125 | 2 | 10 |
| Depth charges (4p) | 100x100 | 4 | 10 |
| Desert (4p) | 110x110 | 4 | 6 |
| Ice Lake (4p) | 130x120 | 4 | 9 |
| Island freeze (4p) | 205x200 | 4 | 13 |
| Islands (4p) | 110x110 | 4 | 6 |
| Lava Maze (4p) | 110x110 | 4 | 8 |
| Lava Vortex (4p) | 150x150 | 4 | 40 |
| Magma Island (4p) | 180x180 | 4 | 24 |
| Manipulation (4p) | 170x100 | 4 | 16 |
| Nuclear war (4p) | 100x100 | 4 | 48 |

残る 29 マップは 3 人用が 2 個、6 人用が 3 個、8 人用が 15 個、10 人用が 9 個である。寸法は 80x270 から 350x350、資源地点は 9 から 104 である。

**寸法はワールド単位ではタイル数の 20 倍である。** 最小の 100x100 で 2000x2000、最大の 350x350 で 7000x7000 になる。

### 道具

ゲームを起動せずに読める。

```powershell
python tools\Show-MapRegions.py
python tools\Show-UnitCatalog.py
```

`Show-UnitCatalog.py` が読めるのは定義ファイルを持つ種別だけである。完全な一覧は実行中のプロセスから取る。

```powershell
.\tools\windows\Start-RwProbe.ps1 -Count 1 -Speed 5 -Seconds 90 -Map Lake -AgentOptions 'catalog=true'
```
