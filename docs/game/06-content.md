# ユニット種別とマップの内容

ゲームが持つ内容データのうち、指揮の判断に必要なものを記す。ユニットの価格と性能、マップの資源配置と出撃地点である。

観測と行動の対応表([03-observation.md](03-observation.md)、[04-actions.md](04-actions.md))が「どう読むか」であるのに対し、こちらは「何があるか」である。

## 内容データは二箇所にあり、片方だけでは足りない

**ユニット定義はディスク上の `.ini` とゲームコードの両方に存在する。** `assets/units` 以下の定義ファイルはエンジンのパーサ([03-observation.md](03-observation.md) の `custom.ag`)が読むものであり、ゲームを起動せずに解析できる。しかし `commandCenter`、`builder`、`landFactory`、`airFactory`、`seaFactory` のような中核の種別には定義ファイルが存在せず、enum `game.units.ar` の実装としてコードの中にしかない。

したがって**完全な一覧は実行中のプロセスからしか取れない**。エンジンは登録済みの全種別を静的な `ar.ae`(`ArrayList`)に保持しており、ここから列挙できる(実行時確認)。

`-nomods` で起動したときの内訳は次のとおりである。

| 項目 | 数 |
| --- | --- |
| レジストリ `ar.ae` の登録数 | 173 |
| 定義ファイルが組み込みを置き換えている数 | 18 |
| 名前で引き直して重複を除いた実効の種別数 | 155 |

制御エージェントは実効の種別を種別表として HELLO に載せる([../project/05-interface.md](../project/05-interface.md))。

### 名前が二つあり、用途が違う

**`ar.a(String)` が受け付ける名前と、種別自身が `as.v()` で返す名前は一致しないことがある。** 定義ファイルが `overrideAndReplace` で組み込みの種別を置き換えると、引くときの名前は組み込みのもの、返る名前は定義ファイルのものになる。`mammothTank` で引いた種別は `c_mammothTank` を名乗り、実際に生成されたユニットも `c_mammothTank` を名乗る。

**生産とアップグレードの特殊アクション識別子 `"u_" + 内部名`([04-actions.md](04-actions.md))は `as.v()` の側で組み立てる必要がある。** 引くときの名前で組み立てると一致しない。

置き換えが起きている種別では、価格などの値も定義ファイルの側が有効になる。例えば `helicopter` は enum の実装が 650、定義ファイルが 700 で、実効は 700 である。**レジストリの要素をそのまま読むのではなく、名前で引き直した結果を読む。**

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

**`as.l()` は「建設可能か」ではなく「建設を行えるか」である。** 実行時に真を返すのは `builder` と `builderShip` だけである。

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

**定義ファイルを持たない種別にはこれらのいずれも存在しない。** そうした種別の多くは建物か建設機だが、**武装しているものもある。** ホバー戦車(`hoverTank`)がその例で、陸上工場の第 1 段階のメニューに並ぶ。これらの値は種別の実装クラス(`game.units.ar` の列挙子が作るユニットクラス)のコードの中にある。

**エンジンは種別ごとの見本ユニットを持っている。** 静的メソッド `am.a(as)` が静的な `am.bF`(種別から見本への `HashMap`)から見本を返す。表は `am.bL()` が全種別について作り直す(逆アセンブル確認)。見本は盤面のユニットではない。見本から、最大体力はフィールド `am.cv`、最大攻撃距離は武装したユニットの基底 `y` のメソッド `m()` で読める。定義ファイル由来の種別では `m()` は `custom.as.i`、すなわち上の最大攻撃距離を返す。連射間隔は `y.b(int)` で読めるが、1 発のダメージはコード由来の種別では発射の処理の中に定数で書かれており、読む口が無い。

### 見本ユニットから取れる能力

見本は種別ごとのユニットの実体なので、種別のクラスが上書きするメソッドがそのまま答えになる。どれも実行時確認である。

| 項目 | メソッド | 値の例 |
| --- | --- | --- |
| 攻撃できるか | `l()` | 戦車、ホバー戦車、司令部は真。ホバークラフト、ドロップシップ、建設機、建設艇、修理所、レーザー防御は偽 |
| 最大攻撃距離、建設者なら建設の届く距離 | `m()` | 戦車 130、建設艇 240、陸の建設機 30。攻撃できない種別も正の値を返す |
| 速度 | `z()` | ワールド単位 / (1/60 秒)。戦車 1.1、ホバークラフト 0.9、ドロップシップ 2.3 |
| 輸送容量 | `bZ()` | ホバークラフト 4、輸送でなければ -1 |
| 占める枠 | `cw()` | 既定 1、実験戦車 5 |
| 今の積載 | `bY()` | 見本では 0 |
| 生産・配置できる種別 | `N()` の行動のうち、生産と配置の種類のもの | 段階 1 のメニュー。段階 2 以降は強化した実体の `N()` から取れる |
| そのユニットを積めるか | `d(am, false)` | 輸送の見本に乗客の見本を渡すと、実体どうしの積み込みの成否と一致する |

**攻撃できるかは `l()` で決まり、射程の有無では決まらない。** 攻撃できない輸送や建設機も `m()` に正の値を持つので、射程が正なら武装していると読むと、輸送を戦闘ユニットとして扱うことになる。

したがって種別表(HELLO)は、射程を定義ファイル由来の種別ではそのステータスから、それ以外では見本の `m()` から取り、攻撃できるか・速度・輸送容量・枠・メニュー・積める種別・最大体力は見本から取る。対地可否は、定義を持たない種別では見本の射程が正なら真とし、対空可否は偽とする。**種別表は起動時に一度で完成する。実体を見るまで待つ必要はない**([../project/05-interface.md](../project/05-interface.md) の種別表)。

## 主要なユニット

1 対 1 の陸戦で使うものである。価格と技術レベルは実行時のレジストリから名前で引き直した実効値であり、置き換えが起きている種別では定義ファイルの値と一致する。名前は左が引くときの名前、右が名乗る名前である。

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

建造速度は 1 フレームあたりの進捗であり、エンジンの基準 60fps で `1 / (速度 * 60)` 秒かかる。`extractor` なら約 17 秒、`landFactory` なら約 17 秒、`commandCenter` なら約 33 秒である。

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
| `laserTank` | `c_laserTank` | 1600 | 2 | LAND |
| `mammothTank` | `c_mammothTank` | 3900 | 2 | LAND |
| `experimentalTank` | `c_experimentalTank` | 14000 | 2 | LAND |

一覧は計測エージェントの `catalog=true` で出力できる([一覧を取る道具](#一覧を取る道具))。

### 距離の単位

**タイルは 20 ワールド単位である。** 定義ファイルの中に `${core.fogOfWarSightRange * 20 - 40}` という式が残っており、視界がタイル、射程がワールド単位であることと換算率の両方が確認できる。

| 項目 | 値 |
| --- | --- |
| 視界の既定値 | 15 タイル = 300 ワールド単位 |
| 視界の最大(前哨基地) | 44 タイル = 880 ワールド単位 |
| 武装可動ユニットの射程の中央値 | 200 ワールド単位 |
| 同 最大 | 400 ワールド単位 |
| 戦車の射程 | 130 ワールド単位 |

領域の大きさを決めるときの尺度になる([../project/05-interface.md](../project/05-interface.md))。

### 価格と戦闘力の関係

定義ファイルを持つ武装可動ユニットについて、価格と戦闘を決める量の対数相関を取ると次のようになる。

| 対 | 相関 |
| --- | --- |
| 価格 と 最大体力 | 0.89 |
| 価格 と 最大体力 x 毎秒ダメージ | 0.80 |
| 価格 と 毎秒ダメージ | 0.47 |
| 価格 と 射程 | 0.43 |

**価格は耐久をよく代理する。** この事実を軍事価値の定義に使う判断は [../project/06-script-policy.md](../project/06-script-policy.md) にある。

### 生産者のメニュー

メニューは生産者の段階で変わる。段階 1 は見本から、段階 2 は強化した実体から読んだ。

| 生産者 | 段階 1 | 段階 2 で加わるもの |
| --- | --- | --- |
| `landFactory` | builder scout c_tank hoverTank c_artillery | hovercraft missileTank plasmaTank heavyTank combatEngineer heavyArtillery heavyHoverTank c_laserTank c_mammothTank |
| `airFactory` | lightGunship c_interceptor c_helicopter | spyDrone dropship gunShip heavyInterceptor amphibiousJet aaBeamGunship bomber missileAirship |
| `seaFactory` | builderShip gunBoat lightSub missileShip hovercraft battleShip attackSubmarine | |
| `mechFactory` | builder mechGun mechMissile mechArtillery mechBunker | |
| `experimentalLandFactory`(段階 2 で立つ) | combatEngineer fireBee c_experimentalTank nautilusSubmarineLand experimentalHoverTank experimentalDropship experimentalSpider | |
| `commandCenter` | builder scout | |
| `builder` | extractorT1 c_turret_t1 c_antiAirTurret landFactory airFactory seaFactory mechFactory laserDefence repairbay fabricatorT1 experimentalLandFactory nukeLauncherC antiNukeLauncherC | |
| `builderShip` | extractorT1 c_turret_t1 c_antiAirTurret landFactory airFactory seaFactory fabricatorT1 laserDefence repairbay | |

**建設艇は水上から陸に建てる。** 届くのは見本の `m()`(240)の内側で、水から遠い内陸の資源地点への建設の命令は黙って消える。

## 輸送

**輸送は容量 `bZ()` が正の種別である。** 乗客は自分の枠 `cw()` を容量から使う。

| 種別 | 移動 | 容量 | 作れる所 |
| --- | --- | --- | --- |
| `hovercraft`(表示名 Landing Craft) | HOVER | 4 | `seaFactory` 段階 1、`landFactory` 段階 2 |
| `dropship` | AIR | 4 | `airFactory` 段階 2 |
| `experimentalDropship`(Flying Fortress) | AIR | 12 | `experimentalLandFactory` |
| `experimentalGunship` | AIR | 5 | |
| `bugPickup` | AIR | 2 | |
| `nautilusSubmarineSurface` | HOVER | 3 | |

**誰を積めるかはエンジンの `d(am, false)` が決める。** ホバークラフト、ドロップシップ、`bugPickup` は陸の車両、メック、ホバー戦車、建設機を積み、航空機、艦船、他の輸送、5 枠の種別を積まない。`experimentalDropship` と `experimentalGunship` は 5 枠の実験戦車も積む。`nautilusSubmarineSurface` は艦船とホバーを積み、陸の車両を積まない。

**乗っているユニットは全ユニットの一覧(`am.bE`)に残る。** 位置は輸送と同じになり、`am.cN` が乗っている輸送を指し、命令は持たない。**輸送が撃沈されると、乗っていたユニットも死ぬ。** 積み込みと降ろしの命令は [04-actions.md](04-actions.md) にある。

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

### 48 マップの内容

出撃地点の数はファイル名の `[pN]` と一致する。2 人用と 4 人用は次のとおりである。

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

残る 29 マップは 3 人用が 2 個、6 人用が 3 個、8 人用が 15 個、10 人用が 9 個である。寸法(幅 x 高さ)は 270x80 から 350x350、資源地点は 9 から 104 である。

**寸法はワールド単位ではタイル数の 20 倍である。** 最小の 100x100 で 2000x2000、最大の 350x350 で 7000x7000 になる。

### 地形と通行

**どのタイルを通れるかは移動タイプごとに違い、エンジンの経路探索が移動タイプごとの格子で持つ**(`l.bU` が `gameFramework.k.l`、格子は `k.i`、`a(ao, x, y)` が通れないタイルで真)。格子はタイルの属性から決まり、その対応は次のとおりである。タイルは、載っている属性のすべてが許す移動タイプだけを通す。属性を持たないタイルは水上(WATER)以外のすべてを通す。空(AIR)はどのタイルも通る。

| 属性 | LAND | OVER_CLIFF | HOVER | WATER | OVER_CLIFF_WATER |
| --- | --- | --- | --- | --- | --- |
| 無し、`small-rock`、`tree`、`small-cliff` | 通る | 通る | 通る | 通らない | 通る |
| `water` | 通らない | 通らない | 通る | 通る | 通る |
| `cliff-soft`、`cliff` | 通らない | 通る | 通る | 通らない | 通る |
| `large-cliff`、`trees` | 通らない | 通る | 通らない | 通らない | 通る |
| `res_pool` | 通らない | 通る | 通る | 通らない | 通る |
| `lava`、`lava-cliff`、`large-rock` | 通らない | 通らない | 通らない | 通らない | 通らない |

**読む属性は `Ground` と `Items` の層のものだけである。** 32 のマップが持つ `set`(綴りに `Set`、`set-disabled` もある)はエディタの補助の層で、そこに描かれた水や崖をエンジンは読まない。立っている建物は足元のタイルをすべての地上の移動タイプについて塞ぐ。この対応は組み込み 48 マップのすべてで、建物の足元を除く全タイルについてエンジンの格子と一致する(`rwintel/data/terrain.py` の `ALLOWS`)。

**移動タイプの通れるタイルが辺でつながった集まりを、その移動タイプの連結成分と呼ぶ**(`terrain.components`)。成分が違えば、その移動タイプのユニットは歩いて(浮かんで)は行けない。戦車や建設機(LAND)は小さな崖も越えられず、資源地点のタイルにも乗れないが、資源地点の隣までは行ける。建設機は陸を歩くので、本拠地と別の LAND の成分にある資源地点へは、運ばなければ建てに行けない。

Lake と Big Island は、LAND でほぼ全体が 1 つの成分であり、両方の司令部がそこに乗る。Beach landing は島の地図である。中央の本島は LAND で 1 つの成分で、資源地点が 8 つあり、東西に走る崖の壁で仕切られているが壁の端が開いている。両方の司令部は左右の小島にあり、資源地点はそれぞれ 2 つである。初期配置は各側に建設機 3、ホバークラフト 2 と、小島の脇の水上の海軍工場 1 で、周りには資源地点 1 つずつの小島が散らばる。HOVER では地図のほぼ全体が 1 つの成分になる。本島へ渡るには、建設機や部隊をホバークラフトなどの輸送に積んで運ぶ([輸送](#輸送))。

### 一覧を取る道具

ゲームを起動せずに読める。`regions` はマップごとの寸法・人数・資源地点の数と領域の切り出しを、`units` は定義ファイル由来の種別の一覧と上の相関を出す。`terrain` は、指定したマップの地面の内訳と、移動タイプごと(既定は LAND、HOVER、WATER、`--movement` で選ぶ)の大きい順の連結成分(タイル数、範囲、資源地点の数、そこにある初期配置)と、その移動タイプが届かない印を出す。`--png` を付けると、地形を色分けし資源地点と初期配置を重ねた画像を、既定では `local/reports/terrain/` に書く。マップは、`[pN]` の後の名前の書き出しかファイル名の一部、または TMX のパスで指定する。

```bash
python -m rwintel.data regions
python -m rwintel.data units
python -m rwintel.data terrain Beach Lake "Big Island" --png
```

`units` が読めるのは定義ファイルを持つ種別だけである。完全な一覧は実行中のプロセスから取る。

```bash
python -m rwintel.runtime probe --count 1 --speed 5 --seconds 90 --map Lake --agent-options catalog=true
```

`catalog:` の行に、引く名前・名乗る名前・表示名・価格・技術レベル・建物か・建設を行えるか・建造速度・移動タイプ、そして定義ファイルが置き換えたかどうかが種別ごとに出る。
