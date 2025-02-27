import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def calcroute(callback: str=None, uturnavoid: str=None, routeid: str=None, impassablearea: str=None, routeresultid: str=None, uturn: str=None, regulations: str=None, resulttype: str=None, fmt: str=None, tollroad: str=None, passablearea: str=None, daytime: str=None, weight: int=None, date: str=None, height: int=None, travel: str=None, danger: str=None, highwayspeed: int=None, generalroad: str=None, width: int=None, loadage: int=None, vehicletype: str=None, cartype: str=None, tollwayspeed: int=None, normalspeed: int=None, ferry: str=None, via: str=None, startangle: int=None, priority: str=None, destination: str='139.62261961,35.46606942', tollway: str=None, etc: str=None, start: str='139.76730676,35.68095910', smartic: str=None, ferryspeed: int=None, tolltarget: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## ルート検索
		ルート検索機能を提供します。
		
		#### 注記
		ルート検索APIを実行する際、距離の長さや検索条件によってはレスポンスが返却されるまで時間がかかる場合があります。"
    callback: ### JSONPのコールバック用関数名
関数名が指定されている場合はJSONP形式で結果を返却します。
指定されていない場合は、`fmt`で指定された形式で結果を返却します。
        uturnavoid: ### 経由地でのUターン回避
|経由地でのUターン回避|値|
|---|---|
|しない|0|
|する|1|

#### デフォルト
`priority`で指定した値により異なります。
- `priority=3、100～103`の場合
`0`
- `priority`が上記以外の場合
`1`

#### 注記
- `travel=1`とした場合、本設定は指定は出来ません。
- 本パラメータは`uturn`（Uターンのし易さ）とは異なりますのでご注意ください。
        routeid: ### ルート識別子
ルート検索を識別する任意の文字列を指定します。
指定した文字列が、レスポンスにそのまま出力されます。

ルート検索結果に影響はしません。

#### 注記
- URLエンコードされたUTF-8の文字列を指定します。
- 半角のダブルクォーテーションの使用は禁止ですが、それ以外の半角記号や全角文字はURLエンコードを行っていれば使用可能となります。
        impassablearea: ### 通行不可エリア
ルート検索対象外となるエリアを矩形で指定します。
指定されたエリア内を回避するルートが検索されます。

#### 記述形式

##### 通行可能エリア
`[エリア#1]|[エリア#2]|...|[エリア#n]`

##### エリア
`[南西経度の値],[南西緯度の値],[北東経度の値],[北東緯度の値]`

#### 範囲
|n（エリアの個数）|値|
|---|---|
|最大|10|

#### 記述例
`139.6965,35.6867,139.7029,35.6921|139.7000,35.6818,139.7044,35.6879`

#### 注記
- 緯度経度の測地系は、世界測地系（JGD2011）となります。
- 緯度経度の小数点は第8位まで考慮され、それより大きい値は無視されます。
- 指定した矩形と整合性の無い`start`、`destination`を指定した場合、エラー（`I00104`）になります。
- `passablearea`と`impassablearea`で重なった部分については、`impassablearea`が優先されます。
- `travel=1`とした場合、本設定は指定は出来ません。
- 長距離ルート検索時に本パラメータを指定した場合、レスポンスの返却が遅くなる場合があります。
        routeresultid: ### ルート結果ID
`resulttype=1`を指定して事前にルート検索した結果に格納されている`routeresultid`を指定します。
ルート結果ID取得時の条件で検索した結果の探索結果サマリ、誘導データ、経路形状データを出力します。

#### 注記
本パラメータを指定した場合、以下のパラメータが有効となり、それ以外のパラメータは無効となります。
- fmt
- callback
        uturn: ### Uターンのし易さ
|Uターンのし易さ|値|
|---|---|
|Uターンを比較的行う|0|
|Uターンを比較的行わない|1|

#### デフォルト
`0`

#### 注記
本パラメータは`uturnavoid `（経由地でのUターン回避）とは異なりますのでご注意ください。
        regulations: ### その他規制
|設定|値|
|---|---|
|一方通行規制の有効/無効を、<br>generalroad、tollroadの設定に従う|0|
|一方通行規制を有効にする|1|

#### デフォルト
`0`

#### 注記
`daytime`、`generalroad`、`tollroad`との関係性は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ ルート検索 ＞ 補足』をご確認ください。
        resulttype: ### ルート検索結果の出力項目設定
|出力項目設定|値|
|---|---|
|標準のルート検索結果の取得|0|
|ルート結果IDの取得|1|

#### デフォルト
`0`

#### 注記
- `resulttype=0`は、探索結果サマリ、誘導データ、経路形状データを出力します。
- `resulttype=1`を指定して取得したルート結果IDの有効期限は、1ヶ月となります。
        fmt: ### 出力フォーマット
|フォーマット|値|
|---|---|
|JSON|json |
|XML|xml|

#### デフォルト
`json`

#### 注記
`callback`が設定されている場合、本パラメータの設定が破棄され、JSONP形式が適用されます。
        tollroad: ### 高速道規制
無効とした場合、高速道の規制を無視したルート検索が可能となります。

|設定|値|
|---|---|
|高速道規制を無効にする|0|
|高速道規制を有効にする|1|

#### デフォルト
`1`

#### 注記
- `daytime`、`regulations`との関係性は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ ルート検索 ＞ 補足』をご確認ください。
- 高速道規制の対象となる道は、道路種別100番台と11番の道です。道路種別は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ 付則 ＞ ルート機能 ＞ 道路種別一覧』をご確認ください。
        passablearea: ### 通行可能エリア
ルート検索対象となるエリアを矩形で指定します。
指定されたエリア内のみを通行するルートが検索されます。

#### 記述形式

##### 通行可能エリア
`[エリア#1]|[エリア#2]|...|[エリア#n]`

##### エリア
`[南西経度の値],[南西緯度の値],[北東経度の値],[北東緯度の値]`

#### 範囲
|n（エリアの個数）|値|
|---|---|
|最大|10|

#### 記述例
`139.6965,35.6867,139.7029,35.6921|139.7000,35.6818,139.7044,35.6879`

#### 注記
- 緯度経度の測地系は、世界測地系（JGD2011）となります。
- 緯度経度の小数点は第8位まで考慮され、それより大きい値は無視されます。
- 指定した矩形と整合性の無い`start`、`destination`を指定した場合、エラー（`I00104`）になります。
- `passablearea`と`impassablearea`で重なった部分については、`impassablearea`が優先されます。
- `travel=1`とした場合、本設定は指定は出来ません。
- 長距離ルート検索時に本パラメータを指定した場合、レスポンスの返却が遅くなる場合があります。
        daytime: ### 日時規制
|設定|値|
|---|---|
|日時規制を無効にする|0|
|日時規制の有効/無効を、<br>generalroad、tollroad、regulationsの設定に従う|1|

#### デフォルト
`1`

#### 注記
`generalroad`、`tollroad`、`regulations`との関係性は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ ルート検索 ＞ 補足』をご確認ください。
        weight: ### 車重
道路規制で考慮する車重を指定します。

#### 範囲
|範囲|値|
|---|---|
|最小|0|

#### 注記
単位は、kg です。
        date: ### 出発日時
ルート検索時の交通規制を考慮するときの日時を指定します。

#### 記述形式
`[年4桁][月2桁][日2桁]_[時2桁][分2桁][秒2桁]` （`yyyyMMdd_HHmmss`）

#### 記述例
`20220420_170505` （2022年4月20日午後5時5分5秒 ）

#### デフォルト
API実行日時を適用して、ルート検索を行います。
        height: ### 車高
道路規制で考慮する車高を指定します。

#### 範囲
|範囲|値|
|---|---|
|最小|0|

#### 注記
単位は、cm です。
        travel: ### 巡回ルート設定
有効とした場合、効率よく経由地を回るルート検索を行うことができます。

|設定|値|
|---|---|
|巡回ルートによる検索を無効にする|0|
|巡回ルートによる検索を有効にする|1|

#### デフォルト
`0`

#### 注記
以下のパラメータが指定されている場合、巡回ルートによる検索を有効にする指定は出来ません。
- `priority=100～103`
- `passablearea`
- `impassablearea`
- `uturnavoid`
        danger: ### 危険物積載車両
道路規制で考慮する危険物積載車両であるかを指定します。

|設定|値|
|---|---|
|危険物積載車両でない|0|
|危険物積載車両である|1|

#### デフォルト
`0`
        highwayspeed: ### 高速道の走行速度
高速道の走行時間の計算に用います。

### 記述例
`90.0`

#### 注記
- 単位は、km/h です。
- 経路種別との関係およびデフォルト値は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ 付則 ＞ ルート機能 ＞ 経路種別一覧』をご確認ください。
        generalroad: ### 一般道規制
無効とした場合、一般道の規制を無視したルート検索が可能となります。

|設定|値|
|---|---|
|一般道規制を無効にする|0|
|一般道規制を有効にする|1|

#### デフォルト
`1`

#### 注記
`daytime`、`regulations`との関係性は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ ルート検索 ＞ 補足』をご確認ください。
        width: ### 車幅
道路規制で考慮する車幅を指定します。

#### 範囲
|範囲|値|
|---|---|
|最小|0|

#### 注記
単位は、cm です。
        loadage: ### 積載量
道路規制で考慮する車両の最大積載量を指定します。

#### 範囲
|範囲|値|
|---|---|
|最小|0|

#### 注記
単位は、kg です。
        vehicletype: ### 車種
道路規制を考慮する車種を指定します。

|車種|値|
|---|---|
|指定なし|0|
|大型乗用自動車|1|
|大型貨物自動車|6|
|大型特殊自動車|11|

#### デフォルト
`0`
        cartype: ### 有料道路利用時の車種
有料道路の料金計算のみに使用します。

|車種|値|
|---|---|
|軽自動車|0|
|普通車|1|
|中型車|2|
|大型車|3|
|特大車|4|

#### デフォルト
`1`
        tollwayspeed: ### 有料道の走行速度
有料道の走行時間の計算に用います。

### 記述例
`55.0`

#### 注記
- 単位は、km/h です。
- 経路種別との関係およびデフォルト値は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ 付則 ＞ ルート機能 ＞ 経路種別一覧』をご確認ください。
        normalspeed: ### 一般道の走行速度
一般道の走行時間の計算に用います。

### 記述例
`25.0`

#### 注記
- 単位は、km/h です。
- 経路種別との関係およびデフォルト値は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ 付則 ＞ ルート機能 ＞ 経路種別一覧』をご確認ください。
        ferry: ### フェリー使用条件
|フェリー使用|値|
|---|---|
|標準|0|
|優先|1|
|回避|2|
|絶対回避|3|

#### デフォルト
`0`

#### 注記
- フェリーを使わないとルートが引けない条件で、
    - `ferry=2` とした場合、フェリーを使ったルートが結果として返ります。
    - `ferry=3` とした場合、エラー（`I00107`）となります。
        via: ### 経由地緯度経度
#### 記述形式
##### 経由地緯度経度
`[経由地#1]|[経由地#2|...|[経由地#n]`

##### 経由地
`[経度の値],[緯度の値],[地点種別],[滞在時間],[横付け優先]`
※`[地点種別]`、`[滞在時間]`、`[横付け優先]`は省略可能です。

|地点種別|値|
|---|---|
|一般道・有料道|0|
|一般道のみ|1|
|有料道のみ|2|

|横付け優先|値|
|---|---|
|横付け優先なし|0|
|横付け優先あり|1|

#### 範囲
|n（経由地の数）|値|
|---|---|
|最大|30|

|滞在時間（分）|値|
|---|---|
|最小|0|
|最大|10080|

#### デフォルト

##### 地点種別
`0`

##### 滞在時間（分）
`0`

##### 横付け優先
`0`

#### 記述例

##### 経由地を3地点指定
`139.75723347,35.66593095|139.73893125,35.62911118|139.69695771,35.53136405`

##### 緯度経度のみ指定
`139.75723347,35.66593095`

##### 地点種別も合わせて指定
`139.75723347,35.66593095,1`

##### 滞在時間も合わせて指定
`139.75723347,35.66593095,1,60`

##### 横付け優先も合わせて指定
`139.75723347,35.66593095,1,60,1`

##### 地点種別指定：なし、滞在時間指定：あり
`139.75723347,35.66593095,,60`

##### 地点種別指定：なし、滞在時間指定：なし、横付け優先指定：あり
`139.75723347,35.66593095,,,1`

#### 注記
- 緯度経度の測地系は、世界測地系（JGD2011）となります。
- 緯度経度の小数点は第8位まで考慮され、それより大きい値は無視されます。
- 指定地点から約5kmの範囲に道路がある場合にルート検索が可能です。道路が見つからない場合、エラー（`I00104`）となります。
歩行者（`priority=100～103`）の場合も同様です。
        startangle: ### 出発方向
出発地の最近傍リンクが双方向に進行可能な場合に、進行方向を指定する事ができます。

指定した場合、最近傍リンクの進行可能な方向に対し、指定角度に近似している方向に進行します。
指定がない場合、最近傍リンクの進行可能な方向のいずれかが、ルートの状況に応じて採用されます。真北を0度とし、時計回りに増加します。

#### 範囲
|範囲|値|
|---|---|
|最小|0（北上）|
|最大|359|

#### デフォルト
`0`

#### 注記
- 整数のみ指定可能です。
- 時計回りに増加します。
        priority: ### 基本条件
|基本条件|値|
|---|---|
|標準|0|
|距離優先|1|
|直進優先|2|
|簡易歩行者|3|
|道幅優先|4|
|歩行者標準（分かり易い）|100|
|歩行者距離優先|101|
|歩行者屋根優先（屋根が多い）|102|
|歩行者段差回避（階段が少ない）|103|

#### 注記
- 巡回ルートによる検索を有効とした場合、歩行者（`priority=100～103`）の指定は出来ません。
- 各条件の詳細は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ ルート検索 ＞ 補足 ＞ ルート基本条件 (priority)』をご確認ください。
        destination: ### 目的地緯度経度

#### 記述形式
`[経度の値],[緯度の値],[地点種別],[横付け優先]`
※`[地点種別]`、`[横付け優先]`は省略可能です。

|地点種別|値|
|---|---|
|一般道・有料道|0|
|一般道のみ|1|
|有料道のみ|2|

|横付け優先|値|
|---|---|
|横付け優先なし|0|
|横付け優先あり|1|

#### デフォルト
##### 地点種別
`0`

##### 横付け優先
`0`

#### 記述例

##### 緯度経度のみ指定
`139.62261961,35.46606942`

##### 地点種別も合わせて指定
`139.62261961,35.46606942,1`

##### 横付け優先も合わせて指定
`139.62261961,35.46606942,1,1`

##### 地点種別指定：なし、横付け優先指定：あり
`139.62261961,35.46606942,,1`

#### 注記
- `routeresultid`が指定されている場合のみ、本パラメータは省略可能です。
- 緯度経度の測地系は、世界測地系（JGD2011）となります。
- 緯度経度の小数点は第8位まで考慮され、それより大きい値は無視されます。
- 指定地点から約5kmの範囲に道路がある場合にルート検索が可能です。道路が見つからない場合、エラー（`I00104`）となります。
歩行者（`priority=100～103`）の場合も同様です。
        tollway: ### 有料道路使用条件
|有料道路使用|値|
|---|---|
|標準|0|
|優先|1|
|回避|2|
|絶対回避|3|

#### デフォルト
`0`

#### 注記
- 有料道路を使わないとルートが引けない条件で、
    - `tollway=2` とした場合、有料道路を使ったルートが結果として返ります。
    - `tollway=3` とした場合、エラー（`I00108`）となります。
        etc: ### ETC専用施設利用有無
|ETC専用施設利用|値|
|---|---|
|利用しない|0|
|利用する|1|

#### デフォルト
`1`

#### 注記
- `smartic` パラメータの代わりに本パラメータをご利用下さい。本パラメータはスマートICにも適用されます。
- 本パラメータが未指定、かつ、 `smartic` パラメータが指定されている場合に限り、本パラメータには `smartic` と同じ値が設定されます。
        start: ### 出発地緯度経度
#### 記述形式

`[経度の値],[緯度の値],[地点種別]`
※`[地点種別]`は省略可能です。

|地点種別|値|
|---|---|
|一般道・有料道|0|
|一般道のみ|1|
|有料道のみ|2|

#### デフォルト

##### 地点種別
`0`

#### 記述例
##### 緯度経度のみ指定
`139.76730676,35.68095910`

##### 地点種別も合わせて指定
`139.76730676,35.68095910,1`

#### 注記
- `routeresultid`が指定されている場合のみ、本パラメータは省略可能です。
- 緯度経度の測地系は、世界測地系（JGD2011）となります。
- 緯度経度の小数点は第8位まで考慮され、それより大きい値は無視されます。
- 指定地点から約5kmの範囲に道路がある場合にルート検索が可能です。道路が見つからない場合、エラー（`I00104`）となります。
歩行者（`priority=100～103`）の場合も同様です。
        smartic: ## 本パラメータは非推奨パラメータです。 `etc` をご使用ください。

### スマートIC利用有無
|スマートIC利用|値|
|---|---|
|利用しない|0|
|利用する|1|

#### デフォルト
`1`
        ferryspeed: ### フェリーの航行速度
フェリーの航行時間の計算に用います。

### 記述例
`27.0`

#### 注記
- 単位は、km/h です。
- 経路種別との関係およびデフォルト値は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ 付則 ＞ ルート機能 ＞ 経路種別一覧』をご確認ください。
        tolltarget: ### 料金計算対象
|料金計算対象|値|
|---|---|
|通常料金|0|
|通常料金＋ETC割引料金|1|

#### デフォルト
`0`
        
    """
    url = f"https://mapfanapi-route.p.rapidapi.com/calcroute"
    querystring = {}
    if callback:
        querystring['callback'] = callback
    if uturnavoid:
        querystring['uturnavoid'] = uturnavoid
    if routeid:
        querystring['routeid'] = routeid
    if impassablearea:
        querystring['impassablearea'] = impassablearea
    if routeresultid:
        querystring['routeresultid'] = routeresultid
    if uturn:
        querystring['uturn'] = uturn
    if regulations:
        querystring['regulations'] = regulations
    if resulttype:
        querystring['resulttype'] = resulttype
    if fmt:
        querystring['fmt'] = fmt
    if tollroad:
        querystring['tollroad'] = tollroad
    if passablearea:
        querystring['passablearea'] = passablearea
    if daytime:
        querystring['daytime'] = daytime
    if weight:
        querystring['weight'] = weight
    if date:
        querystring['date'] = date
    if height:
        querystring['height'] = height
    if travel:
        querystring['travel'] = travel
    if danger:
        querystring['danger'] = danger
    if highwayspeed:
        querystring['highwayspeed'] = highwayspeed
    if generalroad:
        querystring['generalroad'] = generalroad
    if width:
        querystring['width'] = width
    if loadage:
        querystring['loadage'] = loadage
    if vehicletype:
        querystring['vehicletype'] = vehicletype
    if cartype:
        querystring['cartype'] = cartype
    if tollwayspeed:
        querystring['tollwayspeed'] = tollwayspeed
    if normalspeed:
        querystring['normalspeed'] = normalspeed
    if ferry:
        querystring['ferry'] = ferry
    if via:
        querystring['via'] = via
    if startangle:
        querystring['startangle'] = startangle
    if priority:
        querystring['priority'] = priority
    if destination:
        querystring['destination'] = destination
    if tollway:
        querystring['tollway'] = tollway
    if etc:
        querystring['etc'] = etc
    if start:
        querystring['start'] = start
    if smartic:
        querystring['smartic'] = smartic
    if ferryspeed:
        querystring['ferryspeed'] = ferryspeed
    if tolltarget:
        querystring['tolltarget'] = tolltarget
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapfanapi-route.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def multicalcroute(ferry: str=None, impassablearea: str=None, tollroad: str=None, regulations: str=None, fmt: str=None, uturn: str=None, uturnavoid: str=None, generalroad: str=None, passablearea: str=None, callback: str=None, weight: int=None, width: int=None, loadage: int=None, danger: str=None, height: int=None, highwayspeed: int=None, ferryspeed: int=None, normalspeed: int=None, tollwayspeed: int=None, cartype: str=None, vehicletype: str=None, etc: str=None, destination: str='139.62261961,35.46606942', date: str=None, smartic: str=None, start: str='139.76730676,35.68095910', startangle: int=None, via: str=None, routeid: str=None, daytime: str=None, tolltarget: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## 複数ルート検索
		複数ルート検索機能を提供します。
		
		一度のAPIの実行で、以下3つの条件で実行した`calcroute`と同等のルート検索結果を同時に取得できます。
		
		|ルート番号|内容|priority|tollway|
		|---|---|---|---|
		|1|標準|0（標準）|0（標準）|
		|2|有料回避|0（標準）|2（回避）|
		|3|距離優先|1（距離優先）|0（標準）|
		
		#### 注記
		- レスポンスに誘導データは含まれません。誘導データは、レスポンスの`routeResultId`を用いて`calcroute`を実行して取得してください。
		- ルート検索APIを実行する際、距離の長さや検索条件によってはレスポンスが返却されるまで時間がかかる場合があります。"
    ferry: ### フェリー使用条件
|フェリー使用|値|
|---|---|
|標準|0|
|優先|1|
|回避|2|
|絶対回避|3|

#### デフォルト
`0`

#### 注記
- フェリーを使わないとルートが引けない条件で、
    - `ferry=2` とした場合、フェリーを使ったルートが結果として返ります。
    - `ferry=3` とした場合、エラー（`I00107`）となります。
        impassablearea: ### 通行不可エリア
ルート検索対象外となるエリアを矩形で指定します。
指定されたエリア内を回避するルートが検索されます。

#### 記述形式

##### 通行可能エリア
`[エリア#1]|[エリア#2]|...|[エリア#n]`

##### エリア
`[南西経度の値],[南西緯度の値],[北東経度の値],[北東緯度の値]`

#### 範囲
|n（エリアの個数）|値|
|---|---|
|最大|10|

#### 記述例
`139.6965,35.6867,139.7029,35.6921|139.7000,35.6818,139.7044,35.6879`

#### 注記
- 緯度経度の測地系は、世界測地系（JGD2011）となります。
- 緯度経度の小数点は第8位まで考慮され、それより大きい値は無視されます。
- 指定した矩形と整合性の無い`start`、`destination`を指定した場合、エラー（`I00104`）になります。
- `passablearea`と`impassablearea`で重なった部分については、`impassablearea`が優先されます。
- `travel=1`とした場合、本設定は指定は出来ません。
- 長距離ルート検索時に本パラメータを指定した場合、レスポンスの返却が遅くなる場合があります。
        tollroad: ### 高速道規制
無効とした場合、高速道の規制を無視したルート検索が可能となります。

|設定|値|
|---|---|
|高速道規制を無効にする|0|
|高速道規制を有効にする|1|

#### デフォルト
`1`

#### 注記
- `daytime`、`regulations`との関係性は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ ルート検索 ＞ 補足』をご確認ください。
- 高速道規制の対象となる道は、道路種別100番台と11番の道です。道路種別は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ 付則 ＞ ルート機能 ＞ 道路種別一覧』をご確認ください。
        regulations: ### その他規制
|設定|値|
|---|---|
|一方通行規制の有効/無効を、<br>generalroad、tollroadの設定に従う|0|
|一方通行規制を有効にする|1|

#### デフォルト
`0`

#### 注記
`daytime`、`generalroad`、`tollroad`との関係性は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ ルート検索 ＞ 補足』をご確認ください。
        fmt: ### 出力フォーマット
|フォーマット|値|
|---|---|
|JSON|json |
|XML|xml|

#### デフォルト
`json`

#### 注記
`callback`が設定されている場合、本パラメータの設定が破棄され、JSONP形式が適用されます。
        uturn: ### Uターンのし易さ
|Uターンのし易さ|値|
|---|---|
|Uターンを比較的行う|0|
|Uターンを比較的行わない|1|

#### デフォルト
`0`

#### 注記
本パラメータは`uturnavoid `（経由地でのUターン回避）とは異なりますのでご注意ください。
        uturnavoid: ### 経由地でのUターン回避
|経由地でのUターン回避|値|
|---|---|
|しない|0|
|する|1|

#### デフォルト
`priority`で指定した値により異なります。
- `priority=3、100～103`の場合
`0`
- `priority`が上記以外の場合
`1`

#### 注記
- `travel=1`とした場合、本設定は指定は出来ません。
- 本パラメータは`uturn`（Uターンのし易さ）とは異なりますのでご注意ください。
        generalroad: ### 一般道規制
無効とした場合、一般道の規制を無視したルート検索が可能となります。

|設定|値|
|---|---|
|一般道規制を無効にする|0|
|一般道規制を有効にする|1|

#### デフォルト
`1`

#### 注記
`daytime`、`regulations`との関係性は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ ルート検索 ＞ 補足』をご確認ください。
        passablearea: ### 通行可能エリア
ルート検索対象となるエリアを矩形で指定します。
指定されたエリア内のみを通行するルートが検索されます。

#### 記述形式

##### 通行可能エリア
`[エリア#1]|[エリア#2]|...|[エリア#n]`

##### エリア
`[南西経度の値],[南西緯度の値],[北東経度の値],[北東緯度の値]`

#### 範囲
|n（エリアの個数）|値|
|---|---|
|最大|10|

#### 記述例
`139.6965,35.6867,139.7029,35.6921|139.7000,35.6818,139.7044,35.6879`

#### 注記
- 緯度経度の測地系は、世界測地系（JGD2011）となります。
- 緯度経度の小数点は第8位まで考慮され、それより大きい値は無視されます。
- 指定した矩形と整合性の無い`start`、`destination`を指定した場合、エラー（`I00104`）になります。
- `passablearea`と`impassablearea`で重なった部分については、`impassablearea`が優先されます。
- `travel=1`とした場合、本設定は指定は出来ません。
- 長距離ルート検索時に本パラメータを指定した場合、レスポンスの返却が遅くなる場合があります。
        callback: ### JSONPのコールバック用関数名

関数名が指定されている場合はJSONP形式で結果を返却します。
指定されていない場合は、 `fmt` で指定された形式で結果を返却します。
        weight: ### 車重
道路規制で考慮する車重を指定します。

#### 範囲
|範囲|値|
|---|---|
|最小|0|

#### 注記
単位は、kg です。
        width: ### 車幅
道路規制で考慮する車幅を指定します。

#### 範囲
|範囲|値|
|---|---|
|最小|0|

#### 注記
単位は、cm です。
        loadage: ### 積載量
道路規制で考慮する車両の最大積載量を指定します。

#### 範囲
|範囲|値|
|---|---|
|最小|0|

#### 注記
単位は、kg です。
        danger: ### 危険物積載車両
道路規制で考慮する危険物積載車両であるかを指定します。

|設定|値|
|---|---|
|危険物積載車両でない|0|
|危険物積載車両である|1|

#### デフォルト
`0`
        height: ### 車高
道路規制で考慮する車高を指定します。

#### 範囲
|範囲|値|
|---|---|
|最小|0|

#### 注記
単位は、cm です。
        highwayspeed: ### 高速道の走行速度
高速道の走行時間の計算に用います。

### 記述例
`90.0`

#### 注記
- 単位は、km/h です。
- 経路種別との関係およびデフォルト値は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ 付則 ＞ ルート機能 ＞ 経路種別一覧』をご確認ください。
        ferryspeed: ### フェリーの航行速度
フェリーの航行時間の計算に用います。

### 記述例
`27.0`

#### 注記
- 単位は、km/h です。
- 経路種別との関係およびデフォルト値は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ 付則 ＞ ルート機能 ＞ 経路種別一覧』をご確認ください。
        normalspeed: ### 一般道の走行速度
一般道の走行時間の計算に用います。

### 記述例
`25.0`

#### 注記
- 単位は、km/h です。
- 経路種別との関係およびデフォルト値は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ 付則 ＞ ルート機能 ＞ 経路種別一覧』をご確認ください。
        tollwayspeed: ### 有料道の走行速度
有料道の走行時間の計算に用います。

### 記述例
`55.0`

#### 注記
- 単位は、km/h です。
- 経路種別との関係およびデフォルト値は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ 付則 ＞ ルート機能 ＞ 経路種別一覧』をご確認ください。
        cartype: ### 有料道路利用時の車種
有料道路の料金計算のみに使用します。

|車種|値|
|---|---|
|軽自動車|0|
|普通車|1|
|中型車|2|
|大型車|3|
|特大車|4|

#### デフォルト
`1`
        vehicletype: ### 車種
道路規制を考慮する車種を指定します。

|車種|値|
|---|---|
|指定なし|0|
|大型乗用自動車|1|
|大型貨物自動車|6|
|大型特殊自動車|11|

#### デフォルト
`0`
        etc: ### ETC専用施設利用有無
|ETC専用施設利用|値|
|---|---|
|利用しない|0|
|利用する|1|

#### デフォルト
`1`

#### 注記
- `smartic` パラメータの代わりに本パラメータをご利用下さい。本パラメータはスマートICにも適用されます。
- 本パラメータが未指定、かつ、 `smartic` パラメータが指定されている場合に限り、本パラメータには `smartic` と同じ値が設定されます。
        destination: ### 目的地緯度経度

#### 記述形式
`[経度の値],[緯度の値],[地点種別],[横付け優先]`
※`[地点種別]`、`[横付け優先]`は省略可能です。

|地点種別|値|
|---|---|
|一般道・有料道|0|
|一般道のみ|1|
|有料道のみ|2|

|横付け優先|値|
|---|---|
|横付け優先なし|0|
|横付け優先あり|1|

#### デフォルト
##### 地点種別
`0`

##### 横付け優先
`0`

#### 記述例

##### 緯度経度のみ指定
`139.62261961,35.46606942`

##### 地点種別も合わせて指定
`139.62261961,35.46606942,1`

##### 横付け優先も合わせて指定
`139.62261961,35.46606942,1,1`

##### 地点種別指定：なし、横付け優先指定：あり
`139.62261961,35.46606942,,1`

#### 注記
- `routeresultid`が指定されている場合のみ、本パラメータは省略可能です。
- 緯度経度の測地系は、世界測地系（JGD2011）となります。
- 緯度経度の小数点は第8位まで考慮され、それより大きい値は無視されます。
- 指定地点から約5kmの範囲に道路がある場合にルート検索が可能です。道路が見つからない場合、エラー（`I00104`）となります。
歩行者（`priority=100～103`）の場合も同様です。
        date: ### 出発日時
ルート検索時の交通規制を考慮するときの日時を指定します。

#### 記述形式
`[年4桁][月2桁][日2桁]_[時2桁][分2桁][秒2桁]` （`yyyyMMdd_HHmmss`）

#### 記述例
`20220420_170505` （2022年4月20日午後5時5分5秒 ）

#### デフォルト
API実行日時を適用して、ルート検索を行います。
        smartic: ## 本パラメータは非推奨パラメータです。 `etc` をご使用ください。

### スマートIC利用有無
|スマートIC利用|値|
|---|---|
|利用しない|0|
|利用する|1|

#### デフォルト
`1`
        start: ### 出発地緯度経度
#### 記述形式

`[経度の値],[緯度の値],[地点種別]`
※`[地点種別]`は省略可能です。

|地点種別|値|
|---|---|
|一般道・有料道|0|
|一般道のみ|1|
|有料道のみ|2|

#### デフォルト

##### 地点種別
`0`

#### 記述例
##### 緯度経度のみ指定
`139.76730676,35.68095910`

##### 地点種別も合わせて指定
`139.76730676,35.68095910,1`

#### 注記
- `routeresultid`が指定されている場合のみ、本パラメータは省略可能です。
- 緯度経度の測地系は、世界測地系（JGD2011）となります。
- 緯度経度の小数点は第8位まで考慮され、それより大きい値は無視されます。
- 指定地点から約5kmの範囲に道路がある場合にルート検索が可能です。道路が見つからない場合、エラー（`I00104`）となります。
歩行者（`priority=100～103`）の場合も同様です。
        startangle: ### 出発方向
出発地の最近傍リンクが双方向に進行可能な場合に、進行方向を指定する事ができます。

指定した場合、最近傍リンクの進行可能な方向に対し、指定角度に近似している方向に進行します。
指定がない場合、最近傍リンクの進行可能な方向のいずれかが、ルートの状況に応じて採用されます。真北を0度とし、時計回りに増加します。

#### 範囲
|範囲|値|
|---|---|
|最小|0（北上）|
|最大|359|

#### デフォルト
`0`

#### 注記
- 整数のみ指定可能です。
- 時計回りに増加します。
        via: ### 経由地緯度経度
#### 記述形式
##### 経由地緯度経度
`[経由地#1]|[経由地#2|...|[経由地#n]`

##### 経由地
`[経度の値],[緯度の値],[地点種別],[滞在時間],[横付け優先]`
※`[地点種別]`、`[滞在時間]`、`[横付け優先]`は省略可能です。

|地点種別|値|
|---|---|
|一般道・有料道|0|
|一般道のみ|1|
|有料道のみ|2|

|横付け優先|値|
|---|---|
|横付け優先なし|0|
|横付け優先あり|1|

#### 範囲
|n（経由地の数）|値|
|---|---|
|最大|30|

|滞在時間（分）|値|
|---|---|
|最小|0|
|最大|10080|

#### デフォルト

##### 地点種別
`0`

##### 滞在時間（分）
`0`

##### 横付け優先
`0`

#### 記述例

##### 経由地を3地点指定
`139.75723347,35.66593095|139.73893125,35.62911118|139.69695771,35.53136405`

##### 緯度経度のみ指定
`139.75723347,35.66593095`

##### 地点種別も合わせて指定
`139.75723347,35.66593095,1`

##### 滞在時間も合わせて指定
`139.75723347,35.66593095,1,60`

##### 横付け優先も合わせて指定
`139.75723347,35.66593095,1,60,1`

##### 地点種別指定：なし、滞在時間指定：あり
`139.75723347,35.66593095,,60`

##### 地点種別指定：なし、滞在時間指定：なし、横付け優先指定：あり
`139.75723347,35.66593095,,,1`

#### 注記
- 緯度経度の測地系は、世界測地系（JGD2011）となります。
- 緯度経度の小数点は第8位まで考慮され、それより大きい値は無視されます。
- 指定地点から約5kmの範囲に道路がある場合にルート検索が可能です。道路が見つからない場合、エラー（`I00104`）となります。
歩行者（`priority=100～103`）の場合も同様です。
        routeid: ### ルート識別子
ルート検索を識別する任意の文字列を指定します。
指定した文字列が、レスポンスにそのまま出力されます。

ルート検索結果に影響はしません。

#### 注記
- URLエンコードされたUTF-8の文字列を指定します。
- 半角のダブルクォーテーションの使用は禁止ですが、それ以外の半角記号や全角文字はURLエンコードを行っていれば使用可能となります。
        daytime: ### 日時規制
|設定|値|
|---|---|
|日時規制を無効にする|0|
|日時規制の有効/無効を、<br>generalroad、tollroad、regulationsの設定に従う|1|

#### デフォルト
`1`

#### 注記
`generalroad`、`tollroad`、`regulations`との関係性は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ ルート検索 ＞ 補足』をご確認ください。
        tolltarget: ### 料金計算対象
|料金計算対象|値|
|---|---|
|通常料金|0|
|通常料金＋ETC割引料金|1|

#### デフォルト
`0`
        
    """
    url = f"https://mapfanapi-route.p.rapidapi.com/multicalcroute"
    querystring = {}
    if ferry:
        querystring['ferry'] = ferry
    if impassablearea:
        querystring['impassablearea'] = impassablearea
    if tollroad:
        querystring['tollroad'] = tollroad
    if regulations:
        querystring['regulations'] = regulations
    if fmt:
        querystring['fmt'] = fmt
    if uturn:
        querystring['uturn'] = uturn
    if uturnavoid:
        querystring['uturnavoid'] = uturnavoid
    if generalroad:
        querystring['generalroad'] = generalroad
    if passablearea:
        querystring['passablearea'] = passablearea
    if callback:
        querystring['callback'] = callback
    if weight:
        querystring['weight'] = weight
    if width:
        querystring['width'] = width
    if loadage:
        querystring['loadage'] = loadage
    if danger:
        querystring['danger'] = danger
    if height:
        querystring['height'] = height
    if highwayspeed:
        querystring['highwayspeed'] = highwayspeed
    if ferryspeed:
        querystring['ferryspeed'] = ferryspeed
    if normalspeed:
        querystring['normalspeed'] = normalspeed
    if tollwayspeed:
        querystring['tollwayspeed'] = tollwayspeed
    if cartype:
        querystring['cartype'] = cartype
    if vehicletype:
        querystring['vehicletype'] = vehicletype
    if etc:
        querystring['etc'] = etc
    if destination:
        querystring['destination'] = destination
    if date:
        querystring['date'] = date
    if smartic:
        querystring['smartic'] = smartic
    if start:
        querystring['start'] = start
    if startangle:
        querystring['startangle'] = startangle
    if via:
        querystring['via'] = via
    if routeid:
        querystring['routeid'] = routeid
    if daytime:
        querystring['daytime'] = daytime
    if tolltarget:
        querystring['tolltarget'] = tolltarget
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapfanapi-route.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def altcalcroute(fmt: str=None, passablearea: str=None, callback: str=None, saveresult: str=None, daytime: str=None, generalroad: str=None, uturnavoid: str=None, tollroad: str=None, regulations: str=None, impassablearea: str=None, routeid: str=None, uturn: str=None, danger: str=None, width: int=None, weight: int=None, highwayspeed: int=None, height: int=None, loadage: int=None, normalspeed: int=None, tollwayspeed: int=None, ferryspeed: int=None, vehicletype: str=None, tolltarget: str=None, smartic: str=None, cartype: str=None, date: str=None, tollway: str=None, etc: str=None, ferry: str=None, destination: str='139.62261961,35.46606942', via: str=None, startangle: int=None, priority: str=None, start: str='139.76730676,35.68095910', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## 複数推奨ルート検索
		複数推奨ルート検索機能を提供します。
		
		1組のルート検索条件に対し、`calcroute`の結果を「推奨1」として、その他のルート案を「推奨2」「推奨3」として最大3件を返却します。その他の推奨ルートが見つからない場合は、「推奨2」「推奨3」は返却されません。
		
		#### 注記
		ルート検索APIを実行する際、距離の長さや検索条件によってはレスポンスが返却されるまで時間がかかる場合があります。"
    fmt: ### 出力フォーマット
|フォーマット|値|
|---|---|
|JSON|json |
|XML|xml|

#### デフォルト
`json`

#### 注記
`callback`が設定されている場合、本パラメータの設定が破棄され、JSONP形式が適用されます。
        passablearea: ### 通行可能エリア
ルート検索対象となるエリアを矩形で指定します。
指定されたエリア内のみを通行するルートが検索されます。

#### 記述形式

##### 通行可能エリア
`[エリア#1]|[エリア#2]|...|[エリア#n]`

##### エリア
`[南西経度の値],[南西緯度の値],[北東経度の値],[北東緯度の値]`

#### 範囲
|n（エリアの個数）|値|
|---|---|
|最大|10|

#### 記述例
`139.6965,35.6867,139.7029,35.6921|139.7000,35.6818,139.7044,35.6879`

#### 注記
- 緯度経度の測地系は、世界測地系（JGD2011）となります。
- 緯度経度の小数点は第8位まで考慮され、それより大きい値は無視されます。
- 指定した矩形と整合性の無い`start`、`destination`を指定した場合、エラー（`I00104`）になります。
- `passablearea`と`impassablearea`で重なった部分については、`impassablearea`が優先されます。
- `travel=1`とした場合、本設定は指定は出来ません。
- 長距離ルート検索時に本パラメータを指定した場合、レスポンスの返却が遅くなる場合があります。
        callback: ### JSONPのコールバック用関数名
関数名が指定されている場合はJSONP形式で結果を返却します。
指定されていない場合は、`fmt`で指定された形式で結果を返却します。
        saveresult: ### ルート検索結果の保存有無
ルート検索結果を保存するとレスポンスに'routeResultId'が格納されます。

|ルート検索結果の保存|値|
|---|---|
|保存しない|0|
|保存する|1|

#### デフォルト
`0`

#### 注記
`routeResultId`の有効期限は1ヶ月となります。
        daytime: ### 日時規制
|設定|値|
|---|---|
|日時規制を無効にする|0|
|日時規制の有効/無効を、<br>generalroad、tollroad、regulationsの設定に従う|1|

#### デフォルト
`1`

#### 注記
`generalroad`、`tollroad`、`regulations`との関係性は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ ルート検索 ＞ 補足』をご確認ください。
        generalroad: ### 一般道規制
無効とした場合、一般道の規制を無視したルート検索が可能となります。

|設定|値|
|---|---|
|一般道規制を無効にする|0|
|一般道規制を有効にする|1|

#### デフォルト
`1`

#### 注記
`daytime`、`regulations`との関係性は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ ルート検索 ＞ 補足』をご確認ください。
        uturnavoid: ### 経由地でのUターン回避
|経由地でのUターン回避|値|
|---|---|
|しない|0|
|する|1|

#### デフォルト
`priority`で指定した値により異なります。
- `priority=3、100～103`の場合
`0`
- `priority`が上記以外の場合
`1`

#### 注記
- `travel=1`とした場合、本設定は指定は出来ません。
- 本パラメータは`uturn`（Uターンのし易さ）とは異なりますのでご注意ください。
        tollroad: ### 高速道規制
無効とした場合、高速道の規制を無視したルート検索が可能となります。

|設定|値|
|---|---|
|高速道規制を無効にする|0|
|高速道規制を有効にする|1|

#### デフォルト
`1`

#### 注記
- `daytime`、`regulations`との関係性は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ ルート検索 ＞ 補足』をご確認ください。
- 高速道規制の対象となる道は、道路種別100番台と11番の道です。道路種別は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ 付則 ＞ ルート機能 ＞ 道路種別一覧』をご確認ください。
        regulations: ### その他規制
|設定|値|
|---|---|
|一方通行規制の有効/無効を、<br>generalroad、tollroadの設定に従う|0|
|一方通行規制を有効にする|1|

#### デフォルト
`0`

#### 注記
`daytime`、`generalroad`、`tollroad`との関係性は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ ルート検索 ＞ 補足』をご確認ください。
        impassablearea: ### 通行不可エリア
ルート検索対象外となるエリアを矩形で指定します。
指定されたエリア内を回避するルートが検索されます。

#### 記述形式

##### 通行可能エリア
`[エリア#1]|[エリア#2]|...|[エリア#n]`

##### エリア
`[南西経度の値],[南西緯度の値],[北東経度の値],[北東緯度の値]`

#### 範囲
|n（エリアの個数）|値|
|---|---|
|最大|10|

#### 記述例
`139.6965,35.6867,139.7029,35.6921|139.7000,35.6818,139.7044,35.6879`

#### 注記
- 緯度経度の測地系は、世界測地系（JGD2011）となります。
- 緯度経度の小数点は第8位まで考慮され、それより大きい値は無視されます。
- 指定した矩形と整合性の無い`start`、`destination`を指定した場合、エラー（`I00104`）になります。
- `passablearea`と`impassablearea`で重なった部分については、`impassablearea`が優先されます。
- `travel=1`とした場合、本設定は指定は出来ません。
- 長距離ルート検索時に本パラメータを指定した場合、レスポンスの返却が遅くなる場合があります。
        routeid: ### ルート識別子
ルート検索を識別する任意の文字列を指定します。
指定した文字列が、レスポンスにそのまま出力されます。

ルート検索結果に影響はしません。

#### 注記
- URLエンコードされたUTF-8の文字列を指定します。
- 半角のダブルクォーテーションの使用は禁止ですが、それ以外の半角記号や全角文字はURLエンコードを行っていれば使用可能となります。
        uturn: ### Uターンのし易さ
|Uターンのし易さ|値|
|---|---|
|Uターンを比較的行う|0|
|Uターンを比較的行わない|1|

#### デフォルト
`0`

#### 注記
本パラメータは`uturnavoid `（経由地でのUターン回避）とは異なりますのでご注意ください。
        danger: ### 危険物積載車両
道路規制で考慮する危険物積載車両であるかを指定します。

|設定|値|
|---|---|
|危険物積載車両でない|0|
|危険物積載車両である|1|

#### デフォルト
`0`
        width: ### 車幅
道路規制で考慮する車幅を指定します。

#### 範囲
|範囲|値|
|---|---|
|最小|0|

#### 注記
単位は、cm です。
        weight: ### 車重
道路規制で考慮する車重を指定します。

#### 範囲
|範囲|値|
|---|---|
|最小|0|

#### 注記
単位は、kg です。
        highwayspeed: ### 高速道の走行速度
高速道の走行時間の計算に用います。

### 記述例
`90.0`

#### 注記
- 単位は、km/h です。
- 経路種別との関係およびデフォルト値は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ 付則 ＞ ルート機能 ＞ 経路種別一覧』をご確認ください。
        height: ### 車高
道路規制で考慮する車高を指定します。

#### 範囲
|範囲|値|
|---|---|
|最小|0|

#### 注記
単位は、cm です。
        loadage: ### 積載量
道路規制で考慮する車両の最大積載量を指定します。

#### 範囲
|範囲|値|
|---|---|
|最小|0|

#### 注記
単位は、kg です。
        normalspeed: ### 一般道の走行速度
一般道の走行時間の計算に用います。

### 記述例
`25.0`

#### 注記
- 単位は、km/h です。
- 経路種別との関係およびデフォルト値は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ 付則 ＞ ルート機能 ＞ 経路種別一覧』をご確認ください。
        tollwayspeed: ### 有料道の走行速度
有料道の走行時間の計算に用います。

### 記述例
`55.0`

#### 注記
- 単位は、km/h です。
- 経路種別との関係およびデフォルト値は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ 付則 ＞ ルート機能 ＞ 経路種別一覧』をご確認ください。
        ferryspeed: ### フェリーの航行速度
フェリーの航行時間の計算に用います。

### 記述例
`27.0`

#### 注記
- 単位は、km/h です。
- 経路種別との関係およびデフォルト値は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ 付則 ＞ ルート機能 ＞ 経路種別一覧』をご確認ください。
        vehicletype: ### 車種
道路規制を考慮する車種を指定します。

|車種|値|
|---|---|
|指定なし|0|
|大型乗用自動車|1|
|大型貨物自動車|6|
|大型特殊自動車|11|

#### デフォルト
`0`
        tolltarget: ### 料金計算対象
|料金計算対象|値|
|---|---|
|通常料金|0|
|通常料金＋ETC割引料金|1|

#### デフォルト
`0`
        smartic: ## 本パラメータは非推奨パラメータです。 `etc` をご使用ください。

### スマートIC利用有無
|スマートIC利用|値|
|---|---|
|利用しない|0|
|利用する|1|

#### デフォルト
`1`
        cartype: ### 有料道路利用時の車種
有料道路の料金計算のみに使用します。

|車種|値|
|---|---|
|軽自動車|0|
|普通車|1|
|中型車|2|
|大型車|3|
|特大車|4|

#### デフォルト
`1`
        date: ### 出発日時
ルート検索時の交通規制を考慮するときの日時を指定します。

#### 記述形式
`[年4桁][月2桁][日2桁]_[時2桁][分2桁][秒2桁]` （`yyyyMMdd_HHmmss`）

#### 記述例
`20220420_170505` （2022年4月20日午後5時5分5秒 ）

#### デフォルト
API実行日時を適用して、ルート検索を行います。
        tollway: ### 有料道路使用条件
|有料道路使用|値|
|---|---|
|標準|0|
|優先|1|
|回避|2|
|絶対回避|3|

#### デフォルト
`0`

#### 注記
- 有料道路を使わないとルートが引けない条件で、
    - `tollway=2` とした場合、有料道路を使ったルートが結果として返ります。
    - `tollway=3` とした場合、エラー（`I00108`）となります。
        etc: ### ETC専用施設利用有無
|ETC専用施設利用|値|
|---|---|
|利用しない|0|
|利用する|1|

#### デフォルト
`1`

#### 注記
- `smartic` パラメータの代わりに本パラメータをご利用下さい。本パラメータはスマートICにも適用されます。
- 本パラメータが未指定、かつ、 `smartic` パラメータが指定されている場合に限り、本パラメータには `smartic` と同じ値が設定されます。
        ferry: ### フェリー使用条件
|フェリー使用|値|
|---|---|
|標準|0|
|優先|1|
|回避|2|
|絶対回避|3|

#### デフォルト
`0`

#### 注記
- フェリーを使わないとルートが引けない条件で、
    - `ferry=2` とした場合、フェリーを使ったルートが結果として返ります。
    - `ferry=3` とした場合、エラー（`I00107`）となります。
        destination: ### 目的地緯度経度

#### 記述形式
`[経度の値],[緯度の値],[地点種別],[横付け優先]`
※`[地点種別]`、`[横付け優先]`は省略可能です。

|地点種別|値|
|---|---|
|一般道・有料道|0|
|一般道のみ|1|
|有料道のみ|2|

|横付け優先|値|
|---|---|
|横付け優先なし|0|
|横付け優先あり|1|

#### デフォルト
##### 地点種別
`0`

##### 横付け優先
`0`

#### 記述例

##### 緯度経度のみ指定
`139.62261961,35.46606942`

##### 地点種別も合わせて指定
`139.62261961,35.46606942,1`

##### 横付け優先も合わせて指定
`139.62261961,35.46606942,1,1`

##### 地点種別指定：なし、横付け優先指定：あり
`139.62261961,35.46606942,,1`

#### 注記
- `routeresultid`が指定されている場合のみ、本パラメータは省略可能です。
- 緯度経度の測地系は、世界測地系（JGD2011）となります。
- 緯度経度の小数点は第8位まで考慮され、それより大きい値は無視されます。
- 指定地点から約5kmの範囲に道路がある場合にルート検索が可能です。道路が見つからない場合、エラー（`I00104`）となります。
歩行者（`priority=100～103`）の場合も同様です。
        via: ### 経由地緯度経度
#### 記述形式
##### 経由地緯度経度
`[経由地#1]|[経由地#2|...|[経由地#n]`

##### 経由地
`[経度の値],[緯度の値],[地点種別],[滞在時間],[横付け優先]`
※`[地点種別]`、`[滞在時間]`、`[横付け優先]`は省略可能です。

|地点種別|値|
|---|---|
|一般道・有料道|0|
|一般道のみ|1|
|有料道のみ|2|

|横付け優先|値|
|---|---|
|横付け優先なし|0|
|横付け優先あり|1|

#### 範囲
|n（経由地の数）|値|
|---|---|
|最大|30|

|滞在時間（分）|値|
|---|---|
|最小|0|
|最大|10080|

#### デフォルト

##### 地点種別
`0`

##### 滞在時間（分）
`0`

##### 横付け優先
`0`

#### 記述例

##### 経由地を3地点指定
`139.75723347,35.66593095|139.73893125,35.62911118|139.69695771,35.53136405`

##### 緯度経度のみ指定
`139.75723347,35.66593095`

##### 地点種別も合わせて指定
`139.75723347,35.66593095,1`

##### 滞在時間も合わせて指定
`139.75723347,35.66593095,1,60`

##### 横付け優先も合わせて指定
`139.75723347,35.66593095,1,60,1`

##### 地点種別指定：なし、滞在時間指定：あり
`139.75723347,35.66593095,,60`

##### 地点種別指定：なし、滞在時間指定：なし、横付け優先指定：あり
`139.75723347,35.66593095,,,1`

#### 注記
- 緯度経度の測地系は、世界測地系（JGD2011）となります。
- 緯度経度の小数点は第8位まで考慮され、それより大きい値は無視されます。
- 指定地点から約5kmの範囲に道路がある場合にルート検索が可能です。道路が見つからない場合、エラー（`I00104`）となります。
歩行者（`priority=100～103`）の場合も同様です。
        startangle: ### 出発方向
出発地の最近傍リンクが双方向に進行可能な場合に、進行方向を指定する事ができます。

指定した場合、最近傍リンクの進行可能な方向に対し、指定角度に近似している方向に進行します。
指定がない場合、最近傍リンクの進行可能な方向のいずれかが、ルートの状況に応じて採用されます。真北を0度とし、時計回りに増加します。

#### 範囲
|範囲|値|
|---|---|
|最小|0（北上）|
|最大|359|

#### デフォルト
`0`

#### 注記
- 整数のみ指定可能です。
- 時計回りに増加します。
        priority: ### 基本条件
|基本条件|値|
|---|---|
|標準|0|
|距離優先|1|
|直進優先|2|
|簡易歩行者|3|
|道幅優先|4|

#### 注記
- `altcalcroute`では、`priority=100～103`は指定できません。
- 各条件の詳細は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ ルート検索 ＞ 補足 ＞ ルート基本条件 (priority)』をご確認ください。
        start: ### 出発地緯度経度
#### 記述形式

`[経度の値],[緯度の値],[地点種別]`
※`[地点種別]`は省略可能です。

|地点種別|値|
|---|---|
|一般道・有料道|0|
|一般道のみ|1|
|有料道のみ|2|

#### デフォルト

##### 地点種別
`0`

#### 記述例
##### 緯度経度のみ指定
`139.76730676,35.68095910`

##### 地点種別も合わせて指定
`139.76730676,35.68095910,1`

#### 注記
- `routeresultid`が指定されている場合のみ、本パラメータは省略可能です。
- 緯度経度の測地系は、世界測地系（JGD2011）となります。
- 緯度経度の小数点は第8位まで考慮され、それより大きい値は無視されます。
- 指定地点から約5kmの範囲に道路がある場合にルート検索が可能です。道路が見つからない場合、エラー（`I00104`）となります。
歩行者（`priority=100～103`）の場合も同様です。
        
    """
    url = f"https://mapfanapi-route.p.rapidapi.com/altcalcroute"
    querystring = {}
    if fmt:
        querystring['fmt'] = fmt
    if passablearea:
        querystring['passablearea'] = passablearea
    if callback:
        querystring['callback'] = callback
    if saveresult:
        querystring['saveresult'] = saveresult
    if daytime:
        querystring['daytime'] = daytime
    if generalroad:
        querystring['generalroad'] = generalroad
    if uturnavoid:
        querystring['uturnavoid'] = uturnavoid
    if tollroad:
        querystring['tollroad'] = tollroad
    if regulations:
        querystring['regulations'] = regulations
    if impassablearea:
        querystring['impassablearea'] = impassablearea
    if routeid:
        querystring['routeid'] = routeid
    if uturn:
        querystring['uturn'] = uturn
    if danger:
        querystring['danger'] = danger
    if width:
        querystring['width'] = width
    if weight:
        querystring['weight'] = weight
    if highwayspeed:
        querystring['highwayspeed'] = highwayspeed
    if height:
        querystring['height'] = height
    if loadage:
        querystring['loadage'] = loadage
    if normalspeed:
        querystring['normalspeed'] = normalspeed
    if tollwayspeed:
        querystring['tollwayspeed'] = tollwayspeed
    if ferryspeed:
        querystring['ferryspeed'] = ferryspeed
    if vehicletype:
        querystring['vehicletype'] = vehicletype
    if tolltarget:
        querystring['tolltarget'] = tolltarget
    if smartic:
        querystring['smartic'] = smartic
    if cartype:
        querystring['cartype'] = cartype
    if date:
        querystring['date'] = date
    if tollway:
        querystring['tollway'] = tollway
    if etc:
        querystring['etc'] = etc
    if ferry:
        querystring['ferry'] = ferry
    if destination:
        querystring['destination'] = destination
    if via:
        querystring['via'] = via
    if startangle:
        querystring['startangle'] = startangle
    if priority:
        querystring['priority'] = priority
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapfanapi-route.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nearroadinfo(lonlat: str, callback: str=None, fmt: str=None, radius: int=None, date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## 周辺道路情報取得
		指定された緯度経度を中心とした周辺の道路情報の検索機能を提供します。"
    lonlat: ### 緯度経度

#### 記述形式
` [経度の値],[緯度の値]`

#### 記述例
`139.767231,35.681196`

#### 注記
緯度経度の測地系は、世界測地系（JGD2011）となります。
        callback: ### JSONPのコールバック用関数名
関数名が指定されている場合はJSONP形式で結果を返却します。
指定されていない場合は、`fmt`で指定された形式で結果を返却します。
        fmt: ### 出力フォーマット
|フォーマット|値|
|---|---|
|JSON|json |
|XML|xml|

#### デフォルト
`json`

#### 注記
`callback`が設定されている場合、本パラメータの設定が破棄され、JSONP形式が適用されます。
        radius: ### 検索範囲
検索半径をメートル単位で指定します。

#### 範囲
|範囲|値|
|---|---|
|最小|1（1m）|
|最大|10000（10km）|

#### デフォルト
`5000`（5km）

#### 注記
整数のみ指定可能です。
        date: ### 検索日時
以下のレスポンスで考慮されます。
- `isPassableForward`（通行可能フラグ（順方向））
- `impassableCodeForward`（通行不可詳細（順方向））
- `isPassableBackward`（通行可能フラグ（逆方向））
- `impassableCodeBackward`（通行不可詳細（逆方向））

#### 記述形式
`[年4桁][月2桁][日2桁]_[時2桁][分2桁][秒2桁]` （`yyyyMMdd_HHmmss`）

#### 記述例
`20220420_170505` （2022年4月20日午後5時5分5秒 ）

#### デフォルト
API実行日時を適用して、検索を行います。
        
    """
    url = f"https://mapfanapi-route.p.rapidapi.com/nearroadinfo"
    querystring = {'lonlat': lonlat, }
    if callback:
        querystring['callback'] = callback
    if fmt:
        querystring['fmt'] = fmt
    if radius:
        querystring['radius'] = radius
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapfanapi-route.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

