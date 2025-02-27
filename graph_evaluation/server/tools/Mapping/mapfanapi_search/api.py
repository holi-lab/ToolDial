import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def stalist(stepno: str, typecd: str=None, linecd: str=None, lineseq: int=None, corpcd: str=None, fmt: str=None, callback: str=None, pos: int=None, num: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## 駅階層検索
		駅階層検索機能を提供します。"
    stepno: ### 検索階層 
|検索階層|値|例|
|---|---|---|
|鉄道分類一覧|0|ＪＲ, 私鉄, 地下鉄, モノレール…|
|鉄道会社一覧|1|新幹線, 幹線, 小田急電鉄, 阪急電鉄…|
|鉄道路線一覧|2|山手線, 湘南新宿ライン, 小田急小田原線, 小田急多摩線…|
|駅一覧|3|東京駅（山手線）, 新橋駅（山手線）, 高輪ゲートウェイ駅（山手線）…|
|駅出入口一覧|4|丸の内中央口, 八重洲中央口, ３, ５, １４…|
        typecd: ### 鉄道分類コード(2桁)
`stepno=0（鉄道分類一覧） `のレスポンス（`typecd`）から取得します。

#### 注記
`stepno=1, 2, 3, 4`を指定した場合は必須です。
        linecd: ### 鉄道会社コード(3桁)
`stepno=2（鉄道路線一覧） `のレスポンス（`linecd`）から取得します。

#### 注記
`stepno=3, 4`を指定した場合は必須です。
        lineseq: ### 鉄道会社コード(3桁)
`stepno=3（駅一覧） `のレスポンス（`lineseq`）から取得します。

#### 注記
`stepno=4`を指定した場合は必須です。
        corpcd: ### 鉄道会社コード(3桁)
`stepno=1（鉄道会社一覧） `のレスポンス（`corpcd`）から取得します。

#### 注記
`stepno=2, 3, 4`を指定した場合は必須です。
        fmt: ### 出力フォーマット
|フォーマット|値|
|---|---|
|JSON|json |
|XML|xml|

#### デフォルト
`json`

#### 注記
`callback`が設定されている場合、本パラメータの設定が破棄され、JSONP形式が適用されます。
        callback: ### JSONPのコールバック用関数名
関数名が指定されている場合はJSONP形式で結果を返却します。
指定されていない場合は、`fmt`で指定された形式で結果を返却します。
        pos: ### 取得開始位置
検索結果の返却開始位置を指定します。

#### デフォルト
`1`
※1件目から `num` で指定された件数分の結果を返却します。
        num: ### 取得件数
検索結果の返却件数を指定します。

#### 範囲
|範囲|値|
|---|---|
|最大|500|

#### デフォルト
全件取得
        
    """
    url = f"https://mapfanapi-search.p.rapidapi.com/stalist"
    querystring = {'stepno': stepno, }
    if typecd:
        querystring['typecd'] = typecd
    if linecd:
        querystring['linecd'] = linecd
    if lineseq:
        querystring['lineseq'] = lineseq
    if corpcd:
        querystring['corpcd'] = corpcd
    if fmt:
        querystring['fmt'] = fmt
    if callback:
        querystring['callback'] = callback
    if pos:
        querystring['pos'] = pos
    if num:
        querystring['num'] = num
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapfanapi-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def addrlist(level: str, citycd: str=None, fmt: str=None, num: int=None, pos: int=None, prefcd: str=None, callback: str=None, tyocd: str=None, bancd: str=None, gov: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## 住所階層検索
		住所階層検索機能を提供します。"
    level: ### 検索階層
※例：`東京都文京区本駒込２－２８－８`

|階層|値|例|
|---|---|---|
|都道府県一覧|0|東京都|
|市区町村一覧|1|文京区|
|条町丁目一覧|2|本駒込２|
|番地一覧|3|２８|
|号一覧|4|８|
        citycd: ### 市区町村コード(数字3桁)
`level=1（市区町村一覧）` のレスポンス（`citycd`）から取得します。

#### 注記
`level=2, 3, 4` を指定した場合は必須。
        fmt: ### 出力フォーマット
|フォーマット|値|
|---|---|
|JSON|json |
|XML|xml|

#### デフォルト
`json`

#### 注記
`callback`が設定されている場合、本パラメータの設定が破棄され、JSONP形式が適用されます。
        num: ### 取得件数
検索結果の返却件数を指定します。

#### 範囲
|範囲|値|
|---|---|
|最大|500|

#### デフォルト
`500`
        pos: ### 取得開始位置
検索結果の返却開始位置を指定します。

#### デフォルト
`1`
※1件目から `num` で指定された件数分の結果を返却します。
        prefcd: ### 都道府県コード(数字2桁)
`level=0（都道府県一覧） `のレスポンス（`prefcd`）から取得します。

#### 注記
`level=1, 2, 3, 4`を指定した場合は必須です。
        callback: ### JSONPのコールバック用関数名
関数名が指定されている場合はJSONP形式で結果を返却します。
指定されていない場合は、`fmt`で指定された形式で結果を返却します。
        tyocd: ### 条町丁目コード(英数6桁)
`level=2（条町丁目一覧）` のレスポンス（`tyocd`）から取得します。

#### 注記
`level=3, 4` を指定した場合は必須です。
        bancd: ### 番地コード(英数5桁)
`level=3（番地一覧）` のレスポンス（`bancd`）から取得します。

#### 注記
`level=4` を指定した場合は必須です。
        gov: ### 政令指定都市表示有無設定
検索結果に、行政区（横浜市西区 等）に加えて、政令指定都市（横浜市）を追加で表示するかを制御します。

|設定|値|
|---|---|
|表示する|0 |
|表示しない|1|

#### デフォルト
`0`
        
    """
    url = f"https://mapfanapi-search.p.rapidapi.com/addrlist"
    querystring = {'level': level, }
    if citycd:
        querystring['citycd'] = citycd
    if fmt:
        querystring['fmt'] = fmt
    if num:
        querystring['num'] = num
    if pos:
        querystring['pos'] = pos
    if prefcd:
        querystring['prefcd'] = prefcd
    if callback:
        querystring['callback'] = callback
    if tyocd:
        querystring['tyocd'] = tyocd
    if bancd:
        querystring['bancd'] = bancd
    if gov:
        querystring['gov'] = gov
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapfanapi-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def zip(zipcd: str, pos: int=None, fmt: str=None, ot: str=None, num: int=None, callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## 郵便番号検索
		郵便番号による住所検索機能を提供します。"
    zipcd: ### 検索対象の郵便番号
3桁または7桁の郵便番号を指定します。
7桁の場合、半角ハイフンを含む郵便番号、ハイフンなしの郵便番号の入力が可能です。
        pos: ### 取得開始位置
検索結果の返却開始位置を指定します。

#### デフォルト
`1`
※1件目から `num` で指定された件数分の結果を返却します。
        fmt: ### 出力フォーマット
|フォーマット|値|
|---|---|
|JSON|json |
|XML|xml|

#### デフォルト
`json`

#### 注記
`callback`パラメータが設定されている場合、本パラメータの設定が破棄され、JSONP形式が適用されます。
        ot: ### 住所出力形式
|設定|値|
|---|---|
|大口事業所、ビル名または住所|0 |
|住所のみを出力|1|

#### デフォルト
`0` ※大口事業所、ビル名を優先して出力します。
        num: ### 取得件数
検索結果の返却件数を指定します。

#### 範囲
|範囲|値|
|---|---|
|最大|500|

#### デフォルト
`50`
        callback: ### JSONPのコールバック用関数名
関数名が指定されている場合はJSONP形式で結果を返却します。
指定されていない場合は、`fmt`で指定された形式で結果を返却します。
        
    """
    url = f"https://mapfanapi-search.p.rapidapi.com/zip"
    querystring = {'zipcd': zipcd, }
    if pos:
        querystring['pos'] = pos
    if fmt:
        querystring['fmt'] = fmt
    if ot:
        querystring['ot'] = ot
    if num:
        querystring['num'] = num
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapfanapi-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def spot(name: str, openinghour: str=None, facility: str=None, routepoint: str=None, field: str=None, gnrcd: str=None, citycd: str=None, num: int=None, fmt: str=None, prefcd: str=None, callback: str=None, pos: int=None, phonebook: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## スポット検索
		スポット検索機能を提供します。"
    name: ### 検索対象の文字列
スペース区切りで複数のキーワードを指定可能です。複数指定した場合は、AND検索となります。

#### 注記
URLエンコードされたUTF-8の文字列を指定します。
        openinghour: ### 営業時間出力有無
|設定|値|
|---|---|
|出力しない|0|
|出力する|1|

#### デフォルト
`0`

#### 注記
- 営業時間の詳細は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ スポット検索 ＞ 補足』をご参照ください。
- 当該情報を取得可能なデータ・ジャンルにつきましては、『API仕様書：MapFanAPI_ジャンルコード一覧』をご参照ください。
        facility: ### 付帯設備・サービス情報出力有無
|設定|値|
|---|---|
|出力しない|0|
|出力する|1|

#### デフォルト
`0`

#### 注記
- 以下の情報を出力します。
	- ATM設置有無
	- 酒取り扱い有無
	- たばこ取り扱い有無
	- ドライブスルー設置有無
	- 駐車場設置有無
- 当該情報を取得可能なデータ・ジャンルにつきましては、『API仕様書：MapFanAPI_ジャンルコード一覧』をご参照ください。
        routepoint: ### ルート探索用ポイント出力有無
|設定|値|
|---|---|
|出力しない|0|
|出力する|1|

#### デフォルト
`0`

#### 注記
ルート探索用ポイントの詳細は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ スポット検索 ＞ 補足』をご参照ください。
        field: ### 検索フィールド
スポットの特定の属性のみを対象としてキーワード検索を行う場合に、対象とする属性（検索フィールド）を指定します。
カンマ区切りで複数の検索フィールドを指定可能です。複数指定した場合はOR検索となります。

|指定可能な検索フィールド|フィールド名|
|---|
|name|スポット名|
|yomi|スポット名読み|
|gnr_name|ジャンル名|
|addr|住所|
|tel|電話番号|
|comment|コメント|
|access|アクセス情報<br>※スポットの最寄りの鉄道路線・駅、インターチェンジ|
        gnrcd: ### ジャンルコード
特定ジャンルのスポットのみを対象として検索を行う場合に、検索対象のジャンルコードを指定します。
カンマ区切りで複数のジャンルコードを指定可能です。複数指定した場合はOR検索となります。

#### 注記
指定可能なジャンルコードは、『API仕様書：MapFanAPI_ジャンルコード一覧』をご確認ください。
        citycd: ### 市区町村コード(数字3桁)
特定の市区町村のスポットのみを対象として検索を行う場合に、検索対象の市区町村コードを指定します。
        num: ### 取得件数
検索結果の返却件数を指定します。

#### 範囲
|範囲|値|
|---|---|
|最大|500|

#### デフォルト
`50`
        fmt: ### 出力フォーマット
|フォーマット|値|
|---|---|
|JSON|json |
|XML|xml|

#### デフォルト
`json`

#### 注記
`callback`が設定されている場合、本パラメータの設定が破棄され、JSONP形式が適用されます。
        prefcd: ### 都道府県コード(数字2桁)
特定の都道府県のスポットのみを対象として検索を行う場合に、検索対象の都道府県コードを指定します。

#### 注記
`citycd`を指定した場合は必須です。
        callback: ### JSONPのコールバック用関数名
関数名が指定されている場合はJSONP形式で結果を返却します。
指定されていない場合は、`fmt`で指定された形式で結果を返却します。
        pos: ### 取得開始位置
検索結果の返却開始位置を指定します。

#### デフォルト
`1`
※1件目から `num` で指定された件数分の結果を返却します。
        phonebook: ### 電話帳データ検索オプション
|設定|値|
|---|---|
|電話帳データを検索対象としない<br>（弊社独自整備POIのみ検索）|0|
|弊社独自整備POIとMapFan Directory（※）を検索対象とする|1|

※MapFan Directoryは、グリーンページ®α(日本ソフト販売株式会社提供)を基に、弊社が制作した電話帳データベースです。

#### デフォルト
`0`
        
    """
    url = f"https://mapfanapi-search.p.rapidapi.com/spot"
    querystring = {'name': name, }
    if openinghour:
        querystring['openinghour'] = openinghour
    if facility:
        querystring['facility'] = facility
    if routepoint:
        querystring['routepoint'] = routepoint
    if field:
        querystring['field'] = field
    if gnrcd:
        querystring['gnrcd'] = gnrcd
    if citycd:
        querystring['citycd'] = citycd
    if num:
        querystring['num'] = num
    if fmt:
        querystring['fmt'] = fmt
    if prefcd:
        querystring['prefcd'] = prefcd
    if callback:
        querystring['callback'] = callback
    if pos:
        querystring['pos'] = pos
    if phonebook:
        querystring['phonebook'] = phonebook
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapfanapi-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def staarea(lonlat: str, rad: int=None, fmt: str=None, num: int=None, callback: str=None, gateway: str=None, pos: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## 最寄駅検索
		指定された緯度経度を中心とした周辺駅の検索機能を提供します。"
    lonlat: ### 緯度経度

#### 記述形式
` [経度の値],[緯度の値]`

#### 記述例
`139.767231,35.681196`

#### 注記
緯度経度の測地系は、世界測地系（JGD2011）となります。
        rad: ### 検索範囲
検索半径をメートル単位で指定します。

#### 範囲
|範囲|値|
|---|---|
|最大|50000（50km）|

#### デフォルト
`1000`（1km）

#### 注記
整数のみ指定可能です。
        fmt: ### 出力フォーマット
|フォーマット|値|
|---|---|
|JSON|json |
|XML|xml|

#### デフォルト
`json`

#### 注記
`callback`が設定されている場合、本パラメータの設定が破棄され、JSONP形式が適用されます。
        num: ### 取得件数
検索結果の返却件数を指定します。

#### 範囲
|範囲|値|
|---|---|
|最大|500|

#### デフォルト
`50`

#### 注記
`gateway=1`を指定した場合（駅出入口を検索対象に含めた場合）、駅検索結果、駅出入口検索結果それぞれの返却件数に適用されます。
        callback: ### JSONPのコールバック用関数名
関数名が指定されている場合はJSONP形式で結果を返却します。
指定されていない場合は、`fmt`で指定された形式で結果を返却します。
        gateway: ### 駅出入口検索設定
駅出入口を検索対象に含めるかを指定します。

|設定|値|
|---|---|
|含めない|0|
|含める|1|

#### デフォルト
`0`
        pos: ### 取得開始位置
検索結果の返却開始位置を指定します。

#### デフォルト
`1`
※1件目から `num` で指定された件数分の結果を返却します。
        
    """
    url = f"https://mapfanapi-search.p.rapidapi.com/staarea"
    querystring = {'lonlat': lonlat, }
    if rad:
        querystring['rad'] = rad
    if fmt:
        querystring['fmt'] = fmt
    if num:
        querystring['num'] = num
    if callback:
        querystring['callback'] = callback
    if gateway:
        querystring['gateway'] = gateway
    if pos:
        querystring['pos'] = pos
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapfanapi-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def addrname(lonlat: str, fmt: str=None, callback: str=None, level: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## 住所逆引き検索
		緯度経度から住所情報を取得する住所逆引き検索機能を提供します。"
    lonlat: ### 緯度経度

#### 記述形式
` [経度の値],[緯度の値]`

#### 記述例
`139.767231,35.681196`

#### 注記
緯度経度の測地系は、世界測地系（JGD2011）となります。
        fmt: ### 出力フォーマット
|フォーマット|値|
|---|---|
|JSON|json |
|XML|xml|

#### デフォルト
`json`

#### 注記
`callback`が設定されている場合、本パラメータの設定が破棄され、JSONP形式が適用されます。
        callback: ### JSONPのコールバック用関数名
関数名が指定されている場合はJSONP形式で結果を返却します。
指定されていない場合は、`fmt`で指定された形式で結果を返却します。
        level: ### 精度レベル
取得したい住所文字列の精度を指定します。

|精度レベル|値|
|---|---|
|町・丁目レベル|2|
|番地・号レベル|4|

#### デフォルト
`2`

#### 注記
番地・号レベルでの検索には制限事項があります。詳細は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ 住所逆引き検索 ＞ 補足』をご参照ください。
        
    """
    url = f"https://mapfanapi-search.p.rapidapi.com/addrname"
    querystring = {'lonlat': lonlat, }
    if fmt:
        querystring['fmt'] = fmt
    if callback:
        querystring['callback'] = callback
    if level:
        querystring['level'] = level
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapfanapi-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def addr(addr: str, pos: int=None, callback: str=None, fmt: str=None, gov: str=None, num: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## 住所検索
		住所検索機能を提供します。"
    addr: ### 検索対象の住所文字列

#### 注記
- URLエンコードされたUTF-8の文字列を指定します。
- 住所文字列の入出力仕様の詳細は、『API仕様書：MapFanAPI_住所検索入出力パターン』をご参照ください。
        pos: ### 取得開始位置
検索結果の返却開始位置を指定します。

#### デフォルト
`1`
※1件目から `num` で指定された件数分の結果を返却します。
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
`callback`パラメータが設定されている場合、本パラメータの設定が破棄され、JSONP形式が適用されます。
        gov: ### 政令指定都市表示有無設定
|設定|値|”仙台”を検索した場合|
|---|---|---|
|行政区を表示|0 |宮城県仙台市青葉区<br>宮城県仙台市宮城野区<br>宮城県仙台市若林区<br>宮城県仙台市太白区<br>宮城県仙台市泉区|
|政令市を表示|1|宮城県仙台市|

#### デフォルト
`0`
        num: ### 取得件数
検索結果の返却件数を指定します。

#### 範囲
|範囲|値|
|---|---|
|最大|500|

#### デフォルト
`500`
        
    """
    url = f"https://mapfanapi-search.p.rapidapi.com/addr"
    querystring = {'addr': addr, }
    if pos:
        querystring['pos'] = pos
    if callback:
        querystring['callback'] = callback
    if fmt:
        querystring['fmt'] = fmt
    if gov:
        querystring['gov'] = gov
    if num:
        querystring['num'] = num
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapfanapi-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sta(name: str, pos: int=None, fmt: str=None, gateway: str=None, callback: str=None, num: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## 駅検索
		駅検索機能を提供します。"
    name: ### 検索対象の駅名称文字列
スペース区切りで複数のキーワードを指定可能です。複数指定した場合は、AND検索となります。

#### 注記
- URLエンコードされたUTF-8の文字列を指定します。
- 末尾の “駅”は省略可能ですが、省略することで異なる検索結果となる場合があります。
        pos: ### 取得開始位置
検索結果の返却開始位置を指定します。

#### デフォルト
`1`
※1件目から `num` で指定された件数分の結果を返却します。
        fmt: ### 出力フォーマット
|フォーマット|値|
|---|---|
|JSON|json |
|XML|xml|

#### デフォルト
`json`

#### 注記
`callback`が設定されている場合、本パラメータの設定が破棄され、JSONP形式が適用されます。
        gateway: ### 駅出入口検索設定
駅出入口を検索対象に含めるかを指定します。

|設定|値|
|---|---|
|含めない|0|
|含める|1|

#### デフォルト
`0`
        callback: ### JSONPのコールバック用関数名
関数名が指定されている場合はJSONP形式で結果を返却します。
指定されていない場合は、`fmt`で指定された形式で結果を返却します。
        num: ### 取得件数
検索結果の返却件数を指定します。

#### 範囲
|範囲|値|
|---|---|
|最大|500|

#### デフォルト
`50`

#### 注記
`gateway=1`を指定した場合（駅出入口を検索対象に含めた場合）、駅検索結果、駅出入口検索結果それぞれの返却件数に適用されます。
        
    """
    url = f"https://mapfanapi-search.p.rapidapi.com/sta"
    querystring = {'name': name, }
    if pos:
        querystring['pos'] = pos
    if fmt:
        querystring['fmt'] = fmt
    if gateway:
        querystring['gateway'] = gateway
    if callback:
        querystring['callback'] = callback
    if num:
        querystring['num'] = num
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapfanapi-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def spotarea(lonlat: str, openinghour: str=None, facility: str=None, num: int=None, callback: str=None, gnrcd: str=None, routepoint: str=None, name: str=None, fmt: str=None, rad: int=None, phonebook: str=None, pos: int=None, field: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## スポット周辺検索
		指定された緯度経度を中心とした周辺スポットの検索機能を提供します。"
    lonlat: ### 緯度経度

#### 記述形式
` [経度の値],[緯度の値]`

#### 記述例
`139.767231,35.681196`

#### 注記
緯度経度の測地系は、世界測地系（JGD2011）となります。
        openinghour: ### 営業時間出力有無
|設定|値|
|---|---|
|出力しない|0|
|出力する|1|

#### デフォルト
`0`

#### 注記
- 営業時間の詳細は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ スポット検索 ＞ 補足』をご参照ください。
- 当該情報を取得可能なデータ・ジャンルにつきましては、『API仕様書：MapFanAPI_ジャンルコード一覧』をご参照ください。
        facility: ### 付帯設備・サービス情報出力有無
|設定|値|
|---|---|
|出力しない|0|
|出力する|1|

#### デフォルト
`0`

#### 注記
- 以下の情報を出力します。
	- ATM設置有無
	- 酒取り扱い有無
	- たばこ取り扱い有無
	- ドライブスルー設置有無
	- 駐車場設置有無
- 当該情報を取得可能なデータ・ジャンルにつきましては、『API仕様書：MapFanAPI_ジャンルコード一覧』をご参照ください。
        num: ### 取得件数
検索結果の返却件数を指定します。

#### 範囲
|範囲|値|
|---|---|
|最大|500|

#### デフォルト
`50`
        callback: ### JSONPのコールバック用関数名
関数名が指定されている場合はJSONP形式で結果を返却します。
指定されていない場合は、`fmt`で指定された形式で結果を返却します。
        gnrcd: ### ジャンルコード
特定ジャンルのスポットのみを対象として検索を行う場合に、検索対象のジャンルコードを指定します。
カンマ区切りで複数のジャンルコードを指定可能です。複数指定した場合はOR検索となります。

#### 注記
指定可能なジャンルコードは、『API仕様書：MapFanAPI_ジャンルコード一覧』をご確認ください。
        routepoint: ### ルート探索用ポイント出力有無
|設定|値|
|---|---|
|出力しない|0|
|出力する|1|

#### デフォルト
`0`

#### 注記
ルート探索用ポイントの詳細は、『API仕様書：MapFanAPI_サーバーAPI技術仕様書 ＞ スポット検索 ＞ 補足』をご参照ください。
        name: ### 検索対象の絞り込み文字列
スペース区切りで複数のキーワードを指定可能です。複数指定した場合は、AND検索となります。

#### 注記
- URLエンコードされたUTF-8の文字列を指定します。
- `field`を指定した場合は必須です。
        fmt: ### 出力フォーマット
|フォーマット|値|
|---|---|
|JSON|json |
|XML|xml|

#### デフォルト
`json`

#### 注記
`callback`が設定されている場合、本パラメータの設定が破棄され、JSONP形式が適用されます。
        rad: ### 検索範囲
検索半径をメートル単位で指定します。

#### 範囲
|範囲|値|
|---|---|
|最大|50000（50km）|

#### デフォルト
`1000`（1km）

#### 注記
整数のみ指定可能です。
        phonebook: ### 電話帳データ検索オプション
|設定|値|
|---|---|
|電話帳データを検索対象としない<br>（弊社独自整備POIのみ検索）|0|
|弊社独自整備POIとMapFan Directory（※）を検索対象とする|1|

※MapFan Directoryは、グリーンページ®α(日本ソフト販売株式会社提供)を基に、弊社が制作した電話帳データベースです。

#### デフォルト
`0`
        pos: ### 取得開始位置
検索結果の返却開始位置を指定します。

#### デフォルト
`1`
※1件目から `num` で指定された件数分の結果を返却します。
        field: ### 検索フィールド
スポットの特定の属性のみを対象としてキーワード検索を行う場合に、対象とする属性（検索フィールド）を指定します。
カンマ区切りで複数の検索フィールドを指定可能です。複数指定した場合はOR検索となります。

|指定可能な検索フィールド|フィールド名|
|---|
|name|スポット名|
|yomi|スポット名読み|
|gnr_name|ジャンル名|
|addr|住所|
|tel|電話番号|
|comment|コメント|
|access|アクセス情報<br>※スポットの最寄りの鉄道路線・駅、インターチェンジ|
        
    """
    url = f"https://mapfanapi-search.p.rapidapi.com/spotarea"
    querystring = {'lonlat': lonlat, }
    if openinghour:
        querystring['openinghour'] = openinghour
    if facility:
        querystring['facility'] = facility
    if num:
        querystring['num'] = num
    if callback:
        querystring['callback'] = callback
    if gnrcd:
        querystring['gnrcd'] = gnrcd
    if routepoint:
        querystring['routepoint'] = routepoint
    if name:
        querystring['name'] = name
    if fmt:
        querystring['fmt'] = fmt
    if rad:
        querystring['rad'] = rad
    if phonebook:
        querystring['phonebook'] = phonebook
    if pos:
        querystring['pos'] = pos
    if field:
        querystring['field'] = field
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapfanapi-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

