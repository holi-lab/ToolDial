import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def map(tilerow: int, tilematrix: str, tilecol: int, rotationangle: int=None, mapstyle: str=None, landmarkicon: str=None, resolution: str=None, contour: str=None, bldgname: str=None, logo: str=None, target: str=None, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## 地図画像取得
		WMTS形式の地図画像を提供します。
		`tilematrix`, `tilerow`, `tilecol` で指定した位置の地図画像を返します。
		
		本エンドポイントを用いてスクロール地図機能を実装する方法につきましては、 [こちらのチュートリアル](https://rapidapi.com/ja/geotechnologies12-geotechnologies-default/api/mapfanapi-map/tutorials/leaflet%E3%81%AE%E3%82%BF%E3%82%A4%E3%83%AB%E3%83%AC%E3%82%A4%E3%83%A4%E3%83%BCurl%E3%81%AB%E3%83%AA%E3%82%AF%E3%82%A8%E3%82%B9%E3%83%88%E3%83%98%E3%83%83%E3%83%80%E3%83%BC%E3%82%92%E8%A8%AD%E5%AE%9A%E3%81%99%E3%82%8B%EF%BC%88rapidapi%E3%81%AEmap%E3%82%A8%E3%83%B3%E3%83%89%E3%83%9D%E3%82%A4%E3%83%B3%E3%83%88%E3%81%AE%E5%88%A9%E7%94%A8%EF%BC%89) をご確認ください。"
    tilerow: ### WMTSタイル番号（緯度方向）
地図取得位置の緯度をWMTSタイル番号に変換した値
        tilematrix: ### タイルマトリックス

#### 記述形式
`[EPSGコード]:[スケール値]`

|項目|値|
|---|---|
|EPSGコード|EPSG:3857<br>EPSG:900913<br>のいずれか|
|スケール|6 ～ 21<br>のいずれか|
        tilecol: ### WMTSタイル番号（経度方向）
地図取得位置の経度をWMTSタイル番号に変換した値
        rotationangle: ### 角度パラメータ
注記文字列を指定した角度に傾けた画像が返されます。

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
        mapstyle: ### 地図デザイン指定用パラメータ

#### 記述形式
`[デザイン]_[注記サイズ]`
|デザイン|値|
|---|---|
|標準|std|
|グレースケール|gray|
|RPG風地図|rpg|
|古地図風地図|antique|
|道路強調地図|hlightroad|
|鉄道強調地図|hlightrail|
|印刷用白黒デザイン|monochrome|
|おもてなしマップ|tourism|

|注記サイズ|値|
|---|---|
|中|pc|
|大|sp|

#### デフォルト
`std_pc`

#### 注記
詳細は、『API仕様書：MapFanAPI_地図デザイン設定方法と表示イメージ』をご確認ください。
        landmarkicon: ### ランドマークアイコンの表示/非表示指定パラメータ
|表示設定|値|
|---|---|
|非表示|off|
|表示|on|

#### デフォルト
`off`

#### 注記
mapstyleパラメータの一部の値を指定した場合に本パラメータを on を指定すると、`[E03004] map parameter error (invalid combination)`が返却されます。
mapstyleと他パラメータの指定可否の組み合わせは、『API仕様書：MapFanAPI_地図デザイン設定方法と表示イメージ』 をご確認ください。
        resolution: ### 解像度パラメータ
解像度に2を指定した場合、標準の2倍のサイズの画像が返され、4を指定した場合、標準の4倍のサイズの画像が返されます。

|解像度|値|画像サイズ|
|---|---|---|
|標準|1|256×256|
|2倍|2|512×512|
|4倍|4|1024×1024|

#### デフォルト
`1`
        contour: ### 等高線の表示/非表示指定パラメータ
|表示設定|値|
|---|---|
|非表示|off|
|表示|on|

#### デフォルト
`on`

#### 注記
mapstyleパラメータの一部の値を指定した場合に本パラメータを on を指定すると、`[E03004] map parameter error (invalid combination)`が返却されます。
mapstyleと他パラメータの指定可否の組み合わせは、『API仕様書：MapFanAPI_地図デザイン設定方法と表示イメージ』をご確認ください。
        bldgname: ### ビル名の表示/非表示指定パラメータ
|表示設定|値|
|---|---|
|非表示|off|
|表示|on|

#### デフォルト
`off`

#### 注記
- mapstyleパラメータの一部の値を指定した場合に本パラメータを on を指定すると、`[E03004] map parameter error (invalid combination)`が返却されます。
mapstyleと他パラメータの指定可否の組み合わせは、『API仕様書：MapFanAPI_地図デザイン設定方法と表示イメージ』をご確認ください。
- ビル名称は、スケールが19以上の場合のみ描画されます。
        logo: ### ブランドロゴの表示/非表示指定パラメータ
|表示設定|値|
|---|---|
|非表示|off|
|表示|on|

#### デフォルト
`off`

#### 注記
mapstyleパラメータの一部の値を指定した場合に本パラメータを on を指定すると、`[E03004] map parameter error (invalid combination)`が返却されます。
mapstyleと他パラメータの指定可否の組み合わせは、『API仕様書：MapFanAPI_地図デザイン設定方法と表示イメージ』をご確認ください。
        target: ### 地図の表示内容
|表示内容|値|
|---|---|
|背景のみを表示|1|
|注記のみを表示|2|
|背景 + 注記を表示|3|

#### デフォルト
`3`
        format: ### 出力する画像形式

|画像形式|値|
|---|---|
|png|image/png|
|jpeg|image/jpeg|

#### デフォルト
`image/png`
        
    """
    url = f"https://mapfanapi-map.p.rapidapi.com/map"
    querystring = {'tilerow': tilerow, 'tilematrix': tilematrix, 'tilecol': tilecol, }
    if rotationangle:
        querystring['rotationangle'] = rotationangle
    if mapstyle:
        querystring['mapstyle'] = mapstyle
    if landmarkicon:
        querystring['landmarkicon'] = landmarkicon
    if resolution:
        querystring['resolution'] = resolution
    if contour:
        querystring['contour'] = contour
    if bldgname:
        querystring['bldgname'] = bldgname
    if logo:
        querystring['logo'] = logo
    if target:
        querystring['target'] = target
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapfanapi-map.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mapimage(lonlat: str, scale: int, size: str, icon: str=None, centericon: str=None, logo: str=None, landmarkicon: str=None, format: str=None, resolution: str=None, rotationangle: int=None, routeresultid: str=None, routeicon: str=None, scaler: str=None, bldgname: str=None, contour: str=None, target: str=None, mapstyle: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## サイズ指定地図画像取得
		指定された緯度経度、サイズ、縮尺の地図画像を提供します。
		`size`, `lonlat`, `scale` で指定した画像サイズ、位置、スケールの地図画像を返します。
		resolutionで解像度に2を指定した場合、sizeで指定した画像サイズの2倍の画像サイズで返されます。"
    lonlat: ### 中心緯度経度

#### 記述形式
`経度,緯度`

#### 記述例
`139.767231,35.681196`

#### 範囲
|範囲|値|
|---|---|
|最小|経度 120, 緯度20|
|最大|経度 155, 緯度46|

#### 注記
緯度経度の測地系は、世界測地系（JGD2011）となります。
        scale: ### 地図縮尺

#### 範囲
|範囲|値|
|---|---|
|最小|6|
|最大|21|
        size: ### 画像サイズ

#### 記述形式
`[画像幅（ピクセル値）],[画像高さ（ピクセル値）]`

#### 記述例
`320,240`

#### 範囲
|範囲|サイズ|
|---|---|
|最小値|1px, 1px|
|最大値|2048px, 2048px|
        icon: ### アイコン設定

#### 記述形式
`[アイコン定義#1]|[アイコン定義#2]|…|[アイコン定義#n]`

#### 記述形式（アイコン定義）
`[アイコン種別コード],[経度の値],[緯度の値]`

#### 記述例
`1,139.7672,35.6811|102,139.7671,35.6799`

#### 範囲
|n（アイコン定義の件数）|値|
|---|---|
|最大|50|

|経度、緯度|値|
|---|---|
|最小|経度120, 緯度20|
|最大|経度155, 緯度46|

#### 注記
- scaler・centericonがonの場合、表示されるスケーラ・中心点アイコンもそれぞれアイコン定義として件数にカウントされます。
- アイコン種別コードの設定値については、付則. アイコン種別一覧を参照。
        centericon: ### 中心点アイコン表示
|表示設定|値|
|---|---|
|非表示|off|
|表示|on|

#### デフォルト
`off`
        logo: ### ブランドロゴの表示/非表示指定パラメータ

|表示設定|値|
|---|---|
|非表示|off|
|表示|on|

#### デフォルト
`off`

#### 注記
mapstyleパラメータの一部の値を指定した場合に本パラメータを on を指定すると、`[E03004] map parameter error (invalid combination)`が返却されます。
mapstyleと他パラメータの指定可否の組み合わせは、『API仕様書：MapFanAPI_地図デザイン設定方法と表示イメージ』をご確認ください。
        landmarkicon: ### ランドマークアイコンの表示/非表示指定パラメータ

|表示設定|値|
|---|---|
|非表示|off|
|表示|on|

#### デフォルト
`off`

#### 注記
mapstyleパラメータの一部の値を指定した場合に本パラメータを on を指定すると、`[E03004] map parameter error (invalid combination)`が返却されます。
mapstyleと他パラメータの指定可否の組み合わせは、『API仕様書：MapFanAPI_地図デザイン設定方法と表示イメージ』をご確認ください。
        format: ### 出力する画像形式
|画像形式|値|
|---|---|
|png|image/png|
|jpeg|image/jpeg|

#### デフォルト
`image/png`
        resolution: ### 解像度パラメータ
解像度に2を指定した場合、標準の2倍のサイズの画像が返されます。

|解像度|値|
|---|---|
|標準|1|
|2倍|2|

#### デフォルト
`1`
        rotationangle: ### 角度パラメータ
地図を指定した角度に傾けた画像が返されます。

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
        routeresultid: ### ルート結果ID
指定したルート結果IDに紐付くルートを表示します。
ルート結果IDはルート検索APIで取得します。
        routeicon: ### ルートアイコン種別
|設定|値|
|---|---|
|ルートアイコンを表示しない|0|
|標準のルートアイコンを表示する|1|

#### デフォルト
`1`
        scaler: ### スケーラ表示
|表示設定|値|
|---|---|
|非表示|off|
|表示|on|

#### デフォルト
`off`
        bldgname: ### ビル名の表示/非表示指定パラメータ

|表示設定|値|
|---|---|
|非表示|off|
|表示|on|

#### デフォルト
`off`

#### 注記
- mapstyleパラメータの一部の値を指定した場合に本パラメータを on を指定すると、`[E03004] map parameter error (invalid combination)`が返却されます。
mapstyleと他パラメータの指定可否の組み合わせは、『API仕様書：MapFanAPI_地図デザイン設定方法と表示イメージ』をご確認ください。
- ビル名称は、スケールが19以上の場合のみ描画されます。
        contour: ### 等高線の表示/非表示指定パラメータ

|表示設定|値|
|---|---|
|非表示|off|
|表示|on|

#### デフォルト
`on`

#### 注記
mapstyleパラメータの一部の値を指定した場合に本パラメータを on を指定すると、`[E03004] map parameter error (invalid combination)`が返却されます。
mapstyleと他パラメータの指定可否の組み合わせは、『API仕様書：MapFanAPI_地図デザイン設定方法と表示イメージ』をご確認ください。
        target: ### 地図の表示内容
|表示内容|値|
|---|---|
|背景のみを表示|1|
|注記のみを表示|2|
|背景 + 注記を表示|3|

#### デフォルト
`3`
        mapstyle: ### 地図デザイン指定用パラメータ

#### 記述形式
`[デザイン]_[注記サイズ]`

|デザイン|値|
|---|---|
|標準|std|
|グレースケール|gray|
|RPG風地図|rpg|
|古地図風地図|antique|
|道路強調地図|hlightroad|
|鉄道強調地図|hlightrail|
|印刷用白黒デザイン|monochrome|
|おもてなしマップ|tourism|

|注記サイズ|値|
|---|---|
|中|pc|
|大|sp|

#### デフォルト
`std_pc`

#### 注記
詳細は、『API仕様書：MapFanAPI_地図デザイン設定方法と表示イメージ』をご確認ください。
        
    """
    url = f"https://mapfanapi-map.p.rapidapi.com/mapimage"
    querystring = {'lonlat': lonlat, 'scale': scale, 'size': size, }
    if icon:
        querystring['icon'] = icon
    if centericon:
        querystring['centericon'] = centericon
    if logo:
        querystring['logo'] = logo
    if landmarkicon:
        querystring['landmarkicon'] = landmarkicon
    if format:
        querystring['format'] = format
    if resolution:
        querystring['resolution'] = resolution
    if rotationangle:
        querystring['rotationangle'] = rotationangle
    if routeresultid:
        querystring['routeresultid'] = routeresultid
    if routeicon:
        querystring['routeicon'] = routeicon
    if scaler:
        querystring['scaler'] = scaler
    if bldgname:
        querystring['bldgname'] = bldgname
    if contour:
        querystring['contour'] = contour
    if target:
        querystring['target'] = target
    if mapstyle:
        querystring['mapstyle'] = mapstyle
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapfanapi-map.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

