import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def railway_stations(order: str=None, search: str=None, format: str='json', radius: int=10, unit: str='km', lat: str='35', limit: int=10, key: str=None, lng: int=135, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "鉄道駅を検索して、所属する鉄道路線・鉄道事業者を含めたデータを返却します。"
    order: 検索データの取得ソート順。文字列としての数字、または特定のキーで判定を行う。
        search: ラベル検索に用いる文字列
        format: RESTレスポンスの返却データ形式の指定。 xml または jsonで返却
        radius: 緯度経度に対して検索する範囲
        unit: radiusの範囲 以下のいずれかを選択 "km: キロメートル (1000m) mi: 地理マイル (1853.7936m) yd: ヤード (0.9144m)"
        lat: 経度の指定
        limit: 返却数の指定
        key: APIを使用する際に使用するAPIキー
        lng: 緯度の指定
        
    """
    url = f"https://geoloco.p.rapidapi.com/v1/location/railway/stations"
    querystring = {}
    if order:
        querystring['order'] = order
    if search:
        querystring['search'] = search
    if format:
        querystring['format'] = format
    if radius:
        querystring['radius'] = radius
    if unit:
        querystring['unit'] = unit
    if lat:
        querystring['lat'] = lat
    if limit:
        querystring['limit'] = limit
    if key:
        querystring['key'] = key
    if lng:
        querystring['lng'] = lng
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geoloco.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

