import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v2_api_coords(lat: int=None, airport: str='NRT', lon: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "天候情報に基づくファッションレコメンデーション情報"
    lat: 緯度
        airport: 空港のコードを指定します。(指定可能なコードは概要欄を御覧ください)
        lon: 経度
        
    """
    url = f"https://tnql-coords-v2.p.rapidapi.com/v2/api/coords"
    querystring = {}
    if lat:
        querystring['lat'] = lat
    if airport:
        querystring['airport'] = airport
    if lon:
        querystring['lon'] = lon
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tnql-coords-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

