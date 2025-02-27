import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v2_api_coords_trial(airport: str='NRT', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "天候情報に基づくファッションレコメンデーション情報"
    airport: 空港のコードを指定します(指定可能なコードは概要欄を御覧ください)
        
    """
    url = f"https://tnql-coords-trial-v2.p.rapidapi.com/v2/api/coords_trial"
    querystring = {}
    if airport:
        querystring['airport'] = airport
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tnql-coords-trial-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

