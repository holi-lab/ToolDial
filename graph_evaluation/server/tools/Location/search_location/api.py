import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def index_php(kecamatan: int=1, kelurahan: int=2001, provinsi: int=35, kabupatenkota: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://daneswara-search-location-v1.p.rapidapi.com/index.php"
    querystring = {}
    if kecamatan:
        querystring['kecamatan'] = kecamatan
    if kelurahan:
        querystring['kelurahan'] = kelurahan
    if provinsi:
        querystring['provinsi'] = provinsi
    if kabupatenkota:
        querystring['kabupatenkota'] = kabupatenkota
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "daneswara-search-location-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

