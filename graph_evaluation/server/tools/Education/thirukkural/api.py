import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def kural_number_appid_appid_format_format(number: str, appid: str, format: str='xml', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Thirukkural. Valid {number} values: 1) integer between 1 to 1330 2) 'rnd' 3) 10-20 (range)"
    number: Valid values: 1) integer between 1 to 1330 2) rnd 3) 10-20 (range)
        appid: API key
        format: Return format JSON or XML
        
    """
    url = f"https://thirukkural.p.rapidapi.com/kural/rnd"
    querystring = {'number': number, 'appid': appid, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thirukkural.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

