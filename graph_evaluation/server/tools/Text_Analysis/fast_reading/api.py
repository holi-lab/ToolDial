import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def go(txt: str, fix: int=5, lng: str='en', model: str='m1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert raw text to Bionized Reading."
    txt: Raw text to be converted
        fix: Word fixation
        lng: Language: EN, RU, ES
        model: Model For Text Processing: m1,m2,...m11
        
    """
    url = f"https://fast-reading.p.rapidapi.com/api/"
    querystring = {'txt': txt, }
    if fix:
        querystring['fix'] = fix
    if lng:
        querystring['lng'] = lng
    if model:
        querystring['model'] = model
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fast-reading.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

