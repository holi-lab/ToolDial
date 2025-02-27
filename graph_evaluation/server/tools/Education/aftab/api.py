import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def aftavb(aftab: str='3333', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "dsafadasd"
    aftab: dsdasdas
        aftab: dsdasdas
        
    """
    url = f"https://aftab.p.rapidapi.comhttp://q-study.com/"
    querystring = {}
    if aftab:
        querystring['aftab'] = aftab
    if aftab:
        querystring['aftab'] = aftab
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aftab.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

