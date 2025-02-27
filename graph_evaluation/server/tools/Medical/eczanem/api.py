import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def nobetci(ilce: str, il: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "il ve ilçe bilgisine göre nöbetçi eczaneleri görüntüleyin!"
    
    """
    url = f"https://eczanem.p.rapidapi.com/eczane"
    querystring = {'ilce': ilce, 'il': il, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eczanem.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

