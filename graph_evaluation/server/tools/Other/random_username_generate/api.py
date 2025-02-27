import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def identity(minage: int, locale: str, maxage: int, domain: str='ugener.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Random Username Generate"
    
    """
    url = f"https://random-username-generate.p.rapidapi.com/"
    querystring = {'minAge': minage, 'locale': locale, 'maxAge': maxage, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-username-generate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

