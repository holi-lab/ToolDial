import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def similar_sites(domain: str, key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find similar sites for a given domain"
    domain: The domain to find similar sites for
        key: your api key
        
    """
    url = f"https://webpageanalyse-webpageanalyse.p.rapidapi.com/simsites"
    querystring = {'domain': domain, 'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webpageanalyse-webpageanalyse.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

