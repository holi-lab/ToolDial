import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def domain_and_keyword(domain: str, keyword: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It accepts website domain name and keyword."
    
    """
    url = f"https://keyword-rank-checker-in-google-yahoo-bing.p.rapidapi.com/data"
    querystring = {'domain': domain, 'keyword': keyword, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "keyword-rank-checker-in-google-yahoo-bing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

