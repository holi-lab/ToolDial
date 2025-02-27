import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_result(q: str, lang: str='en', country: str='in', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "we provide search results in real-time according to your search query."
    
    """
    url = f"https://search-engine5.p.rapidapi.com/index.php"
    querystring = {'q': q, }
    if lang:
        querystring['lang'] = lang
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "search-engine5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

