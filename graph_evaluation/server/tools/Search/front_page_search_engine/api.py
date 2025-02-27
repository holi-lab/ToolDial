import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(pageno: int, country: str, lang: str, search: str, perpage: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "takes any search string in any language."
    
    """
    url = f"https://front-page-search-engine.p.rapidapi.com/search.php"
    querystring = {'pageno': pageno, 'country': country, 'lang': lang, 'search': search, }
    if perpage:
        querystring['perpage'] = perpage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "front-page-search-engine.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

