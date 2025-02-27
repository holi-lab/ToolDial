import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(query: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search"
    query: Search query. You can use Google Search parameters.

e.g. adele site:youtube.com
e.g. harry potter filetype:pdf
e.g. inurl:store
e.g. ronaldo -cristiano
        page: Page Number
        
    """
    url = f"https://unlimited-google-web-search.p.rapidapi.com/"
    querystring = {'query': query, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unlimited-google-web-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

