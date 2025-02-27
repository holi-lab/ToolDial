import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_search_results(searchurl: str, page: str, docids: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetches the search results from given page"
    
    """
    url = f"https://justdial-scraper.p.rapidapi.com/fetchSearchResults"
    querystring = {'searchUrl': searchurl, 'page': page, }
    if docids:
        querystring['docIds'] = docids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "justdial-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

