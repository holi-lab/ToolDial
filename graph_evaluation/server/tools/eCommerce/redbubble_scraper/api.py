import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def redbubble(page: str, category: str='all-departments', search_term: str='orange cat', sort_order: str='relevant', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get items per page, you can optionally specify a search term, specific categories or ordering"
    
    """
    url = f"https://redbubble-scraper.p.rapidapi.com/redbubble"
    querystring = {'page': page, }
    if category:
        querystring['category'] = category
    if search_term:
        querystring['search_term'] = search_term
    if sort_order:
        querystring['sort_order'] = sort_order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "redbubble-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

