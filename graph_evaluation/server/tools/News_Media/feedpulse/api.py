import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_feeds(per_page: int=None, country: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    per_page: Number of items per page
        country: Country code
        page: Page number
        
    """
    url = f"https://feedpulse2.p.rapidapi.com/api/feeds"
    querystring = {}
    if per_page:
        querystring['per_page'] = per_page
    if country:
        querystring['country'] = country
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "feedpulse2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

