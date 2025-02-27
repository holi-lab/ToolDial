import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tag_counter_endpoint(page: str='https://myanimelist.net/anime/genre/1/Action', is_id: str='a[class="link-title"]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/"
    
    """
    url = f"https://any-web-scraping-data.p.rapidapi.com/tag-counter/result/api.php"
    querystring = {}
    if page:
        querystring['page'] = page
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "any-web-scraping-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

