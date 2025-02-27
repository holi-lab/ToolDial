import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def myanimelist_endpoint(is_id: str, page: str='https://myanimelist.net/anime/16498', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/"
    
    """
    url = f"https://myanimelist-unofficial-api-get-tittle-description-and-genre.p.rapidapi.com/tag-counter/result/api.php"
    querystring = {'id': is_id, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "myanimelist-unofficial-api-get-tittle-description-and-genre.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

