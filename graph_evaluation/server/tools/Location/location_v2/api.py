import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_result(q: str, latitude: str='24.562394', longitude: str='73.7124675', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "We are providing a location search result according to your search query,1"
    
    """
    url = f"https://location52.p.rapidapi.com/index.php"
    querystring = {'q': q, }
    if latitude:
        querystring['latitude'] = latitude
    if longitude:
        querystring['longitude'] = longitude
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location52.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

