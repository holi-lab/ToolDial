import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(applicationid: str='1068023759784411308', genreid: int=559887, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Items"
    
    """
    url = f"https://rakutensupportdefaultteam.p.rapidapi.com/Search/20140222"
    querystring = {}
    if applicationid:
        querystring['applicationId'] = applicationid
    if genreid:
        querystring['genreId'] = genreid
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rakutensupportdefaultteam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

