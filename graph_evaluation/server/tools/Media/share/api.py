import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_share(service: str, link: str, apitype: int, v: int, shortener: str='google', title: str='TechCrunch', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Share API"
    service: Destination Service (lookup codes @ shareapi.com)
        link: Link to the shared
        apitype: apitype=1 (1=Redirect, 2=Pingback, 3=1x1 pixel)
        v: API Version
        shortener: URL shortener
        title: Title
        
    """
    url = f"https://shareaholic-share-v1.p.rapidapi.com/api/share/"
    querystring = {'service': service, 'link': link, 'apitype': apitype, 'v': v, }
    if shortener:
        querystring['shortener'] = shortener
    if title:
        querystring['title'] = title
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shareaholic-share-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

