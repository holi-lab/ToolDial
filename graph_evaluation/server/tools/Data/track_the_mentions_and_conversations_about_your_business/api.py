import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getmentions(query: str, page: int=1, period: int=14, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find your brand, competitor, or any other query mentions across the web"
    
    """
    url = f"https://track-the-mentions-and-conversations-about-your-business.p.rapidapi.com/"
    querystring = {'query': query, }
    if page:
        querystring['page'] = page
    if period:
        querystring['period'] = period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "track-the-mentions-and-conversations-about-your-business.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

