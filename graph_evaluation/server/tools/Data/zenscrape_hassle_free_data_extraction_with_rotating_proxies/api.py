import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get(url: str, location: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetches HTML from a URL"
    url: The URL to fetch
        
    """
    url = f"https://zenscrape-hassle-free-data-extraction-with-rotating-proxies.p.rapidapi.com/get"
    querystring = {'url': url, }
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zenscrape-hassle-free-data-extraction-with-rotating-proxies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

