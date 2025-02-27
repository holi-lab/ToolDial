import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def google_serp(is_id: str, keyword: str, site: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the top 100 Google serp results for any keyword including domain and page authority."
    
    """
    url = f"https://live-serpapi.p.rapidapi.com/api.php"
    querystring = {'id': is_id, 'keyword': keyword, 'site': site, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-serpapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

