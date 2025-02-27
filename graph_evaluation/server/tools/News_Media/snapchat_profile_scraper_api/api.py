import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def profile_api(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Send a request to `https://snapchat-profile-scraper-api.p.rapidapi.com/profile/?username=YOUR_SNAPCHAT_USERNAME` with your correct authentication headers. Nothing more! Quick & Easy"
    
    """
    url = f"https://snapchat-profile-scraper-api.p.rapidapi.com/profile"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snapchat-profile-scraper-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

