import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def insights(user: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Personality insights from Twitter"
    user: This is the UserName of the Twitter user you wish to analyze minus the @ symbol.
        
    """
    url = f"https://broadlistening-com-personality-insights-from-twitter-v1.p.rapidapi.com/blapi?user={user}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "broadlistening-com-personality-insights-from-twitter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

