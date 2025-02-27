import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1(time: str, filter1: str='phishing', filter2: str='url', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Version 1 Endpoint."
    time: Select 1 of these timeframes:

**today** - Today starting 00:00 UTC)
**week** - Last 7 days
**month** - Last 30 days
**year** - Last 365 days
        filter1: Can be an specific user, type or tag:

Type → **url** / **domain** / **ip** / **sha256** / **md5**
Tag → **phishing** / **ransomware** / **CobaltStrike** ...
User → **@malwrhunterteam** / **@1ZRR4H** / **@MBThreatIntel** / ... (don't forget the @)
        filter2: Can be an specific user, type or tag:

Type → **url** / **domain** / **ip** / **sha256** / **md5**
Tag → **phishing** / **ransomware** / **CobaltStrike** ...
User → **@malwrhunterteam** / **@1ZRR4H** / **@MBThreatIntel** / ... (don't forget the @)
        
    """
    url = f"https://tweetfeed.p.rapidapi.com/v1/{time}/{filter1}/{filter2}"
    querystring = {}
    if filter1:
        querystring['filter1'] = filter1
    if filter2:
        querystring['filter2'] = filter2
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tweetfeed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

