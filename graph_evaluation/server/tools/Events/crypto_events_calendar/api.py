import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def events(pagesize: str, page: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get crypto events like listing, airdrops, release, tokenomics, partnershiop, other.
		Grouped by days!"
    pagesize: Max pageSize = 10
        
    """
    url = f"https://crypto-events-calendar.p.rapidapi.com/index"
    querystring = {'pageSize': pagesize, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-events-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

