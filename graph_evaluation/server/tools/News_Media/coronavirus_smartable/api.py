import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getnews(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest coronavirus news for the globe or your selected country or state."
    location: ISO 3166-2 location code.  For example, use "FR" for coronavirus news in France; use "US" for coronavirus news in US;  use "US-CA" for coronavirus news in the US California state.  Use "global" to retrieve global news. 
        
    """
    url = f"https://coronavirus-smartable.p.rapidapi.com/news/v1/{location}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coronavirus-smartable.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstats(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest and historic coronavirus stats data (confirmed cases, deaths and recovered cases) for the globe or your selected country or state."
    location: ISO 3166-2 location code.  For example, use "FR" for coronavirus stats in France; use "US" for coronavirus stats in US;  use "US-CA" for coronavirus stats in the US California state.  Use "global" to retrieve global stats. 
        
    """
    url = f"https://coronavirus-smartable.p.rapidapi.com/stats/v1/{location}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coronavirus-smartable.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

