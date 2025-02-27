import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ligue_1_standings(season: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ligue 1 Standings"
    season: e.g. `2022`
e.g. `2021`
        
    """
    url = f"https://ligue-1-standings.p.rapidapi.com/"
    querystring = {}
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ligue-1-standings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

