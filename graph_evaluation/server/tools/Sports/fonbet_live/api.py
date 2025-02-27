import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_games_odds(sportname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get games odds"
    sportname: available sports: Basketball, Football, IceHockey, Tennis

Event->period:
**Main** - Without overtime
**MainFull** - With overtime

        
    """
    url = f"https://fonbet-live.p.rapidapi.com/odds/{sportname}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fonbet-live.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

