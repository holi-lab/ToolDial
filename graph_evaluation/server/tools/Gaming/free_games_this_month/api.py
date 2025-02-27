import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_recent_gift_games(period: str='7d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It returns the same of `available` endpoint but it will trim a period date. This is useful, for example, to create alerts and notifications with ONLY recent data."
    period: \\\"1d\\\" | \\\"3d\\\" | \\\"7d\\\" | \\\"15d\\\" | \\\"30d\\\"
Default: \\\"7d\\\"
        
    """
    url = f"https://free-games-this-month.p.rapidapi.com/games/gifts/recent"
    querystring = {}
    if period:
        querystring['period'] = period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "free-games-this-month.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_available_gift_games(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Full list of current available free games."
    
    """
    url = f"https://free-games-this-month.p.rapidapi.com/games/gifts/available"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "free-games-this-month.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

