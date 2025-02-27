import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getmetalprice(metal: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the current price for a precious metals such as gold, silver, platinum, palladium, rhodium, and iridium."
    metal: The metals corresponding integer. (Gold = 0, Silver = 1, Platinum = 3, Palladium = 4)
        
    """
    url = f"https://current-precious-metal-price.p.rapidapi.com/metals/v1/{metal}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "current-precious-metal-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

