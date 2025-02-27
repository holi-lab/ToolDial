import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def best_kirby_and_the_forgotten_land_bosses(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Best Kirby and the Forgotten Land Bosses"
    
    """
    url = f"https://kirby-forgotten-land-bosses1.p.rapidapi.com/kOBgcu/kirbyforgottenlandbosses"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kirby-forgotten-land-bosses1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

