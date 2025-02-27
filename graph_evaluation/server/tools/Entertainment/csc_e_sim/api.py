import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def battle(server: str, battle_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It will shows the battle statistic of given ID"
    server: e-Sim server
        battle_id: It's clear.
        
    """
    url = f"https://lrdarc-unofficial-e-sim-csc.p.rapidapi.com/<server>/battle/<battle_id>-<round_id>/.<format>"
    querystring = {'Server': server, 'Battle ID': battle_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lrdarc-unofficial-e-sim-csc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

