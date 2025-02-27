import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_by_weight(input_weight: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<br /><br />
		<b>Authentication:</b> not required"
    
    """
    url = f"https://player-data.p.rapidapi.com/search_by_weight"
    querystring = {}
    if input_weight:
        querystring['input_weight'] = input_weight
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "player-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchplayer(player_input: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<br /><br />
		<b>Authentication:</b> not required"
    
    """
    url = f"https://player-data.p.rapidapi.com/SearchPlayer"
    querystring = {}
    if player_input:
        querystring['player_input'] = player_input
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "player-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

