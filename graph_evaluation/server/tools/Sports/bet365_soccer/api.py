import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def inplay(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get in play events for soccer"
    
    """
    url = f"https://bet365-soccer.p.rapidapi.com/v1/soccer/bet365/inplay"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bet365-soccer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def implay_odds(uid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Match Result 1x2, Double Chance, Both Teams to Score, Draw No Bet, Europien Handicap, Asian handicap, Total Goals - Over/Under,  Corect score,  Half Time/Full Time"
    
    """
    url = f"https://bet365-soccer.p.rapidapi.com/v1/soccer/bet365/inplayodds"
    querystring = {'uid': uid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bet365-soccer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

