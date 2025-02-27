import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def way2mean_goals(under: int, over: int, way2: int, wayx: int, way1: int, spec: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return mean number of goals for home and away teams. The input is 3way and under/over odds. They could with or without margin."
    under: Odds that total number of goals less than "spec"
        over: Odds that total number of goals more than "spec"
        way2: Odds for away team to win
        wayx: Odds for home team to draw
        way1: Odds for home team to win
        spec: Threshold for total goals in under/over odds. Could be only 0.5, 1.5, 2.5, 3.5, 4.5
        
    """
    url = f"https://football-odds.p.rapidapi.com/way2mu"
    querystring = {'under': under, 'over': over, 'way2': way2, 'wayx': wayx, 'way1': way1, 'spec': spec, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

