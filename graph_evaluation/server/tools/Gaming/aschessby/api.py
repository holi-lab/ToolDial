import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def position(fen: str, level: int=20, depth: int=15, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Send a FEN position and receive the best move"
    
    """
    url = f"https://aschessby.p.rapidapi.com/api/position"
    querystring = {'fen': fen, }
    if level:
        querystring['level'] = level
    if depth:
        querystring['depth'] = depth
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aschessby.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

