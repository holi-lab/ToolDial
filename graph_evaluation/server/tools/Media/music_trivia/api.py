import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getgamelevel(level: int=None, output: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get a game level with a level parameter. The level this does not imply difficulty"
    level: paginated level
        output: set output format type , 'xml' or 'json'
        
    """
    url = f"https://roomtek-music-trivia-v1.p.rapidapi.com/getgamelevel"
    querystring = {}
    if level:
        querystring['level'] = level
    if output:
        querystring['output'] = output
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "roomtek-music-trivia-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

