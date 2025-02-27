import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def basic_color(type: str='popular', color: str='f34573', random: bool=None, limit: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Basic Color**
		Return default 50 palettes of colors based on query.
		type: **popular** | **hot**
		random: **true** | **false**
		limit: **number**
		color:  return palette which included this color - color need to be in hex without '#'"
    
    """
    url = f"https://color-hunter.p.rapidapi.com/api/v1/color"
    querystring = {}
    if type:
        querystring['type'] = type
    if color:
        querystring['color'] = color
    if random:
        querystring['random'] = random
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "color-hunter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

