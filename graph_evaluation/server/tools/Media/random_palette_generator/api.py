import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_palettes_using_random_mode(paletteno: str, colorno: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Palettes Using random Scheme, defined Number of Palettes and defined Number of Colors Per Palettes"
    
    """
    url = f"https://random-palette-generator.p.rapidapi.com/palette/{paletteno}/{colorno}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-palette-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_full(type: str, paletteno: str, colorno: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Palettes Using Scheme, Number of Palettes and Number of Colors Per Palettes"
    
    """
    url = f"https://random-palette-generator.p.rapidapi.com/palette/{type}/{paletteno}/{colorno}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-palette-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

