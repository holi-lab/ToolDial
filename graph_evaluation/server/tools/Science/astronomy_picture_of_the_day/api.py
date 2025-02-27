import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def picture_of_the_day(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Endpoint returns an astronomy image of the univers with caption and explanation from professionals."
    
    """
    url = f"https://astronomy-picture-of-the-day.p.rapidapi.com/apod?api_key=nWYhQQdmCKwd0cVvrfyge124OrW4fnVOEL7QDdJH"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astronomy-picture-of-the-day.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

