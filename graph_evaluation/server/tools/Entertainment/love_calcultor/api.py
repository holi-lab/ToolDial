import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def calculate_percentage(sname: str, fname: str, content_type: str='application/json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "calculate Percentage"
    
    """
    url = f"https://love-calcultor.p.rapidapi.com/api/loveCalculator"
    querystring = {'sname': sname, 'fname': fname, }
    if content_type:
        querystring['Content-Type'] = content_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "love-calcultor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

