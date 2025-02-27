import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def bmi(weight: int, height: int, system: str='metric', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate BMI using weight and height."
    
    """
    url = f"https://bmi-calculator6.p.rapidapi.com/bmi"
    querystring = {'weight': weight, 'height': height, }
    if system:
        querystring['system'] = system
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bmi-calculator6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

