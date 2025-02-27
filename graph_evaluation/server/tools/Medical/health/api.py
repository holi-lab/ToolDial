import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def bmi(height: int, weight: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the physical state of the body and the number of total muscle mass."
    height: height of a person. Accept meters or feet.
        weight: Weight of a person. Accept kilos or pounds.
        
    """
    url = f"https://gabamnml-health-v1.p.rapidapi.com/bmi"
    querystring = {'height': height, 'weight': weight, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gabamnml-health-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

