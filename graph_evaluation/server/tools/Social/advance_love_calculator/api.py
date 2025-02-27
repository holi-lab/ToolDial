import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def calulatelove(content_type: str, secondname: str, firstname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "CalulateLove"
    
    """
    url = f"https://advance-love-calculator.p.rapidapi.com/CalulateLove"
    querystring = {'Content-Type': content_type, 'secondName': secondname, 'firstName': firstname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "advance-love-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

