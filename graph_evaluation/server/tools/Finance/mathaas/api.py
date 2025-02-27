import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getdivision(divisor: int, dividend: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Divides two numbers (dividend and divisor) provided as parameters"
    divisor: The second term in a division: <dividend> / <divisor>
        dividend: The first term in a division: <dividend> / <divisor>
        
    """
    url = f"https://mathaas.p.rapidapi.com/divide"
    querystring = {'divisor': divisor, 'dividend': dividend, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mathaas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

