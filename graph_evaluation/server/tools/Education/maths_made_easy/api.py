import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def arithmatic(num1: str, num2: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the result of all arithmetic operation between two integers."
    num1: Any integer value
        
    """
    url = f"https://maths-made-easy.p.rapidapi.com/arithmatic"
    querystring = {'num1': num1, 'num2': num2, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maths-made-easy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

