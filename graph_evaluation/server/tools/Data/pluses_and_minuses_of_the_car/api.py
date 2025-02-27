import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def pluses_and_minuses_of_the_car(year: str, make: str, model: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pluses and minuses of the car"
    
    """
    url = f"https://pluses-and-minuses-of-the-car.p.rapidapi.com/b2b/plusminus"
    querystring = {'year': year, 'make': make, 'model': model, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pluses-and-minuses-of-the-car.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

