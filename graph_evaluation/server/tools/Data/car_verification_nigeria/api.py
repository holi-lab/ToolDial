import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def verify_plate_number_location(platecodeloc: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter the first 3-digit of any Nigeria plate number to know the state and LGA it comes from. For example, if a plate number is GGE123YZ. you will pass the parameter as GGE or gge (not case sensitive) and the state, LGA and state slogan will be displayed including its unique id. Feel free to try it out."
    
    """
    url = f"https://car-verification-nigeria.p.rapidapi.com/platenumberloc.php"
    querystring = {'platecodeloc': platecodeloc, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-verification-nigeria.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

