import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_caloriesburned(activity: str, weight: int=None, duration: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Calories Burned API endpoint."
    activity: name of the given activity. This value can be partial (e.g. ski will match water skiing and downhill skiing)
        weight: weight of the user performing the activity in pounds. Must be between 50 and 500. Default value is 160.
        duration: how long the activity was performed in minutes. Must be 1 or greater. Default value is 60 (1 hour).
        
    """
    url = f"https://calories-burned-by-api-ninjas.p.rapidapi.com/v1/caloriesburned"
    querystring = {'activity': activity, }
    if weight:
        querystring['weight'] = weight
    if duration:
        querystring['duration'] = duration
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "calories-burned-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

