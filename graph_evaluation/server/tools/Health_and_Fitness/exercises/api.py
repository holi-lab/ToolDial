import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def exercises(count: int=None, equipmentused: str=None, muscletarget: str=None, bodypart: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access thousands of exercises with individual exercise data and animated demonstrations."
    count: Min 1, Max 100
        
    """
    url = f"https://exercises2.p.rapidapi.com/"
    querystring = {}
    if count:
        querystring['count'] = count
    if equipmentused:
        querystring['equipmentUsed'] = equipmentused
    if muscletarget:
        querystring['muscleTarget'] = muscletarget
    if bodypart:
        querystring['bodyPart'] = bodypart
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "exercises2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

