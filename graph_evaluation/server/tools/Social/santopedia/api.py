import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def names(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the saint related to the specified name"
    name: Separated by commas, the names you want to search for. No default: this field is mandatory
        
    """
    url = f"https://santopedia.p.rapidapi.com/names/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "santopedia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def saints_of_the_day(day: str='07-31', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It returns the saints of a given day"
    day: Optional (if not included, TODAY is default). The date you want to know the saints from (format MM-DD). If you only want to recieve 1-3 relevant saints, add ",important" after the MM-DD date, like 07-31,important
        
    """
    url = f"https://santopedia.p.rapidapi.com/days/{day}"
    querystring = {}
    if day:
        querystring['day'] = day
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "santopedia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

