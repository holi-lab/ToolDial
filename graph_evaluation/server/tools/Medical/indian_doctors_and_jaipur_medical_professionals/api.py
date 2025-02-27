import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def doctor_pages(no: int, get_1: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint return doctors details by page no."
    get_1: This end point return the doctors details of page 1
        
    """
    url = f"https://indian-doctors-and-jaipur-medical-professionals.p.rapidapi.com/dentist/{no}"
    querystring = {}
    if get_1:
        querystring['1'] = get_1
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indian-doctors-and-jaipur-medical-professionals.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

