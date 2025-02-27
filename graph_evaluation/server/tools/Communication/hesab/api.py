import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def hesab_yarat(mail: str, password: str, name: str, birthday: int=23, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Yoxdur"
    
    """
    url = f"https://hesab.p.rapidapi.com/user.ru"
    querystring = {'Mail': mail, 'Password': password, 'Name': name, }
    if birthday:
        querystring['Birthday'] = birthday
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hesab.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

