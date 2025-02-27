import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def businesssearch(yplocation: str, yppage: int, ypkeyword: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "you can search any business at any city or state"
    
    """
    url = f"https://yellow-page-us.p.rapidapi.com/"
    querystring = {'yplocation': yplocation, 'yppage': yppage, 'ypkeyword': ypkeyword, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yellow-page-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

