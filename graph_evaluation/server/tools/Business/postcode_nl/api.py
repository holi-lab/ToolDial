import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def zipcode(pc: str, mode: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Street and City based on zipcode"
    pc: Zipcode
        mode: Mode, xml or json otherwhise you get json in plain text
        
    """
    url = f"https://edwardbrosens-postcode-nl-v1.p.rapidapi.com/"
    querystring = {'pc': pc, }
    if mode:
        querystring['mode'] = mode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "edwardbrosens-postcode-nl-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

