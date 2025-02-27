import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def canada_routing_number(rtn: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Canada Routing Number is of 9 digits.  Pass the value of the rtn and get the Routing Number Information"
    rtn: Routing Number
        
    """
    url = f"https://canada-routing-numbers-varification-api.p.rapidapi.com/"
    querystring = {'rtn': rtn, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "canada-routing-numbers-varification-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

