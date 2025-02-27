import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def usa_routing_numbers_validation(rtn: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Validate USA Routing Numbers in order to avoid transactions failure. In USA Routing Numbers are required for domestic  payment transactions& international payment receipts."
    
    """
    url = f"https://usa-routing-number-validation.p.rapidapi.com/"
    querystring = {'rtn': rtn, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "usa-routing-number-validation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

