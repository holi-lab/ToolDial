import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def bangladesh_routing_numbers_validation(rtn: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Validates Bangladesh Routing Numbers by the method prescribed for verifying the Bangladesh Routing Numbers. This API helps in ensuring accuracy of Bangladesh Routing Numbers."
    
    """
    url = f"https://bangladesh-routing-numbers-validation.p.rapidapi.com/"
    querystring = {'rtn': rtn, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bangladesh-routing-numbers-validation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

