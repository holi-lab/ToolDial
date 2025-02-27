import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_global_number_info(intlnumber: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides global telephone number demographic data for the given telephone number"
    intlnumber: Global Telephone Number with Country Code
        
    """
    url = f"https://global-telephone-number-information.p.rapidapi.com/getglobalnumberinfo"
    querystring = {'intlnumber': intlnumber, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-telephone-number-information.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

