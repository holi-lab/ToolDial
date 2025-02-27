import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def home(iso: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Simply pass an iso query parameter with the alpha-3 ISO country code of your choice to receive data for that country
		
		[List of ISO codes](https://www.iban.com/country-codes)
		
		example:  ?iso=USA"
    
    """
    url = f"https://covid-19-world-vaccination-data.p.rapidapi.com/"
    querystring = {'iso': iso, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-world-vaccination-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

