import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def returnisocountry(countrystring: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return ISO country code from any short or long country name in more than 100 languages"
    countrystring: County short or long form from any language will return ISO code of the country
        
    """
    url = f"https://country-name-to-iso.p.rapidapi.com/ReturnISOCountry/{countrystring}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "country-name-to-iso.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

