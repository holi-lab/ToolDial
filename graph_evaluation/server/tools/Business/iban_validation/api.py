import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def validation_endpoint(iban: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Validating IBAN's is a critical step to reducing costly payment and other financially-related errors."
    iban: The IBAN to validate. Note that the API will accept white spaces, so BE71 0961 2345 6769 is considered as valid as BE71096123456769.
        
    """
    url = f"https://iban-validation2.p.rapidapi.com/v1"
    querystring = {'iban': iban, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iban-validation2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

