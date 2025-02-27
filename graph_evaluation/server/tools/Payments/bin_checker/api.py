import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def card_check(expyear: str, cvc: str, ccnumber: str, expmonth: str, zip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "CCN, CCV 0$, CCV 1$"
    
    """
    url = f"https://bin-checker3.p.rapidapi.com/card-auth/check"
    querystring = {'expYear': expyear, 'cvc': cvc, 'ccNumber': ccnumber, 'expMonth': expmonth, 'zip': zip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bin-checker3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

