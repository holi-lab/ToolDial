import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ccv_0(ccnumber: int, zip: int, expmonth: int, expyear: int, cvc: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    ccnumber: required
        zip: optional
        expmonth: required
        expyear: required
        cvc: required
        
    """
    url = f"https://bin-credit-card-checker.p.rapidapi.com/card-auth/check"
    querystring = {'ccNumber': ccnumber, 'zip': zip, 'expMonth': expmonth, 'expYear': expyear, 'cvc': cvc, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bin-credit-card-checker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ccv_1(zip: int, expyear: int, ccnumber: int, cvc: int, expmonth: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    zip: optional
        expyear: required
        ccnumber: required
        cvc: required
        expmonth: required
        
    """
    url = f"https://bin-credit-card-checker.p.rapidapi.com/card-charge/check"
    querystring = {'zip': zip, 'expYear': expyear, 'ccNumber': ccnumber, 'cvc': cvc, 'expMonth': expmonth, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bin-credit-card-checker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ccn(ccnumber: int, zip: int, expmonth: int, expyear: int, cvc: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    ccnumber: required
        zip: optional
        expmonth: required
        expyear: required
        cvc: required
        
    """
    url = f"https://bin-credit-card-checker.p.rapidapi.com/card-ccn/check"
    querystring = {'ccNumber': ccnumber, 'zip': zip, 'expMonth': expmonth, 'expYear': expyear, 'cvc': cvc, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bin-credit-card-checker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

