import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def receive_the_credit_card_data(visa_type: str='visa', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive the requested credit card data set consisting of:
		
		- type
		- firstname
		- lastname
		- cc
		- valid_date
		- cvc
		
		**The following providers are valid and can be generated**
		***amex,diners,discover',jcb',jcb15,jcb16,maestro,mastercard,visa,visa13,visa16,visa19***"
    
    """
    url = f"https://fake-valid-cc-data-generator.p.rapidapi.com/request/"
    querystring = {}
    if visa_type:
        querystring['visa_type'] = visa_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-valid-cc-data-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

