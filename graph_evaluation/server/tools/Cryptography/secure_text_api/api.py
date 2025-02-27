import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getkey(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will generated publickey and privatekey.
		It response **msgCode** as six character string.
		
		Client can use this code to verify owner of key, 
		when calling encrypt and decrypt endpoint.
		
		*** After decrypt ciphertext this code will inactive, 
		client need to call getKey again to get new code"
    
    """
    url = f"https://secure-text-api.p.rapidapi.com/getKey"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "secure-text-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

