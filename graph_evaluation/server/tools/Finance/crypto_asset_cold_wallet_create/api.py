import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getactivecurrencylist(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Current Active Currency List REST API allows clients to retrieve a list of currently active currencies by sending a GET request to the endpoint. The endpoint should be in the following format:"
    
    """
    url = f"https://crypto-asset-cold-wallet-create.p.rapidapi.com/api/v1/currency"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-asset-cold-wallet-create.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

