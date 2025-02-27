import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def create(blockchain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Create Crypto Wallet
		Simply get request to the desired chain 
		EX: https://crypto-wallet-create-daddy.p.rapidapi.com/bitcoin"
    
    """
    url = f"https://crypto-wallet-create-daddy.p.rapidapi.com/{blockchain}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-wallet-create-daddy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

