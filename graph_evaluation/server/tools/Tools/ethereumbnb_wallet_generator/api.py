import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def eth_bnb_wallet_generator(generate_wallet: str='/generate-wallet', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "this endpoint takes a 'GET' request with "/generate-wallet" parameter and returns an Address and the respective private key"
    
    """
    url = f"https://ethereum-bnb-wallet-generator.p.rapidapi.com/generate-wallet"
    querystring = {}
    if generate_wallet:
        querystring['generate-wallet'] = generate_wallet
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ethereum-bnb-wallet-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

