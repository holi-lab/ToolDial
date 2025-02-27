import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def balance(wallet: str, blockchain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get crypto wallet balance, supported blockchains:     algorand,    bitcoin,    bsc,    celo,    ethereum,    flow,    one,    klaytn,    kcs,    litecoin,    polygon,    solana,    vet,    xdc,
		Send GET request to https://check-crypto-balance-daddy.p.rapidapi.com/[blockchain]/[wallet]
		for example :
		https://check-crypto-balance-daddy.p.rapidapi.com/ethereum/0xc1e42f862d202b4a0ed552c1145735ee088f6ccf"
    
    """
    url = f"https://check-crypto-balance-daddy.p.rapidapi.com/{blockchain}/{wallet}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "check-crypto-balance-daddy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

