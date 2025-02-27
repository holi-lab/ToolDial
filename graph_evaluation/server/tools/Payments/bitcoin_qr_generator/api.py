import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_qr_photo(amount: str, address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate qr photo for your bitcoin transactions, there are two required parameters: address & amount, the photo will be availlabe on: crypto.hirak.site/btc/ADDRESSAMOUNT.png<br/>
		exemple: crypto.hirak.site/1BtS4rSSi12RZcNY79q7hWEUSw2L5Ach5G0.0001.png"
    
    """
    url = f"https://bitcoin-qr-generator.p.rapidapi.com/"
    querystring = {'amount': amount, 'address': address, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitcoin-qr-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

