import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_public_key_wif_address(new: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "generate private key (wif format) & address
		the wallet qr image will be availlable on: crypto.hirak.site/YOUR-WIF.png<br/>
		exemple:  crypto.hirak.site/L22FQFiLgL7HsYa6JDBizuLT6cBngjj5Wm8cuz3HP15zVGraSuCw.png.png"
    new: two possible parameters:
register: register new wif (generate new one)
exist: generate address from existing wif
        
    """
    url = f"https://bitcoin-address-generator.p.rapidapi.com/"
    querystring = {'new': new, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitcoin-address-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_qr_transaction(amount: str, address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "generate bitcoin qr transaction, the image will be availlable on: <br/>crypto.hirak.site/YOUR-ADDRESSAMOUNT.png<br/>
		exemple: crypto.hirak.site/1BtS4rSSi12RZcNY79q7hWEUSw2L5Ach5G0.0001.png"
    
    """
    url = f"https://bitcoin-address-generator.p.rapidapi.com/"
    querystring = {'amount': amount, 'address': address, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitcoin-address-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

