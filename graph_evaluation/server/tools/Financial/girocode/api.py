import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate(information: str, iban: str, currency: str, amount: int, name: str, bic: str=None, dimensions: int=256, reference: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    information: What is the reason for this transaction?
        iban: The IBAN of the payee
        currency: The currency
        amount: The amount
        name: The payee
        bic: The BIC of the payee
        dimensions: The desired width and height of the generated qr code in px
        reference: ISO 11649 RF Creditor Reference
        
    """
    url = f"https://girocode2.p.rapidapi.com/generate"
    querystring = {'information': information, 'iban': iban, 'currency': currency, 'amount': amount, 'name': name, }
    if bic:
        querystring['bic'] = bic
    if dimensions:
        querystring['dimensions'] = dimensions
    if reference:
        querystring['reference'] = reference
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "girocode2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

