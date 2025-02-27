import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_fx_rates_for_a_bank(bank_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this Endpoint to get list of currencies and rates for a certain bank"
    bank_code: Bank Codes abbreviations of the bank name for a full list go to :  https://gist.github.com/amrfarid140/bd65a40f101cda713716930135b82304
        
    """
    url = f"https://amrfarid140-egypt-fx-v1.p.rapidapi.com/banks/{bank_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amrfarid140-egypt-fx-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_currency_rates_in_a_specific_bank(bank_code: str, currency_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Buy and Sell rates for a single currency for a specific bank"
    bank_code: Bank Codes abbreviations of the bank name for a full list go to :  https://gist.github.com/amrfarid140/bd65a40f101cda713716930135b82304
        currency_code: Example for Currency codes are : USD,EGP,EUR and so on
        
    """
    url = f"https://amrfarid140-egypt-fx-v1.p.rapidapi.com/banks/{bank_code}/{currency_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amrfarid140-egypt-fx-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

