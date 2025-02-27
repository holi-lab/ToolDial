import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def validate_beneficiary_details(nickname: str, acct_ccy: str, beneficiary_name: str, destination_country_code: str, token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Validates the bank details you provide, which needs to be all the details required for the relevant country to be able to create a beneficiary and make a payment."
    
    """
    url = f"https://community-the-currency-cloud.p.rapidapi.com/{token}/beneficiary/validate_details"
    querystring = {'nickname': nickname, 'acct_ccy': acct_ccy, 'beneficiary_name': beneficiary_name, 'destination_country_code': destination_country_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-the-currency-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def beneficiaries_required_fields(ccy: str, destination_country_code: str, token: str, payment_type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a set of required fields which should be provided when a bank account is created."
    payment_type: One of - 'standard' (default), 'local'
        
    """
    url = f"https://community-the-currency-cloud.p.rapidapi.com/{token}/beneficiaries/required_fields"
    querystring = {'ccy': ccy, 'destination_country_code': destination_country_code, }
    if payment_type:
        querystring['payment_type'] = payment_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-the-currency-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

