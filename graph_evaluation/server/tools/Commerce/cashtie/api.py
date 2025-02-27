import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def check_an_account_status_and_balance(account_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Any time your application needs to check the Account status, balance or related payments, the following request should be made:"
    
    """
    url = f"https://community-cashtie.p.rapidapi.com/accounts/{account_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-cashtie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_merchants(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "In order to let your consumers make a payment for a particular Account, your system should be able to provide information about stores where payments are accepted. Cashtie™ API provides this information via Merchant resource:"
    
    """
    url = f"https://community-cashtie.p.rapidapi.com/merchants"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-cashtie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check_amount_of_payment(account_id: str, payment_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Once you receive a webhook notification that a payment is made, your application should send a request to the Cashtie™ API to check the amount of payment:"
    
    """
    url = f"https://community-cashtie.p.rapidapi.com/accounts/{account_id}/payments/{payment_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-cashtie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

