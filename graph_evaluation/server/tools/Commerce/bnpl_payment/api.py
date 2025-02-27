import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_bnpl_customer_details_by_loyalty_card_number(loycardnumber: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get BNPL Customer Details by Loyalty Card Number"
    
    """
    url = f"https://bnpl-payment2.p.rapidapi.com/v1/bnpl-gateway/customer/loyalty/{loycardnumber}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bnpl-payment2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_bnpl_customer_details_by_umico_master_phone_number(phone_number: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get BNPL Customer Details by Umico Master Phone Number"
    
    """
    url = f"https://bnpl-payment2.p.rapidapi.com/v1/bnpl-gateway/customer/{phone_number}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bnpl-payment2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

