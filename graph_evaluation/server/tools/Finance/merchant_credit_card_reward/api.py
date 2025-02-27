import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def merchant_credit_card_reward_lookup(query: str, country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is an API designed to retrieve comprehensive information about merchants, in addition to providing associated data on qualifying credit card rewards."
    
    """
    url = f"https://merchant-credit-card-reward.p.rapidapi.com/api/rapidapi/merchant_reward"
    querystring = {'query': query, 'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "merchant-credit-card-reward.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

