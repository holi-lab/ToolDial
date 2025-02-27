import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_axie_details_of_an_account(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is to get list of axies of an account. It includes picture of the axie and its skills and more information. Add address
		
		0x7bad5c65f2e2c53e65a5c32d330c070c337ce066"
    
    """
    url = f"https://axie-infinity.p.rapidapi.com/axie/get-axies/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-infinity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_adventure_slp_mmr_total_slp_v2(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "add ronin address 0x26d252724d08a30151ab5c87bd6b4fb5eadb1500"
    
    """
    url = f"https://axie-infinity.p.rapidapi.com/axie/get-update/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-infinity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

