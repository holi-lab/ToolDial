import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_payment_link(transactionrefnumber: str, authorization: str='Authorization', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate payment Link"
    
    """
    url = f"https://be-pfh.p.rapidapi.com/api/v0/SISTIC/generatepaymentlink"
    querystring = {'transactionrefnumber': transactionrefnumber, }
    if authorization:
        querystring['Authorization'] = authorization
    if authorization:
        querystring['Authorization'] = authorization
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "be-pfh.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

