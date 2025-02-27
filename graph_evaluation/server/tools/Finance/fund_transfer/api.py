import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def resendotpft(body: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "In case you miss the otp from our side."
    body: Give customerMobileNo in json format
        
    """
    url = f"https://mano2468-fund-transfer-v1.p.rapidapi.com/resend_ot_ps"
    querystring = {'Body': body, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mano2468-fund-transfer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbankdetailsft(bankcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "By accessing  this API we can get all the bank details of a particular customer. For this a bank code is required."
    
    """
    url = f"https://mano2468-fund-transfer-v1.p.rapidapi.com/getbankdetail/{bankcode}.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mano2468-fund-transfer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findcustomerdetailsft(customermobileno: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API permits us to  find a particular customer's details. e.g-customer's name, customer's mobile number etc."
    
    """
    url = f"https://mano2468-fund-transfer-v1.p.rapidapi.com/getcustomerdetails/{customermobileno}.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mano2468-fund-transfer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

