import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def polygon_fraud(wallet: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    wallet: Wallet address for which you want to check the fraud probabilty
        
    """
    url = f"https://crypto-fraud-detection2.p.rapidapi.com/Polygon/fraud/"
    querystring = {'Wallet': wallet, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-fraud-detection2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bnb_fraud(wallet: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    wallet: Wallet address for which you want to check the fraud probabilty
        
    """
    url = f"https://crypto-fraud-detection2.p.rapidapi.com/BNB/fraud/"
    querystring = {'Wallet': wallet, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-fraud-detection2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def eth_fraud(wallet: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    wallet: Wallet address for which you want to check the fraud probabilty
        
    """
    url = f"https://crypto-fraud-detection2.p.rapidapi.com/ETH/fraud/"
    querystring = {'Wallet': wallet, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-fraud-detection2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

