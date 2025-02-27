import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_chain_id(chain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns chainId from chain name."
    chain: Available chains:
**ethereum, bsc, okc, heco, polygon, arbitrum, optimism, avalanche, fantom, kcc, gnosis, tron, linea, harmony, zkysnc, ethw, fon, cronos**
        
    """
    url = f"https://token-scanner-multichain-honeypot-checker-multichain.p.rapidapi.com/get_chain_id"
    querystring = {'chain': chain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "token-scanner-multichain-honeypot-checker-multichain.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scan_token(token_address: str, chain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns token scan result."
    chain: Available chains:
**ethereum, bsc, okc, heco, polygon, arbitrum, optimism, avalanche, fantom, kcc, gnosis, tron, linea, harmony, zkysnc, ethw, fon, cronos**
        
    """
    url = f"https://token-scanner-multichain-honeypot-checker-multichain.p.rapidapi.com/scan"
    querystring = {'token_address': token_address, 'chain': chain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "token-scanner-multichain-honeypot-checker-multichain.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

