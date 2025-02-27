import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def toptier_summary(limit: int, page: int, tsym: str, assetclass: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This response returns an ascending crypto coins the rank varies the way you choose limit if you choose 10 as limit you will get the top 10 cryptos
		->limit= Options are: 1-->100
		->page= Options are: 1-->100 based on limit rate 
		->assetClass= Options are: ALL,DEFI,NFT
		->tsym" typically stands for "to symbol", which represents the cryptocurrency that the conversion rate is being calculated for."
    
    """
    url = f"https://sciphercrypto.p.rapidapi.com/api/toptier"
    querystring = {'limit': limit, 'page': page, 'tsym': tsym, 'assetClass': assetclass, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sciphercrypto.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def conversion(tsym: str, fsyms: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "example from "BTC" (Bitcoin) to "USDT" (Tether)
		This JSON response contains data for the cryptocurrency example:Ethereum (ETH), including its name, algorithm, block number, launch date, and market performance ratings. It also includes conversion information for ETH to USDT, such as the current market cap, total volume, and conversion rate. use the correct symbol in order to get an accurate data."
    
    """
    url = f"https://sciphercrypto.p.rapidapi.com/api/coin/conversion"
    querystring = {'tsym': tsym, 'fsyms': fsyms, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sciphercrypto.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

