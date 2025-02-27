import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def btcregapi(buy: int=35000, amount: int=1729, simbol: str='BTC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives you the profit of your buy. If you buy some crypto and want to know how much you won, try this. Many providers offer a view of your crypto assets but not the performance for each purchase you have made. This api offers that you can quickly get the performance for each purchase you have made, for example if you bought 1729 dollars of BTC when the price was close to 36000 USDT Now with BTC at 40000 today you would have 1921.11 dollars and a profit of 192.11 dollars.
		Default values:
		BUY=1
		AMOUNT=1
		simbol=BTC"
    
    """
    url = f"https://criptoreg.p.rapidapi.com/btcregapi"
    querystring = {}
    if buy:
        querystring['buy'] = buy
    if amount:
        querystring['amount'] = amount
    if simbol:
        querystring['simbol'] = simbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "criptoreg.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

