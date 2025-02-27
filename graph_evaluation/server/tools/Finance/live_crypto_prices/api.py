import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_live_price_feed_data_for_hundreds_of_cryptocurrencies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "this endpoint will return crypto data.
		for example:
		
		`{
		"Id": "1",
		"Logo": "https://assets.coingecko.com/coins/images/1/thumb/bitcoin.png?1547033579",
		"Symbol": "BTC",
		"CoinName": "Bitcoin",
		"Price": "$39,667.18",
		"1hChange": "0.1%",
		"24hChange": "-2.5%",
		"7dChange": "-0.7%",
		"Volume24h": "$47,766,388,041",
		"MarketCapital": "$755,344,578,139"
		}`"
    
    """
    url = f"https://live-crypto-prices.p.rapidapi.com/pricefeed"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-crypto-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

