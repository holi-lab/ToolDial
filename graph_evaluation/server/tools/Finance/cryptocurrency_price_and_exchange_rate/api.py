import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cryptocurrency_price_and_exchange_rate(symbols: str='btc-usdt-binance', platform: str='binance', coin: str='btc', market: str='usdt', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get cryptocurrency price and exchange rate"
    symbols: example:
symbols[]=btc-usdt-binance&symbols[]=btc-usdt-okex&symbols[]=btc-usdt-huobi&symbols[]=btc-usdt-mxc&symbols[]=btc-usdt-ftx&symbols[]=btc-usdt-coinbase&symbols[]=cny-usd-huilv

        platform: support:
binacne
huobi
okex
coinbase
ftx
mxc
huilv(exchange rate)
        
    """
    url = f"https://cryptocurrency-price-and-exchange-rate.p.rapidapi.com/"
    querystring = {}
    if symbols:
        querystring['symbols'] = symbols
    if platform:
        querystring['platform'] = platform
    if coin:
        querystring['coin'] = coin
    if market:
        querystring['market'] = market
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrency-price-and-exchange-rate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

