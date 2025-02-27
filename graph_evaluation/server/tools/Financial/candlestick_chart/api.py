import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def binance_charts(symbol: str, content_type: str='application/x-www-form-urlencoded', limit: int=16, interval: str='1m', lastprice: int=57500, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint creates candlestick charts for any cryptocurrency listed on [Binance](https://www.binance.com) that you want!"
    symbol: Symbol for the traiding pair

You can see every traiding pair available [here](https://coinmarketcap.com/en/exchanges/binance)
        limit: Amount of candles in the chart

- Default: 16
- Type: Integer. Maximum 1000
        interval: Time interval for each candle.

- Default: 1m

## Time intervals
- m: Minutes
- h: Hours
- d: Days
- w: Weeks
- M: Months

List of intervals:
- 1m
- 3m
- 5m
- 15m
- 30m
- 1h
- 2h
- 4h
- 6h
- 8h
- 12h
- 1d
- 3d
- 1w
- 1M
        lastprice: The last price that the chart must have. This could be useful if there is some delay between your analysis and the call to this API, and that delay could make a difference between the numbers and the chart. If not given, the chart will be created with Binance last price

- Type: Number
        
    """
    url = f"https://candlestick-chart.p.rapidapi.com/binance"
    querystring = {'symbol': symbol, }
    if content_type:
        querystring['Content-Type'] = content_type
    if limit:
        querystring['limit'] = limit
    if interval:
        querystring['interval'] = interval
    if lastprice:
        querystring['lastPrice'] = lastprice
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "candlestick-chart.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def health(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Health check"
    
    """
    url = f"https://candlestick-chart.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "candlestick-chart.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

