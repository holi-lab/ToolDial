import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_latest_quote(tradingsymbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The /getLatestQuote API returns latest realtime quote for for given trading symbol. The tradingSymbol parameter lets you select any valid trading symbol of stocks that are part of the Nifty 500 or the top 500 stocks, giving you access to the data you need for the stocks you're interested in.
		See the list of trading symbols [here](https://stockseyes.com/group/all)."
    tradingsymbol: Check the list of all supported tradingSymbols at [https://stockseyes.com/group/all](https://stockseyes.com/group/all) , you can download as csv also. Mainly, all major stocks which are part of nifty500, nifty 50, nifty next 50, nifty midcap50, nifty sector indices, nifty bank are supported. For exact details check the link.

        
    """
    url = f"https://stockseyes.p.rapidapi.com/latestQuote"
    querystring = {'tradingSymbol': tradingsymbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stockseyes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_last_n_candles_nse(numberofcandles: int, tradingsymbol: str, timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The /getLastNCandles API returns latest n number of candles formed for given trading symbol and time frame. The timeFrame parameter allows you to select the time frame of candleStick, with options of 1m, 5m, 15m, or 1h for 1 minute candle, 5 minute candle, 15 minute candle, 1 hour candle respectively. The tradingSymbol parameter lets you select any valid trading symbol of stocks that are part of the Nifty 500 or the top 500 stocks, giving you access to the data you need for the stocks you're interested in."
    numberofcandles: Number of candles to be fetched (max 100).
        tradingsymbol: Check the list of all supported tradingSymbols at [https://stockseyes.com/group/all](https://stockseyes.com/group/all) , you can download as csv also. Mainly, all major stocks which are part of nifty500, nifty 50, nifty next 50, nifty midcap50, nifty sector indices, nifty bank are supported. For exact details check the link.

        timeframe: 1m -> 1 minute candle
5m -> 5 minute candle
15m -> 15 minute candle
1h -> 1 hour candle
        
    """
    url = f"https://stockseyes.p.rapidapi.com/getLastNCandles"
    querystring = {'numberOfCandles': numberofcandles, 'tradingSymbol': tradingsymbol, 'timeFrame': timeframe, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stockseyes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_latest_candle_nse(tradingsymbol: str, timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The /getLatestCandle API returns latest realtime candle formed for given trading symbol and time frame. The timeFrame parameter allows you to select the time frame of candleStick, with options of 1m, 5m, 15m, or 1h for 1 minute candle, 5 minute candle, 15 minute candle, 1 hour candle respectively. The tradingSymbol parameter lets you select any valid trading symbol of stocks that are part of the Nifty 500 or the top 500 stocks, giving you access to the data you need for the stocks you're interested in."
    tradingsymbol: Check the list of all supported tradingSymbols at [https://stockseyes.com/group/all](https://stockseyes.com/group/all) , you can download as csv also. Mainly, all major stocks which are part of nifty500, nifty 50, nifty next 50, nifty midcap50, nifty sector indices, nifty bank are supported. For exact details check the link.

        timeframe: 1m -> 1 minute candle
5m -> 5 minute candle
15m -> 15 minute candle
1h -> 1 hour candle
        
    """
    url = f"https://stockseyes.p.rapidapi.com/getLatestCandle"
    querystring = {'tradingSymbol': tradingsymbol, 'timeFrame': timeframe, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stockseyes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

