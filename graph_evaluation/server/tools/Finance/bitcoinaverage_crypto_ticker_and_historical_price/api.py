import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def crypto_exchange_ticker_price(exchange: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest price data for specific cryptocurrency exchange."
    
    """
    url = f"https://bitcoinaverage-crypto-ticker-and-historical-price.p.rapidapi.com/exchanges/ticker/{exchange}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitcoinaverage-crypto-ticker-and-historical-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_price_data_for_period(symbol_set: str, symbol: str, period: str='day', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns history price for specific symbol for certain period.
		Works in parallel to the Ticker endpoint where both symbol set and market symbol need to be specified.
		This endpoint additionally accepts the period query parameter that specifies the resolution of the data.
		Period can be: minute, hour or day."
    
    """
    url = f"https://bitcoinaverage-crypto-ticker-and-historical-price.p.rapidapi.com/indices/{symbol_set}/history/{symbol}"
    querystring = {}
    if period:
        querystring['period'] = period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitcoinaverage-crypto-ticker-and-historical-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def history_data_since_timestamp(symbol: str, symbol_set: str, since: str='1620643516', resolution: str='hour', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides historical data since specific timestamp.
		Accepts the regular symbol set and symbol parameter to specify the crypto market.
		Also accepts query parameters to specify the timestamp for querying and the resolution of the data.
		The resolution can be minute, hour or day.
		Minute resolution goes back 24 hours.
		Hour resolution can go back to 30 days.
		Day resolution can go back years, for BTC it goes back to 2010."
    
    """
    url = f"https://bitcoinaverage-crypto-ticker-and-historical-price.p.rapidapi.com/indices/{symbol_set}/history/{symbol}"
    querystring = {}
    if since:
        querystring['since'] = since
    if resolution:
        querystring['resolution'] = resolution
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitcoinaverage-crypto-ticker-and-historical-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cryptocurrency_index_ticker_price(symbol: str, symbol_set: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest Ticker price for thousands of cryptocurrencies.
		Our Ticker data includes the latest price, bid, ask, 24h volume, moving average and price changes."
    symbol: The shorthand symbol of the market you are requesting data for.
A full list of supported symbols grouped by symbol set can be found [here.](https://apiv2.bitcoinaverage.com/info/indices/ticker)
        symbol_set: Symbol set can be one of: global, local, crypto, tokens and light
        
    """
    url = f"https://bitcoinaverage-crypto-ticker-and-historical-price.p.rapidapi.com/indices/{symbol_set}/ticker/{symbol}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitcoinaverage-crypto-ticker-and-historical-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_all_supported_crypto_markets(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all supported cryptocurrency markets by the BitcoinAverage API.
		New cryptos or tokens are added on a monthly basis."
    
    """
    url = f"https://bitcoinaverage-crypto-ticker-and-historical-price.p.rapidapi.com/info/indices/ticker"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitcoinaverage-crypto-ticker-and-historical-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def history_price_at_a_point_in_time(symbol: str, symbol_set: str, resolution: str='day', at: str='1620643516', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch the exact price of a cryptocurrency at a specific timestamp in the past."
    
    """
    url = f"https://bitcoinaverage-crypto-ticker-and-historical-price.p.rapidapi.com/indices/{symbol_set}/history/{symbol}"
    querystring = {}
    if resolution:
        querystring['resolution'] = resolution
    if at:
        querystring['at'] = at
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitcoinaverage-crypto-ticker-and-historical-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

