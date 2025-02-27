import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def top_long_short_account_ratio(pair: str, period: str, starttime: int=None, limit: int=None, endtime: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the top long to short account ratio
		
		If **startTime** and **endTime** are not sent, the most recent data is returned."
    pair: e.g bnbusdt or BNBUSDT. Cases insensitive
        period: time period for each stat
        starttime: Timestamp in ms (unix time value).
        limit: number value of the limit of number of stats to be returned (default=30, max=500)
        endtime: Timestamp in ms (unix time value).
        
    """
    url = f"https://binance-api8.p.rapidapi.com/top-long-short-account-ratio/{pair}/{period}"
    querystring = {}
    if starttime:
        querystring['startTime'] = starttime
    if limit:
        querystring['limit'] = limit
    if endtime:
        querystring['endTime'] = endtime
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-api8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def candle_stick(interval: str, pair: str, starttime: int=None, limit: int=None, endtime: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows you to get information about one or many candle sticks
		
		Valid **Intervals** are ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]"
    interval: candle stick interval
        pair: e.g bnbusdt or BNBUSDT. Cases insensitive
        starttime: Timestamp in ms (unix time value).
        limit: number value of the limit of number of canldles to be returned (default=500, max=1500)
        endtime: Timestamp in ms (unix time value).
        
    """
    url = f"https://binance-api8.p.rapidapi.com/candlestick/{pair}"
    querystring = {'interval': interval, }
    if starttime:
        querystring['startTime'] = starttime
    if limit:
        querystring['limit'] = limit
    if endtime:
        querystring['endTime'] = endtime
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-api8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_order_book(pair: str, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the most recent trades in order book for the market pair provided
		
		Limits that are set must be among these [5, 10, 20, 50, 100, 500, 1000]
		
		For each element in the bids/asks response (See example response)
		[
		          "19952.90",  #price
		          "15.393"        #quantity
		]
		**E** is the message output time
		**T** is the transaction time"
    limit: number value of the limit of number of trades to be returned (default=500)
Value must be among these [5, 10, 20, 50, 100, 500, 1000]
        
    """
    url = f"https://binance-api8.p.rapidapi.com/market-order-book/{pair}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-api8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_long_short_position_ratio(pair: str, period: str, starttime: int=None, endtime: int=None, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the top long to short positions ratio
		
		If **startTime** and **endTime** are not sent, the most recent data is returned."
    pair: e.g bnbusdt or BNBUSDT. Case insensitive
        period: time period for each stat
        starttime: Timestamp in ms (unix time value).
        endtime: Timestamp in ms (unix time value).
        limit: number value of the limit of number of stats to be returned (default=30, max=500)
        
    """
    url = f"https://binance-api8.p.rapidapi.com/top-long-short-position-ratio/{pair}/{period}"
    querystring = {}
    if starttime:
        querystring['startTime'] = starttime
    if endtime:
        querystring['endTime'] = endtime
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-api8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def contract_candlestick(pair: str, contract: str, interval: str, starttime: int=None, limit: int=None, endtime: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get candle sticks from a particular contract
		
		**contract** types are PERPETUAL, CURRENT_QUARTER and NEXT_QUARTER
		Valid **Intervals** are ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]"
    contract: The type of contract to interact with

        interval: candle stick interval
        starttime: Timestamp in ms (unix time value).
        limit: number value of the limit of number of candles to be returned (default=500, max=1500)
        endtime: Timestamp in ms (unix time value).
        
    """
    url = f"https://binance-api8.p.rapidapi.com/contract-candlestick/{pair}/{contract}/{interval}"
    querystring = {}
    if starttime:
        querystring['startTime'] = starttime
    if limit:
        querystring['limit'] = limit
    if endtime:
        querystring['endTime'] = endtime
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-api8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def funding_rate_history(pair: str=None, limit: int=None, endtime: int=None, starttime: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns funding rate history
		
		
		**endTime** is a timestamp in ms
		**limit** is the number of candle sticks to get (default=500)
		
		If no **pair** is provided, it returns all current funding rates.
		If **startTime** and **endTime **are not sent, the most recent limit data are returned."
    pair: e.g bnbusdt or BNBUSDT. Cases insensitive
        limit: number value of the limit of number of pairs to be returned (default=100, max=1000)
        endtime: Timestamp in ms (unix time value).
        starttime: Timestamp in ms (unix time value).
        
    """
    url = f"https://binance-api8.p.rapidapi.com/funding-rate"
    querystring = {}
    if pair:
        querystring['pair'] = pair
    if limit:
        querystring['limit'] = limit
    if endtime:
        querystring['endTime'] = endtime
    if starttime:
        querystring['startTime'] = starttime
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-api8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mark_price_candlestick(interval: str, pair: str, starttime: int=None, endtime: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the mark price candle sticks for the specified pairs
		
		Valid **intervals** are ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
		
		If startTime and endTime are not sent, the most recent candle sticks are returned"
    interval: candle stick interval

        pair: e.g bnbusdt or BNBUSDT. Cases insensitive
        starttime: Timestamp in ms (unix time value).
        endtime: Timestamp in ms (unix time value).
        limit: number value of the limit of number of pairs to be returned (default=100, max=1000)
        
    """
    url = f"https://binance-api8.p.rapidapi.com/mark-price-candlestick/{pair}/{interval}"
    querystring = {}
    if starttime:
        querystring['startTime'] = starttime
    if endtime:
        querystring['endTime'] = endtime
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-api8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_24_hr_stats(pair: str='ethusdt', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "24 hour rolling window price change statistics.
		Be Careful when accessing this with no **pair**."
    pair: e.g bnbusdt or BNBUSDT. Cases insensitive
        
    """
    url = f"https://binance-api8.p.rapidapi.com/24hr-stats"
    querystring = {}
    if pair:
        querystring['pair'] = pair
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-api8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recent_trades(pair: str, limit: int=200, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the most recent trades of a particular pair
		
		Set **limit** as a query parameter to specify the number of trades to get (default=500, max=1000)."
    pair: e.g bnbusdt or BNBUSDT. Cases insensitive
        limit: number value of the limit of number of trades to be returned (default=100, max=1000)
        
    """
    url = f"https://binance-api8.p.rapidapi.com/trades/{pair}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-api8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def open_interest(pair: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns present open interest of a specific pair."
    pair: e.g bnbusdt or BNBUSDT. Case insensitive
        
    """
    url = f"https://binance-api8.p.rapidapi.com/open-interest/{pair}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-api8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exchange_info(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the rules of Binance exchange"
    
    """
    url = f"https://binance-api8.p.rapidapi.com/exchange-info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-api8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def index_info(pair: str='bnbusdt', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Index info of a **pair**
		
		If a **pair** is not provided, index info for all the pairs will be returned"
    pair: e.g bnbusdt or BNBUSDT. Case insensitive
        
    """
    url = f"https://binance-api8.p.rapidapi.com/index-info"
    querystring = {}
    if pair:
        querystring['pair'] = pair
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-api8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def order_book(pair: str='rayusdt', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the order book of the **pair** provided.
		
		If no pair is provided, order books of all pairs will be returned in an array."
    pair: e.g bnbusdt or BNBUSDT. Cases insensitive
        
    """
    url = f"https://binance-api8.p.rapidapi.com/order-book"
    querystring = {}
    if pair:
        querystring['pair'] = pair
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-api8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def taker_long_short_ratio(pair: str, period: str, limit: int=None, endtime: int=None, starttime: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns takers long to short positions ratio
		
		If **startTime** and **endTime** are not sent, the most recent data is returned.
		Only the data of the latest 30 days is available."
    limit: number value of the limit of number of stats to be returned (default=30, max=500)
        endtime: Timestamp in ms (unix time value).
        starttime: Timestamp in ms (unix time value).
        
    """
    url = f"https://binance-api8.p.rapidapi.com/taker-long-short-ratio/{pair}/{period}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if endtime:
        querystring['endTime'] = endtime
    if starttime:
        querystring['startTime'] = starttime
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-api8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def global_long_short_account_ratio(period: str, pair: str, limit: int=None, endtime: int=None, starttime: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the global long to short account ratio
		
		If **startTime** and **endTime** are not sent, the most recent data is returned."
    period: time period for each stat
        pair: e.g bnbusdt or BNBUSDT. Case insensitive
        limit: number value of the limit of number of stats to be returned (default=30, max=500)
        endtime: Timestamp in ms (unix time value).
        starttime: Timestamp in ms (unix time value).
        
    """
    url = f"https://binance-api8.p.rapidapi.com/global-long-short-account-ratio/{pair}/{period}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if endtime:
        querystring['endTime'] = endtime
    if starttime:
        querystring['startTime'] = starttime
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-api8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def asset_index(pair: str='bnbusdt', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns asset index for Multi-Assets mode
		
		If no **pair** is provided, it returns data for all the pairs"
    pair: e.g AUDUSD or audusd. Case insensitive
        
    """
    url = f"https://binance-api8.p.rapidapi.com/asset-info"
    querystring = {}
    if pair:
        querystring['pair'] = pair
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-api8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def blvt_nav_candlestick(interval: str, pair: str, limit: str=None, starttime: str=None, endtime: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The BLVT NAV system is based on Binance Futures
		Returns  LVT candlesticks"
    limit: The limit of number of candles to be returned (default=500, max=1000)
        starttime: Timestamp in ms (unix time value).
        endtime: Timestamp in ms (unix time value).
        
    """
    url = f"https://binance-api8.p.rapidapi.com/blvt-nav-candlestick/{pair}/{interval}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if starttime:
        querystring['startTime'] = starttime
    if endtime:
        querystring['endTime'] = endtime
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-api8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def open_interest_stats(pair: str, interval: str, limit: int=None, endtime: int=None, starttime: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns open interest statistics with the period provided
		
		If **startTime** and **endTime** are not sent, the most recent data is returned.
		Only the data of the latest 30 days is available."
    pair: e.g bnbusdt or BNBUSDT. Cases insensitive
        interval: time period for each stat
        limit: number value of the limit of number of stats to be returned (default=30, max=500)
        endtime: Timestamp in ms (unix time value).
        starttime: Timestamp in ms (unix time value).
        
    """
    url = f"https://binance-api8.p.rapidapi.com/open-interest-hist/{pair}/{interval}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if endtime:
        querystring['endTime'] = endtime
    if starttime:
        querystring['startTime'] = starttime
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-api8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def premium_index(pair: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the preimum index and funding rate of a pair
		If no pair is given, it returns the premium indexes and funding rates of all the pairs"
    pair: e.g bnbusdt or BNBUSDT. Cases insensitive
        
    """
    url = f"https://binance-api8.p.rapidapi.com/premium-index"
    querystring = {}
    if pair:
        querystring['pair'] = pair
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-api8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def index_price_candle_stick(pair: str, interval: str, limit: int=20, starttime: int=None, endtime: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the index price candle sticks of the specified pair
		
		Valid **Intervals** are ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]"
    pair: e.g bnbusdt or BNBUSDT. Cases insensitive
        interval: candle stick interval
        limit: number value of the limit of number of pairs to be returned (default=500, max=1500)
        starttime: Timestamp in ms (unix time value).
        endtime: Timestamp in ms (unix time value).
        
    """
    url = f"https://binance-api8.p.rapidapi.com/index-price-candlestick/{pair}/{interval}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if starttime:
        querystring['startTime'] = starttime
    if endtime:
        querystring['endTime'] = endtime
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-api8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def price(pair: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the price of an asset
		
		If no **pair** is provided, prices for all symbols will be returned in an array."
    pair: e.g bnbusdt or BNBUSDT. Cases insensitive
        
    """
    url = f"https://binance-api8.p.rapidapi.com/price"
    querystring = {}
    if pair:
        querystring['pair'] = pair
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-api8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def server_time(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current time on the server.
		Also used to check if you are connected too the API"
    
    """
    url = f"https://binance-api8.p.rapidapi.com/server-time"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-api8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

