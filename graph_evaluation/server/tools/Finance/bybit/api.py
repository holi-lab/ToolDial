import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def latest_symbol_info(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Latest Symbol Info"
    symbol: Contract name
        
    """
    url = f"https://bybit4.p.rapidapi.com/perpetual/usdc/openapi/public/v1/tick"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def contract_info(limit: int=None, direction: str=None, symbol: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Contract Info"
    limit: Number of data per page; the max. value is 200, and the default value is 200
        direction: prev: prev page, next: next page.
        symbol: Contract name
        
    """
    url = f"https://bybit4.p.rapidapi.com/perpetual/usdc/openapi/public/v1/symbols"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if direction:
        querystring['direction'] = direction
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def order_book(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "OrderBook"
    symbol: Contract name
        
    """
    url = f"https://bybit4.p.rapidapi.com/perpetual/usdc/openapi/public/v1/order-book"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_symbol_info_by_basecoin(basecoin: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all ticker info by passed base coin"
    basecoin: Base coin. Returns all records with base coin. If not passed, it returns records with BTC by default
        
    """
    url = f"https://bybit4.p.rapidapi.com/option/usdc/openapi/public/v1/all-tickers"
    querystring = {}
    if basecoin:
        querystring['baseCoin'] = basecoin
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_historical_volatility(starttime: int=1659283200000, endtime: int=1660060800000, basecoin: str=None, period: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The data is in hourly.
		If time field is not passed, it returns the recent 1 hour data by default.
		It could be any timeframe by inputting startTime & endTime, but it must satisfy [endTime - startTime] <= 30 days.
		It returns all data in 2 years when startTime & endTime are not passed.
		Both startTime & endTime entered together or both are left blank"
    starttime: Starting timestamp (unit: ms)
        endtime: Ending timestamp (unit: ms)
        basecoin: Base coin. If not passed, BTC returned by default
        period: Period. if not passed, it returns 7 days by default
        
    """
    url = f"https://bybit4.p.rapidapi.com/option/usdc/openapi/public/v1/query-historical-volatility"
    querystring = {}
    if starttime:
        querystring['startTime'] = starttime
    if endtime:
        querystring['endTime'] = endtime
    if basecoin:
        querystring['baseCoin'] = basecoin
    if period:
        querystring['period'] = period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_last_500_trades(category: str, optiontype: str=None, symbol: str=None, basecoin: str=None, limit: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returned records are Taker Buy in default."
    category: Type. OPTION or PERPETUAL
        optiontype: Call or put, required for Options
        symbol: Contract name
        basecoin: Base coin. Returns all records with base coin. If not passed, it returns records with BTC by default
        limit: default 500
        
    """
    url = f"https://bybit4.p.rapidapi.com/option/usdc/openapi/public/v1/query-trade-latest"
    querystring = {'category': category, }
    if optiontype:
        querystring['optionType'] = optiontype
    if symbol:
        querystring['symbol'] = symbol
    if basecoin:
        querystring['baseCoin'] = basecoin
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def delivery_price(basecoin: str=None, limit: int=None, direction: str=None, cursor: str=None, symbol: str='BTC-3NOV21-58000-C', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Delivery Price"
    basecoin: Base coin. Returns all records with base coin. If not passed, it returns records with BTC by default
        limit: Number of data per page; the max. value is 200, and the default value is 50
        direction: prev: prev page, next: next page.
        cursor: API pass-through
        symbol: Contract name
        
    """
    url = f"https://bybit4.p.rapidapi.com/option/usdc/openapi/public/v1/delivery-price"
    querystring = {}
    if basecoin:
        querystring['baseCoin'] = basecoin
    if limit:
        querystring['limit'] = limit
    if direction:
        querystring['direction'] = direction
    if cursor:
        querystring['cursor'] = cursor
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_symbol_info(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Latest Symbol Info"
    symbol: Contract name
        
    """
    url = f"https://bybit4.p.rapidapi.com/option/usdc/openapi/public/v1/tick"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def contract_info(symbol: str='BTC-30SEP22-400000-C', status: str='ONLINE', direction: str=None, basecoin: str=None, limit: int=None, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query for all if blank."
    symbol: Contract name
        status: Status, can be WAITING_ONLINE, ONLINE, DELIVERING, or OFFLINE
        direction: prev: prev page, next: next page.
        basecoin: Base coin. Returns all records with base coin. If not passed, it returns records with BTC by default
        limit: Number of data per page; the max. value is 1000, and the default value is 500
        cursor: API pass-through
        
    """
    url = f"https://bybit4.p.rapidapi.com/option/usdc/openapi/public/v1/symbols"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    if status:
        querystring['status'] = status
    if direction:
        querystring['direction'] = direction
    if basecoin:
        querystring['baseCoin'] = basecoin
    if limit:
        querystring['limit'] = limit
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def orderbook(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query order book info. Each side has a depth of 25 orders."
    symbol: Contract name
        
    """
    url = f"https://bybit4.p.rapidapi.com/option/usdc/openapi/public/v1/order-book"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_public_trading_history(symbol: str, category: str, optiontype: str=None, limit: int=1, basecoin: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get recent trades."
    symbol: Symbol
        category: Derivatives products category.For now, *linear inverse* including inverse futures are available
        optiontype: Call/Put Trading type of option. Call/Put
        limit: Limit for data size per page, max size is 1000. Default as showing 500 pieces of data per page
        basecoin: Base coin. Only valid when category=option. If not passed, BTC by default.
        
    """
    url = f"https://bybit4.p.rapidapi.com/derivatives/v3/public/recent-trade"
    querystring = {'symbol': symbol, 'category': category, }
    if optiontype:
        querystring['optionType'] = optiontype
    if limit:
        querystring['limit'] = limit
    if basecoin:
        querystring['baseCoin'] = basecoin
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_open_interest(symbol: str, interval: str, category: str, starttime: int=None, limit: int=None, endtime: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the total amount of unsettled contracts. In other words, the total number of contracts held in open positions."
    symbol: Symbol
        interval: Interval. 5min 15min 30min 1h 4h 1d
        category: Derivatives products category.For now, *linear inverse* including inverse futures are available
        starttime: Start timestamp point for result, in milliseconds
        limit: Limit for data size per page, max size is 200. Default as showing 50 pieces of data per page
        endtime: End timestamp point for result, in milliseconds
        
    """
    url = f"https://bybit4.p.rapidapi.com/derivatives/v3/public/open-interest"
    querystring = {'symbol': symbol, 'interval': interval, 'category': category, }
    if starttime:
        querystring['startTime'] = starttime
    if limit:
        querystring['limit'] = limit
    if endtime:
        querystring['endTime'] = endtime
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_risk_limit(symbol: str, category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Risk Limit"
    symbol: Symbol
        category: Derivatives products category.For now, *linear inverse* including inverse futures are available
        
    """
    url = f"https://bybit4.p.rapidapi.com/derivatives/v3/public/risk-limit/list"
    querystring = {'symbol': symbol, 'category': category, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_latest_information_for_symbol(category: str, symbol: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all latest information for symbol"
    category: Derivatives products category.For now, *linear* *inverse* *option* are available
        symbol: Name of the trading pair. If category is *option* REQUIRED
        
    """
    url = f"https://bybit4.p.rapidapi.com/derivatives/v3/public/tickers"
    querystring = {'category': category, }
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_kline(symbol: str, end: int, start: int, category: str, interval: str, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get kline."
    symbol: Symbol
        end: End timestamp point for result, in milliseconds
        start: Start timestamp point for result, in milliseconds
        category: Derivatives products category.For now, *linear* *inverse* including inverse futures are available
        interval: Kline interval. enum: 1 3 5 15 30 60 120 240 360 720 \"D\" \"M\" \"W\" 
        limit: Limit for data size per page, max size is 200. Default as showing 200 pieces of data per page
        
    """
    url = f"https://bybit4.p.rapidapi.com/derivatives/v3/public/kline"
    querystring = {'symbol': symbol, 'end': end, 'start': start, 'category': category, 'interval': interval, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_instrument_info(category: str, symbol: str=None, limit: int=None, cursor: str=None, basecoin: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query launched instruments info list"
    category: Derivatives products category.For now, *linear inverse option* are available
        symbol: Symbol
        limit: Limit for data size per page, max size is 1000. Default as showing 500 pieces of data per page
        cursor: API pass-through
        basecoin: Base coin. Only valid when category=*option*. If not passed, BTC by default.
        
    """
    url = f"https://bybit4.p.rapidapi.com/derivatives/v3/public/instruments-info"
    querystring = {'category': category, }
    if symbol:
        querystring['symbol'] = symbol
    if limit:
        querystring['limit'] = limit
    if cursor:
        querystring['cursor'] = cursor
    if basecoin:
        querystring['baseCoin'] = basecoin
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_last_funding_rate(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The funding rate is generated every 8 hours at 00:00 UTC, 08:00 UTC and 16:00 UTC. For example, if a request is sent at 12:00 UTC, the funding rate generated earlier that day at 08:00 UTC will be sent."
    symbol: Symbol
        
    """
    url = f"https://bybit4.p.rapidapi.com/public/linear/funding/prev-funding-rate"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_index_price_kline(symbol: str, interval: str, is_from: int, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Index price kline. Tracks BTC spot prices, with a frequency of every second."
    symbol: Symbol
        interval: Data refresh interval. Enum : 1 3 5 15 30 60 120 240 360 720 \"D\" \"M\" \"W\"
        from: From timestamp in seconds
        limit: Limit for data size, max size is 200. Default as showing 200 pieces of data
        
    """
    url = f"https://bybit4.p.rapidapi.com/v2/public/index-price-kline"
    querystring = {'symbol': symbol, 'interval': interval, 'from': is_from, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_kline_usdt(symbol: str, interval: str, is_from: int, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get kline."
    symbol: Symbol
        interval: Data refresh interval. Enum : 1 3 5 15 30 60 120 240 360 720 "D" "M" "W"
        from: From timestamp in seconds
        limit: Limit for data size, max size is 200. Default as showing 200 pieces of data
        
    """
    url = f"https://bybit4.p.rapidapi.com/public/linear/kline"
    querystring = {'symbol': symbol, 'interval': interval, 'from': is_from, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_premium_index_kline_usdt(is_from: int, symbol: str, interval: str, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Premium index kline. Tracks the premium / discount of BTC perpetual contracts relative to the mark price per minute."
    from: From timestamp in seconds
        symbol: Symbol
        interval: Data refresh interval. Enum : 1 3 5 15 30 60 120 240 360 720 \\\"D\\\" \\\"M\\\" \\\"W\\\"
        limit: Limit for data size, max size is 200. Default as showing 200 pieces of data
        
    """
    url = f"https://bybit4.p.rapidapi.com/public/linear/premium-index-kline"
    querystring = {'from': is_from, 'symbol': symbol, 'interval': interval, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_index_price_kline_usdt(interval: str, is_from: int, symbol: str, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Index price kline. Tracks BTC spot prices, with a frequency of every second."
    interval: Data refresh interval. Enum : 1 3 5 15 30 60 120 240 360 720 \"D\" \"M\" \"W\"
        from: From timestamp in seconds
        symbol: Symbol
        limit: Limit for data size, max size is 200. Default as showing 200 pieces of data
        
    """
    url = f"https://bybit4.p.rapidapi.com/public/linear/index-price-kline"
    querystring = {'interval': interval, 'from': is_from, 'symbol': symbol, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_mark_price_kline_usdt(symbol: str, is_from: int, interval: str, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query mark price kline"
    symbol: Symbol
        from: From timestamp in seconds
        interval: Data refresh interval. Enum : 1 3 5 15 30 60 120 240 360 720 "D" "M" "W"
        limit: Limit for data size, max size is 200. Default as showing 200 pieces of data
        
    """
    url = f"https://bybit4.p.rapidapi.com/public/linear/mark-price-kline"
    querystring = {'symbol': symbol, 'from': is_from, 'interval': interval, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_big_deal(symbol: str, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain filled orders worth more than 500,000 USD within the last 24h.
		
		This endpoint may return orders which are over the maximum order qty for the symbol you call. For instance, the maximum order qty for BTCUSD is 1 million contracts, but in the event of the liquidation of a position larger than 1 million this endpoint returns this "impossible" order size."
    symbol: Symbol
        limit: Limit for data size per page, max size is 1000. Default as showing 500 pieces of data per page
        
    """
    url = f"https://bybit4.p.rapidapi.com/v2/public/big-deal"
    querystring = {'symbol': symbol, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def public_trading_records_usdt(symbol: str, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get recent trades."
    symbol: Symbol
        limit: Number of results. Default 500; max 1000
        
    """
    url = f"https://bybit4.p.rapidapi.com/public/linear/recent-trading-records"
    querystring = {'symbol': symbol, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_premium_index_kline(symbol: str, interval: str, is_from: int, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Premium index kline. Tracks the premium / discount of BTC perpetual contracts relative to the mark price per minute."
    symbol: Symbol
        interval: Data refresh interval. Enum : 1 3 5 15 30 60 120 240 360 720 \\\"D\\\" \\\"M\\\" \\\"W\\\"
        from: From timestamp in seconds
        limit: Limit for data size, max size is 200. Default as showing 200 pieces of data
        
    """
    url = f"https://bybit4.p.rapidapi.com/v2/public/premium-index-kline"
    querystring = {'symbol': symbol, 'interval': interval, 'from': is_from, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def public_trading_records(symbol: str, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get recent trades."
    symbol: Symbol
        limit: Limit for data size, max size is 1000. Default size is 500
        
    """
    url = f"https://bybit4.p.rapidapi.com/v2/public/trading-records"
    querystring = {'symbol': symbol, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_symbol(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get symbol info."
    
    """
    url = f"https://bybit4.p.rapidapi.com/v2/public/symbols"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def long_short_ratio(symbol: str, period: str, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the Bybit user accounts' long-short ratio."
    period: Data recording period. 5min, 15min, 30min, 1h, 4h, 1d
        limit: Limit for data size per page, max size is 500. Default as showing 50 pieces of data per page
        
    """
    url = f"https://bybit4.p.rapidapi.com/v2/public/account-ratio"
    querystring = {'symbol': symbol, 'period': period, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def best_bid_ask_price(symbol: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "If symbol is not specified, the best order price from all symbols will be returned"
    symbol: Name of the trading pair
        
    """
    url = f"https://bybit4.p.rapidapi.com/spot/v3/public/quote/ticker/bookTicker"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def last_traded_price(symbol: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "If symbol is not specified, the price from all symbols will be returned"
    symbol: Name of the trading pair
        
    """
    url = f"https://bybit4.p.rapidapi.com/spot/v3/public/quote/ticker/price"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_information_for_symbol(symbol: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "If symbol is not specified, the data from all symbols will be returned"
    symbol: Name of the trading pair
        
    """
    url = f"https://bybit4.p.rapidapi.com/spot/v3/public/quote/ticker/24hr"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_kline(symbol: str, interval: str, starttime: int=None, limit: int=None, endtime: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query Kline"
    symbol: Name of the trading pair
        interval: Chart interval
        starttime: Start time, unit in millisecond
        limit: Default value is 1000, max 1000
        endtime: End time, unit in millisecond
        
    """
    url = f"https://bybit4.p.rapidapi.com/spot/v3/public/quote/kline"
    querystring = {'symbol': symbol, 'interval': interval, }
    if starttime:
        querystring['startTime'] = starttime
    if limit:
        querystring['limit'] = limit
    if endtime:
        querystring['endTime'] = endtime
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def public_trading_records(symbol: str, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Public Trading Records"
    symbol: Name of the trading pair
        limit: Default value is 60, max 60
        
    """
    url = f"https://bybit4.p.rapidapi.com/spot/v3/public/quote/trades"
    querystring = {'symbol': symbol, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def merged_order_book(symbol: str, scale: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Merged Order Book"
    symbol: Name of the trading pair
        scale: Precision of the merged orderbook, 1 means 1 digit
        limit: Default value is 100. Max value is 200
        
    """
    url = f"https://bybit4.p.rapidapi.com/spot/v3/public/quote/depth/merged"
    querystring = {'symbol': symbol, }
    if scale:
        querystring['scale'] = scale
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def order_book(symbol: str, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Order Book"
    symbol: Name of the trading pair
        limit: Default value is 100. Max value is 200.
        
    """
    url = f"https://bybit4.p.rapidapi.com/spot/v3/public/quote/depth"
    querystring = {'symbol': symbol, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_symbols(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get All Symbols"
    
    """
    url = f"https://bybit4.p.rapidapi.com/spot/v3/public/symbols"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_option_delivery_price(direction: str=None, basecoin: str='option', category: str='option', symbol: str='BTC-14JUL22-18000-C', cursor: str=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get option delivery price"
    direction: prev: prev, next: next. 
        basecoin: Base coin. Only valid when category=*option*. If not passed, BTC by default.
        category: Derivatives products category.For now, *linear inverse option* are available
        symbol: Symbol
        cursor: API pass-through
        limit: Limit for data size per page, max size is 200. Default as showing 50 pieces of data per page
        
    """
    url = f"https://bybit4.p.rapidapi.com/derivatives/v3/public/delivery-price"
    querystring = {}
    if direction:
        querystring['direction'] = direction
    if basecoin:
        querystring['baseCoin'] = basecoin
    if category:
        querystring['category'] = category
    if symbol:
        querystring['symbol'] = symbol
    if cursor:
        querystring['cursor'] = cursor
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_funding_rate_history(interval: str, start: int, category: str, end: int, symbol: str, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The funding rate is generated every 8 hours at 00:00 UTC, 08:00 UTC and 16:00 UTC. For example, if a request is sent at 12:00 UTC, the funding rate generated earlier that day at 08:00 UTC will be sent."
    interval: Kline interval. enum: 1 3 5 15 30 60 120 240 360 720 \\\\\\\"D\\\\\\\" \\\\\\\"M\\\\\\\" \\\\\\\"W\\\\\\\" 
        start: Start timestamp point for result, in milliseconds
        category: Derivatives products category.For now, *linear inverse* are available
        end: End timestamp point for result, in milliseconds
        symbol: Symbol
        limit: Limit for data size per page, max size is 200. Default as showing 200 pieces of data per page
        
    """
    url = f"https://bybit4.p.rapidapi.com/derivatives/v3/public/funding/history-funding-rate"
    querystring = {'interval': interval, 'start': start, 'category': category, 'end': end, 'symbol': symbol, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_index_price_kline(symbol: str, category: str, interval: str, end: int, start: int, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query Index Price Kline"
    symbol: Symbol
        category: Derivatives products category.For now, *linear inverse* including inverse futures are available
        interval: Kline interval. enum: 1 3 5 15 30 60 120 240 360 720 \\\"D\\\" \\\"M\\\" \\\"W\\\" 
        end: End timestamp point for result, in milliseconds
        start: Start timestamp point for result, in milliseconds
        limit: Limit for data size per page, max size is 200. Default as showing 200 pieces of data per page
        
    """
    url = f"https://bybit4.p.rapidapi.com/derivatives/v3/public/index-price-kline"
    querystring = {'symbol': symbol, 'category': category, 'interval': interval, 'end': end, 'start': start, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_mark_price_kline(interval: str, start: int, symbol: str, category: str, end: int, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query mark price kline."
    interval: Kline interval. enum: 1 3 5 15 30 60 120 240 360 720 \"D\" \"M\" \"W\" 
        start: Start timestamp point for result, in milliseconds
        symbol: Symbol
        category: Derivatives products category.For now, *linear inverse* including inverse futures are available
        end: End timestamp point for result, in milliseconds
        limit: Limit for data size per page, max size is 200. Default as showing 200 pieces of data per page
        
    """
    url = f"https://bybit4.p.rapidapi.com/derivatives/v3/public/mark-price-kline"
    querystring = {'interval': interval, 'start': start, 'symbol': symbol, 'category': category, 'end': end, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_order_book(symbol: str, category: str, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the orderbook. Each side has a depth of 25.
		Enable 500 orders for orderbook API."
    symbol: Symbol
        category: Derivatives products category. For now, *linear* *inverse* *option* are available
        limit: Optional value 25 or 500. Default 25.
        
    """
    url = f"https://bybit4.p.rapidapi.com/derivatives/v3/public/order-book/L2"
    querystring = {'symbol': symbol, 'category': category, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def announcement(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Bybit OpenAPI announcements in the last 30 days in reverse order."
    
    """
    url = f"https://bybit4.p.rapidapi.com/v2/public/announcement"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def server_time(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Bybit server time."
    
    """
    url = f"https://bybit4.p.rapidapi.com/v2/public/time"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def open_interest(symbol: str, period: str, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the total amount of unsettled contracts. In other words, the total number of contracts held in open positions."
    symbol: Symbol
        period: Data recording period. 5min, 15min, 30min, 1h, 4h, 1d
        limit: Limit for data size per page, max size is 200. Default as showing 50 pieces of data per page
        
    """
    url = f"https://bybit4.p.rapidapi.com/v2/public/open-interest"
    querystring = {'symbol': symbol, 'period': period, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_mark_price_kline(interval: str, is_from: int, symbol: str, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query mark price kline."
    interval: Data refresh interval. Enum : 1 3 5 15 30 60 120 240 360 720 \"D\" \"M\" \"W\"
        from: From timestamp in seconds
        symbol: Symbol
        limit: Limit for data size, max size is 200. Default as showing 200 pieces of data
        
    """
    url = f"https://bybit4.p.rapidapi.com/v2/public/mark-price-kline"
    querystring = {'interval': interval, 'from': is_from, 'symbol': symbol, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_information_for_symbol(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest information for symbol."
    symbol: Symbol
        
    """
    url = f"https://bybit4.p.rapidapi.com/v2/public/tickers"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_kline(symbol: str, is_from: int, interval: str, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get kline."
    symbol: Symbol
        from: From timestamp in seconds
        interval: Data refresh interval. Enum : 1 3 5 15 30 60 120 240 360 720 \"D\" \"M\" \"W\"
        limit: Limit for data size per page, max size is 200. Default as showing 200 pieces of data per page
        
    """
    url = f"https://bybit4.p.rapidapi.com/v2/public/kline/list"
    querystring = {'symbol': symbol, 'from': is_from, 'interval': interval, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def order_book(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the orderbook. Each side has a depth of 25."
    symbol: Symbol
        
    """
    url = f"https://bybit4.p.rapidapi.com/v2/public/orderBook/L2"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

