import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ohlcv_vwap_spot(market_venue: str, start: str, base: str, symbol: str, accept: str='application/json', gran: str='1d', sort: str='asc', end: str='2023-05-30T10:05:00', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Price and volume metrics for spot markets"
    market_venue: The name of an exchange or a venue
        start: Start of the requested time period, *%Y-%m-%dT%H:%M:%S* UTC timezone
        base: The second listed currency of a currency pair
        symbol: The first listed currency of a currency pair
        accept: Output format: `application/json`, `application/csv`
        gran: Available granularities: `1m`, `15m`, `1h`, `1d`
        sort: The ordering of events: `asc` (from earliest to latest), `desc` (from latest to earliest) 
        end: End of the requested time period, *%Y-%m-%dT%H:%M:%S* UTC timezone
        
    """
    url = f"https://cryptocurrency-financial-data.p.rapidapi.com/metrics/ohlcv"
    querystring = {'market_venue': market_venue, 'start': start, 'base': base, 'symbol': symbol, }
    if accept:
        querystring['Accept'] = accept
    if gran:
        querystring['gran'] = gran
    if sort:
        querystring['sort'] = sort
    if end:
        querystring['end'] = end
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrency-financial-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ohlcv_vwap_futures(symbol: str, start: str, base: str, market_venue: str, accept: str='application/json', delivery_date: str=None, gran: str='1d', sort: str='asc', end: str='2023-05-06T10:05:00', expiration: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Price and volume metrics for futures markets"
    symbol: The first listed currency of a currency pair
        start: Start of the requested time period, UTC timezone
        base: The second listed currency of a currency pair
        market_venue: The name of exchange or venue
        accept: Output format: `application/json`, `application/csv`
        delivery_date: The last day when a future contract is valid - *YYYY-MM-DD*
        gran: Available granularities: `1m`, `15m`, `1h`, `1d`
        sort: The ordering of events: `asc` (from earliest to latest), `desc` (from latest to earliest) 
        end: End of the requested time period, UTC timezone
        expiration: The lifespan of a futures contract. Allowed values: `perpetual`(default), `weekly`, `quarterly`, `monthly`
        
    """
    url = f"https://cryptocurrency-financial-data.p.rapidapi.com/metrics/ohlcv/futures"
    querystring = {'symbol': symbol, 'start': start, 'base': base, 'market_venue': market_venue, }
    if accept:
        querystring['Accept'] = accept
    if delivery_date:
        querystring['delivery_date'] = delivery_date
    if gran:
        querystring['gran'] = gran
    if sort:
        querystring['sort'] = sort
    if end:
        querystring['end'] = end
    if expiration:
        querystring['expiration'] = expiration
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrency-financial-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def metadata(asset_type: str='spot', market_venue: str='BINANCE', symbol: str='BTC', base: str='USDT', data_type: str='metrics', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The full list of supported markets"
    
    """
    url = f"https://cryptocurrency-financial-data.p.rapidapi.com/metadata"
    querystring = {}
    if asset_type:
        querystring['asset_type'] = asset_type
    if market_venue:
        querystring['market_venue'] = market_venue
    if symbol:
        querystring['symbol'] = symbol
    if base:
        querystring['base'] = base
    if data_type:
        querystring['data_type'] = data_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrency-financial-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trades_spot(start: str, market_venue: str, symbol: str, base: str, accept: str='application/json', end: str='2023-05-06T10:05:00', limit: int=100, sort: str='asc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Trades endpoint available upon request"
    start: Start of the requested time period, *%Y-%m-%dT%H:%M:%S* UTC timezone
        market_venue: The name of an exchange or a venue
        symbol: The first or base currency in the traded pair
        base: The second or quote currency in the traded pair
        accept: Output format: `application/json`, `application/csv`
        end: End of the requested time period, *%Y-%m-%dT%H:%M:%S* UTC timezone
        limit: Maximum number of records to return, max `10000`
        sort: The ordering of results: `asc` (from earliest to latest), `desc` (from latest to earliest)
        
    """
    url = f"https://cryptocurrency-financial-data.p.rapidapi.com/trades"
    querystring = {'start': start, 'market_venue': market_venue, 'symbol': symbol, 'base': base, }
    if accept:
        querystring['Accept'] = accept
    if end:
        querystring['end'] = end
    if limit:
        querystring['limit'] = limit
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrency-financial-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trades_futures(market_venue: str, base: str, symbol: str, accept: str='application/json', limit: int=100, end: str='2023-05-06T10:05:00', expiration: str=None, delivery_date: str=None, start: str='2023-05-05T10:05:00', sort: str='asc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Derivatives contracts transactions"
    market_venue: The name of an exchange or a venue
        base: The second listed currency of a currency pair
        symbol: The first listed currency of a currency pair
        accept: Output format: `application/json`, `application/csv`
        limit: Maximum number of records to return, max `10000`
        end: End of the requested time period, *%Y-%m-%dT%H:%M:%S* UTC timezone
        expiration: The lifespan of a futures contract. Allowed values: `perpetual`(default), `weekly`, `quarterly`, `monthly`
        delivery_date: The last day when a future contract is valid - *YYYY-MM-DD*
        start: Start of the requested time period, *%Y-%m-%dT%H:%M:%S* UTC timezone
        sort: The ordering of results: `asc` (from earliest to latest), `desc` (from latest to earliest)
        
    """
    url = f"https://cryptocurrency-financial-data.p.rapidapi.com/trades/futures"
    querystring = {'market_venue': market_venue, 'base': base, 'symbol': symbol, }
    if accept:
        querystring['Accept'] = accept
    if limit:
        querystring['limit'] = limit
    if end:
        querystring['end'] = end
    if expiration:
        querystring['expiration'] = expiration
    if delivery_date:
        querystring['delivery_date'] = delivery_date
    if start:
        querystring['start'] = start
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrency-financial-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

