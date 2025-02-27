import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getexchangestatus(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint for getting the exchange status."
    
    """
    url = f"https://kalshi-trading-api.p.rapidapi.com/exchange/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kalshi-trading-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmarket(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint for getting data about a specific market.
		
		The value for the ticker path parameter should match the ticker of the target market."
    ticker: Market ticker for the market being retrieved.
        
    """
    url = f"https://kalshi-trading-api.p.rapidapi.com/markets/{ticker}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kalshi-trading-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettrades(min_ts: int=None, limit: int=None, ticker: str=None, cursor: str=None, max_ts: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint for getting all trades for all markets."
    min_ts: Restricts the response to trades after a timestamp.
        limit: Parameter to specify the number of results per page. Defaults to 100.
        ticker: Parameter to specify a specific market to get trades from.
        cursor: The Cursor represents a pointer to the next page of records in the pagination.
So this optional parameter, when filled, should be filled with the cursor string returned in a previous request to this end-point.
Filling this would basically tell the api to get the next page containing the number of records passed on the limit parameter.
On the other side not filling it tells the api you want to get the first page for another query.
The cursor does not store any filters, so if any filter parameters like ticker, max_ts or min_ts were passed in the original query they must be passed again.
        max_ts: Restricts the response to trades before a timestamp.
        
    """
    url = f"https://kalshi-trading-api.p.rapidapi.com/markets/trades"
    querystring = {}
    if min_ts:
        querystring['min_ts'] = min_ts
    if limit:
        querystring['limit'] = limit
    if ticker:
        querystring['ticker'] = ticker
    if cursor:
        querystring['cursor'] = cursor
    if max_ts:
        querystring['max_ts'] = max_ts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kalshi-trading-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getevents(series_ticker: str=None, status: str=None, cursor: str=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint for getting data about all events."
    series_ticker: Series ticker to retrieve contracts for.
        status: Restricts the events to those with certain statuses, as a comma separated list.
The following values are accepted: open, closed, settled.
        cursor: The Cursor represents a pointer to the next page of records in the pagination.
So this optional parameter, when filled, should be filled with the cursor string returned in a previous request to this end-point.
Filling this would basically tell the api to get the next page containing the number of records passed on the limit parameter.
On the other side not filling it tells the api you want to get the first page for another query.
The cursor does not store any filters, so if any filter parameters like series_ticker was passed in the original query they must be passed again.
        limit: Parameter to specify the number of results per page. Defaults to 100.
        
    """
    url = f"https://kalshi-trading-api.p.rapidapi.com/events"
    querystring = {}
    if series_ticker:
        querystring['series_ticker'] = series_ticker
    if status:
        querystring['status'] = status
    if cursor:
        querystring['cursor'] = cursor
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kalshi-trading-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmarketorderbook(ticker: str, depth: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint for getting the orderbook for a market."
    ticker: Market ticker.
        depth: Depth specifies the maximum number of orderbook price levels on either side.
        
    """
    url = f"https://kalshi-trading-api.p.rapidapi.com/markets/{ticker}/orderbook"
    querystring = {}
    if depth:
        querystring['depth'] = depth
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kalshi-trading-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getseries(series_ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint for getting data about a series by its ticker."
    series_ticker: Should be filled with the ticker of the series.
        
    """
    url = f"https://kalshi-trading-api.p.rapidapi.com/series/{series_ticker}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kalshi-trading-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmarkets(cursor: str=None, event_ticker: str=None, limit: int=None, status: str=None, tickers: str=None, min_close_ts: int=None, series_ticker: str=None, max_close_ts: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint for listing and discovering markets on Kalshi."
    cursor: The Cursor represents a pointer to the next page of records in the pagination.
So this optional parameter, when filled, should be filled with the cursor string returned in a previous request to this end-point.
Filling this would basically tell the api to get the next page containing the number of records passed on the limit parameter.
On the other side not filling it tells the api you want to get the first page for another query.
The cursor does not store any filters, so if any filter parameters like tickers, max_ts or min_ts were passed in the original query they must be passed again.
        event_ticker: Event ticker to retrieve markets for.
        limit: Parameter to specify the number of results per page. Defaults to 100.
        status: Restricts the markets to those with certain statuses, as a comma separated list.
The following values are accepted: open, closed, settled.
        tickers: Restricts the markets to those with certain tickers, as a comma separated list.
        min_close_ts: Restricts the markets to those that are closing in or after this timestamp.
        series_ticker: Series ticker to retrieve contracts for.
        max_close_ts: Restricts the markets to those that are closing in or before this timestamp.
        
    """
    url = f"https://kalshi-trading-api.p.rapidapi.com/markets"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if event_ticker:
        querystring['event_ticker'] = event_ticker
    if limit:
        querystring['limit'] = limit
    if status:
        querystring['status'] = status
    if tickers:
        querystring['tickers'] = tickers
    if min_close_ts:
        querystring['min_close_ts'] = min_close_ts
    if series_ticker:
        querystring['series_ticker'] = series_ticker
    if max_close_ts:
        querystring['max_close_ts'] = max_close_ts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kalshi-trading-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getevent(event_ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint for getting data about an event by its ticker."
    event_ticker: Should be filled with the ticker of the event.
        
    """
    url = f"https://kalshi-trading-api.p.rapidapi.com/events/{event_ticker}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kalshi-trading-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmarkethistory(ticker: str, cursor: str=None, min_ts: int=None, limit: int=None, max_ts: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint for getting the statistics history for a market.
		
		The value for the ticker path parameter should match the ticker of the target market.
		The min_ts parameter is optional, and will restrict statistics to those after provided timestamp.
		The min_ts is inclusive, which means a market history point at min_ts will be returned."
    ticker: Market ticker
        cursor: The Cursor represents a pointer to the next page of records in the pagination.
So this optional parameter, when filled, should be filled with the cursor string returned in a previous request to this end-point.
Filling this would basically tell the api to get the next page containing the number of records passed on the limit parameter.
On the other side not filling it tells the api you want to get the first page for another query.
The cursor does not store any filters, so if any filter parameters like max_ts or min_ts were passed in the original query they must be passed again.
        min_ts: If provided, MinTs restricts history to trades starting from MinTs.  Default value: 1 hour ago.
        limit: Parameter to specify the number of results per page. Defaults to 100.
        max_ts: If provided, MaxTs restricts history to trades up until MaxTs
        
    """
    url = f"https://kalshi-trading-api.p.rapidapi.com/markets/{ticker}/history"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if min_ts:
        querystring['min_ts'] = min_ts
    if limit:
        querystring['limit'] = limit
    if max_ts:
        querystring['max_ts'] = max_ts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kalshi-trading-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getportfoliosettlements(limit: int=None, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint for getting the logged-in member's settlements historical track."
    limit: Parameter to specify the number of results per page. Defaults to 100.
        cursor: The Cursor represents a pointer to the next page of records in the pagination.
So this optional parameter, when filled, should be filled with the cursor string returned in a previous request to this end-point.
Filling this would basically tell the api to get the next page containing the number of records passed on the limit parameter.
On the other side not filling it tells the api you want to get the first page for another query.
        
    """
    url = f"https://kalshi-trading-api.p.rapidapi.com/portfolio/settlements"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kalshi-trading-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getorders(limit: int=None, max_ts: int=None, ticker: str=None, min_ts: int=None, status: str=None, event_ticker: str=None, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint for getting all orders for the logged-in member."
    limit: Parameter to specify the number of results per page. Defaults to 100.
        max_ts: Restricts the response to orders before a timestamp, formatted as a Unix Timestamp.
        ticker: Restricts the response to orders in a single market.
        min_ts: Restricts the response to orders after a timestamp, formatted as a Unix Timestamp.
        status: Restricts the response to orders that have a certain status: resting, canceled, or executed.
        event_ticker: Restricts the response to orders in a single event.
        cursor: The Cursor represents a pointer to the next page of records in the pagination.
So this optional parameter, when filled, should be filled with the cursor string returned in a previous request to this end-point.
Filling this would basically tell the api to get the next page containing the number of records passed on the limit parameter.
On the other side not filling it tells the api you want to get the first page for another query.
The cursor does not store any filters, so if any filter parameters like ticker, max_ts or min_ts were passed in the original query they must be passed again.
        
    """
    url = f"https://kalshi-trading-api.p.rapidapi.com/portfolio/orders"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if max_ts:
        querystring['max_ts'] = max_ts
    if ticker:
        querystring['ticker'] = ticker
    if min_ts:
        querystring['min_ts'] = min_ts
    if status:
        querystring['status'] = status
    if event_ticker:
        querystring['event_ticker'] = event_ticker
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kalshi-trading-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpositions(ticker: str=None, event_ticker: str=None, limit: int=None, settlement_status: str='all', cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint for getting all market positions for the logged-in member."
    ticker: Ticker of desired positions.
        event_ticker: Event ticker of desired positions.
        limit: Parameter to specify the number of results per page. Defaults to 100.
        settlement_status: Settlement status of the markets to return. Defaults to unsettled.
        cursor: The Cursor represents a pointer to the next page of records in the pagination.
So this optional parameter, when filled, should be filled with the cursor string returned in a previous request to this end-point.
Filling this would basically tell the api to get the next page containing the number of records passed on the limit parameter.
On the other side not filling it tells the api you want to get the first page for another query.
The cursor does not store any filters, so if any filter parameters like settlement_status, ticker, or event_ticker were passed in the original query they must be passed again.
        
    """
    url = f"https://kalshi-trading-api.p.rapidapi.com/portfolio/positions"
    querystring = {}
    if ticker:
        querystring['ticker'] = ticker
    if event_ticker:
        querystring['event_ticker'] = event_ticker
    if limit:
        querystring['limit'] = limit
    if settlement_status:
        querystring['settlement_status'] = settlement_status
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kalshi-trading-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbalance(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint for getting the balance of the logged-in member."
    
    """
    url = f"https://kalshi-trading-api.p.rapidapi.com/portfolio/balance"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kalshi-trading-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getorder(order_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint for getting a single order."
    order_id: Order_id input for the current order.
        
    """
    url = f"https://kalshi-trading-api.p.rapidapi.com/portfolio/orders/{order_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kalshi-trading-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfills(cursor: str=None, ticker: str=None, order_id: str=None, max_ts: int=None, limit: int=None, min_ts: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint for getting all fills for the logged-in member."
    cursor: The Cursor represents a pointer to the next page of records in the pagination.
So this optional parameter, when filled, should be filled with the cursor string returned in a previous request to this end-point.
Filling this would basically tell the api to get the next page containing the number of records passed on the limit parameter.
On the other side not filling it tells the api you want to get the first page for another query.
The cursor does not store any filters, so if any filter parameters like ticker, max_ts or min_ts were passed in the original query they must be passed again.
        ticker: Restricts the response to trades in a specific market.
        order_id: Restricts the response to trades related to a specific order.
        max_ts: Restricts the response to trades before a timestamp.
        limit: Parameter to specify the number of results per page. Defaults to 100.
        min_ts: Restricts the response to trades after a timestamp.
        
    """
    url = f"https://kalshi-trading-api.p.rapidapi.com/portfolio/fills"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if ticker:
        querystring['ticker'] = ticker
    if order_id:
        querystring['order_id'] = order_id
    if max_ts:
        querystring['max_ts'] = max_ts
    if limit:
        querystring['limit'] = limit
    if min_ts:
        querystring['min_ts'] = min_ts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kalshi-trading-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

