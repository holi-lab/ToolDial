import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_api_sentiments_all(period: str, date: str=None, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the raw bullish and bearish values for all stocks and specific date aggregated by period. Defaults to most recent data available for your API access level."
    period: Period of sentiments aggregation. It can be d for daily values or 2 for 2 minutes values. (Required)
        date: Specific date of the request. The date can be formatted as YYYY-MM-DD. Date can be modified depending on your API access level. (Optional)
        format: Output format. Accepted output formats are json or csv. Defaults to JSON (Optional)
        
    """
    url = f"https://mashaper-psychsignal.p.rapidapi.com/sentiments/all"
    querystring = {'period': period, }
    if date:
        querystring['date'] = date
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashaper-psychsignal.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_api_alerts_stream(max_id: str=None, limit: str=None, symbol: str=None, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns paginated alerts triggered on the system from most recent to oldest. It can be stock specific and it is limited to 100 alerts per request. It's also limited by your API level access."
    max_id: Alert ID. Alerts returned will be older than the specified id. (Optional)
        limit: Number of alerts to be returned. Max and default to 100 alerts. (Optional)
        symbol: Stock symbol. Examples: AAPL, IBM, FB. For the aggregate of all market sentiment, use MARKET as symbol. (Optional)
        format: Output format. Accepted output formats are json or csv. Defaults to JSON (Optional)
        
    """
    url = f"https://mashaper-psychsignal.p.rapidapi.com/alerts-stream"
    querystring = {}
    if max_id:
        querystring['max_id'] = max_id
    if limit:
        querystring['limit'] = limit
    if symbol:
        querystring['symbol'] = symbol
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashaper-psychsignal.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_api_ratio(symbol: str=None, is_from: str=None, to: str=None, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns daily bullish / bearish messages volume ratio for a specific symbol. Defaults to most recent data available for your API access level."
    symbol: Stock symbol. Examples: AAPL, IBM, FB. For the aggregate of all market sentiment, use MARKET as symbol. Defaults to MARKET. (optional)
        is_from: Start date of the returned range. The date must be formatted as YYYY-MM-DD. Range can be modified depending on your API access level. (Optional)
        to: End date of the returned range. The date must be formatted as YYYY-MM-DD. Range can be modified depending on your API access level. (Optional)
        format: Output format. Accepted output formats are json or csv. Defaults to JSON (Optional)
        
    """
    url = f"https://mashaper-psychsignal.p.rapidapi.com/ratio"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    if is_from:
        querystring['from'] = is_from
    if to:
        querystring['to'] = to
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashaper-psychsignal.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_api_ratio_all(date: str=None, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns daily bullish / bearish messages volume ratio for all symbols. Defaults to most recent data available for your API access level."
    date: Specific date of the request. The date can be formatted as YYYY-MM-DD. Date can be modified depending on your API access level. (Optional)
        format: Output format. Accepted output formats are json or csv. Defaults to JSON (Optional)
        
    """
    url = f"https://mashaper-psychsignal.p.rapidapi.com/ratio/all"
    querystring = {}
    if date:
        querystring['date'] = date
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashaper-psychsignal.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_api_sentiments(symbol: str, period: str, is_from: str=None, to: str=None, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the raw bullish and bearish values for a specific stock and specified date range aggregated by period. Defaults to most recent data available for your API access level."
    symbol: Stock symbol. Examples: AAPL, IBM, FB. For the aggregate of all market sentiment, use MARKET as symbol. (Required)
        period: Period of sentiments aggregation. It can be d for daily values or 2 for 2 minutes values. (Required)
        is_from: Start date of the returned range. The date can be formatted as YYYY-MM-DD or YYYY-MM-DD HH:MM:SS. Range can be modified depending on your API access level. (Optional)
        to: End date of the returned range. The date can be formatted as YYYY-MM-DD or YYYY-MM-DD HH:MM:SS. Range can be modified depending on your API access level. (Optional)
        format: Output format. Accepted output formats are json or csv. Defaults to JSON (Optional)
        
    """
    url = f"https://mashaper-psychsignal.p.rapidapi.com/sentiments"
    querystring = {'symbol': symbol, 'period': period, }
    if is_from:
        querystring['from'] = is_from
    if to:
        querystring['to'] = to
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashaper-psychsignal.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_api_alerts(is_from: str=None, to: str=None, symbol: str=None, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns alerts triggered on the system for the requested date range. Defaults to most recent data available for your API access level. It can be stock specific and the date range is limited to 30 days."
    is_from: Start date of the returned range. The date can be formatted as YYYY-MM-DD or YYYY-MM-DD HH:MM:SS. Range can be modified depending on your API access level. (Optional)
        to: End date of the returned range. The date can be formatted as YYYY-MM-DD or YYYY-MM-DD HH:MM:SS. Range can be modified depending on your API access level. (Optional)
        symbol: Stock symbol. Examples: AAPL, IBM, FB. For the aggregate of all market sentiment, use MARKET as symbol. (Optional)
        format: Output format. Accepted output formats are json or csv. Defaults to JSON (Optional)
        
    """
    url = f"https://mashaper-psychsignal.p.rapidapi.com/alerts"
    querystring = {}
    if is_from:
        querystring['from'] = is_from
    if to:
        querystring['to'] = to
    if symbol:
        querystring['symbol'] = symbol
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashaper-psychsignal.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

