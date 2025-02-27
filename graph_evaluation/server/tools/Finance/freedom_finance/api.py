import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def live_stock_statistics(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the real time statistics about the stock (market cap, etc)"
    
    """
    url = f"https://freedom-finance.p.rapidapi.com/stock-statistics"
    querystring = {'Symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "freedom-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_stock_metadata(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the real time metadata about the stock  (currency, trend averages etc)"
    
    """
    url = f"https://freedom-finance.p.rapidapi.com/stock-metadata"
    querystring = {'Symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "freedom-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historic_stock_prices(startdateinclusive: str, enddateinclusive: str, symbol: str, orderby: str='Ascending', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all end of day quotes (open, close, high, low, volume etc) for given stock symbol"
    
    """
    url = f"https://freedom-finance.p.rapidapi.com/stock-prices"
    querystring = {'StartDateInclusive': startdateinclusive, 'EndDateInclusive': enddateinclusive, 'Symbol': symbol, }
    if orderby:
        querystring['OrderBy'] = orderby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "freedom-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quarterly_cashflow_statements(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Quarterly cashflow statements"
    
    """
    url = f"https://freedom-finance.p.rapidapi.com/financialstatements/quarterly-cashflow"
    querystring = {'Symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "freedom-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quarterly_balance_sheets(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Quarterly balance sheet statements"
    
    """
    url = f"https://freedom-finance.p.rapidapi.com/financialstatements/quarterly-balancesheet"
    querystring = {'Symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "freedom-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quarterly_income_statements(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Quarterly income statements"
    
    """
    url = f"https://freedom-finance.p.rapidapi.com/financialstatements/quarterly-income"
    querystring = {'Symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "freedom-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def yearly_balance_sheets(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Annual balance sheet statements"
    
    """
    url = f"https://freedom-finance.p.rapidapi.com/financialstatements/annual-balancesheet"
    querystring = {'Symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "freedom-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def yearly_income_statements(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Annual income statements"
    
    """
    url = f"https://freedom-finance.p.rapidapi.com/financialstatements/annual-income"
    querystring = {'Symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "freedom-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def yearly_cashflow_statements(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Annual cashflow statements"
    
    """
    url = f"https://freedom-finance.p.rapidapi.com/financialstatements/annual-cashflow"
    querystring = {'Symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "freedom-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_splits(symbol: str, orderby: str='Ascending', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all stock splits for given stock symbol"
    
    """
    url = f"https://freedom-finance.p.rapidapi.com/stock-splits"
    querystring = {'Symbol': symbol, }
    if orderby:
        querystring['OrderBy'] = orderby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "freedom-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dividends(symbol: str, orderby: str='Ascending', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download dividend history by specific stock symbol"
    
    """
    url = f"https://freedom-finance.p.rapidapi.com/dividends"
    querystring = {'Symbol': symbol, }
    if orderby:
        querystring['OrderBy'] = orderby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "freedom-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def companies_by_exchange(exchangecode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convenient User-Friendly Manually Populated List Of Common Stocks Per Exchange Code. Not Guaranteed To Be Up To Date."
    
    """
    url = f"https://freedom-finance.p.rapidapi.com/companies/list-by-exchange"
    querystring = {'ExchangeCode': exchangecode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "freedom-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_exchanges(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all exchanges known to have stocks associated with them"
    
    """
    url = f"https://freedom-finance.p.rapidapi.com/exchanges"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "freedom-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

