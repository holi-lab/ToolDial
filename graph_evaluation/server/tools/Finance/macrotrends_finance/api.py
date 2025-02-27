import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def history_prices(range: str='1y', symbol: str='TSLA', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This gives you the Date, Open, High, Low, Close, Adj Close, Volume prices of the stock
		
		QUERY PARAMETER: symbol the ticker symbol of the company you want to see.
		
		QUERY PARAMETER: range = ['1y', '5y', 'max']"
    
    """
    url = f"https://macrotrends-finance.p.rapidapi.com/quotes/history-price"
    querystring = {}
    if range:
        querystring['range'] = range
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "macrotrends-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def financial_cash_statement(symbol: str, formstyle: str='dataframe', freq: str='Q', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives 10 years data of cash statement by the given ticker symbol.
		
		To choose for a time format put a QUERY PARAMETER (freq) either "A" for annual or "Q" for quarterly
		
		(symbol) To choose what ticker symbol to see the company financials."
    
    """
    url = f"https://macrotrends-finance.p.rapidapi.com/statements/cash"
    querystring = {'symbol': symbol, }
    if formstyle:
        querystring['formstyle'] = formstyle
    if freq:
        querystring['freq'] = freq
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "macrotrends-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def financial_income_statement(symbol: str, formstyle: str='dataframe', freq: str='Q', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives 10 years data of cash statement by the given ticker symbol.
		
		To choose for a time format put a QUERY PARAMETER (freq) either "A" for annual or "Q" for quarterly
		
		(symbol) To choose what ticker symbol to see the company financials."
    
    """
    url = f"https://macrotrends-finance.p.rapidapi.com/statements/income"
    querystring = {'symbol': symbol, }
    if formstyle:
        querystring['formstyle'] = formstyle
    if freq:
        querystring['freq'] = freq
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "macrotrends-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def financial_balance_sheet(symbol: str, formstyle: str='dataframe', freq: str='Q', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives 10 years data of cash statement by the given ticker symbol.
		
		To choose for a time format put a QUERY PARAMETER (freq) either "A" for annual or "Q" for quarterly
		
		(symbol) To choose what ticker symbol to see the company financials."
    
    """
    url = f"https://macrotrends-finance.p.rapidapi.com/statements/balance"
    querystring = {'symbol': symbol, }
    if formstyle:
        querystring['formstyle'] = formstyle
    if freq:
        querystring['freq'] = freq
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "macrotrends-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

