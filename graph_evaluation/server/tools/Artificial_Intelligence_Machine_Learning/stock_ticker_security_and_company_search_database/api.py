import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def security_price_volume(date: str, ticker: str, past_days: str='2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns end of day Price and Volume for a security"
    date: It needs a date from which the historical price and volume will be provided.
If on that date there is an exchange holiday or data is not available, then it will return previous days data
        ticker: The ticker should having following pattern:
<Exchange_Code>:<Security_Ticker>
        
    """
    url = f"https://stock-ticker-security-and-company-search-database.p.rapidapi.com/security_price_volume"
    querystring = {'date': date, 'ticker': ticker, }
    if past_days:
        querystring['past_days'] = past_days
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-ticker-security-and-company-search-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sp500_active_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all the SP500 active companies"
    
    """
    url = f"https://stock-ticker-security-and-company-search-database.p.rapidapi.com/sp500_active_list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-ticker-security-and-company-search-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_search_get_result_by_ticker_or_security_term(security: str='Apple Inc', ticker: str='AAPL_sdfs', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "In a single call search by ticker OR security_term"
    
    """
    url = f"https://stock-ticker-security-and-company-search-database.p.rapidapi.com/all_search"
    querystring = {}
    if security:
        querystring['security'] = security
    if ticker:
        querystring['ticker'] = ticker
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-ticker-security-and-company-search-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def company_sentiment(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extract securities and sentiment towards those securities from a free form text using an advanced AI"
    
    """
    url = f"https://stock-ticker-security-and-company-search-database.p.rapidapi.com/company_sentiment"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-ticker-security-and-company-search-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sp500_search_by_ticker(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check if security is currently listed in the S&P 500 by Ticker (or Symbol)"
    
    """
    url = f"https://stock-ticker-security-and-company-search-database.p.rapidapi.com/sp500_search"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-ticker-security-and-company-search-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_search_by_ticker(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search security by ticker
		
		You can also input different exchanges, for example:
		
		```
		ticker=NASDAQ:AAPL
		ticker=NYSE:BA
		```"
    
    """
    url = f"https://stock-ticker-security-and-company-search-database.p.rapidapi.com/all_search"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-ticker-security-and-company-search-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sp500_search_by_security_term(security: str, exchange: str='NASDAQ', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search security by providing the first few words for the security or the company name"
    
    """
    url = f"https://stock-ticker-security-and-company-search-database.p.rapidapi.com/sp500_search"
    querystring = {'security': security, }
    if exchange:
        querystring['exchange'] = exchange
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-ticker-security-and-company-search-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_search_by_exchange_and_ticker(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search security by ticker
		
		You can also input different exchanges, for example:
		
		```
		ticker=NASDAQ:AAPL
		ticker=NYSE:BA
		```"
    
    """
    url = f"https://stock-ticker-security-and-company-search-database.p.rapidapi.com/all_search"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-ticker-security-and-company-search-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_search_by_security_term(security: str, exchange: str='NASDAQ', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search security by providing the first few words for the security or the company name"
    
    """
    url = f"https://stock-ticker-security-and-company-search-database.p.rapidapi.com/all_search"
    querystring = {'security': security, }
    if exchange:
        querystring['exchange'] = exchange
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-ticker-security-and-company-search-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_search_get_single_security_by_term_search(security: str, result_type: str='short', exchange: str='NASDAQ', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search security by providing the first few words for the security or the company name
		
		Use optional  query param 
		`result_type=short`, which will return the top-most similar only one security, which matches your search term.
		Otherwise, it will return all the security, which starts with the term you provided."
    
    """
    url = f"https://stock-ticker-security-and-company-search-database.p.rapidapi.com/all_search"
    querystring = {'security': security, }
    if result_type:
        querystring['result_type'] = result_type
    if exchange:
        querystring['exchange'] = exchange
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-ticker-security-and-company-search-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

