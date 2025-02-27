import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def balance_sheet_statement(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get company balance sheet statement by year.
		
		All numbers in thousands."
    
    """
    url = f"https://financial-statements.p.rapidapi.com/api/v1/resources/balance-sheet"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-statements.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cash_flow_statement(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get company cash flow statement by year (ttm = Trailing Twelve Months).
		
		All numbers in thousands."
    
    """
    url = f"https://financial-statements.p.rapidapi.com/api/v1/resources/cash-flow"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-statements.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def income_statement(ticker: str='AAPL', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get company income statement by year (ttm = Trailing Twelve Months).
		
		All numbers in thousands."
    
    """
    url = f"https://financial-statements.p.rapidapi.com/api/v1/resources/income-statement"
    querystring = {}
    if ticker:
        querystring['ticker'] = ticker
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-statements.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

