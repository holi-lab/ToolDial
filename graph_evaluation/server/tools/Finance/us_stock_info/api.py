import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_balance_sheet(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns the annual and annual balance sheet statements for the company of interest. Append the ticker of the company of interest along with your Rapid API key to the endpoint.
		
		**Example:**
		
		> https://us-stock-info.p.rapidapi.com/api/v1/getBalanceSheetData/AAPL?rapidapi-key=YOUR_RAPID_API_KEY"
    
    """
    url = f"https://us-stock-info.p.rapidapi.com/api/v1/getBalanceSheetData/{ticker}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-stock-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_income_statement(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns the annual and annual income statements for the company of interest. Append the ticker of the company of interest along with your Rapid API key to the endpoint.
		
		**Example:**
		
		> https://us-stock-info.p.rapidapi.com/api/v1/getIncomeStatementData/AAPL?rapidapi-key=YOUR_RAPID_API_KEY"
    
    """
    url = f"https://us-stock-info.p.rapidapi.com/api/v1/getIncomeStatementData/{ticker}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-stock-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_cash_flow_statement(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns the annual and annual cash flow statements for the company of interest. Append the ticker of the company of interest along with your Rapid API key to the endpoint.
		
		**Example:**
		
		> https://us-stock-info.p.rapidapi.com/api/v1/getCashFlowData/AAPL?rapidapi-key=YOUR_RAPID_API_KEY"
    
    """
    url = f"https://us-stock-info.p.rapidapi.com/api/v1/getCashFlowData/{ticker}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-stock-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_industry_info_list(exchange: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tickers by each industry of a US exchanges. 
		
		exchanges: NASDAQ, NYSE, AMEX, OTC"
    
    """
    url = f"https://us-stock-info.p.rapidapi.com/api/v1/getIndustryInfoList/{exchange}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-stock-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_company_info(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get company info for ticker"
    
    """
    url = f"https://us-stock-info.p.rapidapi.com/api/v1/getCompanyInfo/{ticker}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-stock-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_industry_list(exchange: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get industry list for a US exchange. 
		
		exchanges: NASDAQ, NYSE, AMEX, OTC"
    
    """
    url = f"https://us-stock-info.p.rapidapi.com/api/v1/getIndustryList/{exchange}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-stock-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sector_info_list(exchange: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tickers by each sectors of a US exchanges. 
		
		exchanges: NASDAQ, NYSE, AMEX, OTC"
    
    """
    url = f"https://us-stock-info.p.rapidapi.com/api/v1/getSectorInfoList/{exchange}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-stock-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sector_list(exchange: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get sector list for a US exchange. 
		
		exchanges: NASDAQ, NYSE, AMEX, OTC"
    
    """
    url = f"https://us-stock-info.p.rapidapi.com/api/v1/getSectorList/{exchange}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-stock-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_quarterly_cash_flow_statement(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns the annual and quarterly cash flow statements for the company of interest. Append the ticker of the company of interest along with your Rapid API key to the endpoint.
		
		**Example:**
		
		> https://us-stock-info.p.rapidapi.com/api/v1/getCashFlowData/quarterly/AAPL?rapidapi-key=YOUR_RAPID_API_KEY"
    
    """
    url = f"https://us-stock-info.p.rapidapi.com/api/v1/getCashFlowData/quarterly/{ticker}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-stock-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_quarterly_balance_sheet(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns the annual and quarterly balance sheet statements for the company of interest. Append the ticker of the company of interest along with your Rapid API key to the endpoint.
		
		**Example:**
		
		> https://us-stock-info.p.rapidapi.com/api/v1/getBalanceSheetData/quarterly/AAPL?rapidapi-key=YOUR_RAPID_API_KEY"
    
    """
    url = f"https://us-stock-info.p.rapidapi.com/api/v1/getBalanceSheetData/quarterly/{ticker}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-stock-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_quarterly_income_statement(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns the annual and quarterly income statements for the company of interest. Append the ticker of the company of interest along with your Rapid API key to the endpoint.
		
		**Example:**
		
		> https://us-stock-info.p.rapidapi.com/api/v1/getIncomeStatementData/quarterly/AAPL?rapidapi-key=YOUR_RAPID_API_KEY"
    
    """
    url = f"https://us-stock-info.p.rapidapi.com/api/v1/getIncomeStatementData/quarterly/{ticker}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-stock-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

