import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_cashflow_data(companyname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns the annual and annual cash flow statements for the company of interest. Append the ticker of the company of interest along with your Rapid API key to the endpoint
		
		**Example:**
		
		> https://indian-stock-info.p.rapidapi.com/api/v1/getCashflowData/ASHOKLEY.NS?rapidapi-key=YOUR_RAPID_API_KEY"
    
    """
    url = f"https://indian-stock-info.p.rapidapi.com/api/v1/getCashFlowData/{companyname}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indian-stock-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_balance_sheet(companyname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns the annual and annual balance sheet statements for the company of interest. Append the ticker of the company of interest along with your Rapid API key to the endpoint.
		
		**Example:**
		
		> https://indian-stock-info.p.rapidapi.com/api/v1/getBalanceSheetData/ASHOKLEY.NS?rapidapi-key=YOUR_RAPID_API_KEY"
    
    """
    url = f"https://indian-stock-info.p.rapidapi.com/api/v1/getBalanceSheetData/{companyname}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indian-stock-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_industry_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get industry list for a NSE."
    
    """
    url = f"https://indian-stock-info.p.rapidapi.com/api/v1/getIndustryList/NSE"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indian-stock-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sector_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get sector list for a NSE."
    
    """
    url = f"https://indian-stock-info.p.rapidapi.com/api/v1/getSectorList/NSE"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indian-stock-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_income_statement(companyname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns the annual and annual income statements for the company of interest. Append the ticker of the company of interest along with your Rapid API key to the endpoint.
		
		**Example:**
		
		> https://indian-stock-info.p.rapidapi.com/api/v1/getIncomeStatementData/ASHOKLEY.NS?rapidapi-key=YOUR_RAPID_API_KEY"
    
    """
    url = f"https://indian-stock-info.p.rapidapi.com/api/v1/getIncomeStatementData/{companyname}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indian-stock-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

