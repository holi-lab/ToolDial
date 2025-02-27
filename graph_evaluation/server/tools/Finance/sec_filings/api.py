import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def trending_stocks(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the top  trending stocks on Last10K.com right now with the most viewed <strong>10-K / 20-F / 40-F Annual Reports</strong> and <strong>10-Q Quarterly SEC Filings</strong>.
		
		View live data at: https://last10k.com/stock-screeners/trending"
    
    """
    url = f"https://last10k-company-v1.p.rapidapi.com/v1/app/analytics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "last10k-company-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def items(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the content of each item in an entire 10-K Annual Report"
    
    """
    url = f"https://last10k-company-v1.p.rapidapi.com/v1/company/items"
    querystring = {'ticker': ticker, }
    if ticker:
        querystring['ticker'] = ticker
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "last10k-company-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def operations(ticker: str, filingorder: str='0', formtype: str='10-k', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a company's most recent statement of operations showing how they balance costs with revenue"
    ticker: Publicly traded company's stock symbol
        filingorder: Retrieve an older financial statement by specifying number of filings to go back. For example, to retrieve the second to most recent filing, specify a filingOrder value of 1. By default, the most recent filing is returned (filingOrder = 0)
        formtype: 10-K = Annual Report (default) | 10-Q = Quarterly Report
        
    """
    url = f"https://last10k-company-v1.p.rapidapi.com/v1/company/operations"
    querystring = {'ticker': ticker, }
    if filingorder:
        querystring['filingOrder'] = filingorder
    if formtype:
        querystring['formType'] = formtype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "last10k-company-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def income(ticker: str, formtype: str='10-k', filingorder: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a company's most recent income statement detailing what they earn for providing their goods and services OR a statement of loss explaining where lost income occurred"
    ticker: Publicly traded company's stock symbol
        formtype: 10K = Annual Report (default) | 10-Q = Quarterly Report
        filingorder: Retrieve an older financial statement by specifying number of filings to go back. For example, to retrieve the 10-K filed last year, specify a filingOrder value of 1. By default, the most recent filing is returned (filingOrder = 0)
        
    """
    url = f"https://last10k-company-v1.p.rapidapi.com/v1/company/income"
    querystring = {'ticker': ticker, }
    if formtype:
        querystring['formType'] = formtype
    if filingorder:
        querystring['filingorder'] = filingorder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "last10k-company-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def manager_ownership(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns institutional managers positions in a stock including the number of managers that sold, increased, decreased and added a stock to their portfolio by comparing current and previous 13F SEC Filings"
    ticker: Publicly traded company's stock symbol
        
    """
    url = f"https://last10k-company-v1.p.rapidapi.com/v1/company/manager-ownership"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "last10k-company-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def manager_holdings(cik: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an institutional manager's quarterly portfolio holdings including new, sold and changed positions by comparing current and previous 13F SEC Filings"
    cik: Central Index Key (CIK)
        
    """
    url = f"https://last10k-company-v1.p.rapidapi.com/v1/company/manager-holdings"
    querystring = {'cik': cik, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "last10k-company-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insider_trades(ticker: str='eeft', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an aggregated and summarized collection of the most recent BUY, SELL and OPTION EXERCISE insider trades filed with Form 4 SEC Filings.
		
		Visit [https://last10k.com/stock-screeners/insider-trading](https://last10k.com/stock-screeners/insider-trading) to view this data as a web page."
    ticker: Publicly traded company's stock symbol. If not specified in the request, the endpoint will return insider trades for all companies on the most recent trading day,
        
    """
    url = f"https://last10k-company-v1.p.rapidapi.com/v1/company/insider-transactions"
    querystring = {}
    if ticker:
        querystring['ticker'] = ticker
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "last10k-company-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def financial_statements_disclosures(cik: str, accessionnumber: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of financial statements and disclosures from a company's Annual or Quarterly report.
		<p><i>Use the CIK and Accession Number values returned from the <b>SEC Filings</b> endpoint for this endpoint's Request Parameters.</i></p>"
    cik: Central Index Key (CIK)
        accessionnumber: SEC Filing Identifier
        
    """
    url = f"https://last10k-company-v1.p.rapidapi.com/v1/company/sections"
    querystring = {'cik': cik, 'accessionNumber': accessionnumber, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "last10k-company-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cash_flows(ticker: str, filingorder: int=0, formtype: str='10-k', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a company's most recent cash flow statement showing the amount of cash used and generated"
    ticker: Publicly traded company's stock symbol
        filingorder: Retrieve an older financial statement by specifying number of filings to go back. For example, to retrieve the second to most recent filing, specify a filingOrder value of 1. By default, the most recent filing is returned (filingOrder = 0)
        formtype: 10-K = Annual Report (default) | 10-Q = Quarterly Report
        
    """
    url = f"https://last10k-company-v1.p.rapidapi.com/v1/company/cashflows"
    querystring = {'ticker': ticker, }
    if filingorder:
        querystring['filingOrder'] = filingorder
    if formtype:
        querystring['formType'] = formtype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "last10k-company-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sec_filings(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of SEC Filings for the requested company."
    identifier: Publicly traded company's stock symbol or Central Index Key (CIK)
        
    """
    url = f"https://last10k-company-v1.p.rapidapi.com/v1/company/filings"
    querystring = {'identifier': identifier, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "last10k-company-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def late_sec_filings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns recent 8-K Material Event SEC Filings made after the 5:30pm EST deadline on the previous filing day. The feed is updated live on Monday mornings because these filings do not become publicly available until Monday at 6am EST."
    
    """
    url = f"https://last10k-company-v1.p.rapidapi.com/v1/company/late-filings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "last10k-company-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def balance_sheet(ticker: str, formtype: str='10-k', filingorder: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a company's most recent balance sheet summarizing their liabilities, assets and shareholders' equity"
    ticker: Publicly traded company's stock symbol
        formtype: 10-K = Annual Report (default) | 10-Q = Quarterly Report
        filingorder: Retrieve an older financial statement by specifying number of filings to go back. For example, to retrieve the second to most recent filing, specify a filingOrder value of 1. By default, the most recent filing is returned (filingOrder = 0)
        
    """
    url = f"https://last10k-company-v1.p.rapidapi.com/v1/company/balancesheet"
    querystring = {'ticker': ticker, }
    if formtype:
        querystring['formtype'] = formtype
    if filingorder:
        querystring['filingorder'] = filingorder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "last10k-company-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

