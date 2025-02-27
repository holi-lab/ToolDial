import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_company_facts(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Complete Company Financial Statements Data"
    id: Company CIK Code
        
    """
    url = f"https://sec-edgar.p.rapidapi.com/Filing/GetCompanyFacts/"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sec-edgar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_specific_month_filings(effectivedate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Specific Month Filings"
    
    """
    url = f"https://sec-edgar.p.rapidapi.com/Filing/GetMonthFilings"
    querystring = {'effectiveDate': effectivedate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sec-edgar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_latest_filings_latest_200(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Latest 200 Filings as soon as they are available on the SEC EDGAR System."
    
    """
    url = f"https://sec-edgar.p.rapidapi.com/Filing/GetLatestFilings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sec-edgar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_latest_filings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get All Latest Filings as soon as they are available on the SEC EDGAR System."
    
    """
    url = f"https://sec-edgar.p.rapidapi.com/Filing/GetLatestFilings_ALL"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sec-edgar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_filings(is_id: str, fromdate: str='2019-12-31', todate: str='2020-06-30', code: str='10-K', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get List of Company, Institutional Owners, Company Insider Filings."
    id: This id is the Company's SEC EDGAR CIK Code.
        fromdate: Please note the Date Format: YYYY-MM-DD. 

Default date is Beginning of the Year, if no date is provided.
        todate: Please note the Date Format: YYYY-MM-DD. 

Default date is Previous Month End, if no date is provided.
        code: This code is the Filing Code from the SEC EDGAR System.

Default value is 10-K, 10-K/A, if no value is provided. 

Multiple filing codes are allowed. Multiple codes to be comma delimited.
        
    """
    url = f"https://sec-edgar.p.rapidapi.com/Filing/"
    querystring = {'id': is_id, }
    if fromdate:
        querystring['fromdate'] = fromdate
    if todate:
        querystring['todate'] = todate
    if code:
        querystring['code'] = code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sec-edgar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sec_edgar_cik_code_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to obtain list of Tickers and SEC EDGAR CIK Codes for Entities and Individuals. This makes it much more easier to allow you to obtain SEC EDGAR CIK Codes needed to retrieve filings."
    
    """
    url = f"https://sec-edgar.p.rapidapi.com/Company/GetTickers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sec-edgar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_specific_day_filings(effectivedate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Specific Day Filings - Within Last 5 Business Days"
    
    """
    url = f"https://sec-edgar.p.rapidapi.com/Filing/GetDayFilings"
    querystring = {'effectiveDate': effectivedate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sec-edgar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_balance_sheet_income_statement_cash_flow_statement_and_other_financial_statement_data(is_id: str, code: str, todate: str='2021-01-31', fromdate: str='2020-01-01', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Company's most recent Balance Sheet, Income Statement, Cash Flow Statement and other Financial Statement data."
    id: This id is the Company's SEC EDGAR CIK Code.
        code: This code is the Filing Code from the SEC EDGAR System.

Default value is 10-K, 10-K/A, if no value is provided. 

Multiple filing codes are allowed. Multiple codes to be comma delimited.
        todate: Please note the Date Format: YYYY-MM-DD. 

Default date is Previous Month End, if no date is provided.
        fromdate: Please note the Date Format: YYYY-MM-DD. 

Default date is Beginning of the Year, if no date is provided.
        
    """
    url = f"https://sec-edgar.p.rapidapi.com/Filing/ExtractFilingData"
    querystring = {'id': is_id, 'code': code, }
    if todate:
        querystring['todate'] = todate
    if fromdate:
        querystring['fromdate'] = fromdate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sec-edgar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_company_insiders_filings(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get List of Company Insiders so you are able to much easily obtain their filings using the SEC EDGAR CIK Codes. 
		
		Company Insiders filings gives special insights to the transactions of Company Insiders like the CEO, CFO and other C-Suite Executives."
    id: This id parameter is the Company's CIK Code.
        
    """
    url = f"https://sec-edgar.p.rapidapi.com/Insiders"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sec-edgar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_company_institutional_owners_filings(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get List of Institutional Owners so you are able to much easily obtain their filings using the SEC EDGAR CIK Codes. 
		
		Institutional Owners filings gives special insights to the transactions of Institutional Owners."
    id: This id parameter is the Company's CIK Code.
        
    """
    url = f"https://sec-edgar.p.rapidapi.com/Issuers"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sec-edgar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

