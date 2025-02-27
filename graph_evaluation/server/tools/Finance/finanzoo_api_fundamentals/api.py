import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_dividend(is_from: str, shareuid: int, api_dividend: bool, to: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Contains the annual dividends (split-adjusted without special dividend)
		
		API_Dividend
		Syntax:
		
		shareuid = unique identifier for the share searched
		from = Start date of the searched period in American notation year-month-day with leading 0
		to = End date of the searched period in American notation year-month-day with leading 0"
    
    """
    url = f"https://finanzoo-api_fundamentals.p.rapidapi.com/"
    querystring = {'from': is_from, 'shareuid': shareuid, 'API_Dividend': api_dividend, 'to': to, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finanzoo-api_fundamentals.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_index_isin(api_index: bool, isin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Determine shareuid
		To query the fundamental data via our API, you need the Shareuid for the share you are looking for. You can either determine this yourself via the API, or take it from the Excel table.
		IMPORTANT! The currency that is output in the API_Fundamentals results is also included here.
		
		Syntax:
		
		isin = search shareuid with ISIN"
    
    """
    url = f"https://finanzoo-api_fundamentals.p.rapidapi.com/"
    querystring = {'API_Index': api_index, 'isin': isin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finanzoo-api_fundamentals.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_index_name(name: str, api_index: bool, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Determine shareuid
		To query the fundamental data via our API, you need the Shareuid for the share you are looking for. You can either determine this yourself via the API, or take it from the Excel table.
		IMPORTANT! The currency that is output in the API_Fundamentals results is also included here.
		
		Syntax:
		
		name = search shareuid with company name"
    
    """
    url = f"https://finanzoo-api_fundamentals.p.rapidapi.com/"
    querystring = {'name': name, 'API_Index': api_index, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finanzoo-api_fundamentals.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_index_wkn(wkn: str, api_index: bool, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Determine shareuid
		To query the fundamental data via our API, you need the Shareuid for the share you are looking for. You can either determine this yourself via the API, or take it from the Excel table.
		IMPORTANT! The currency that is output in the API_Fundamentals results is also included here.
		
		Syntax:
		
		wkn = search shareuid with WKN"
    
    """
    url = f"https://finanzoo-api_fundamentals.p.rapidapi.com/"
    querystring = {'wkn': wkn, 'API_Index': api_index, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finanzoo-api_fundamentals.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_pricetobook(api_pricetobook: bool, shareuid: int, to: str, is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "contains the price-book ratio (P / B ratio) for each past trading day"
    
    """
    url = f"https://finanzoo-api_fundamentals.p.rapidapi.com/"
    querystring = {'API_Pricetobook': api_pricetobook, 'shareuid': shareuid, 'to': to, 'from': is_from, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finanzoo-api_fundamentals.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_pricetocashflow(to: str, is_from: str, shareuid: int, api_pricetocashflow: bool, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Contains the price-cash-flow ratio (P / C ratio) for each past trading day
		
		API_Pricetocashflow
		Syntax:
		 
		
		shareuid = unique identifier for the share searched
		from = Start date of the searched period in American notation year-month-day with leading 0
		to = End date of the searched period in American notation year-month-day with leading 0"
    
    """
    url = f"https://finanzoo-api_fundamentals.p.rapidapi.com/"
    querystring = {'to': to, 'from': is_from, 'shareuid': shareuid, 'API_Pricetocashflow': api_pricetocashflow, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finanzoo-api_fundamentals.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_pricetoearning(to: str, api_pricetoearning: bool, is_from: str, shareuid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Contains the price-earnings ratio (P / E ratio) for each past trading day
		
		API_Pricetoearning
		Syntax:
		 
		
		shareuid = unique identifier for the share searched
		from = Start date of the searched period in American notation year-month-day with leading 0
		to = End date of the searched period in American notation year-month-day with leading 0"
    
    """
    url = f"https://finanzoo-api_fundamentals.p.rapidapi.com/"
    querystring = {'to': to, 'API_Pricetoearning': api_pricetoearning, 'from': is_from, 'shareuid': shareuid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finanzoo-api_fundamentals.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_pricetosales(shareuid: int, to: str, api_pricetosales: bool, is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Contains the price turnover ratio (P / S ratio) for each past trading day.
		
		API_Pricetosales
		Syntax:
		 
		
		shareuid = unique identifier for the share searched
		from = Start date of the searched period in American notation year-month-day with leading 0
		to = End date of the searched period in American notation year-month-day with leading 0"
    
    """
    url = f"https://finanzoo-api_fundamentals.p.rapidapi.com/"
    querystring = {'shareuid': shareuid, 'to': to, 'API_Pricetosales': api_pricetosales, 'from': is_from, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finanzoo-api_fundamentals.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_roa(api_roa: bool, to: str, is_from: str, shareuid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Contains the profitability (ROA ratio) for the financial year.
		
		API_Roa
		Syntax:
		 
		
		shareuid = unique identifier for the share searched
		from = Start date of the searched period in American notation year-month-day with leading 0
		to = End date of the searched period in American notation year-month-day with leading 0"
    
    """
    url = f"https://finanzoo-api_fundamentals.p.rapidapi.com/"
    querystring = {'API_Roa': api_roa, 'to': to, 'from': is_from, 'shareuid': shareuid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finanzoo-api_fundamentals.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_roe(is_from: str, shareuid: int, to: str, api_roe: bool, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Contains the equity ratio (ROE ratio) for the financial year.
		
		API_Roe
		Syntax:
		 
		
		shareuid = unique identifier for the share searched
		from = Start date of the searched period in American notation year-month-day with leading 0
		to = End date of the searched period in American notation year-month-day with leading 0"
    
    """
    url = f"https://finanzoo-api_fundamentals.p.rapidapi.com/"
    querystring = {'from': is_from, 'shareuid': shareuid, 'to': to, 'API_Roe': api_roe, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finanzoo-api_fundamentals.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_pricetofreecashflow(api_key: str, api_pricetofreecashflow: bool, shareuid: int, is_from: str, to: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Contains the price-free-cash-flow ratio (P / FC ratio) for each past trading day.
		
		API_Pricetofreecashflow
		Syntax:
		 
		
		shareuid = unique identifier for the share searched
		from = Start date of the searched period in American notation year-month-day with leading 0
		to = End date of the searched period in American notation year-month-day with leading 0"
    
    """
    url = f"https://finanzoo-api_fundamentals.p.rapidapi.com/"
    querystring = {'API_key': api_key, 'API_Pricetofreecashflow': api_pricetofreecashflow, 'shareuid': shareuid, 'from': is_from, 'to': to, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finanzoo-api_fundamentals.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_fundamentals(is_from: str, to: str, api_fundamentals: bool, shareuid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "contains fundamental data from the annual reports:
		
		- Diluted earnings per share
		- total capital
		- equity
		- net operating cash flow
		- capital expenditure
		- free cash flow
		- Profit loss
		- Shares outstanding
		- Net sales
		
		Determine shareuid
		To query the fundamental data via our API, you need the Shareuid for the share you are looking for. You can either determine this yourself via the API, or take it from the Excel table.
		IMPORTANT! The currency that is output in the API_Fundamentals results is also included here.
		
		Syntax:
		
		name = search shareuid with company name
		wkn = search shareuid with WKN
		isin = search shareuid with ISIN
		token = your Finanzoo token (https://www.finanzoo.de/account/api.html) -> You need a premium account to use.
		
		 
		
		Example calls:
		
		https://api.finanzoo.de/v1/public/?API_Index&name=ibm&token=YOURTOKEN
		https://api.finanzoo.de/v1/public/?API_Index&wkn=851399&token=YOURTOKEN
		https://api.finanzoo.de/v1/public/?API_Index&isin=US4592001014&token=YOURTOKEN
		
		Example answer:
		
		{"shareuid":"2","isin":"US4592001014 ","wkn":"851399","currency":"USD","shortname":"IBM","name":"International Business Machines Corp.","country":"Vereinigte Staaten von Amerika","indizes":"Dow Jones Industrial Average,S&P 500","sector":"Technology","lastbalanceupdate":"2020-01-26"}
		
		Shareuid Excel Table
		Find the shareuid for your query.
		[https://www.finanzoo.de/en/fundamental-api.html](url)
		is required for the API queries"
    
    """
    url = f"https://finanzoo-api_fundamentals.p.rapidapi.com/"
    querystring = {'from': is_from, 'to': to, 'API_Fundamentals': api_fundamentals, 'shareuid': shareuid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finanzoo-api_fundamentals.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

