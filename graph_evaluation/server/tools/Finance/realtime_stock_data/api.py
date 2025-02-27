import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def asset_profile(symb: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Asset profiles provide information about each known asset in your network, including what services are running on each asset. Asset profile information is used for correlation purposes to help reduce false positives.
		this endpoint returns  **information about the company including companies physical address , contact information ,and number of full time employees and much more**"
    
    """
    url = f"https://realtime-stock-data.p.rapidapi.com/asset-profile/{symb}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtime-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def finance_analytics(symb: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint also returns **AI recommendation**
		Financial analysis refers to an assessment of the viability, stability, and profitability of a business, sub-business or project. It is performed by professionals who prepare reports using ratios and other techniques, that make use of information taken from financial statements and other reports."
    
    """
    url = f"https://realtime-stock-data.p.rapidapi.com/finance-analytics/{symb}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtime-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def balance_sheet(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A **balance sheet** is a financial statement that communicates the so-called “book value” of an organization, as calculated by subtracting all of the company's liabilities and shareholder equity from its total assets."
    
    """
    url = f"https://realtime-stock-data.p.rapidapi.com/balance-sheet/{symbol}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtime-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def earnings_trend(symbl: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This reflects the insurance company's profits over given periods of time.
		This endpoint also gives **experts** & **AI supported predictions** about stock future behavior"
    
    """
    url = f"https://realtime-stock-data.p.rapidapi.com/earnings-trend/{symbl}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtime-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historic_data(range: str, interval: str, symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return historic data
		interval - The time interval between two data points. Can be 1m 2m 5m 15m 30m 60m 90m 1h 1d 5d 1wk 1mo 3mo
		range - The range for which the data is returned."
    
    """
    url = f"https://realtime-stock-data.p.rapidapi.com/historic/{symbol}/{interval}/{range}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtime-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def earnings(symb: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A company's earnings are its after-tax net income. This is the company's bottom line or its profits. Earnings are perhaps the single most important and most closely studied number in a company's financial statements.
		This API **also** returns **historical earnings**"
    
    """
    url = f"https://realtime-stock-data.p.rapidapi.com/earnings/{symb}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtime-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def options(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Option chain data is **the complete picture pertaining to option strikes of a particular stock or index in a single frame**. In the Option chain frame, the strike price is at the center and all data pertaining to calls and puts on the same strike are presented next to each other."
    
    """
    url = f"https://realtime-stock-data.p.rapidapi.com/options/{symbol}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtime-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def price(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the Realtime Stock Price Information for the symbol passed as parameter"
    
    """
    url = f"https://realtime-stock-data.p.rapidapi.com/price/{symbol}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtime-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def index_trend(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Trend Analysis is a relatively straightforward quantitative demand forecasting technique that uses the historical relationship between the operational index (i.e. sales level) and the number of employees required by the organization (demand for labour) to forecast future requirements."
    
    """
    url = f"https://realtime-stock-data.p.rapidapi.com/index-trend/{symbol}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtime-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def esg_score(symb: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Simply explained, an ESG score is a measure of a company's exposure to long-term environmental, social, and governance risks that are often overlooked during traditional financial analyses."
    
    """
    url = f"https://realtime-stock-data.p.rapidapi.com/esg-score/{symb}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtime-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

