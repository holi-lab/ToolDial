import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def spac_top_10_rankings(sortby: str, period: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Top-10 gainers, losers and volume leaders for a given period and equity type (i.e., common, unit, warrants). 
		
		Accepts three query parameters:
		
		**period: **daily, weekly, monthly
		**type:** common, unit, warrant
		**sortby:** gainers, losers, volume"
    
    """
    url = f"https://spachero-spac-database.p.rapidapi.com/top10/"
    querystring = {'sortby': sortby, 'period': period, 'type': type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spachero-spac-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def spac_summary(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve SPAC summary including important dates, status and target names."
    
    """
    url = f"https://spachero-spac-database.p.rapidapi.com/summary/{symbol}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spachero-spac-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_spac_deals_ipos_closings(event: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of latest SPAC events, such as deals, ipos, closings, rumors.
		
		**Available options are: ** ipo, closings, deals, rumor"
    
    """
    url = f"https://spachero-spac-database.p.rapidapi.com/news/latest/"
    querystring = {'event': event, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spachero-spac-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def spac_calendar(type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get important SPAC dates such as merger meetings dates, warrant redemption deadlines, definitive agreement dates, unit split (estimate), SPAC deadline (estimate).
		
		Available options: merger, split, deadline, redemption"
    
    """
    url = f"https://spachero-spac-database.p.rapidapi.com/calendar/"
    querystring = {'type': type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spachero-spac-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def spac_sec_filings(symbol: str='ITAC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest SPAC SEC filings, filter by SPAC symbol."
    
    """
    url = f"https://spachero-spac-database.p.rapidapi.com/secfilings/latest/"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spachero-spac-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def spac_price_targets(symbol: str='SOFI', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest SPAC price targets from analysts, filter by SPAC symbol."
    
    """
    url = f"https://spachero-spac-database.p.rapidapi.com/pricetargets/latest/"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spachero-spac-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

