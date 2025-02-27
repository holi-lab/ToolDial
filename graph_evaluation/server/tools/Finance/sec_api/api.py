import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_fundamentals(ticker: str, is_from: str='2022-01-01', to: str='2022-06-01', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get company fundamentals data as listed in SEC filings from historic or current date ranges. Includes data for: 
		Symbol, Start, End, Filed, Form, Revenue, Net Income, Comprehensive Income, EPS, Diluted EPS, Shares, Diluted, Shares, Assets, Current Assets, Liabilities, Current Liabilities, Stockholders Equity, Operating Activities, Investing, Activities, Financing Activities."
    is_from: Earliest date of fundamentals company data requested. This is the start of the period being considered by the SEC filing and not the date on which it was filed.
        to: latest date of company fundamentals data being requested.
        
    """
    url = f"https://sec-api2.p.rapidapi.com/SEC_API"
    querystring = {'ticker': ticker, }
    if is_from:
        querystring['from'] = is_from
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sec-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

