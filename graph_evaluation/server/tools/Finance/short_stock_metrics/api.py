import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def short_volume_specific_date(date: str, ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the short volume for a specific date for a ticker."
    
    """
    url = f"https://short-stock-metrics.p.rapidapi.com/get-specific-date"
    querystring = {'date': date, 'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "short-stock-metrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def summary_stats(enddate: str, startdate: str, ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get summary stats about short positions for a specific period (e.g., average short percentage, standard deviation, etc)."
    startdate: You can go far back as 2018-11-01
        
    """
    url = f"https://short-stock-metrics.p.rapidapi.com/get-summary-stats-for-date-range"
    querystring = {'endDate': enddate, 'startDate': startdate, 'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "short-stock-metrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

