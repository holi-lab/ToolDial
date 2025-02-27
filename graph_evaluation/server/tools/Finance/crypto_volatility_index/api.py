import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tick(currency: str, datetime: str='2021-10-01-10-31-45', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find index value at a certain point in time. Returns the closest value prior to specified datetime.
		
		If no datetime is specified, returns the latest available index value.
		
		In case if the request is prior to the first publication date, returns the oldest index value."
    
    """
    url = f"https://crypto-volatility-index.p.rapidapi.com/tick/{currency}/{datetime}"
    querystring = {}
    if datetime:
        querystring['datetime'] = datetime
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-volatility-index.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily_snapshot(currency: str, snaptime: str='17-00-00', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Daily index snapshots at a custom time, full history. 
		Default is midnight UTC."
    
    """
    url = f"https://crypto-volatility-index.p.rapidapi.com/history_EOD/{currency}/{snaptime}"
    querystring = {}
    if snaptime:
        querystring['snaptime'] = snaptime
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-volatility-index.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def period(currency: str, frequency: str='minute', datetime_end: str='2021-10-05-05-10-00', datetime_start: str='2021-10-05-05-00-00', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get index values with specified frequency, for a specified period. 
		
		Available frequencies and period limits: 
		day - snapped at midnight UTC, unlimited
		hour - 30 days
		minute - 24 hours
		tick (default) - 10 hours (every 15 seconds) 
		
		If no time is specified, returns maximum period for this frequency, looking backwards from now.
		If only starting time is specified, returns the maximal allowed period for this frequency starting at that time."
    
    """
    url = f"https://crypto-volatility-index.p.rapidapi.com/history/{currency}/{frequency}/{datetime_start}/{datetime_end}"
    querystring = {}
    if frequency:
        querystring['frequency'] = frequency
    if datetime_end:
        querystring['datetime_end'] = datetime_end
    if datetime_start:
        querystring['datetime_start'] = datetime_start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-volatility-index.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

