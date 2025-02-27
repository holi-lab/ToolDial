import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_events_using_get_method(month: str, day: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives the events and link related to the events. Needed Month and Day passed as params in url
		
		1) pass the valid day of month
		2) pass the month in as full name in lowercase  ex: "january""
    
    """
    url = f"https://history-events-of-a-day.p.rapidapi.com/api/getevents/{month}/{day}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "history-events-of-a-day.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

