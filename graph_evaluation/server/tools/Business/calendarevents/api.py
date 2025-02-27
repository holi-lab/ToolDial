import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def calendar_events(src: str, months: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve Google Calendar Events from a public calendar"
    src: Calendar ID (ending in `@group.calendar.google.com`).
You can find it in the `src` parameter of the embed URL
        months: Number of months of events to retireve.
Default is `2`. Max is `12`.
        
    """
    url = f"https://calendarevents.p.rapidapi.com/calendar/{src}/{months}"
    querystring = {}
    if months:
        querystring['months'] = months
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "calendarevents.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

