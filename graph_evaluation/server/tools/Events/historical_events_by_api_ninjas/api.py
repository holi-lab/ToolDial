import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_historicalevents(text: str='roman empire', month: int=None, day: int=None, year: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Historical Events API endpoint. Returns a list of up to 10 events that match the search parameters. At least one of the following parameters is required: text, year, month, day."
    text: query text to search events by. Use keywords or short phrases for best match results.
        month: integer month (e.g. 3 for March).
        day: calendar day of the month.
        year: 4-digit year (e.g. 1776). For BC/BCE years, use a negative integer (e.g. -351 for 351 BC).
        offset: number of results to offset (for pagination).
        
    """
    url = f"https://historical-events-by-api-ninjas.p.rapidapi.com/v1/historicalevents"
    querystring = {}
    if text:
        querystring['text'] = text
    if month:
        querystring['month'] = month
    if day:
        querystring['day'] = day
    if year:
        querystring['year'] = year
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "historical-events-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

