import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def holidays(year: int, country: str, pretty: bool=None, day: str=None, format: str='json', public: bool=None, month: str=None, lang: str=None, previous: bool=None, upcoming: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get public holidays and observances list"
    year: 4-digit year (YYYY)
        country: ISO 3166-1 alpha-2 country code from Countries section.
        pretty: Output as human-readable text
        day: 1 or 2 digit day identifier (DD or D)
        format: Define return data format (available: json, xml)
        public: Return only public holidays
        month: 1 or 2 digit month identifier (MM or M)
        lang: Choose holiday name output language
        previous: Return only holidays before given date (default: 0)
        upcoming: Return only holidays after given date (default: 0)
        
    """
    url = f"https://holidayapi1.p.rapidapi.com/holidays"
    querystring = {'year': year, 'country': country, }
    if pretty:
        querystring['pretty'] = pretty
    if day:
        querystring['day'] = day
    if format:
        querystring['format'] = format
    if public:
        querystring['public'] = public
    if month:
        querystring['month'] = month
    if lang:
        querystring['lang'] = lang
    if previous:
        querystring['previous'] = previous
    if upcoming:
        querystring['upcoming'] = upcoming
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holidayapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

