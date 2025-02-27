import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def download(country: str='US', date: str='2017-09-03', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Takes a country code and a date and returns a URL to download corresponding Twitter trending topics archive file in CSV format."
    country: Country code for the trends to be downloaded. Leave blank for global trends.
        date: Date of the trends to be returned in YYYY-MM-DD format; must be equal to or later than 2017-09-03. If not provided, the latest available hourly trend file of the current date will be returned.
        
    """
    url = f"https://onurmatik-twitter-trends-archive-v1.p.rapidapi.com/download"
    querystring = {}
    if country:
        querystring['country'] = country
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onurmatik-twitter-trends-archive-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

