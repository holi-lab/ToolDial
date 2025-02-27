import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(terms: str, date_range: str='before:2008-01-01', google_domain: str='google.fr', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search and Get Results from Major Search Engines"
    date_range: Specify a date period for the results.
Details: https://searchengineland.com/search-google-by-date-with-new-before-and-after-search-commands-315184
        google_domain: Specify which google domain you'd like to use.
Available domains here: https://www.google.com/supported_domains

By default: google.com
        
    """
    url = f"https://siphon-ai.p.rapidapi.com/search"
    querystring = {'terms': terms, }
    if date_range:
        querystring['date_range'] = date_range
    if google_domain:
        querystring['google_domain'] = google_domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "siphon-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

