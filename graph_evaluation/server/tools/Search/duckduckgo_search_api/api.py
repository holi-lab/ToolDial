import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def html_search(q: str, kl: str=None, df: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/htmlSearch allows you to perform a search query of [DuckDuckGo's HTML-only site](https://html.duckduckgo.com) and retrieve the results in JSON format.
		
		Optionally easily filter your results by a language+location and date range."
    q: The **q** parameter is required for the /htmlSearch endpoint and is used to specify the search query that you want to lookup.
        kl: The **kl** parameter is optional for the /htmlSearch endpoint and is used to specify the location and language code of the search results. (format: countrycode-languagecode).  

By default (if no value is passed for this parameter) the search results returned will be based on worldwide data without any location or language filters.
        df: The **df** parameter is optional for the /htmlSearch endpoint and is allows you to retrieve search results from specific timeframes - from the past day, week, month, or year.

By default (if no value is passed for this parameter) the search results returned will include data from all time periods.
        
    """
    url = f"https://duckduckgo-search-api.p.rapidapi.com/htmlSearch"
    querystring = {'q': q, }
    if kl:
        querystring['kl'] = kl
    if df:
        querystring['df'] = df
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "duckduckgo-search-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def html_search_with_icons(q: str, kl: str=None, df: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/htmlSearchWithIcons allows you to perform a search query of [DuckDuckGo's HTML-only site](https://html.duckduckgo.com) and retrieve the results including available favicons (encoded as base64, ready for you to save)
		
		Optionally easily filter your results by a language+location and date range."
    q: The **q** parameter is required for the /htmlSearchWithIcons endpoint and is used to specify the search query that you want to lookup.
        kl: The **kl** parameter is optional for the /htmlSearchWithIcons endpoint and is used to specify the location and language code of the search results (format: countrycode-languagecode).  

By default (if no value is passed for this parameter) the search results returned will be based on worldwide data without any location or language filters.
        df: The **df** parameter is optional for the /htmlSearchWithIcons endpoint and allows you to retrieve search results from specific timeframes - from the past day, week, month, or year.

By default (if no value is passed for this parameter) the search results returned will include data from all time periods.
        
    """
    url = f"https://duckduckgo-search-api.p.rapidapi.com/htmlSearchWithIcons"
    querystring = {'q': q, }
    if kl:
        querystring['kl'] = kl
    if df:
        querystring['df'] = df
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "duckduckgo-search-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

