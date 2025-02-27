import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def autocomplete(query: str, region: str='us', cursor_pointer: str=None, user_agent: str=None, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get query suggestions from Google Search, including Knowledge Graph information when available."
    query: Autocomplete / typeahead search query.
        region: The country / region from which to make the query.

**Allowed values:** 2-letter country code, see [ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2)
**Default:** us.
        cursor_pointer: Cursor pointer defines the position of cursor for the query provided, position starts from 0 which is a case where cursor is placed before the query. If not provided acts as cursor is placed in the end of query (Google's *cp* parameter).
        user_agent: Device type to use for the search.

**Default:** desktop.
        language: Set the language of the results.

**Allowed values:** 2-letter language code, see [ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)
**Default:** en.
        
    """
    url = f"https://web-search-autocomplete.p.rapidapi.com/autocomplete"
    querystring = {'query': query, }
    if region:
        querystring['region'] = region
    if cursor_pointer:
        querystring['cursor_pointer'] = cursor_pointer
    if user_agent:
        querystring['user_agent'] = user_agent
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "web-search-autocomplete.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

