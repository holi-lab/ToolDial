import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(query: str, filterresult: str='0', languagecode: str='en', sortby: str='date', safe: str='off', returnpage: str='1', countrycode: str='us', returnresult: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter data to search. GET-request."
    filterresult: Controls whether the duplicate content filter is enabled or disabled. '0' - Disables the duplicate content filter. '1' - Enables the duplicate content filter. The default setting is '0'.
        languagecode: Sets the language of the user interface. The default is 'en'.
        sortby: The sorting expression applied to the results. Example: sort='date'. Default sort=' ' (empty string, sort relevance)
        safe: Search security level. Valid values: 'active' : enables SafeSearch filtering. 'off' : disables SafeSearch filtering (default).
        returnpage: The index of the first result returned. The default number of results on the page is 10
        countrycode: End-user geolocation. The parameter value is a two-letter country code. The parameter boosts search results whose country of origin matches the parameter value. The default is 'us'.
        returnresult: The number of search results returned. Valid values are integers from 1 to 10
        
    """
    url = f"https://google-search-results1.p.rapidapi.com/"
    querystring = {'query': query, }
    if filterresult:
        querystring['filterResult'] = filterresult
    if languagecode:
        querystring['languageCode'] = languagecode
    if sortby:
        querystring['sortBy'] = sortby
    if safe:
        querystring['safe'] = safe
    if returnpage:
        querystring['returnPage'] = returnpage
    if countrycode:
        querystring['countryCode'] = countrycode
    if returnresult:
        querystring['returnResult'] = returnresult
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-search-results1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

