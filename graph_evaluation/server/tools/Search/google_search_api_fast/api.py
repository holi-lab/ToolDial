import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(query: str, sort: str='relevance', offset: int=0, count: int=10, location: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is specifically designed for conducting web searches. It allows you to retrieve and access up-to date information available on the web, directly from Google's search engine."
    query: The query parameter is used to specify the search term that the API should use to retrieve information.
        sort: Sort by `relevance` or `date`, default is `relevance`.
        offset: Parameter defines the result offset. It skips the given number of results. It's used for pagination. (e.g., `0` (default) is the first page of results, `10` is the 2nd page of results, `20` is the 3rd page of results, etc.).

        count: Parameter defines the maximum number of results to return.  (e.g., `10` (default) returns 10 results, `40` returns 40 results, and `100` returns 100 results).
        location: Parameter defines the country to use for the Google search. It's a two-letter country code. (e.g., us for the United States, uk for United Kingdom, or fr for France). It will prioritize website search results based on the country specified.
        
    """
    url = f"https://google-search-api-fast.p.rapidapi.com/search"
    querystring = {'query': query, }
    if sort:
        querystring['sort'] = sort
    if offset:
        querystring['offset'] = offset
    if count:
        querystring['count'] = count
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-search-api-fast.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

