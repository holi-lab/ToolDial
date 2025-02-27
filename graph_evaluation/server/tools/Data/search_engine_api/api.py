import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_search_query(query: str, numofpages: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint executes a search query against a predefined search source and returns the search results. 
		It accepts a search query string as a parameter and optionally a number of pages to return.
		If the number of pages is not provided, it defaults to 1. The maximum number of pages allowed is 10.
		If the search query is successful, the search results are returned as a list in JSON format.
		
		Sample request:
		            
		    GET /api/Search/{query}?numOfPages={numOfPages}
		            
		Sample response:
		
		    Status code: 200 OK
		    
		    [
		        {
		            "Title": "Wikipedia",
		            "Link": "https://www.wikipedia.org",
		            "Description": "Wikipedia is a free online encyclopedia, created and edited by volunteers around the world and hosted by the Wikimedia Foundation."
		        },
		        {
		            "Title": "History of Wikipedia",
		            "Link": "https://en.wikipedia.org/wiki/History_of_Wikipedia",
		            "Description": "The history of Wikipedia, a free, web-based, collaborative, multilingual encyclopedia project supported by the non-profit Wikimedia Foundation, started in 2001."
		        }
		    ]"
    query: Search query string
        numofpages: Number of pages to return, default is 1, maximum is 10
        
    """
    url = f"https://search-engine-api.p.rapidapi.com/api/Search/{query}"
    querystring = {}
    if numofpages:
        querystring['numOfPages'] = numofpages
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "search-engine-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

