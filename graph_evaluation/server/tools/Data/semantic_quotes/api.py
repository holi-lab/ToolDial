import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def random_quote(max_length: int=4000, tags: str='inspirational,motivational', min_length: int=0, limit: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get random quotes by length and/or tags.  **NOTE:** up to 10 results can be returned, they may have similar meanings. If you want want absolute random, just pick the first result (or set `limit` to 1) and call this API again to get another quote."
    max_length: Maximum length of the quote.
        tags: Comma seperated tags for filtering, for example, 
                `faith`,
                `god`,
                `humor`, 
                `hope`,
                `inspirational`, 
                `love`, 
                `life`,
                `motivational`,
                `philosophy`,
                `religion`,
                `science`,
                `spirituality`,
                `success`,
                `truth`,
                `wisdom`,
                `poetry`,
                and much more.
        
        min_length: Minimum length of the quote.
        limit: Number of results returned.
        
    """
    url = f"https://semantic-quotes.p.rapidapi.com/random"
    querystring = {}
    if max_length:
        querystring['max_length'] = max_length
    if tags:
        querystring['tags'] = tags
    if min_length:
        querystring['min_length'] = min_length
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "semantic-quotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_quotes(q: str, tags: str='inspirational,motivational', max_length: int=4000, limit: int=3, min_length: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search quotes which have similar meaning/intent to the query sentences/words. We allow user to flexibly filter the results by the length of the quote, and by a large variety of tags. The results are returned in order of similarity."
    q: Search term.
        tags: Comma seperated tags for filtering, for example, 
                `faith`,
                `god`,
                `humor`, 
                `hope`,
                `inspirational`, 
                `love`, 
                `life`,
                `motivational`,
                `philosophy`,
                `religion`,
                `science`,
                `spirituality`,
                `success`,
                `truth`,
                `wisdom`,
                `poetry`,
                and much more.
        
        max_length: Maximum length of the quote.
        limit: Number of results returned.
        min_length: Minimum length of the quote.
        
    """
    url = f"https://semantic-quotes.p.rapidapi.com/search"
    querystring = {'q': q, }
    if tags:
        querystring['tags'] = tags
    if max_length:
        querystring['max_length'] = max_length
    if limit:
        querystring['limit'] = limit
    if min_length:
        querystring['min_length'] = min_length
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "semantic-quotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

