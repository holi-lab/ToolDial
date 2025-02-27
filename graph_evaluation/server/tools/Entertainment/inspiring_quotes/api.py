import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_multiple_inspiring_quotes(count: int=50, author: str='Einstein', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This request gets you number of random quotes specified by the query parameter 'count' that are all different and inspiring in their way."
    count: number of random quotes to be returned
note: number must be between 3 and 50
        author: filters quotes to the ones that contains this query parameter value
        
    """
    url = f"https://inspiring-quotes.p.rapidapi.com/multiple"
    querystring = {}
    if count:
        querystring['count'] = count
    if author:
        querystring['author'] = author
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "inspiring-quotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_random_quote(author: str='Albert', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This request gets you one random quote that inspire you to continue through this rough life."
    author: filters quotes to the ones that contains this query parameter value
        
    """
    url = f"https://inspiring-quotes.p.rapidapi.com/random"
    querystring = {}
    if author:
        querystring['author'] = author
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "inspiring-quotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

