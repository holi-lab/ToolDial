import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def top_10_serp_common_themes_query(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Submit a search query. Use provided token to get results with the other endpoint, or get results with callback URL."
    q: define search
        
    """
    url = f"https://common-themese-from-top-10-serp.p.rapidapi.com/top-ten-themes-for-search.php"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "common-themese-from-top-10-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_10_serp_common_themes_results(q: str, token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve results for previously submitted query (first wait 1 minute). Token is valid for about 15 minutes after query submission."
    q: define search
        token: token was provided as a response to submission of this query
        
    """
    url = f"https://common-themese-from-top-10-serp.p.rapidapi.com/async_top-ten-themes-for-search.php"
    querystring = {'q': q, 'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "common-themese-from-top-10-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

