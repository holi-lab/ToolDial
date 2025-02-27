import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def youtubesearchresults(q: str, next: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the YouTube Search Results Items and their Information! No limits and no Google API Key needed!"
    q: The URL Encoded search Query
        next: Leave empty if you want the results from the first page.
Use the value from [nextpageRef] from the previous request (for example first page) to get the next page (for example 2nd page)
MUST BE URL ENCODED
        
    """
    url = f"https://youtube-search-results.p.rapidapi.com/youtube-search/"
    querystring = {'q': q, }
    if next:
        querystring['next'] = next
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-search-results.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

