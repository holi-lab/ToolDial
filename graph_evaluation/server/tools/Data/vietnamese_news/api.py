import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def query_by_date(datestring: str, offset: str='2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "- Show all articles and all of their metadata fields that have the publish date match the provided string.
		- For the URL, `date` can be shortened to `d`.
		- Each query will return **10 results**, sorted by date incrementally. Add `/1`, `/2`, etc. at the end of the URL to fetch more. Default is equal to `/0`."
    
    """
    url = f"https://vietnamese-news.p.rapidapi.com/date/{datestring}/{offset}"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vietnamese-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

