import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_collocations(query: str='prestar', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Looks up a word in a spanish collocation database with the top spanish collocations and returns the best spanish collocations (ordered by significance) and 2 or 3 example sentences for each collocation."
    
    """
    url = f"https://linguatools-top-spanish-collocations.p.rapidapi.com/"
    querystring = {}
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "linguatools-top-spanish-collocations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

