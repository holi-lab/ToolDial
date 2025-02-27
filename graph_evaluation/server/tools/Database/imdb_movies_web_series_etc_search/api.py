import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_anything_on_imdb_api(search_query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search anything on imdb api by entering the movie, webseries , etc title or film"
    search_query: Its **Very Important** need to add **.json** as a suffix to every serch_query that you send or else you will get an error.

*example if you are searching for salman add .json to it i.e salman.json *

        
    """
    url = f"https://imdb-movies-web-series-etc-search.p.rapidapi.com/{search_query}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imdb-movies-web-series-etc-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

