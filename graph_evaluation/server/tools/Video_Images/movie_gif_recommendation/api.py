import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def moviegifs(searchstring: str, locale: str, page: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get available Movie GIFs based on input search string"
    searchstring: Input string for which the Movie GIFs will be returned
        locale: Locale which can be used for personalisation
        page: The page number of the content to be fetched.
        limit: The maximum number of results to return
        
    """
    url = f"https://movie-gif-recommendation.p.rapidapi.com/movieGifs"
    querystring = {'searchString': searchstring, 'locale': locale, }
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "movie-gif-recommendation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

