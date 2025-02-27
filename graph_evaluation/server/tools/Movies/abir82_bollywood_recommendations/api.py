import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fetch_movies(year: str, genre: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Endpoint allows to get a list of movies from 2000 to 2019 based on the genre selected by the user."
    
    """
    url = f"https://abir82-bollywood-recommendations.p.rapidapi.com/"
    querystring = {'year': year, 'genre': genre, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abir82-bollywood-recommendations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

