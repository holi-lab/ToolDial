import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def quotes(movie: str=None, category: str=None, character: str=None, year: str=None, actor: str='Al Pacino', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch well known quotes from more than 500 movies."
    category: E.g.: "action", "crime", "drama", etc.
        year: Filter quotes by year or years.
        
    """
    url = f"https://juanroldan1989-moviequotes-v1.p.rapidapi.com/api/v1/quotes"
    querystring = {}
    if movie:
        querystring['movie'] = movie
    if category:
        querystring['category'] = category
    if character:
        querystring['character'] = character
    if year:
        querystring['year'] = year
    if actor:
        querystring['actor'] = actor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "juanroldan1989-moviequotes-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

