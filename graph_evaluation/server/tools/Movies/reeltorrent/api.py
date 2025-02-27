import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_movies(search: str, cat: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can search movie name"
    
    """
    url = f"https://reeltorrent.p.rapidapi.com/search"
    querystring = {'search': search, 'cat': cat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reeltorrent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movies(title: str=None, year: str=None, quality: str=None, writer: str=None, country: str=None, directed: str=None, genres: str=None, order: str=None, limit: str=None, lang: str=None, movie_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You will get the movies Title and download link"
    
    """
    url = f"https://reeltorrent.p.rapidapi.com/flims"
    querystring = {}
    if title:
        querystring['title'] = title
    if year:
        querystring['year'] = year
    if quality:
        querystring['quality'] = quality
    if writer:
        querystring['writer'] = writer
    if country:
        querystring['country'] = country
    if directed:
        querystring['directed'] = directed
    if genres:
        querystring['genres'] = genres
    if order:
        querystring['order'] = order
    if limit:
        querystring['limit'] = limit
    if lang:
        querystring['lang'] = lang
    if movie_name:
        querystring['movie_name'] = movie_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reeltorrent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

