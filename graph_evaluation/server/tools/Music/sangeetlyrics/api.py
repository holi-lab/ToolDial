import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def song_lyrics(movie: str='Selfiee', label: str=None, singer: str='Yo Yo Honey Singh', title: str=None, lang: str='hindi', limit: str='2', year: str='2023', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "SangeetLyrics API provides a way to retrieve lyrics by specifying parameters such as limit, order, language, movie, singer, label, and title.
		
		- The limit parameter specifies the maximum number of lyrics to retrieve, with a maximum limit of 60.
		- The order parameter specifies the sort order of the lyrics, either ascending (asc) or descending (desc).
		- The lang parameter specifies the language of the lyrics, which can be either Telugu (telugu), Hindi (hindi), or English (english).
		- The movie parameter can be used to retrieve lyrics from a specific movie.
		- The singer parameter can be used to retrieve lyrics sung by a specific singer.
		- The label parameter can be used to retrieve lyrics from a specific record label.
		- The title parameter can be used to retrieve lyrics for a specific song title."
    
    """
    url = f"https://sangeetlyrics.p.rapidapi.com/lyrics"
    querystring = {}
    if movie:
        querystring['movie'] = movie
    if label:
        querystring['label'] = label
    if singer:
        querystring['singer'] = singer
    if title:
        querystring['title'] = title
    if lang:
        querystring['lang'] = lang
    if limit:
        querystring['limit'] = limit
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sangeetlyrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def music_label(label: str, limit: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It will provide you all music labels"
    
    """
    url = f"https://sangeetlyrics.p.rapidapi.com/label"
    querystring = {'label': label, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sangeetlyrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(search: str, cat: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You Can Search anything
		cat can be 'music' or 'movie'
		SangeetLyrics - Song Lyrics, Artists, Albums, Knowledge & More API"
    
    """
    url = f"https://sangeetlyrics.p.rapidapi.com/search"
    querystring = {'search': search, 'cat': cat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sangeetlyrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movie(movie: str, limit: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It will Provide you List of Movies"
    
    """
    url = f"https://sangeetlyrics.p.rapidapi.com/movie"
    querystring = {'movie': movie, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sangeetlyrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def singer(singer: str, limit: str='2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It will provide you the list of singers"
    
    """
    url = f"https://sangeetlyrics.p.rapidapi.com/singer"
    querystring = {'singer': singer, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sangeetlyrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

