import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_songs_from_movie(movieid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return all songs from the specified episode of your choosen movie.
		
		**Return codes:**
		- 0: Everything went as expected. *(If the list of songs is empty, it means that there's no song registered)*
		- 1: Movie not found. *(Please verify the parameter **movieID**)*
		- For any other code returned, [check the meaning of it](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status) before contacting us."
    
    """
    url = f"https://music-show.p.rapidapi.com/GetSongsFromMovie/{movieid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "music-show.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_movieid(search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the movieID of a giving movie title.
		
		If the named researched has "spaces", replace them by "+". Example: "Top Gun" would be "top+gun" *(Uppercases are not important)*.
		
		**Return codes:**
		- 0: Everything went as expected.
		- 1: No match found."
    
    """
    url = f"https://music-show.p.rapidapi.com/getmovieID/{search}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "music-show.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_showid(search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the showID of a giving show name.
		
		If the named researched has "spaces", replace them by "+". Example: "The Walking Dead" would be "the+walking+dead" *(Uppercases are not important)*.
		
		**Return codes:**
		- 0: Everything went as expected.
		- 1: No match found."
    
    """
    url = f"https://music-show.p.rapidapi.com/getshowid/{search}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "music-show.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_songs_from_show(seasons: int, episode: int, showid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return all songs from the specified episode of your choosen show.
		
		**Return codes:**
		- 0: Everything went as expected. *(If the list of songs is empty, it means that there's no song registered)*
		- 1: Show not found. *(Please verify the parameter **showID**)*
		- 2: Season not found. *(Please verify the parameter **season**)*
		- 3: Episode not found. *(Please verify the parameter **episode**)*
		- For any other code returned, [check the meaning of it](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status) before contacting us."
    
    """
    url = f"https://music-show.p.rapidapi.com/getsongsfromshow/{showid}/{seasons}/{episode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "music-show.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

