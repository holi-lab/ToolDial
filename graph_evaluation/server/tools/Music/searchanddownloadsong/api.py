import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def searchsong(searchterm: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter the song title, artist or album that you would like to download. This returns a list of the most relevant search results ."
    
    """
    url = f"https://searchanddownloadsong.p.rapidapi.com/songRequestApi/"
    querystring = {'searchTerm': searchterm, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "searchanddownloadsong.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadlink(songid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter the songId ('id' from  searchSong endpoint) of the song you would like to download to your local storage. This returns the download link."
    
    """
    url = f"https://searchanddownloadsong.p.rapidapi.com/downloadLink/"
    querystring = {'songId': songid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "searchanddownloadsong.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

