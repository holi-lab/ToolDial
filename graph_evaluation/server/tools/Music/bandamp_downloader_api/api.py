import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def download_bandcamp_track(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Supply any Bandcamp track URL & get downloadable links, alongwith:
		
		- Track name
		- Related Album Name
		- Track Published date
		- Duration
		- Thumbnail
		- Artist name
		- Downloadable Mp3 file link"
    
    """
    url = f"https://bandamp-downloader-api.p.rapidapi.com/tracks/"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bandamp-downloader-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_album_details(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch:
		- Album Name
		- 2. Album Description
		- 3. Album Thumbnail URL
		- 4. Artist Name
		- 3. Total Tracks Count
		- 4. Each Track URLs
		- 5. Each Track Names
		- 6. Each Track Durations"
    
    """
    url = f"https://bandamp-downloader-api.p.rapidapi.com/album/"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bandamp-downloader-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_album_details(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch:
		- Album Name
		- 2. Album Description
		- 3. Album Thumbnail URL
		- 4. Artist Name
		- 3. Total Tracks Count
		- 4. Each Track URLs
		- 5. Each Track Names
		- 6. Each Track Durations"
    
    """
    url = f"https://bandamp-downloader-api.p.rapidapi.com/album/"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bandamp-downloader-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

