import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_movie(vsid: str, hash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Downloading of the completed movie"
    vsid: From Start Movie Session API call response
        hash: Returned from Check Status API when "movie_status" = "DONE"
        
    """
    url = f"https://magisto.p.rapidapi.com/video/{hash}"
    querystring = {'vsid': vsid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magisto.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check_movie_status(vsid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check the state of the movie session"
    vsid: From Start Movie Session API call response
        
    """
    url = f"https://magisto.p.rapidapi.com/video/check"
    querystring = {'vsid': vsid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magisto.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

