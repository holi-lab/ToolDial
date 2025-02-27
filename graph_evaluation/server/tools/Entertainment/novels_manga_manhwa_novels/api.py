import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def novel_info(link: str, mode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about selected novel"
    
    """
    url = f"https://novels-manga-manhwa-novels.p.rapidapi.com/"
    querystring = {'link': link, 'mode': mode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "novels-manga-manhwa-novels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_novel(string: str, mode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a novel by title"
    
    """
    url = f"https://novels-manga-manhwa-novels.p.rapidapi.com/"
    querystring = {'string': string, 'mode': mode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "novels-manga-manhwa-novels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def novel_genres(genre: str, mode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of novels by chosen genre"
    genre: can be: 
action, ecchi, horror, martial, romance, seinen, sports, chinese, fantasy, mystery, shoujo, supernatural, adult, comedy, game, historical, mature, psychological, sci-fi, tragedy, adventure, drama, gender, history, lolicon, shounen
        
    """
    url = f"https://novels-manga-manhwa-novels.p.rapidapi.com/"
    querystring = {'genre': genre, 'mode': mode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "novels-manga-manhwa-novels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read_novel(mode: str, link: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "read selected novel chapter"
    
    """
    url = f"https://novels-manga-manhwa-novels.p.rapidapi.com/"
    querystring = {'mode': mode, 'link': link, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "novels-manga-manhwa-novels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def popular_novels(mode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of popular novels"
    
    """
    url = f"https://novels-manga-manhwa-novels.p.rapidapi.com/"
    querystring = {'mode': mode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "novels-manga-manhwa-novels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

