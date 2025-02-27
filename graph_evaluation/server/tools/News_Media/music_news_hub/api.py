import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_random_music_news(amount: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get random music news selections from the available outlets. Optional amount parameter provided."
    
    """
    url = f"https://music-news-hub.p.rapidapi.com/randomNews/{amount}"
    querystring = {}
    if amount:
        querystring['amount'] = amount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "music-news-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_news(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back the latest music news worldwide."
    
    """
    url = f"https://music-news-hub.p.rapidapi.com/allNews"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "music-news-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_individual_news_source(newspaperid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get music news from a specific outlet. 
		Currently available sources and their newspaperId:
		
		- Pitchfork: pitchfork
		- NPR: npr
		- New York Times: nyt
		- KEXP: kexp
		- Downbeat Magazine: downbeat
		- LA Times: latimes
		- NME: nme
		- Mojo: mojo
		- Under the Radar: undertheradar
		- Fact: fact"
    
    """
    url = f"https://music-news-hub.p.rapidapi.com/news/{newspaperid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "music-news-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

