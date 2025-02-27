import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def news_per_newspaper(newspaperid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Getting the climate change news from a defined newspaper from the list of supported newspapers in the API. Below list of supported newspapers in the API. Feel free to suggest me any newspaper that is talking about the climate change to add it to our API.
		
		        cityam: cityam
		        thetimes: thetimes
		        theguardian: guardian
		        telegraph: telegraph
		        nytimes: nyt
		        latimes: latimes 
		        smh: smh
		        un: un
		        bbc: bbc
		        standard: es
		        thesun: sun
		        dailymail: dm
		        nypost: nyp"
    
    """
    url = f"https://climate-change21.p.rapidapi.com/news/{newspaperid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "climate-change21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def climate_change_news(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Getting all of the news related to the climate change from all over there world"
    
    """
    url = f"https://climate-change21.p.rapidapi.com/news"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "climate-change21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

