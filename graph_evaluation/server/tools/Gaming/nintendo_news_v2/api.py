import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_latest_nintendo_articles_by_news_channels(sourceid: str, p: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest NBA news based on its source. 
		You can get from the following gaming websites: 
		
		Nintendo News from various sources -> nintendonews
		PureNintendo -> purenintendo
		GameSpot -> gamespot
		IGN -> ign"
    
    """
    url = f"https://nintendo-news1.p.rapidapi.com/news/{sourceid}"
    querystring = {}
    if p:
        querystring['p'] = p
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nintendo-news1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_latest_nintendo_articles_from_popular_news_channels(p: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back all the latest articles about Nintendo from all popular Nintendo news outlets."
    
    """
    url = f"https://nintendo-news1.p.rapidapi.com/news"
    querystring = {}
    if p:
        querystring['p'] = p
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nintendo-news1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

