import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_the_news_from_cnn_by_category(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest news from CNN as a resource. 
		Available categories : 
		- world
		- africa
		- americas
		- asia
		- australia
		- china
		- europe
		- india
		- middle-east
		- uk"
    
    """
    url = f"https://all-news.p.rapidapi.com/api/cnn/?news=all"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_the_news_from_fow_news_by_category(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest news from Fox News as a resource. 
		Available categories : 
		- politics
		- media
		- opinion
		- entertainment
		- sports
		- lifestyle"
    
    """
    url = f"https://all-news.p.rapidapi.com/api/fox/?news=all"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_the_news(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the latest news with all categories included."
    
    """
    url = f"https://all-news.p.rapidapi.com/api/all/?news=all"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

