import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_news_by_sport(sportname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve news articles for a specific sport. It returns a list of news articles related to the specified sport."
    
    """
    url = f"https://bbc-world-news-api.p.rapidapi.com/sport/{sportname}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bbc-world-news-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sports_news(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve news articles related to sports. It returns a list of news articles covering various sports and sporting events."
    
    """
    url = f"https://bbc-world-news-api.p.rapidapi.com/sport"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bbc-world-news-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_news(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve news articles from various topics. It returns a list of news articles covering a wide range of subjects such as climate, world events, technology, health, and more."
    
    """
    url = f"https://bbc-world-news-api.p.rapidapi.com/all"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bbc-world-news-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_news_on_specific_topic(newsid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves news based on a specific news ID. Users can provide a news ID corresponding to a category or region of interest, and the API will return news items related to that ID. For example, /news/climate will fetch news articles specifically related to climate."
    
    """
    url = f"https://bbc-world-news-api.p.rapidapi.com/news/{newsid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bbc-world-news-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_news_from_world_wide(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches news articles from all categories and regions worldwide. It returns a comprehensive collection of news items, including titles, summaries, URLs, images, timestamps, and sources."
    
    """
    url = f"https://bbc-world-news-api.p.rapidapi.com/news"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bbc-world-news-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

