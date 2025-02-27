import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_detailed_news(news_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news with proper body with specific news id."
    
    """
    url = f"https://live-sports.p.rapidapi.com/news/details/{news_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_list_of_news(category: str=None, page: int=1, tag: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of news. 
		If you want to filter using tags or category, Please pass the category string or tag string.
		Page no is required for pagination. Default page no is 1."
    
    """
    url = f"https://live-sports.p.rapidapi.com/news/news-list"
    querystring = {}
    if category:
        querystring['category'] = category
    if page:
        querystring['page'] = page
    if tag:
        querystring['tag'] = tag
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_latest_news_headlines(limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Hot News headlines. This is perfect for your homepage.
		
		limit is the number of headlines you want to have in the response."
    
    """
    url = f"https://live-sports.p.rapidapi.com/news/headlines"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_upcoming_and_current_match_list(game_type: str='football', result_type: str='current', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get match lists
		
		Args:
		    game_type (str, optional): football/cricket
		    result_type (str, optional): current"
    
    """
    url = f"https://live-sports.p.rapidapi.com/sports/matches"
    querystring = {}
    if game_type:
        querystring['game_type'] = game_type
    if result_type:
        querystring['result_type'] = result_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_live_links_for_a_game(game_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Live Links for a Specific Game.
		Iframe links can be embedded on your website or mobile app."
    
    """
    url = f"https://live-sports.p.rapidapi.com/sports/live-links"
    querystring = {'game_id': game_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_top_upcoming_current_match_list(result_type: str='current', game_type: str='football', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Top Upcoming/Current Match List for your homepage. 
		
		Args:
		    game_type (str, optional): football/cricket
		    result_type (str, optional): current"
    
    """
    url = f"https://live-sports.p.rapidapi.com/sports/top-matches"
    querystring = {}
    if result_type:
        querystring['result_type'] = result_type
    if game_type:
        querystring['game_type'] = game_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

