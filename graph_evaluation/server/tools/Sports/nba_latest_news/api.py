import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_latest_nba_articles(limit: str=None, player: str=None, team: str=None, source: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET /articles
		Returns a list of all the latest nba articles.
		
		Optional params:
		
		source returns articles based on chosen source
		
		options: nba-canada, nba, bleacher-report, yahoo, espn, slam
		
		Example /articles?source=bleacher-report
		
		team returns articles based on chosen team
		
		Example /articles?team=lakers
		
		limit returns the maximum number of articles desired
		
		Example /articles?limit=5
		
		player returns articles based on chosen player
		
		Use dash to seperate names
		
		Searching by players full name seperated by dash produces best results
		
		Example /articles?player=kevin-durant&limit=10"
    
    """
    url = f"https://nba-latest-news.p.rapidapi.com/articles"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if player:
        querystring['player'] = player
    if team:
        querystring['team'] = team
    if source:
        querystring['source'] = source
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-latest-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

