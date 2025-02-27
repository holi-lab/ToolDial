import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_tweets_and_scores(score_below: int=0, score_above: int=None, page: str='1', team: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get All tweets and Scores (only recommended for larger plans)"
    score_below: Filter for only scores only below a specified number (-1 to +1)
        score_above: Query for tweets with scores only above a specified about. Number must be +1 to -1
        page: Page number. Get tweets 1-100, or page 2 101-200 etc
        team: Filter by team abbreviation. For example, NE, DAL, etc.
        
    """
    url = f"https://football-sentiment.p.rapidapi.com/tweets/"
    querystring = {}
    if score_below:
        querystring['score_below'] = score_below
    if score_above:
        querystring['score_above'] = score_above
    if page:
        querystring['Page'] = page
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-sentiment.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

