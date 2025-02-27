import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v4_sports_sport_odds(sport: str, regions: str, oddsformat: str='decimal', markets: str='h2h,spreads', dateformat: str='iso', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of live and upcoming games for a given sport, showing bookmaker odds for the specified region and markets. Set the `sport` to a `sport_key` from the /sports endpoint. Alternatively if `sport=upcoming`, it will return a list of the next 8 upcoming games across all sports, as well as any live games. For more info, see [list of available sports](https://the-odds-api.com/sports-odds-data/sports-apis.html) and [list of available bookmakers](https://the-odds-api.com/sports-odds-data/bookmaker-apis.html)."
    sport: sport key for which to return games and odds
        regions: Determines which bookmakers appear in the response. Can be a comma delimited list of regions. Each region will count as 1 request against the usage quota for each market. Most use cases will only need to specify one region. For a list of bookmakers by region, see https://the-odds-api.com/sports-odds-data/bookmaker-apis.html
        oddsformat: Format of returned odds.
        markets: The odds market to return. Can be a comma delimited list of odds markets. Defaults to h2h (head to head / moneyline). Outrights only avaialable for select sports. Note each market counts as 1 request against the usage quota.
        dateformat: Format of returned timestamps. Can be iso (ISO8601) or unix timestamp (seconds since epoch)
        
    """
    url = f"https://odds.p.rapidapi.com/v4/sports/{sport}/odds"
    querystring = {'regions': regions, }
    if oddsformat:
        querystring['oddsFormat'] = oddsformat
    if markets:
        querystring['markets'] = markets
    if dateformat:
        querystring['dateFormat'] = dateformat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v4_sports_sport_scores(sport: str, daysfrom: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of live and upcoming games for a given sport, and optionally recently completed games. Live and completed games will contain scores. **Currently in beta** and only available for selected sports. For more info, see the [list of available sports](https://the-odds-api.com/sports-odds-data/sports-apis.html)"
    sport: sport key for which to return games and odds
        daysfrom: The number of days in the past from which to return completed games. Valid values are integers from `1` to `3`. If this field is missing, only live and upcoming games are returned.
        
    """
    url = f"https://odds.p.rapidapi.com/v4/sports/{sport}/scores"
    querystring = {}
    if daysfrom:
        querystring['daysFrom'] = daysfrom
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v4_sports(all: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of available sports and tournaments. Use the `sports_key` in requests for /odds and /scores endpoints."
    all: When excluded, only recently updated (in-season) sports appear. Include this paramter to see all available sports
        
    """
    url = f"https://odds.p.rapidapi.com/v4/sports"
    querystring = {}
    if all:
        querystring['all'] = all
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

