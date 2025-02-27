import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_player_odds_for_event(eventid: int, bookieid: str='1:4:5:6:7:8:9:10', decimal: bool=None, marketid: str='1:2:3:4:5:6', best: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get's the odds for the provided eventId from the parameter.
		
		You can send in a lot of parameters in this request, here's what's available.
		
		- eventId (Required). Send in the eventId in order for the API to know which data to showcase.
		- marketId (Optional). If you want to focus on one or more markets but not all, get the id of the market you want from the Get All Markets request and put it as a parameter on this API call. Here's an example, if we want to get the offers only for Assists, we send in the 'marketId=1'. If we want Assists & Points, we do it like so, 'marketId=1:3'. As you can see, you need to separate the markets by a : in between.
		- bookieId (Optional). If you want to focus on one or more bookies but not all, get the id of the bookies you want from the Get All Bookies request and put it as a parameter on this API call. Here's an example, if we want to get the offers only from bet365, we send in the 'bookieId=10'. If we want BetMGM & bet365, we do it like so, 'bookieId=8:10'. As you can see, you need to separate the bookies by a : in between.
		- best (Optional). The default for this parameter is 'false' meaning that if it's not provided, the API will get all of the lines from all of the available bookies. If you provide it as 'true' if will get the best possible lines from all of the available bookies. (There's a possibility that nothing is return if the best available line isn't present in the bookies sent as a parameter. Meaning if bet365 has the best line and you don't include bet365 as a bookie, you won't get a result because the best result won't be available in the object.
		- decimal (Optional). The default lines are sent back as American Odds (i.e. +130, -145). If you send decimal as a true, you will see results in decimals (i.e. 2.3, 1.4)."
    
    """
    url = f"https://nba-player-props-odds.p.rapidapi.com/get-player-odds-for-event"
    querystring = {'eventId': eventid, }
    if bookieid:
        querystring['bookieId'] = bookieid
    if decimal:
        querystring['decimal'] = decimal
    if marketid:
        querystring['marketId'] = marketid
    if best:
        querystring['best'] = best
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-player-props-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_events_for_today(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all of the available events for today's date. The reason why you want to use this API is to get the 'eventId' from the event that you need. This is useful for the main API call in which it requires you to send the eventId as a parameter in order to know which data to send back."
    
    """
    url = f"https://nba-player-props-odds.p.rapidapi.com/get-events-for-date"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-player-props-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_markets(name: str='assists', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all of the available markets (Assists, Points, Rebounds, etc.) that our API provides.
		
		If you want to sort by name to get the id, simply provide the 'name=<<INSERT LABEL HERE>>' as a query parameter and it will return the filtered list."
    
    """
    url = f"https://nba-player-props-odds.p.rapidapi.com/get-markets"
    querystring = {}
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-player-props-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_bookies(name: str='bet365', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all of the available bookies / gambling websites that our API provides.
		
		If you want to sort by name to get the id, simply provide the 'name=<<INSERT NAME HERE>>' as a query parameter and it will return the filtered list."
    
    """
    url = f"https://nba-player-props-odds.p.rapidapi.com/get-bookies"
    querystring = {}
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-player-props-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

