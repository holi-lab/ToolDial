import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def teams(league: str, conference: str=None, division: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of teams within a specified league, conference, or division."
    league: A valid league code (i.e., NFL, NBA, MLB, NHL, NCAAF, or NCAAB)
        conference: A conference within the specified league
        division: A division within the specified conference
        
    """
    url = f"https://sportspage-feeds.p.rapidapi.com/teams"
    querystring = {'league': league, }
    if conference:
        querystring['conference'] = conference
    if division:
        querystring['division'] = division
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportspage-feeds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_by_id(gameid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a specific game based on its ID."
    gameid: A unique game identifier
        
    """
    url = f"https://sportspage-feeds.p.rapidapi.com/gameById"
    querystring = {'gameId': gameid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportspage-feeds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games(odds: str=None, team: str=None, division: str=None, date: str=None, skip: str=None, status: str=None, league: str=None, conference: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of games."
    odds: A comma-separated filter to select games with one or more of the following odds types: \\\\\\\"spread\\\\\\\", \\\\\\\"moneyline\\\\\\\", \\\\\\\"total\\\\\\\", or \\\\\\\"any\\\\\\\"
        team: A team participating in one or more games
        division: A division within the specified conference
        date: One or two (comma-separated) YYYY-MM-DD- or ISO-formatted dates
        skip: The number of game results to skip, which is required to access results beyond the first 100
        status: A valid status of one of the following: \\\\\\\"scheduled\\\\\\\", \\\\\\\"in progress\\\\\\\", \\\\\\\"final\\\\\\\", \\\\\\\"canceled\\\\\\\", or \\\\\\\"delayed\\\\\\\"
        league: A valid league code (i.e., NFL, NBA, MLB, NHL, NCAAF, or NCAAB)
        conference: A conference within the specified league
        
    """
    url = f"https://sportspage-feeds.p.rapidapi.com/games"
    querystring = {}
    if odds:
        querystring['odds'] = odds
    if team:
        querystring['team'] = team
    if division:
        querystring['division'] = division
    if date:
        querystring['date'] = date
    if skip:
        querystring['skip'] = skip
    if status:
        querystring['status'] = status
    if league:
        querystring['league'] = league
    if conference:
        querystring['conference'] = conference
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportspage-feeds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def conferences(league: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of conferences and divisions within the specified league. Use this endpoint to obtain conference and division names to be used as parameters for other requests."
    league: A valid league code (i.e., NFL, NBA, MLB, NHL, NCAAF, or NCAAB)
        
    """
    url = f"https://sportspage-feeds.p.rapidapi.com/conferences"
    querystring = {'league': league, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportspage-feeds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odds(gameid: str, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the odds history for a game by type."
    gameid: A unique game identifier
        type: An odds type of one of the following: \\\"spread\\\", \\\"moneyline\\\", \\\"total\\\", or \\\"any\\\"
        
    """
    url = f"https://sportspage-feeds.p.rapidapi.com/odds"
    querystring = {'gameId': gameid, }
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportspage-feeds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

