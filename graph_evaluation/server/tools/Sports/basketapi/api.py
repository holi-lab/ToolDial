import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for basketball players, teams, and tournaments using a search term."
    term: The search term to use
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/search/{term}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def managerlastmatches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the last matches for a specific basketball manager."
    page: Zero-based page
        id: The ID of the manager for which you want to retrieve the last matches
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/manager/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerstatisticsoverall(is_id: int, tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the overall statistics for a specific basketball player in a given tournament and season."
    id: The ID of the player for which you want to retrieve the overall statistics
        tournamentid: The unique tournament ID for which you want to retrieve the player's overall statistics
        seasonid: The season ID for which you want to retrieve the player's overall statistics
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/statistics/overall"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playershotactionsplayoffs(is_id: int, seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the shot actios of a player in the basketball league during the playoffs."
    id: The player id you want to retrieve the shot actions.
        seasonid: The season id you want to retrieve the player's shot action.
        tournamentid: The unique tournament id you want to retrieve the player's shot actions.
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/shot-actions/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playernearmatches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the near matches for a specific basketball player."
    id: The ID of the player for which you want to retrieve the near matches
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/player/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playermedia(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the media for a specific basketball player."
    id: The ID of the player for which you want to retrieve the media
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/player/{is_id}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerlastmatches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the last matches of a basketball player using their id."
    page: Zero-based page.
        id: The team id you want to retrieve the last matches.
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/player/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerstatisticsseasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the statistics seasons for a specific basketball player."
    id: The ID of the player for which you want to retrieve the statistics seasons
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/player/{is_id}/statistics/season"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerstatisticsregularseason(seasonid: int, tournamentid: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the regular season statistics for a specific basketball player in a given tournament and season."
    seasonid: The season ID for which you want to retrieve the player's regular season statistics
        tournamentid: The unique tournament ID for which you want to retrieve the player's regular season statistics
        id: The ID of the player for which you want to retrieve the regular season statistics
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/statistics/regularseason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of a specific basketball player."
    id: The ID of the player for which you want to retrieve the details
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/player/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerimage(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the image for a specific basketball player using the player ID. The image is in PNG format."
    id: The player ID for which you want to retrieve the image
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/player/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def managerdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of a specific basketball manager."
    id: The ID of the manager for which you want to retrieve the details
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/manager/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playertransferhistory(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the transfer history for a specific basketball player."
    id: The ID of the player for which you want to retrieve the transfer history
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/player/{is_id}/transfers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playershotactionsregularseason(is_id: int, tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the overall statistics of a player in the basketball league during the regular season."
    id: The player id you want to retrieve the shot actions.
        tournamentid: The unique tournament id you want to retrieve the player's shot actions.
        seasonid: The season id you want to retrieve the player's shot actions.
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/shot-actions/regularSeason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchseries(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get matches in the series for a specific basketball match."
    id: The ID of the match for which you want to get the series
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/match/{is_id}/series"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def managerimage(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the image for a specific basketball manager using the manager ID. The image is in PNG format."
    id: The manager ID for which you want to retrieve the image
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/manager/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchodds(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get odds information for a specific basketball match."
    id: The ID of the match for which you want to get odds
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/match/{is_id}/odds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchplayershotmap(is_id: int, playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the shotmap of a player in a basketball match."
    id: The id of the match you want to get shotmap.
        playerid: The player id of the match you want to get shotmap.
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/match/{is_id}/player/{playerid}/shotmap"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerstatisticsplayoffs(seasonid: int, is_id: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the playoffs statistics for a specific basketball player in a given tournament and season."
    seasonid: The season ID for which you want to retrieve the player's playoffs statistics
        id: The ID of the player for which you want to retrieve the playoffs statistics
        tournamentid: The unique tournament ID for which you want to retrieve the player's playoffs statistics
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/statistics/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchlineups(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get lineups for a specific basketball match."
    id: The ID of the match for which you want to get lineups
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/match/{is_id}/lineups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchbestplayers(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get best players information for a specific basketball match."
    id: The ID of the match for which you want to get the best players
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/match/{is_id}/best-players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamnearmatches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the near matches for a specific basketball team."
    id: The ID of the team for which you want to retrieve the near matches
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/team/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchincidents(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get incidents for a specific basketball match."
    id: The ID of the match for which you want to get incidents
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/match/{is_id}/incidents"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def headtoheadmatches(customid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get head-to-head matches for a specific basketball match using the custom ID."
    customid: The custom ID of the match for which you want to get head-to-head matches. It is obtained by the match.
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/match/{customid}/h2h"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teammedia(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the media for a specific basketball team."
    id: The ID of the team for which you want to retrieve the media
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/team/{is_id}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamlastmatches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the last matches for a specific basketball team."
    page: Zero-based page
        id: The ID of the team for which you want to retrieve the last matches
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/team/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchstatistics(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics for a specific basketball match."
    id: The ID of the match for which you want to get statistics
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/match/{is_id}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamplayers(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the players of a specific basketball team."
    id: The ID of the team for which you want to retrieve the players
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/team/{is_id}/players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchmanagers(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the managers for a specific basketball match."
    id: The ID of the match for which you want to get managers
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/match/{is_id}/managers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamtournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the unique tournaments that a specific basketball team has participated in."
    id: The ID of the team for which you want to retrieve the tournaments
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/team/{is_id}/tournaments"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamstandingsseasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the standings seasons for a specific basketball team."
    id: The ID of the team for which you want to retrieve the standings seasons
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/team/{is_id}/standings/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamplayersstatisticsseasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the players statistics seasons for a specific basketball team."
    id: The ID of the team for which you want to retrieve the players statistics seasons
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/team/{is_id}/players/statistics/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchhighlights(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get highlights for a specific basketball match."
    id: The ID of the match for which you want to get highlights
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/match/{is_id}/highlights"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def prematchform(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get pre-match form information for a specific basketball match."
    id: The ID of the match for which you want to get pre-match form
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/match/{is_id}/form"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of a specific basketball team."
    id: The ID of the team for which you want to retrieve the details
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/team/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchgraph(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get graph data for a specific basketball match."
    id: The ID of the match for which you want to get the graph
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/match/{is_id}/graph"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamnextmatches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the next matches for a specific basketball team."
    page: Zero-based page
        id: The ID of the team for which you want to retrieve the next matches
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/team/{is_id}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categoryflag(flag: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the flag image of a specific category in PNG format."
    flag: The flag name.
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/img/flag/{flag}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamtopplayersregularseason(seasonid: int, is_id: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the top players for a specific basketball team during the regular season."
    seasonid: The season ID for which you want to retrieve the team's best players
        id: The ID of the team for which you want to retrieve the best players
        tournamentid: The unique tournament ID for which you want to retrieve the team's best players
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/team/{is_id}/tournament/{tournamentid}/season/{seasonid}/best-players/regularseason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamlogoimage(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the logo image for a specific basketball team using the team ID. The image is in PNG format."
    id: The team ID for which you want to retrieve the logo image
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/team/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information of a specific basketball match using the match ID."
    id: The ID of the match for which you want to get information
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/match/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchteamshotmap(teamid: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the shotmap of a team in a basketball match."
    teamid: The team id of the match you want to get shotmap.
        id: The id of the match you want to get shotmap.
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/match/{is_id}/team/{teamid}/shotmap"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueseasons(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the seasons for a specific basketball league using the tournament ID."
    tournamentid: The unique tournament ID for which you want to retrieve the league's seasons
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchtvchanneldetails(is_id: int, channid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of a specific TV channel broadcasting a specific Football match."
    id: The ID of the match you want to retrieve the channel details for.
        channid: The ID of the channel you want to retrieve the details for.
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/tv/channel/{channid}/event/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def livematches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of all live basketball matches."
    
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/matches/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamtopplayersplayoffs(tournamentid: int, seasonid: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the top players for a specific basketball team during the playoffs."
    tournamentid: The unique tournament ID for which you want to retrieve the team's best players
        seasonid: The season ID for which you want to retrieve the team's best players
        id: The ID of the team for which you want to retrieve the best players
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/team/{is_id}/tournament/{tournamentid}/season/{seasonid}/best-players/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchschedules(day: int, year: int, month: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of basketball matches scheduled on a specific day."
    day: The day for which you want to retrieve the schedules
        year: The year for which you want to retrieve the schedules
        month: The month for which you want to retrieve the schedules
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/matches/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguehomestandings(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the home standings for a specific basketball league in a given tournament and season."
    seasonid: The season ID for which you want to retrieve the league's home standings
        tournamentid: The unique tournament ID for which you want to retrieve the league's home standings
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/standings/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueshotactionsareasplayoffs(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the shot actions areas for the basketball league during the playoffs."
    tournamentid: The unique tournament id you want to retrieve the league's best players types.
        seasonid: The season id you want to retrieve the league's best players types.
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/shot-actions/areas/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguegroupmatches(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the groups in the basketball league."
    tournamentid: The group tournament ID for which you want to retrieve the matches.
        seasonid: The season ID for which you want to retrieve the matches.
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/group/tournament/{tournamentid}/season/{seasonid}/matches"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguehometeamevents(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get home last 5 matches for a specific basketball league in a given tournament and season."
    tournamentid: The unique tournament ID for which you want to retrieve the league's home team events
        seasonid: The season ID for which you want to retrieve the league's home team events
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/team-events/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueawayteamevents(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get away last 5 matches for a specific basketball league in a given tournament and season."
    tournamentid: The unique tournament ID for which you want to retrieve the league's away team events
        seasonid: The season ID for which you want to retrieve the league's away team events
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/team-events/away"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguetotalstandings(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the total standings for a specific basketball league in a given tournament and season."
    tournamentid: The unique tournament ID for which you want to retrieve the league's total standings
        seasonid: The season ID for which you want to retrieve the league's total standings
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/standings/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguetopplayersregularseason(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the top players of a specific basketball league's regular season using the tournament and season IDs."
    seasonid: The season ID for which you want to retrieve the league's best players
        tournamentid: The unique tournament ID for which you want to retrieve the league's best players
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/best-players/regularseason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueoveralltopplayers(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the best players overall in the basketball league."
    seasonid: The season id you want to retrieve the league's best players.
        tournamentid: The unique tournament id you want to retrieve the league's best players.
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/best-players/overall"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguelastmatches(seasonid: int, page: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the last matches for a specific basketball league using the tournament and season IDs, and a zero-based page number."
    seasonid: The season ID for which you want to retrieve the league's last matches
        page: Zero-based page number
        tournamentid: The unique tournament ID for which you want to retrieve the league's last matches
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/matches/last/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamplaceholderimage(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the team placeholder image in SVG format."
    
    """
    url = f"https://basketapi1.p.rapidapi.com/api/placeholder/team.svg"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguecuptrees(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the cup trees for a specific basketball league using the tournament and season IDs."
    seasonid: The season ID for which you want to retrieve the league's cup trees
        tournamentid: The unique tournament ID for which you want to retrieve the league's cup trees
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/cuptrees"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueplayoffspergametopplayers(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the best players in the basketball league per game during the playoffs."
    seasonid: The season id you want to retrieve the league's best players.
        tournamentid: The unique tournament id you want to retrieve the league's best players.
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/best-players/per-game/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguetopplayerstypes(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the types of best players in the basketball league."
    seasonid: The season id you want to retrieve the league's best players types.
        tournamentid: The unique tournament id you want to retrieve the league's best players types.
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/best-players/types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tvcountries(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of countries and their respective TV channels broadcasting a specific Football match."
    id: The ID of the match you want to retrieve the TV countries for.
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/tv/country/all/event/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueregularseasonpergametopplayers(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the best players in the basketball league per game during the regular season."
    tournamentid: The unique tournament id you want to retrieve the league's best players.
        seasonid: The season id you want to retrieve the league's best players.
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/best-players/per-game/regularSeason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueoveralltopteams(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the best teams overall in the basketball league."
    seasonid: The season id you want to retrieve the league's best teams.
        tournamentid: The unique tournament id you want to retrieve the league's best teams.
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/best-teams/overall"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueregularseasontopteams(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the best teams in the basketball league during the regular season."
    tournamentid: The unique tournament id you want to retrieve the league's best teams.
        seasonid: The season id you want to retrieve the league's best teams.
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/best-teams/regularSeason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueawaystandings(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the away standings for a specific basketball league in a given tournament and season."
    tournamentid: The unique tournament ID for which you want to retrieve the league's away standings
        seasonid: The season ID for which you want to retrieve the league's away standings
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/standings/away"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguetopplayersplayoffs(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the top players of a specific basketball league's playoffs using the tournament and season IDs."
    tournamentid: The unique tournament ID for which you want to retrieve the league's best players
        seasonid: The season ID for which you want to retrieve the league's best players
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/best-players/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguerounds(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the rounds in the basketball league."
    seasonid: The season id you want to retrieve the league's rounds.
        tournamentid: The unique tournament id you want to retrieve the league's rounds.
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/rounds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguelogoimage(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the logo image for a specific basketball league using the tournament ID. The image is in PNG format."
    tournamentid: The unique tournament ID for which you want to retrieve the league logo image
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournamentplaceholderimage(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the tournament placeholder image in PNG format."
    
    """
    url = f"https://basketapi1.p.rapidapi.com/api/placeholder/tournament.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguedetails(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details for a specific basketball league using the tournament ID."
    tournamentid: The unique tournament ID for which you want to retrieve the league's details
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categoryschedules(is_id: int, month: int, year: int, day: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the schedules of the day for a specific basketball category using the category ID and the date."
    id: The category ID for which you want to retrieve the schedules
        month: The month for which you want to retrieve the schedules
        year: The year for which you want to retrieve the schedules
        day: The day for which you want to retrieve the schedules
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/category/{is_id}/events/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamtransfers(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the transfers of a specific basketball team."
    id: The ID of the team for which you want to retrieve the transfers
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/team/{is_id}/transfers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueoverallpergametopplayers(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the best players in the basketball league per game overall."
    tournamentid: The unique tournament id you want to retrieve the league's best players.
        seasonid: The season id you want to retrieve the league's best players.
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/best-players/per-game"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueshotactionsareasregularseason(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the shot actions areas for the basketball league during the regular season."
    tournamentid: The unique tournament id you want to retrieve the league's best players types.
        seasonid: The season id you want to retrieve the league's best players types.
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/shot-actions/areas/regularSeason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguenextmatches(seasonid: int, tournamentid: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the next matches for a specific basketball league using the tournament and season IDs, and a zero-based page number."
    seasonid: The season ID for which you want to retrieve the league's next matches
        tournamentid: The unique tournament ID for which you want to retrieve the league's next matches
        page: Zero-based page number
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchh2hduel(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get head-to-head duel information for a specific basketball match."
    id: The ID of the match for which you want to get head-to-head duel
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/match/{is_id}/duel"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguemedia(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the media for a specific basketball league using the tournament ID."
    tournamentid: The unique tournament ID for which you want to retrieve the league media
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available basketball categories."
    
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguetotalteamevents(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the both home and away last 5 matches for a specific basketball league in a given tournament and season."
    seasonid: The season ID for which you want to retrieve the league's total team events
        tournamentid: The unique tournament ID for which you want to retrieve the league's total team events
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/team-events/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def managerplaceholderimage(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the manager placeholder image in PNG format."
    
    """
    url = f"https://basketapi1.p.rapidapi.com/api/placeholder/player.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguegroups(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the groups in the basketball league."
    seasonid: The season id you want to retrieve the league's groupd.
        tournamentid: The unique tournament id you want to retrieve the league's groups.
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/groups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categorytournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of all leagues belonging to a specific basketball category using the category ID."
    id: The category ID for which you want to retrieve all leagues
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/all/category/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueplayoffstopteams(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the best teams in the basketball league during the playoffs."
    seasonid: The season id you want to retrieve the league's best teams.
        tournamentid: The unique tournament id you want to retrieve the league's best teams.
        
    """
    url = f"https://basketapi1.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/best-teams/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

