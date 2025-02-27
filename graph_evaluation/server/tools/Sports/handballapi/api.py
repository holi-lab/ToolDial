import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation allows you to search for handball players, teams, and tournaments based on the search term provided."
    term: The search term to use when searching for players, teams, and tournaments.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/search/{term}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerimage(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get image of a handball player by ID"
    id: The ID of the player for which you want to retrieve the image.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/player/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get handball match details by ID."
    id: The ID of the match you want to retrieve information for.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/match/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details of a handball player by ID"
    id: The ID of the player you want to retrieve the details for.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/player/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchbestplayers(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get best players for a handball match by ID."
    id: The ID of the match for which you want to get the best players data.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/match/{is_id}/best-players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchheadtoheadduel(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get headtohead resume for a handball match by ID."
    id: The ID of the football match for which you want to get head-to-head summary.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/match/{is_id}/duel"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def prematchform(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get pregame form for a handball match by ID."
    id: The ID of the football match for which you want to get pre-match form.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/match/{is_id}/form"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchincidents(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get incidents for a handball match by ID."
    id: The ID of the match you want to retrieve the incidents for.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/match/{is_id}/incidents"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchschedules(month: int, day: int, year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get handball matches by date."
    month: The month for which you want to retrieve the schedules (1-12).
        day: The day of the month for which you want to retrieve the schedules (1-31).
        year: The year for which you want to retrieve the schedules (e.g., 2022).
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/matches/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playeroverallstatistics(is_id: int, tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get overall player statistics for a handball player in a unique tournament season"
    id: The ID of the player you want to retrieve the overall statistics for.
        tournamentid: The unique tournament ID you want to retrieve the player's overall statistics for.
        seasonid: The season ID you want to retrieve the player's overall statistics for.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/player/{is_id}/unique-tournament/{tournamentid}/season/{seasonid}/statistics/overall"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def livematches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get live handball matches."
    
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/matches/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerstatisticsseasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player statistics seasons for a handball player by ID"
    id: The ID of the player you want to retrieve the statistics seasons for.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/player/{is_id}/statistics/season"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerlastmatches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get last matches of a handball player by ID"
    id: The ID of the player whose last matches you want to retrieve.
        page: The zero-based page number of the results you want to retrieve.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/player/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamnextmatches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get upcoming matches for a handball team by ID and page"
    id: The ID of the team you want to retrieve the next matches for.
        page: Zero-based page number.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/team/{is_id}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamstandingsseasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get seasons standings for a handball team by ID"
    id: The ID of the team you want to retrieve team standings seasons for.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/team/{is_id}/standings/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categoryschedules(month: int, is_id: int, year: int, day: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get handball matches by category and date."
    month: The month for which you want to retrieve the schedules (1-12).
        id: The category ID for which you want to retrieve the schedules.
        year: The year for which you want to retrieve the schedules (e.g., 2022).
        day: The day of the month for which you want to retrieve the schedules (1-31).
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/category/{is_id}/events/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchstatistics(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics for a handball match by ID."
    id: The ID of the match you want to retrieve the statistics for.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/match/{is_id}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchvotes(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get votes for a handball match by ID."
    id: The ID of the match for which you want to get the votes.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/match/{is_id}/votes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playernearmatches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get near matches of a handball player by ID"
    id: The ID of the player you want to retrieve the near matches for.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/player/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def headtoheadmatches(customid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get headtohead events for a handball match by custom ID."
    customid: The custom ID of the match you want to retrieve the head-to-head matches for.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/match/{customid}/h2h"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamlogoimage(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team image"
    id: The ID of the football team for which you want to retrieve the logo image.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/team/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchlineups(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get lineups and player statistics for a handball match by ID."
    id: The ID of the match you want to retrieve the lineups for.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/match/{is_id}/lineups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueawaystandings(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get away standings for a handball unique tournament and season"
    seasonid: The ID of the season for which you want to retrieve the away standings.
        tournamentid: The unique ID of the tournament for which you want to retrieve the away standings.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/unique-tournament/{tournamentid}/season/{seasonid}/standings/away"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueplayerstatisticstype(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player statistics types for a handball unique tournament and season"
    tournamentid: The unique tournament ID you want to retrieve the player's statistics type for.
        seasonid: The season ID you want to retrieve the player's statistics type for.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/unique-tournament/{tournamentid}/season/{seasonid}/player-statistics/types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamplayersstatisticsseasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player statistics seasons for a handball team by ID"
    id: The ID of the team you want to retrieve the players' statistics seasons for.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/team/{is_id}/player-statistics/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueoveralltopplayers(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get overall top players for a handball unique tournament and season"
    tournamentid: The unique ID of the tournament for which you want to retrieve the top players.
        seasonid: The ID of the season for which you want to retrieve the top players.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/unique-tournament/{tournamentid}/season/{seasonid}/top-players/overall"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamplayers(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get players for a handball team by ID"
    id: The ID of the team you want to retrieve the players for.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/team/{is_id}/players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueawayteamevents(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get away team events for a handball unique tournament and season"
    seasonid: The season ID you want to retrieve the league's away team events from.
        tournamentid: The unique tournament ID you want to retrieve the league's away team events from.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/unique-tournament/{tournamentid}/season/{seasonid}/team-events/away"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamnearmatches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get near matches for a handball team by ID"
    id: The ID of the team you want to retrieve the near matches for.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/team/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamtournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get unique tournaments for a handball team by ID"
    id: The ID of the team you want to retrieve the tournaments for.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/team/{is_id}/unique-tournaments"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchstreaks(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team streaks for a handball match by ID."
    id: The ID of the match for which you want to get the streaks data.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/match/{is_id}/streaks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchodds(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get odds for a handball match by ID."
    id: The ID of the match for which you want to get the odds.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/match/{is_id}/odds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details for a handball team by ID"
    id: The ID of the team you want to retrieve the details for.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/team/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguelastmatches(tournamentid: int, page: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get structured cup trees for a handball unique tournament season"
    tournamentid: The unique ID of the tournament for which you want to retrieve the last matches.
        page: The zero-based page number.
        seasonid: The season ID for which you want to retrieve the last matches.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/unique-tournament/{tournamentid}/season/{seasonid}/matches/last/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamlastmatches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get previous matches for a handball team by ID and page"
    page: Zero-based page number.
        id: The ID of the team you want to retrieve the last matches for.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/team/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchhighlights(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get highlights of a handball match by ID."
    id: The ID of the football match for which you want to get highlights.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/match/{is_id}/highlights"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguetotalteamevents(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get total team events for a handball unique tournament and season"
    seasonid: The season ID you want to retrieve the league's total team events from.
        tournamentid: The unique tournament ID you want to retrieve the league's total team events from.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/unique-tournament/{tournamentid}/season/{seasonid}/team-events/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguepergametopplayers(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get overall top players per game for a handball unique tournament and season"
    seasonid: The ID of the season for which you want to retrieve the top players.
        tournamentid: The unique ID of the tournament for which you want to retrieve the top players.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/unique-tournament/{tournamentid}/season/{seasonid}/top-players-per-game/all/overall"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get handball categories"
    
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/tournament/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamoveralltopplayers(is_id: int, tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get overall top players for a handball team in a unique tournament and season"
    id: The ID of the team you want to retrieve the best players for.
        tournamentid: The unique tournament ID you want to retrieve the team's best players for.
        seasonid: The season ID you want to retrieve the team's best players for.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/team/{is_id}/unique-tournament/{tournamentid}/season/{seasonid}/top-players/overall"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguegroupmatches(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get group events for a handball tournament season"
    tournamentid: The group tournament ID for which you want to retrieve the matches.
        seasonid: The season ID for which you want to retrieve the matches.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/tournament/{tournamentid}/season/{seasonid}/matches"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueeventsbyround(seasonid: int, tournamentid: int, round: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get matches by round for a handball unique tournament season"
    seasonid: The season ID for which you want to retrieve the league's events.
        tournamentid: The unique tournament ID for which you want to retrieve the league's events.
        round: The round number.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/unique-tournament/{tournamentid}/season/{seasonid}/matches/round/{round}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueoveralltopteams(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get overall top teams for a handball unique tournament and season"
    tournamentid: The unique ID of the tournament for which you want to retrieve the top teams.
        seasonid: The ID of the season for which you want to retrieve the top teams.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/unique-tournament/{tournamentid}/season/{seasonid}/top-teams/overall"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguelogoimage(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get image of a handball unique tournament by ID"
    tournamentid: The ID of the football league for which you want to retrieve the logo image.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/unique-tournament/{tournamentid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguenextmatches(page: int, tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get structured cup trees for a handball unique tournament season"
    page: The zero-based page number.
        tournamentid: The unique ID of the tournament for which you want to retrieve the next matches.
        seasonid: The season ID for which you want to retrieve the next matches.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/unique-tournament/{tournamentid}/season/{seasonid}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguemedia(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get media for a handball unique tournament"
    tournamentid: The ID of the tournament for which you want to retrieve the media.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/unique-tournament/{tournamentid}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguerounds(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get rounds for a handball unique tournament season"
    tournamentid: The unique ID of the tournament for which you want to retrieve the rounds.
        seasonid: The season ID for which you want to retrieve the rounds.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/unique-tournament/{tournamentid}/season/{seasonid}/rounds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueeventsbyroundandslug(slug: str, tournamentid: int, round: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get matches by round and slug for a handball unique tournament season"
    slug: The round slug.
        tournamentid: The unique tournament ID for which you want to retrieve the league's events.
        round: The round number.
        seasonid: The season ID for which you want to retrieve the league's events.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/unique-tournament/{tournamentid}/season/{seasonid}/matches/round/{round}/slug/{slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguedetails(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details of a handball unique tournament by ID"
    tournamentid: The ID of the football league for which you want to retrieve the details.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/unique-tournament/{tournamentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueseasons(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get seasons of a handball unique tournament by ID"
    tournamentid: The unique tournament ID for which you want to retrieve the league's seasons.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/unique-tournament/{tournamentid}/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguegroups(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get groups for a handball unique tournament season"
    tournamentid: The unique ID of the tournament for which you want to retrieve the groups.
        seasonid: The season ID for which you want to retrieve the groups.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/unique-tournament/{tournamentid}/season/{seasonid}/groups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categorytournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tournaments by category."
    id: The category ID for which you want to retrieve all leagues.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/unique-tournament/all/category/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguecuptrees(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get structured cup trees for a handball unique tournament season"
    seasonid: The ID of the season for which you want to retrieve the cup trees.
        tournamentid: The unique ID of the football league for which you want to retrieve the cup trees.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/unique-tournament/{tournamentid}/season/{seasonid}/cuptrees"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguehometeamevents(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get home team events for a handball unique tournament and season"
    tournamentid: The unique tournament ID you want to retrieve the league's home team events from.
        seasonid: The season ID you want to retrieve the league's home team events from.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/unique-tournament/{tournamentid}/season/{seasonid}/team-events/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguehomestandings(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get home standings for a handball unique tournament and season"
    tournamentid: The unique tournament ID you want to retrieve the league's home standings from.
        seasonid: The season ID you want to retrieve the league's home standings from.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/unique-tournament/{tournamentid}/season/{seasonid}/standings/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguetotalstandings(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get total standings for a handball unique tournament and season"
    seasonid: The season ID you want to retrieve the league's total standings from.
        tournamentid: The unique tournament ID you want to retrieve the league's total standings from.
        
    """
    url = f"https://handballapi.p.rapidapi.com/api/handball/unique-tournament/{tournamentid}/season/{seasonid}/standings/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "handballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

