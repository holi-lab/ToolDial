import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Perform a search for players, teams, and tournaments in Ice Hockey using the provided search term."
    term: The search term to use for finding players, teams, and tournaments.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/search/{term}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
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
    url = f"https://icehockeyapi.p.rapidapi.com/api/placeholder/manager.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
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
    url = f"https://icehockeyapi.p.rapidapi.com/api/tv/channel/{channid}/event/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
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
    url = f"https://icehockeyapi.p.rapidapi.com/api/placeholder/tournament.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
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
    url = f"https://icehockeyapi.p.rapidapi.com/api/placeholder/team.svg"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
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
    url = f"https://icehockeyapi.p.rapidapi.com/api/tv/country/all/event/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def managerdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of a specific Ice Hockey manager using the manager id."
    id: The manager id for which you want to retrieve the details.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/manager/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
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
    url = f"https://icehockeyapi.p.rapidapi.com/api/img/flag/{flag}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def managerimage(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the image for a specific Ice Hockey manager in PNG format by providing the manager ID."
    id: The manager ID for which you want to retrieve the image.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/manager/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def managerlastmatches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the last matches of a specific Ice Hockey manager using the manager id and a zero-based page number."
    page: Zero-based page number.
        id: The manager id for which you want to retrieve the last matches.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/manager/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerplaceholderimage(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the player placeholder image in PNG format."
    
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/placeholder/player.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguetopplayerstypes(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the top players types for a specific Ice Hockey league by providing the tournament ID and season ID."
    seasonid: The season ID for which you want to get the league's best players types.
        tournamentid: The unique tournament ID for which you want to get the league's best players types.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/best-players/types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamlogoimage(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the logo image for a specific Ice Hockey team in PNG format by providing the team ID."
    teamid: The team ID for which you want to retrieve the logo image.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/team/{teamid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueawayteamevents(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the away team events for a specific Ice Hockey league by providing the tournament ID and season ID."
    seasonid: The season ID for which you want to get the league's away team events.
        tournamentid: The unique tournament ID for which you want to get the league's away team events.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/team-events/away"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerplayoffsstatistics(tournamentid: int, seasonid: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the playoffs statistics for a specific Ice Hockey player by providing the player ID, tournament ID, and season ID."
    tournamentid: The unique tournament ID for which you want to get the player's playoffs statistics.
        seasonid: The season ID for which you want to get the player's playoffs statistics.
        id: The ID of the player for which you want to get the playoffs statistics.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/statistics/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguetotalstandings(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the total standings for a specific Ice Hockey league by providing the tournament ID and season ID."
    tournamentid: The unique tournament ID for which you want to get the league's total standings.
        seasonid: The season ID for which you want to get the league's total standings.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/standings/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerregularseasonstatistics(is_id: int, tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the regular season statistics for a specific Ice Hockey player by providing the player ID, tournament ID, and season ID."
    id: The ID of the player for which you want to get the regular season statistics.
        tournamentid: The unique tournament ID for which you want to get the player's regular season statistics.
        seasonid: The season ID for which you want to get the player's regular season statistics.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/statistics/regularSeason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueoveralltopplayerspergame(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the top players types for a specific Ice Hockey league by providing the tournament ID and season ID."
    tournamentid: The unique tournament ID for which you want to get the league's best players.
        seasonid: The season ID for which you want to get the league's best players.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/best-players/per-game/overall"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamplayers(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the players for a specific Ice Hockey team by providing the team ID."
    id: The ID of the team for which you want to get the players.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/team/{is_id}/players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamregularseasontopplayers(seasonid: int, is_id: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the top players of a particular ice hockey team for a given regular season and tournament."
    seasonid: The ID of the season for which you want to retrieve the team's top players.
        id: The ID of the ice hockey team whose top players you want to retrieve.
        tournamentid: The ID of the tournament for which you want to retrieve the team's top players.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/team/{is_id}/tournament/{tournamentid}/season/{seasonid}/best-players/regularSeason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamlastmatches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of last matches played by a specific Ice Hockey team."
    id: The ID of the Ice Hockey team for which you want to retrieve last matches.
        page: The page number (zero-indexed) of the results you want to retrieve.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/team/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playeroverallstatistics(is_id: int, tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the regular season statistics for a specific Ice Hockey player by providing the player ID, tournament ID, and season ID."
    id: The ID of the player for which you want to get the overall statistics.
        tournamentid: The unique tournament ID for which you want to get the player's overall statistics.
        seasonid: The season ID for which you want to get the player's overall statistics.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/statistics/overall"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details for a specific Ice Hockey team by providing the team ID."
    id: The ID of the team for which you want to get the details.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/team/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamnextmatches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of upcoming matches of a specific Ice Hockey team."
    page: The page number (zero-indexed) of the results you want to retrieve.
        id: The ID of the Ice Hockey team for which you want to retrieve next matches.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/team/{is_id}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguemedia(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the media for a specific Ice Hockey league by providing the tournament ID."
    tournamentid: The unique tournament ID for which you want to retrieve the league media.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueplayoffstopplayerspergame(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the top players types for a specific Ice Hockey league by providing the tournament ID and season ID."
    seasonid: The season ID for which you want to get the league's best players.
        tournamentid: The unique tournament ID for which you want to get the league's best players.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/best-players/per-game/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerimage(playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the image for a specific Ice Hockey player in PNG format by providing the player ID."
    playerid: The player ID for which you want to retrieve the image.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/player/{playerid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueawaystandings(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the away standings for a specific Ice Hockey league by providing the tournament ID and season ID."
    tournamentid: The unique tournament ID for which you want to get the league's away standings.
        seasonid: The season ID for which you want to get the league's away standings.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/standings/away"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguenextmatches(seasonid: int, page: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the next matches for a specific Ice Hockey league by providing the tournament ID, season ID, and the page number."
    seasonid: The season ID for which you want to get the league's next matches.
        page: Zero-based page number.
        tournamentid: The unique tournament ID for which you want to get the league's next matches.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueoveralltopplayers(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the top players for a specific Ice Hockey league by providing the tournament ID and season ID."
    seasonid: The season ID for which you want to get the league's best players.
        tournamentid: The unique tournament ID for which you want to get the league's best players.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/best-players/overall"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerstatisticsseasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the statistics seasons for a specific Ice Hockey player by providing the player ID."
    id: The ID of the player for which you want to get the statistics seasons.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/player/{is_id}/statistics/season"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueseasons(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the seasons for a specific Ice Hockey league by providing the tournament ID."
    tournamentid: The unique tournament ID for which you want to retrieve the league's seasons.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguelastmatches(seasonid: int, tournamentid: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the last matches for a specific Ice Hockey league by providing the tournament ID, season ID, and the page number."
    seasonid: The season ID for which you want to get the league's last matches.
        tournamentid: The unique tournament ID for which you want to get the league's last matches.
        page: Zero-based page number.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/matches/last/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueplayoffstopteams(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the top teams for a specific Ice Hockey league by providing the tournament ID and season ID."
    tournamentid: The unique tournament ID for which you want to get the league's best teams.
        seasonid: The season ID for which you want to get the league's best teams.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/best-teams/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categoryschedules(year: int, is_id: int, day: int, month: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the schedules of ice hockey matches for a given date and category, including match timings, teams, and other relevant information."
    year: The year for which you want to retrieve the schedules (e.g., 2022).
        id: The ID of the category for which you want to retrieve the schedules.
        day: The day of the month for which you want to retrieve the schedules (1-31).
        month: The month for which you want to retrieve the schedules (1-12).
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/category/{is_id}/events/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playernearmatches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the near matches for a specific Ice Hockey player by providing the player ID."
    id: The ID of the player for which you want to get the near matches.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/player/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamnearmatches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the near matches for a specific Ice Hockey team by providing the team ID."
    id: The ID of the team for which you want to get the near matches.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/team/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueregularseasontopplayerspergame(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the top players types for a specific Ice Hockey league by providing the tournament ID and season ID."
    tournamentid: The unique tournament ID for which you want to get the league's best players.
        seasonid: The season ID for which you want to get the league's best players.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/best-players/per-game/regularSeason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all ice hockey categories."
    
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/tournament/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueplayoffstopplayers(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the top players for a specific Ice Hockey league during playoffs by providing the tournament ID and season ID."
    seasonid: The season ID for which you want to get the league's best players.
        tournamentid: The unique tournament ID for which you want to get the league's best players.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/best-players/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguetotalteamevents(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the total team events for a specific Ice Hockey league by providing the tournament ID and season ID."
    tournamentid: The unique tournament ID for which you want to get the league's total team events.
        seasonid: The season ID for which you want to get the league's total team events.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/team-events/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguecuptrees(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the cup trees for a specific Ice Hockey league by providing the tournament ID and season ID."
    seasonid: The season ID for which you want to retrieve the league cup trees.
        tournamentid: The unique tournament ID for which you want to retrieve the league cup trees.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/cuptrees"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerlastmatches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of last matches played by a specific Ice Hockey player."
    id: The ID of the Ice Hockey player for which you want to retrieve last matches.
        page: The page number (zero-indexed) of the results you want to retrieve.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/player/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguelogoimage(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the logo image for a specific Ice Hockey league in PNG format by providing the tournament ID."
    tournamentid: The unique tournament ID for which you want to retrieve the league logo image.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguehometeamevents(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the home team events for a specific Ice Hockey league by providing the tournament ID and season ID."
    seasonid: The season ID for which you want to get the league's home team events.
        tournamentid: The unique tournament ID for which you want to get the league's home team events.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/team-events/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playermedia(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the media for a specific Ice Hockey player by providing the player ID."
    id: The ID of the player for which you want to get the media.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/player/{is_id}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchodds(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the odds for a specific Ice Hockey match using the match id."
    id: The id of the match for which you want to retrieve odds.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/match/{is_id}/odds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categorytournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all ice hockey tournaments for a given category ID."
    id: The ID of the ice hockey category for which to retrieve tournaments.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/tournament/all/category/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchvotes(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the votes for a specific Ice Hockey match using the match id."
    id: The id of the match for which you want to retrieve votes.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/match/{is_id}/votes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchstreaks(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the streaks for a specific Ice Hockey match using the match id."
    id: The id of the match for which you want to retrieve match streaks.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/match/{is_id}/streaks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchh2hduel(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the head-to-head duel for a specific Ice Hockey match using the match id."
    id: The id of the match for which you want to retrieve the head-to-head duel.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/match/{is_id}/duel"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchplayerstatistics(playerid: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed statistics for a specific player in an Ice Hockey match using the match ID and player ID."
    playerid: The ID of the player for which you want to get statistics.
        id: The ID of the Ice Hockey match for which you want to get player statistics.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/match/{is_id}/player/{playerid}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueregularseasontopteams(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the top teams for a specific Ice Hockey league by providing the tournament ID and season ID."
    tournamentid: The unique tournament ID for which you want to get the league's best teams.
        seasonid: The season ID for which you want to get the league's best teams.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/best-teams/regularSeason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchlineups(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information on the lineups for a specific Ice Hockey match using the match ID."
    id: The ID of the Ice Hockey match for which you want to get lineups.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/match/{is_id}/lineups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teammedia(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get media for a specific Ice Hockey team by providing the team ID."
    id: The ID of the team for which you want to get the media.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/team/{is_id}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def headtoheadmatches(customid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get head-to-head match data for a specific Ice Hockey match using its custom ID."
    customid: The custom ID of the Ice Hockey match for which you want to get head-to-head matches. This ID is obtained from the match.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/match/{customid}/h2h"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchstreakodds(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the streak odds for a specific Ice Hockey match using the match id."
    id: The id of the match for which you want to retrieve match streaks odds.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/match/{is_id}/streaks/odds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def prematchform(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the pre-match form for a specific Ice Hockey match using the match id."
    id: The id of the match for which you want to retrieve the pre-match form.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/match/{is_id}/form"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchincidents(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation returns incidents for a specific ice hockey match, including red cards, yellow cards, and other relevant data."
    id: The ID of the match for which you want to retrieve incidents.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/match/{is_id}/incidents"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details for a specific Ice Hockey player by providing the player ID."
    id: The ID of the player for which you want to get the details.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/player/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamplayoffstopplayers(tournamentid: int, seasonid: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation returns the best players of an Ice Hockey team for the playoffs of a specific tournament and season."
    tournamentid: The ID of the tournament for which you want to retrieve the team's best players.
        seasonid: The ID of the season for which you want to retrieve the team's best players.
        id: The ID of the team for which you want to retrieve the best players.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/team/{is_id}/tournament/{tournamentid}/season/{seasonid}/best-players/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamplayersstatisticsseasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the statistics of the players in a particular ice hockey team for each season."
    id: The ID of the ice hockey team whose players' statistics seasons you want to retrieve.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/team/{is_id}/players/statistics/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguehomestandings(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the home standings for a specific Ice Hockey league by providing the tournament ID and season ID."
    tournamentid: The unique tournament ID for which you want to get the league's home standings.
        seasonid: The season ID for which you want to get the league's home standings.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/standings/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchstatistics(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation returns statistics for a specific ice hockey match, including player statistics and other relevant data."
    id: The ID of the match for which you want to retrieve statistics.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/match/{is_id}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguedetails(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of a specific Ice Hockey league by providing the tournament ID."
    tournamentid: The unique tournament ID for which you want to get the league's details.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamtournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation returns the unique tournaments in which an Ice Hockey team has participated."
    id: The ID of the team for which you want to retrieve the tournaments.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/team/{is_id}/tournaments"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information for a specific ice hockey match by providing the match ID."
    id: The ID of the ice hockey match for which to retrieve details.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/match/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchschedules(day: int, year: int, month: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the schedules of ice hockey matches for a given date, including match timings, teams, and other relevant information."
    day: The day of the month for which you want to retrieve the schedules (1-31).
        year: The year for which you want to retrieve the schedules (e.g., 2022).
        month: The month for which you want to retrieve the schedules (1-12).
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/matches/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchbestplayers(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the best players for a specific Ice Hockey match using the match id."
    id: The id of the match for which you want to retrieve the best players.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/match/{is_id}/best-players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueregularseasontopplayers(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the top players for a specific Ice Hockey league during the regular season by providing the tournament ID and season ID."
    seasonid: The season ID for which you want to get the league's best players.
        tournamentid: The unique tournament ID for which you want to get the league's best players.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/best-players/regularSeason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchhighlights(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the highlights for a specific Ice Hockey match using the match id."
    id: The id of the match for which you want to retrieve highlights.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/match/{is_id}/highlights"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchmanagers(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the managers for a specific Ice Hockey match using the match id."
    id: The id of the match for which you want to retrieve managers.
        
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/match/{is_id}/managers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def livematches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all live ice hockey matches."
    
    """
    url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/matches/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

