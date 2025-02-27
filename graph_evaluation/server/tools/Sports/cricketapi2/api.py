import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search cricket players, teams, and tournaments using a search term."
    term: The search term to find cricket-related entities.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/search/{term}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamtournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of unique tournaments for a specific Cricket team by providing its team ID."
    id: The ID of the team for which you want to retrieve the tournaments.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/team/{is_id}/tournaments"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamplayers(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of players for a cricket team using their team ID."
    id: The ID of the team for which you want to retrieve the players.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/team/{is_id}/players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teammedia(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get media for a cricket team using their team ID."
    id: The ID of the team for which you want to retrieve the media.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/team/{is_id}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamstandingsseasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the standing seasons for a cricket team by specifying the team ID."
    id: The ID of the cricket team for which you want to retrieve the standing seasons.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/team/{is_id}/standings/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
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
    url = f"https://cricketapi21.p.rapidapi.com/api/img/flag/{flag}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerlastmatches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the last matches for a cricket player by specifying the player ID and the page number (starting from 0)."
    id: The ID of the cricket player for which you want to retrieve the last matches.
        page: The page number (starting from 0) of the last matches you want to retrieve.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/player/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamplayersstatisticsseasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of seasons with player statistics for a cricket team using their team ID."
    id: The ID of the team for which you want to retrieve the players' statistics seasons.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/team/{is_id}/players/statistics/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information about a cricket player using their player ID."
    id: The ID of the player for which you want to retrieve the details.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/player/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamlogoimage(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the logo image for a cricket team in PNG format, using the team's ID."
    teamid: The ID of the cricket team for which you want to retrieve the logo image.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/team/{teamid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguelogoimage(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the logo image for a cricket league in PNG format, using the league's ID."
    tournamentid: The ID of the cricket league for which you want to retrieve the logo image.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/tournament/{tournamentid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
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
    url = f"https://cricketapi21.p.rapidapi.com/api/placeholder/tournament.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerimage(playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the image of a cricket player as a PNG file using their player ID."
    playerid: The ID of the player for which you want to retrieve the image.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/player/{playerid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
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
    url = f"https://cricketapi21.p.rapidapi.com/api/tv/country/all/event/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguemedia(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get media of a cricket league using the unique tournament ID."
    tournamentid: The unique tournament ID for which you want to retrieve the league's media.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/tournament/{tournamentid}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information about a cricket team using their team ID."
    id: The ID of the team for which you want to retrieve the details.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/team/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchodds(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the odds for a specific cricket match."
    id: The ID of the cricket match for which you want to retrieve odds.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/match/{is_id}/odds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchincidents(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the incidents for a specific Cricket match using its match ID."
    id: The ID of the Cricket match for which you want to get the incidents.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/match/{is_id}/incidents"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playeroverallstatistics(is_id: int, seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the overall statistics for a specific cricket player for a given tournament and season."
    id: The ID of the cricket player for which you want to retrieve the statistics.
        seasonid: The season ID for which you want to retrieve the player's overall statistics.
        tournamentid: The unique tournament ID for which you want to retrieve the player's overall statistics.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/statistics/overall"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playernearmatches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of upcoming and recent matches for a cricket player using their player ID."
    id: The ID of the player for which you want to retrieve the near matches.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/player/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def livematches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get live cricket matches that are currently taking place."
    
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/matches/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamnearmatches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of cricket matches that the team with the specified ID has in the near future."
    id: The ID of the team for which you want to retrieve near matches.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/team/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
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
    url = f"https://cricketapi21.p.rapidapi.com/api/placeholder/team.svg"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamnextmatches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the next matches for a cricket team by specifying the team ID and the page number (starting from 0)."
    id: The ID of the cricket team for which you want to retrieve the next matches.
        page: The page number (starting from 0) of the next matches you want to retrieve.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/team/{is_id}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchlineups(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the lineups for a specific Cricket match using its match ID."
    id: The ID of the Cricket match for which you want to get the lineups.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/match/{is_id}/lineups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguetotalstandings(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the total standings for a specific Cricket league for a given season by providing its unique tournament ID and season ID."
    seasonid: The season ID for which you want to retrieve the league's total standings.
        tournamentid: The unique tournament ID for which you want to retrieve the league's total standings.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/tournament/{tournamentid}/season/{seasonid}/standings/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchtvchanneldetails(channid: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of a specific TV channel broadcasting a specific Football match."
    channid: The ID of the channel you want to retrieve the details for.
        id: The ID of the match you want to retrieve the channel details for.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/tv/channel/{channid}/event/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerstatisticsseasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation returns the statistics for a given cricket player by season."
    id: The ID of the cricket player for which you want to retrieve the statistics.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/player/{is_id}/statistics/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categorytournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all cricket leagues associated with a specific category."
    id: The ID of the cricket category for which you want to retrieve all leagues.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/tournament/all/category/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguelastmatches(tournamentid: int, seasonid: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of last matches for a cricket league using the tournament ID, season ID, and page number."
    tournamentid: The unique tournament ID for which you want to retrieve the league's last matches.
        seasonid: The season ID for which you want to retrieve the league's last matches.
        page: Zero-based page number.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/tournament/{tournamentid}/season/{seasonid}/matches/last/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def headtoheadmatches(customid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get head-to-head match data for a specific Cricket match using its custom ID."
    customid: The custom ID of the Cricket match for which you want to get the head-to-head matches.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/match/{customid}/h2h"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguedetails(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details of a cricket league using the unique tournament ID."
    tournamentid: The unique tournament ID for which you want to retrieve the league's details.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/tournament/{tournamentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueseasons(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of seasons for a cricket league using the unique tournament ID."
    tournamentid: The unique tournament ID for which you want to retrieve the league's seasons.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/tournament/{tournamentid}/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchinnings(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the innings for a specific Cricket match using its match ID."
    id: The ID of the Cricket match for which you want to get the innings.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/match/{is_id}/innings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchvotes(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the votes for a specific cricket match."
    id: The ID of the cricket match for which you want to retrieve votes.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/match/{is_id}/votes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categoryschedules(year: int, day: int, is_id: int, month: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get cricket schedules of a specific day for a given category."
    year: The year for which you want to retrieve the schedules.
        day: The day of the month for which you want to retrieve the schedules.
        id: The ID of the cricket category for which you want to retrieve the schedules.
        month: The month for which you want to retrieve the schedules.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/category/{is_id}/events/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueoveralltopplayers(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the overall top players for a specific cricket league in a given season, including player statistics and other relevant information."
    tournamentid: The ID of the cricket tournament for which you want to retrieve the overall top players.
        seasonid: The ID of the cricket season for which you want to retrieve the overall top players.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/tournament/{tournamentid}/season/{seasonid}/top-players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguetotalteamevents(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the total team events for a specific Cricket league for a given season by providing its unique tournament ID and season ID."
    tournamentid: The unique tournament ID for which you want to retrieve the league's total team events.
        seasonid: The season ID for which you want to retrieve the league's total team events.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/tournament/{tournamentid}/season/{seasonid}/team-events/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def prematchform(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get pre-match form information for a specific cricket match."
    id: The ID of the cricket match for which you want to get pre-match form information.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/match/{is_id}/form"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_player_placeholder(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the player placeholder image in PNG format."
    
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/placeholder/player.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of all cricket categories."
    
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/tournament/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguenextmatches(seasonid: int, page: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of next matches for a cricket league using the tournament ID, season ID, and page number."
    seasonid: The season ID for which you want to retrieve the league's next matches.
        page: Zero-based page number.
        tournamentid: The unique tournament ID for which you want to retrieve the league's next matches.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/tournament/{tournamentid}/season/{seasonid}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamlastmatches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the last matches for a cricket team by specifying the team ID and the page number (starting from 0)."
    page: The page number (starting from 0) of the last matches you want to retrieve.
        id: The ID of the cricket team for which you want to retrieve the last matches.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/team/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information for a specific cricket match by providing the match ID."
    id: The ID of the cricket match for which you want to get the details.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/match/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
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
    url = f"https://cricketapi21.p.rapidapi.com/api/placeholder/manager.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchschedules(year: int, day: int, month: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation returns the cricket match schedules for the given date, including match timings, teams, and other relevant information."
    year: The year for which you want to retrieve the cricket match schedules (e.g., 2022).
        day: The day of the month for which you want to retrieve the cricket match schedules (1-31).
        month: The month for which you want to retrieve the cricket match schedules (1-12).
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/matches/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchh2hduel(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get head-to-head duel information for a specific cricket match."
    id: The ID of the cricket match for which you want to get head-to-head duel information.
        
    """
    url = f"https://cricketapi21.p.rapidapi.com/api/cricket/match/{is_id}/duel"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricketapi21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

