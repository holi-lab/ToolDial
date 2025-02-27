import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for E-Sports players, teams, and tournaments by providing a search term."
    term: The search term.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/search/{term}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
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
    url = f"https://esportapi1.p.rapidapi.com/api/placeholder/tournament.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
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
    url = f"https://esportapi1.p.rapidapi.com/api/placeholder/player.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
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
    url = f"https://esportapi1.p.rapidapi.com/api/img/flag/{flag}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
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
    url = f"https://esportapi1.p.rapidapi.com/api/placeholder/team.svg"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
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
    url = f"https://esportapi1.p.rapidapi.com/api/tv/country/all/event/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamlogoimage(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the team logo image in PNG format by providing the team ID."
    teamid: The team ID for which you want to retrieve the logo image.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/team/{teamid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teammedia(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the media for a specific E-Sports team by providing the team ID."
    id: The ID of the team for which you want to retrieve the media.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/team/{is_id}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamlastmatches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the last matches for a specific E-Sports team by providing the team ID and the page number."
    page: The zero-based page number.
        id: The ID of the team for which you want to retrieve the last matches.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/team/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamnearmatches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the near matches for a specific E-Sports team by providing the team ID."
    id: The ID of the team for which you want to retrieve the near matches.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/team/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
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
    url = f"https://esportapi1.p.rapidapi.com/api/tv/channel/{channid}/event/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamnextmatches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the next matches for a specific E-Sports team by providing the team ID and the page number."
    id: The ID of the team for which you want to retrieve the next matches.
        page: The zero-based page number.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/team/{is_id}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def eventstreaks(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get streaks of a specific E-Sports event by providing its ID."
    id: The ID of the event for which you want to get event streaks.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/event/{is_id}/streaks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
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
    url = f"https://esportapi1.p.rapidapi.com/api/placeholder/manager.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueseasoninfo(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the season information for a specific E-Sports league by providing the tournament ID and season ID."
    tournamentid: The unique tournament ID for which you want to retrieve the league's season info.
        seasonid: The season ID for which you want to retrieve the league's season info.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/tournament/{tournamentid}/season/{seasonid}/info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gamelineups(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get game lineups for a specific E-Sports game by providing the game ID."
    id: The ID of the game for which you want to get the lineups.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/game/{is_id}/lineups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gamemapimage(mapid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the map image in PNG format by providing the map ID."
    mapid: The map ID for which you want to retrieve the image.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/map/{mapid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguelastmatches(page: int, seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the last matches for a specific E-Sports league by providing the tournament ID, season ID, and page number."
    page: Zero-based page number.
        seasonid: The season ID for which you want to retrieve the league's last matches.
        tournamentid: The unique tournament ID for which you want to retrieve the league's last matches.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/tournament/{tournamentid}/season/{seasonid}/matches/last/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def characterimage(characterid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the character image in PNG format by providing the character ID."
    characterid: The character ID for which you want to retrieve the character image.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/character/{characterid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details for a specific E-Sports player by providing the player ID."
    id: The ID of the player for which you want to retrieve the details.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/player/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def livematches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get live E-Sports matches that are currently taking place."
    
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/matches/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguedetails(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of a specific E-Sports league by providing the tournament ID."
    tournamentid: The unique tournament ID for which you want to retrieve the league's details.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/tournament/{tournamentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gamestatistics(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get game statistics for a specific E-Sports game by providing the game ID."
    id: The ID of the game for which you want to get the statistics.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/game/{is_id}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchschedules(day: int, year: int, month: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get e-sports schedules for a specific day."
    day: The day you want to retrieve the schedules
        year: The year you want to retrieve the schedules
        month: The month you want to retrieve the schedules
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/matches/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def eventhighlights(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get highlights of a specific E-Sports event by providing its ID."
    id: The ID of the event for which you want to get highlights.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/event/{is_id}/highlights"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information for a specific E-Sports event by providing the event ID."
    id: The ID of the event for which you want to get the details.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/event/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details for a specific E-Sports team by providing the team ID."
    id: The ID of the team for which you want to retrieve the details.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/team/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def eventlineups(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event lineups for a specific E-Sports event by providing the event ID."
    id: The ID of the event for which you want to get the lineups.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/event/{is_id}/lineups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available e-sports categories."
    
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/tournament/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def eventh2hduel(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get head-to-head duel information of a specific E-Sports event by providing its ID."
    id: The ID of the event for which you want to get head-to-head duel information.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/event/{is_id}/duel"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguelogoimage(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the league logo image in PNG format by providing the tournament ID."
    tournamentid: The unique tournament ID for which you want to retrieve the league logo image.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/tournament/{tournamentid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def eventvotes(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get votes of a specific E-Sports event by providing its ID."
    id: The ID of the event for which you want to get votes.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/event/{is_id}/votes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguenextmatches(seasonid: int, page: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the next matches for a specific E-Sports league by providing the tournament ID, season ID, and page number."
    seasonid: The season ID for which you want to retrieve the league's next matches.
        page: Zero-based page number.
        tournamentid: The unique tournament ID for which you want to retrieve the league's next matches.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/tournament/{tournamentid}/season/{seasonid}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categoryschedules(month: int, is_id: int, day: int, year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get e-sports schedules for a specific day from a specific category."
    month: The month you want to retrieve the schedules
        id: The category id you want to retrieve the schedules
        day: The day you want to retrieve the schedules
        year: The year you want to retrieve the schedules
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/category/{is_id}/events/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def eventgames(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event games for a specific E-Sports event by providing the event ID."
    id: The ID of the event for which you want to get the games.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/event/{is_id}/games"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categorytournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all leagues from a specific E-Sports category using the category ID."
    id: The category ID for which you want to retrieve all leagues.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/tournament/all/category/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguetotalstandings(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the total standings for a specific E-Sports league by providing the tournament ID and season ID."
    seasonid: The season ID for which you want to retrieve the league's total standings.
        tournamentid: The unique tournament ID for which you want to retrieve the league's total standings.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/tournament/{tournamentid}/season/{seasonid}/standings/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gamebans(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get game bans for a specific E-Sports game by providing the game ID."
    id: The ID of the game for which you want to get the bans.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/game/{is_id}/bans"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gamerounds(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get game rounds for a specific E-Sports game by providing the game ID."
    id: The ID of the game for which you want to get the rounds.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/game/{is_id}/rounds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerimage(playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the player image in PNG format by providing the player ID."
    playerid: The player ID for which you want to retrieve the image.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/player/{playerid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguemedia(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the media related to a specific E-Sports league by providing the tournament ID."
    tournamentid: The unique tournament ID for which you want to retrieve the league media.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/tournament/{tournamentid}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def headtoheadmatches(customid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get head-to-head match data for a specific E-Sports event using its custom ID."
    customid: The custom ID of the event for which you want to get the head-to-head matches.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/event/{customid}/h2h"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguetotalteamevents(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get last 5 events for a specific E-Sports league by providing the tournament ID and season ID."
    seasonid: The season ID for which you want to retrieve the league's total team events.
        tournamentid: The unique tournament ID for which you want to retrieve the league's total team events.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/tournament/{tournamentid}/season/{seasonid}/team-events/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueseasons(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the seasons of a specific E-Sports league by providing the tournament ID."
    tournamentid: The unique tournament ID for which you want to retrieve the league's seasons.
        
    """
    url = f"https://esportapi1.p.rapidapi.com/api/esport/tournament/{tournamentid}/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esportapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

