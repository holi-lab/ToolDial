import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation allows you to search for football players, teams, and tournaments based on the search term provided."
    term: The search term to use when searching for players, teams, and tournaments.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/search/{term}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def managerdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details of a football manager, including their name, age, nationality, and other relevant information."
    id: The ID of the manager for whom you want to retrieve the details.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/manager/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def managerlastmatches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of the last matches played by a football manager, paginated by the given page number."
    id: The ID of the manager whose last matches you want to retrieve.
        page: The zero-based page number of the results you want to retrieve.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/manager/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playernationalteamstatistics(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the national team statistics for a specific Football player."
    id: The ID of the player you want to retrieve the national team statistics for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/player/{is_id}/statistics/national"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playermedia(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the media of a specific Football player."
    id: The ID of the player you want to retrieve the media for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/player/{is_id}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
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
    url = f"https://footapi7.p.rapidapi.com/api/img/flag/{flag}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerlastyearsummary(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the last year summary for a specific Football player."
    id: The ID of the player you want to retrieve the summary for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/player/{is_id}/summary"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def managernextmatches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details of the next matches for a football manager, including the teams playing, match timings, and other relevant information."
    id: The ID of the manager for whom you want to retrieve the next matches.
        page: The zero-based index of the page of matches you want to retrieve.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/manager/{is_id}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def refereestatistics(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the statistics of a specific Football referee."
    id: The ID of the referee  you want to retrieve the statistics for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/referee/{is_id}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerlastmatches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of the last matches played by a football player, paginated by the given page number."
    page: The zero-based page number of the results you want to retrieve.
        id: The ID of the player whose last matches you want to retrieve.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/player/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerstatisticsseasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the statistics seasons for a specific Football player."
    id: The ID of the player you want to retrieve the statistics seasons for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/player/{is_id}/statistics/season"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerattributes(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the attributes of a specific Football player."
    id: The ID of the player you want to retrieve the attributes for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/player/{is_id}/attribute"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playertransferhistory(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the transfer history of a specific Football player."
    id: The ID of the player you want to retrieve the transfer history for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/player/{is_id}/transfer"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def managerimage(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the image of a football manager with the given ID."
    id: The ID of the manager for which you want to retrieve the image.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/manager/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playernearmatches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the near matches for a specific Football player."
    id: The ID of the player you want to retrieve the near matches for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/player/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
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
    url = f"https://footapi7.p.rapidapi.com/api/tv/channel/{channid}/event/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playeroverallheatmaps(seasonid: int, playerid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the overall heatmaps for a specific football player for a given tournament and season."
    seasonid: The ID of the football season for which to retrieve the player's overall heatmaps.
        playerid: The ID of the football player for whom to retrieve the overall heatmaps.
        tournamentid: The unique ID of the football tournament for which to retrieve the player's overall heatmaps.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/player/{playerid}/tournament/{tournamentid}/season/{seasonid}/heatmap"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerlastratings(is_id: int, seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the last ratings for a specific Football player in a specific tournament and season."
    id: The ID of the player you want to retrieve the last ratings for.
        seasonid: The season ID you want to retrieve the player's last ratings for.
        tournamentid: The unique tournament ID you want to retrieve the player's last ratings for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/last-ratings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playeroverallstatistics(is_id: int, tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the overall statistics for a specific Football player in a specific tournament and season."
    id: The ID of the player you want to retrieve the overall statistics for.
        tournamentid: The unique tournament ID you want to retrieve the player's overall statistics for.
        seasonid: The season ID you want to retrieve the player's overall statistics for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
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
    url = f"https://footapi7.p.rapidapi.com/api/placeholder/tournament.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchschedulestop(day: int, month: int, year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top football schedules for a specific day, month, and year."
    day: The day for which you want to retrieve the schedules
        month: The month for which you want to retrieve the schedules
        year: The year for which you want to retrieve the schedules
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/matches/top/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerpenaltyhistory(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the penalty history of a specific Football player."
    id: The ID of the player you want to retrieve the penalty history for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/player/{is_id}/penalty"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
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
    url = f"https://footapi7.p.rapidapi.com/api/tv/country/all/event/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
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
    url = f"https://footapi7.p.rapidapi.com/api/placeholder/player.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
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
    url = f"https://footapi7.p.rapidapi.com/api/placeholder/team.svg"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamplayers(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the players for a specific Football team."
    id: The ID of the team you want to retrieve the players for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/team/{is_id}/players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamlogoimage(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the logo image of a specific football team using its ID."
    teamid: The ID of the football team for which you want to retrieve the logo image.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/team/{teamid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def managerplaceholderimage(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the player placeholder image in PNG format."
    
    """
    url = f"https://footapi7.p.rapidapi.com/api/placeholder/manager.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamtransfers(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the transfers for a specific Football team."
    id: The ID of the team you want to retrieve the transfers for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/team/{is_id}/transfers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def refereeimage(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the image of a football player with the given ID."
    id: The ID of the referee for which you want to retrieve the image.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/referee/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamlastmatches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the last matches for a specific Football team."
    id: The ID of the team you want to retrieve the last matches for.
        page: Zero-based page number.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/team/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchodds(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get odds data for a specific football match using its ID."
    id: The ID of the match for which you want to get the odds.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/match/{is_id}/odds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teammedia(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the media for a specific Football team."
    id: The ID of the team you want to retrieve the media for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/team/{is_id}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamnearmatches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the near matches for a specific Football team."
    id: The ID of the team you want to retrieve the near matches for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/team/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchheadtoheadduel(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation returns a head-to-head summary for a specific football match, including previous matches and other relevant data."
    id: The ID of the football match for which you want to get head-to-head summary.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/match/{is_id}/duel"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def refereenextmatches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of the last matches by a football referee, paginated by the given page number."
    page: The zero-based page number of the results you want to retrieve.
        id: The ID of the referee whose last matches you want to retrieve.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/referee/{is_id}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchvotes(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get votes data for a specific football match using its ID."
    id: The ID of the match for which you want to get the votes.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/match/{is_id}/votes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchhomeplayerjersey(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the home player jersey for a specific Football match."
    id: The ID of the match you want to retrieve the home player jersey for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/match/{is_id}/player/jersey/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchlineups(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the lineups for a specific Football match."
    id: The ID of the match you want to retrieve the lineups for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/match/{is_id}/lineups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchplayerstatistics(playerid: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the statistics of a specific player in a specific Football match."
    playerid: The ID of the player you want to retrieve statistics for.
        id: The ID of the match you want to retrieve the player statistics for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/match/{is_id}/player/{playerid}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamstatisticsseasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the statistics seasons for a specific Football team."
    id: The ID of the team you want to retrieve team statistics seasons for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/team/{is_id}/statistics/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of a specific Football player."
    id: The ID of the player you want to retrieve the details for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/player/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamoveralltopplayers(seasonid: int, tournamentid: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the best players for a specific Football team in a particular tournament and season."
    seasonid: The season ID you want to retrieve the team's best players for.
        tournamentid: The unique tournament ID you want to retrieve the team's best players for.
        id: The ID of the team you want to retrieve the best players for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/team/{is_id}/tournament/{tournamentid}/season/{seasonid}/best-players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamplayersstatisticsseasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the players' statistics seasons for a specific Football team."
    id: The ID of the team you want to retrieve the players' statistics seasons for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/team/{is_id}/players/statistics/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchplayershotmap(is_id: int, playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the shotmap of a specific player in a specific Football match."
    id: The ID of the match you want to retrieve the player shotmap for.
        playerid: The ID of the player you want to retrieve the shotmap for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/match/{is_id}/player/{playerid}/shotmap"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def footballmatchawayplayerjersey(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the away player jersey for a specific Football match."
    id: The ID of the match you want to retrieve the away player jersey for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/match/{is_id}/player/jersey/away"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def refereelastmatches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of the last matches by a football referee, paginated by the given page number."
    page: The zero-based page number of the results you want to retrieve.
        id: The ID of the referee whose last matches you want to retrieve.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/referee/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamnextmatches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the next matches for a specific Football team."
    page: Zero-based page number.
        id: The ID of the team you want to retrieve the next matches for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/team/{is_id}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playercharacteristics(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the characteristics of a specific Football player."
    id: The ID of the player you want to retrieve the characteristics for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/player/{is_id}/characteristics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchplayersaveragepositions(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the average positions of players for a specific Football match."
    id: The ID of the match you want to retrieve the average positions for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/match/{is_id}/average-positions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchstreaks(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data about the streaks in a specific football match using its ID."
    id: The ID of the match for which you want to get the streaks data.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/match/{is_id}/streaks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchawaygoalkeeperjersey(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the away goalkeeper jersey for a specific Football match."
    id: The ID of the match you want to retrieve the away goalkeeper jersey for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/match/{is_id}/goalkeeper/jersey/away"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguehometeamevents(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get last 5 home matches of a specific Football league."
    tournamentid: The unique tournament ID you want to retrieve the league's home team events from.
        seasonid: The season ID you want to retrieve the league's home team events from.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/team-events/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchgraph(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the graph for a specific Football match."
    id: The ID of the match you want to retrieve the graph for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/match/{is_id}/graph"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchschedulesbottom(year: int, month: int, day: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get less relevant Football match schedules for a specific date, including match timings, teams, and other relevant information."
    year: The year for which you want to retrieve the schedules (e.g., 2022).
        month: The month for which you want to retrieve the schedules (1-12).
        day: The day of the month for which you want to retrieve the schedules (1-31).
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/matches/bottom/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguelogoimage(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the logo image of a specific football league using its ID."
    tournamentid: The ID of the football league for which you want to retrieve the logo image.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def uefaclubrankings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the current UEFA club rankings of Football teams."
    
    """
    url = f"https://footapi7.p.rapidapi.com/api/rankings/uefa/clubs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchshotmap(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the shotmap of a specific Football match."
    id: The ID of the match you want to retrieve the shotmap for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/match/{is_id}/shotmap"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fifarankings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the current FIFA rankings of Football teams."
    
    """
    url = f"https://footapi7.p.rapidapi.com/api/rankings/fifa"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchincidents(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the incidents for a specific Football match."
    id: The ID of the match you want to retrieve the incidents for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/match/{is_id}/incidents"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguemeta(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get meta information for a specific football tournament."
    id: The unique tournament id for which you want to get meta information
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{is_id}/meta"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguetotalteamevents(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get last 5 matches of a specific Football league."
    seasonid: The season ID you want to retrieve the league's total team events from.
        tournamentid: The unique tournament ID you want to retrieve the league's total team events from.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/team-events/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueoveralltopteams(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the overall top teams for the given tournament and season ID."
    seasonid: The ID of the season for which you want to retrieve the top teams.
        tournamentid: The unique ID of the tournament for which you want to retrieve the top teams.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/best-teams"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerimage(playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the image of a football player with the given ID."
    playerid: The ID of the player for which you want to retrieve the image.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/player/{playerid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details for a specific Football team."
    id: The ID of the team you want to retrieve the details for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/team/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchbestplayers(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data about the best players in a specific football match using its ID."
    id: The ID of the match for which you want to get the best players data.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/match/{is_id}/best-players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchhighlights(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation returns highlights of a specific football match, including key moments and player actions."
    id: The ID of the football match for which you want to get highlights.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/match/{is_id}/highlights"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchmanagers(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about the managers for a specific football match using its ID."
    id: The ID of the match for which you want to get the manager information.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/match/{is_id}/managers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamtournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the unique tournaments a specific Football team has participated in."
    id: The ID of the team you want to retrieve the tournaments for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/team/{is_id}/tournaments"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguehomestandings(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the home standings of a specific Football league."
    seasonid: The season ID you want to retrieve the league's home standings from.
        tournamentid: The unique tournament ID you want to retrieve the league's home standings from.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/standings/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information about a specific Football match by providing the match ID."
    id: The ID of the match you want to retrieve information for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/match/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueeventsbyround(seasonid: int, tournamentid: int, round: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation allows you to retrieve events for a specified football league, season, and round."
    seasonid: The season ID for which you want to retrieve the league's events.
        tournamentid: The unique tournament ID for which you want to retrieve the league's events.
        round: The round number.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/matches/round/{round}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def refereedetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of a specific Football referee."
    id: The ID of the referee  you want to retrieve the details for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/referee/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguepowerrankingbyroundbyid(tournamentid: int, round: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation returns the power ranking round for the specified tournament, season and round ID for a football league."
    tournamentid: The unique ID of the tournament for which you want to retrieve the power ranking round.
        round: The round id.
        seasonid: The season ID for which you want to retrieve the power ranking round.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/power-rankings/round/{round}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueseasons(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the available seasons for a specific football league."
    tournamentid: The unique tournament ID for which you want to retrieve the league's seasons.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def managercareerhistory(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the career history of a football manager, including information on the teams they have managed, their win-loss record, and other relevant information."
    id: The ID of the manager for whom you want to retrieve the career history.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/manager/{is_id}/history"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def uefacountryrankings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the current UEFA country rankings of Football teams."
    
    """
    url = f"https://footapi7.p.rapidapi.com/api/rankings/uefa/countries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchhomegoalkeeperjersey(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the home goalkeeper jersey for a specific Football match."
    id: The ID of the match you want to retrieve the home goalkeeper jersey for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/match/{is_id}/goalkeeper/jersey/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def livematches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of currently ongoing live Football matches."
    
    """
    url = f"https://footapi7.p.rapidapi.com/api/matches/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueteamoftheweek(roundid: int, seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the team of the week for a specific round of a football league by providing the tournament ID, season ID, and round ID."
    roundid: The round ID for which you want to retrieve the team of the week.
        seasonid: The season ID for which you want to retrieve the team of the week.
        tournamentid: The unique ID of the tournament for which you want to retrieve the team of the week.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/round/{roundid}/team-of-week"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of Football categories available."
    
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguedetails(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information for a specific football league using its ID."
    tournamentid: The ID of the football league for which you want to retrieve the details.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchplayerheatmap(matchid: int, playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the heatmap of a specific player in a given football match."
    matchid: The ID of the football match for which to retrieve the player heatmap.
        playerid: The ID of the football player for whom to retrieve the heatmap.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/match/{matchid}/player/{playerid}/heatmap"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguetotalstandings(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the total standings of a specific Football league."
    seasonid: The season ID you want to retrieve the league's total standings from.
        tournamentid: The unique tournament ID you want to retrieve the league's total standings from.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/standings/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamleagueperformance(teamid: int, tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the league performance graph for a specific football team within a particular tournament and season."
    teamid: The ID of the team for which you want to retrieve the league performance.
        tournamentid: The unique tournament ID for which you want to retrieve the team's league performance.
        seasonid: The season ID for which you want to retrieve the team's league performance.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/team/{teamid}/performance"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueteamoftheweekrounds(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation returns the team of the week rounds for the specified tournament and season ID for a football league."
    seasonid: The season ID for which you want to retrieve the team of the week rounds.
        tournamentid: The unique ID of the tournament for which you want to retrieve the team of the week rounds.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/tow/rounds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueoveralltopplayers(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the overall top players for the given tournament and season ID."
    tournamentid: The unique ID of the tournament for which you want to retrieve the top players.
        seasonid: The ID of the season for which you want to retrieve the top players.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/best-players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguenextmatches(page: int, tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation returns the next matches of a football league for the specified tournament and season ID, including match timings, teams, and other relevant information."
    page: The zero-based page number.
        tournamentid: The unique ID of the tournament for which you want to retrieve the next matches.
        seasonid: The season ID for which you want to retrieve the next matches.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categorytournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all leagues associated with a specific Football category by providing the category ID."
    id: The category ID for which you want to retrieve all leagues.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/all/category/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguelastmatches(tournamentid: int, seasonid: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation returns the last matches of a football league for the specified tournament and season ID, including match timings, teams, and other relevant information."
    tournamentid: The unique ID of the tournament for which you want to retrieve the last matches.
        seasonid: The season ID for which you want to retrieve the last matches.
        page: The zero-based page number.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/matches/last/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categoryschedules(is_id: int, day: int, month: int, year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Football match schedules for a specific date and category, including match timings, teams, and other relevant information."
    id: The category ID for which you want to retrieve the schedules.
        day: The day of the month for which you want to retrieve the schedules (1-31).
        month: The month for which you want to retrieve the schedules (1-12).
        year: The year for which you want to retrieve the schedules (e.g., 2022).
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/category/{is_id}/events/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamoverallstatistics(seasonid: int, tournamentid: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the overall statistics for a specific Football team in a particular tournament and season."
    seasonid: The season ID you want to retrieve the team's overall statistics for.
        tournamentid: The unique tournament ID you want to retrieve the team's overall statistics for.
        id: The ID of the team you want to retrieve the overall statistics for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/team/{is_id}/tournament/{tournamentid}/season/{seasonid}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguecuptrees(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get cup trees for a specific football league in a given season."
    tournamentid: The unique ID of the football league for which you want to retrieve the cup trees.
        seasonid: The ID of the season for which you want to retrieve the cup trees.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/cuptrees"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def headtoheadmatches(customid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the head-to-head matches for a specific Football match using a custom ID."
    customid: The custom ID of the match you want to retrieve the head-to-head matches for. It is obtained from the match.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/match/{customid}/h2h"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueawayteamevents(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get last 5 away matches of a specific Football league."
    tournamentid: The unique tournament ID you want to retrieve the league's away team events from.
        seasonid: The season ID you want to retrieve the league's away team events from.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/team-events/away"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguegroups(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the groups for a specific season of a football league by providing the tournament ID and season ID."
    tournamentid: The unique ID of the tournament for which you want to retrieve the groups.
        seasonid: The season ID for which you want to retrieve the groups.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/groups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguegroupmatches(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the groups matches for a specific season of a football league by providing the group tournament ID and season ID."
    tournamentid: The group tournament ID for which you want to retrieve the matches.
        seasonid: The season ID for which you want to retrieve the matches.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/group/tournament/{tournamentid}/season/{seasonid}/matches"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueeventsbyroundandslug(slug: str, tournamentid: int, round: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation allows you to retrieve events for a specified football league, season, round and slug."
    slug: The round slug.
        tournamentid: The unique tournament ID for which you want to retrieve the league's events.
        round: The round number.
        seasonid: The season ID for which you want to retrieve the league's events.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/matches/round/{round}/slug/{slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchstatistics(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the statistics for a specific Football match."
    id: The ID of the match you want to retrieve the statistics for.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/match/{is_id}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguerounds(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation returns the rounds for the specified tournament and season ID for a football league."
    tournamentid: The unique ID of the tournament for which you want to retrieve the rounds.
        seasonid: The season ID for which you want to retrieve the rounds.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/rounds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def prematchform(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation returns the pre-match form for a specific football match, including team statistics and player data."
    id: The ID of the football match for which you want to get pre-match form.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/match/{is_id}/form"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguepergametopplayers(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the overall top players per game for the given tournament and season ID."
    tournamentid: The unique ID of the tournament for which you want to retrieve the top players.
        seasonid: The ID of the season for which you want to retrieve the top players.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/best-players/per-game"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchschedules(month: int, day: int, year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get football schedules for a specific day, month, and year."
    month: The month for which you want to retrieve the schedules
        day: The day for which you want to retrieve the schedules
        year: The year for which you want to retrieve the schedules
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/matches/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueawaystandings(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the league's away standings for the given tournament and season ID."
    tournamentid: The unique ID of the tournament for which you want to retrieve the away standings.
        seasonid: The ID of the season for which you want to retrieve the away standings.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/standings/away"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchstreakodds(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get odds data about the streaks in a specific football match using its ID."
    id: The ID of the match for which you want to get the streaks odds data.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/match/{is_id}/streaks/odds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueseasoninfo(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the season information of a specific Football league."
    tournamentid: The unique tournament ID you want to retrieve the league's season information from.
        seasonid: The season ID you want to retrieve the league's season information from.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueplayersstatistics(limit: int, page: int, seasonid: int, tournamentid: int, minapps: bool=None, order: str='-rating', filters: str=None, accumulation: str='total', group: str='summary', fields: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the player statistics of a specific Football league."
    limit: How many players to include per page?
        page: One based page.
        seasonid: The season id you want to retrieve the league players statistics.
        tournamentid: The unique tournament id you want to retrieve the league players statistics.
        minapps: The statitiscs only includes player with minimum appearences. If set to true only the players with half of matches of the player with most appearences will be included.
        order: The sorting order. Based on the response properties. Eg.: For the summary group thedefault order is -rating. If you add - before the property it sends the descending order, if you let the property without any sign before it is ascending. Leave it empty to get default order
        filters: The filters to apply. Only use when not specifying the group. Possible filters are: position.in.G~D~M~F, you can omit any position like: position.in.G~M~F or even position.in.F. type.EQ.home and type.EQ.away to specify if the statistics are for matches played home or away. appearances.GT.5 where GT means greater than and can be replaced by EQ meaning equal and LT  meaning less than and the 5 is the number of appearences of the player. age.GT.25 where GT and 25 logic is the same as appearences.preferredFoot.EQ.Both where Both can be replaced by Right and Left.team.in.1660~1644 where 1660 and 1644 are team ids, same logic as position.in filter.nationality.in.AO~BR where AO and BR are country alpha2, same logic as position.in filter.
        accumulation: The accumulation. It can be total, perGame or per90.
        group: The statistic group. It can be summary, attack, defence, passing, goalkeeper or leave it to apply special filters.
        fields: The fields to include, up to 6. Only use if not specifying the groupt. They can be: successfulDribblesPercentage,goals,blockedShots,penaltyWon,goalsFromOutsideTheBox,hitWoodwork,rating,expectedGoals,totalShots,goalConversionPercentage,shotFromSetPiece,headedGoals, offsides,bigChancesMissed,shotsOnTarget,penaltiesTaken,freeKickGoal,leftFootGoals,penaltyConversion,successfulDribbles,shotsOffTarget,penaltyGoals,goalsFromInsideTheBox,rightFootGoals,setPieceConversion,tackles,errorLeadToGoal,cleanSheet,interceptions,errorLeadToShot,penaltyConceded,ownGoals,clearances,dribbledPast,bigChancesCreated,totalPasses,accurateFinalThirdPasses,accurateLongBalls,assists,accuratePassesPercentage,keyPasses,accurateLongBallsPercentage,accuratePasses,accurateOwnHalfPasses,accurateCrosses,passToAssist,inaccuratePasses,accurateOppositionHalfPasses,accurateCrossesPercentage,saves,savedShotsFromInsideTheBox,punches,crossesNotClaimed,cleanSheet,savedShotsFromOutsideTheBox,runsOut,rating,penaltyFaced,goalsConcededInsideTheBox,successfulRunsOut,penaltySave,goalsConcededOutsideTheBox,highClaims,yellowCards,aerialDuelsWon,minutesPlayed,possessionLost,redCards,aerialDuelsWonPercentage,wasFouled,appearances,groundDuelsWon,totalDuelsWon,fouls,matchesStarted,groundDuelsWonPercentage,totalDuelsWonPercentage,dispossessed.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/statistics"
    querystring = {'limit': limit, 'page': page, }
    if minapps:
        querystring['minApps'] = minapps
    if order:
        querystring['order'] = order
    if filters:
        querystring['filters'] = filters
    if accumulation:
        querystring['accumulation'] = accumulation
    if group:
        querystring['group'] = group
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguemedia(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get media for a football league, including images and videos."
    tournamentid: The ID of the tournament for which you want to retrieve the media.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguepowerrankingrounds(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation returns the power ranking rounds for the specified tournament and season ID for a football league."
    tournamentid: The unique ID of the tournament for which you want to retrieve the power ranking rounds.
        seasonid: The season ID for which you want to retrieve the power ranking rounds.
        
    """
    url = f"https://footapi7.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/power-rankings/rounds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "footapi7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

