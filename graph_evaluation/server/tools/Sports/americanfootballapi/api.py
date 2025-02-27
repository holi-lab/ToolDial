import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def managerimage(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the image for a specific American Football manager in PNG format by providing the manager ID."
    id: The ID of the manager whose image you want to retrieve.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/manager/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def managerlastmatches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation retrieves the last matches for a given American Football manager ID. It returns a paginated list of events (matches), with the latest match first."
    id: The ID of the manager whose last matches you want to retrieve.
        page: The zero-based index of the page of results to retrieve.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/manager/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchstreaks(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the win/loss streaks for the two teams involved in a specific American Football match."
    id: The ID of the match for which you want to retrieve the win/loss streaks.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/match/{is_id}/team-streaks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchvotes(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get votes for a specific American Football match by providing its ID."
    id: The ID of the match for which you want to get the votes.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/match/{is_id}/votes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchschedules(year: int, day: int, month: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation returns the match schedules for the given date, including match timings, teams, and other relevant information."
    year: The year for which you want to retrieve the match schedules (e.g., 2022).
        day: The day of the month for which you want to retrieve the match schedules (1-31).
        month: The month for which you want to retrieve the match schedules (1-12).
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/matches/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchstatistics(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match statistics for a specific American Football match."
    id: The ID of the match for which you want to get match statistics.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/match/{is_id}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchlineups(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the match lineups for a specific American Football match by providing the match ID."
    id: The ID of the match for which you want to get the lineups.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/match/{is_id}/lineups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def prematchform(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get pre-match form data for a specific American Football match by providing the match ID."
    id: The ID of the match for which you want to get the pre-match form data.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/match/{is_id}/form"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def managerdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of a specific American Football manager using the manager id."
    id: The ID of the manager whose details you want to retrieve.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/manager/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
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
    url = f"https://americanfootballapi.p.rapidapi.com/api/placeholder/player.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerregularseasonstatistics(is_id: int, tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves regular season statistics for a specific player in a given tournament and season."
    id: The ID of the player whose regular season statistics are to be retrieved.
        tournamentid: The ID of the unique tournament in which the player participated.
        seasonid: The ID of the season in which the player participated.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/statistics/regularSeason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playermedia(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves media for a specific player."
    id: The ID of the player whose media is to be retrieved.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/player/{is_id}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
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
    url = f"https://americanfootballapi.p.rapidapi.com/api/placeholder/manager.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playernearmatches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the upcoming and recent matches for an American Football player using their ID."
    id: The ID of the player for which you want to retrieve the near matches.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/player/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchodds(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get betting odds for a specific American Football match by providing the match ID."
    id: The ID of the match for which you want to get the odds.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/match/{is_id}/odds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
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
    url = f"https://americanfootballapi.p.rapidapi.com/api/tv/country/all/event/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
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
    url = f"https://americanfootballapi.p.rapidapi.com/api/tv/channel/{channid}/event/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamplayers(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team players."
    id: The ID of the team for which you want to retrieve the players.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/team/{is_id}/players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchincidents(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match incidents for a specific American Football match by providing the match ID."
    id: The ID of the match for which you want to get the incidents.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/match/{is_id}/incidents"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teammedia(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get media information for a specific American Football team by providing its ID."
    id: The ID of the team for which you want to get the media information.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/team/{is_id}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchgraph(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match graph data for a specific American Football match by providing the match ID."
    id: The ID of the match for which you want to get the match graph.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/match/{is_id}/graph"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamplayoffsbestplayers(seasonid: int, tournamentid: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the top players of a specific American Football team during the playoffs by providing the team ID, tournament ID, and season ID."
    seasonid: The ID of the season in which the team's best players will be retrieved.
        tournamentid: The ID of the tournament in which the team's best players will be retrieved.
        id: The ID of the team whose best players you want to retrieve.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/team/{is_id}/tournament/{tournamentid}/season/{seasonid}/best-players/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerstatisticsseasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player statistics seasons."
    id: The ID of the player for which you want to retrieve the statistics seasons.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/player/{is_id}/statistics/season"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
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
    url = f"https://americanfootballapi.p.rapidapi.com/api/placeholder/tournament.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueseasons(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the seasons of an American Football league for a specific tournament, including the start and end dates of each season."
    tournamentid: The unique tournament ID for which you want to retrieve the seasons.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/tournament/{tournamentid}/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchbestplayers(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the best players for a specific American Football match by providing its ID."
    id: The ID of the match for which you want to get the best players.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/match/{is_id}/best-players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerlastmatches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the last matches played by a specific American Football player by providing the team ID and page number."
    id: The ID of the player for which you want to retrieve the last matches.
        page: Zero-based page number.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/player/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def livematches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get live American Football matches that are currently taking place."
    
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/matches/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguecuptrees(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves cup trees for a specific league in a given season."
    seasonid: The ID of the season for which the league cup trees are to be retrieved.
        tournamentid: The ID of the unique tournament for which the league cup trees are to be retrieved.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/tournament/{tournamentid}/season/{seasonid}/cuptrees"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamplayersstatisticsseasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the seasons for which player statistics are available for a specific American Football team by providing the team ID."
    id: The ID of the team for which you want to retrieve the player's statistics seasons.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/team/{is_id}/players/statistics/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguetotalteamevents(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the last 5 matches for a specific league in a given season for both home and away events."
    tournamentid: The ID of the unique tournament for which the league's team events are to be retrieved.
        seasonid: The ID of the season for which the league's total team events are to be retrieved.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/tournament/{tournamentid}/season/{seasonid}/team-events/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchhighlights(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get video highlights for a specific American Football match by providing the match ID."
    id: The ID of the match for which you want to get the highlights.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/match/{is_id}/highlights"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all tournament categories. It's the top level of tournaments and represent countries."
    
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/tournament/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamstandingsseasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the seasons for which standings are available for a specific American Football team by providing the team ID."
    id: The ID of the team for which you want to retrieve the standings seasons.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/team/{is_id}/standings/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamregularseasonbestplayers(tournamentid: int, is_id: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the top players of a specific American Football team during the regular season by providing the team ID, tournament ID, and season ID."
    tournamentid: The ID of the tournament in which the team's best players will be retrieved.
        id: The ID of the team whose best players you want to retrieve.
        seasonid: The ID of the season in which the team's best players will be retrieved.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/team/{is_id}/tournament/{tournamentid}/season/{seasonid}/best-players/regularSeason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueawaystandings(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves away standings for a specific league in a given season."
    seasonid: The ID of the season for which the league's away standings are to be retrieved.
        tournamentid: The ID of the unique tournament for which the league's away standings are to be retrieved.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/tournament/{tournamentid}/season/{seasonid}/standings/away"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguenextmatches(seasonid: int, tournamentid: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the next matches of an American Football league for a specific tournament and season, including match timings, teams, and other relevant information."
    seasonid: The season ID for which you want to retrieve the next matches.
        tournamentid: The unique tournament ID for which you want to retrieve the next matches.
        page: The zero-based page number.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/tournament/{tournamentid}/season/{seasonid}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguehomestandings(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves home standings for a specific league in a given season."
    tournamentid: The ID of the unique tournament for which the league's home standings are to be retrieved.
        seasonid: The ID of the season for which the league's home standings are to be retrieved.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/tournament/{tournamentid}/season/{seasonid}/standings/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamlastmatches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the last matches played by a specific American Football team by providing the team ID and page number."
    id: The ID of the team for which you want to retrieve the last matches.
        page: Zero-based page number.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/team/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information for a specific American Football team by providing its ID."
    id: The ID of the team for which you want to get the details.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/team/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamnearmatches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of near matches for a specific American Football team by providing its ID."
    id: The ID of the team for which you want to get the list of near matches.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/team/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguetopplayersplayoffs(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the top players for a specific league in playoffs during a given season."
    seasonid: The ID of the season for which the league's top players in playoffs are to be retrieved.
        tournamentid: The ID of the unique tournament for which the league's top players in playoffs are to be retrieved.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/tournament/{tournamentid}/season/{seasonid}/best-players/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def headtoheadmatches(customid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get head-to-head match data for a specific American Football match using its custom ID."
    customid: The custom ID of the match for which you want to get the head-to-head matches.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/match/{customid}/h2h"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categorymatchschedules(year: int, is_id: int, month: int, day: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match schedules of the day from the specified American Football category."
    year: The year for which you want to retrieve the match schedules.
        id: The ID of the category for which you want to retrieve the match schedules.
        month: The month of the year for which you want to retrieve the match schedules.
        day: The day of the month for which you want to retrieve the match schedules.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/category/{is_id}/events/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueawayteamevents(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the last 5 matches for a specific league in a given season for away events."
    seasonid: The ID of the season for which the league's away team events are to be retrieved.
        tournamentid: The ID of the unique tournament for which the league's team events are to be retrieved.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/tournament/{tournamentid}/season/{seasonid}/team-events/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categorytournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all leagues from the specified American Football category."
    id: The ID of the category for which you want to retrieve all leagues.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/tournament/all/category/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguetotalstandings(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves total standings for a specific league in a given season."
    seasonid: The ID of the season for which the league's total standings are to be retrieved.
        tournamentid: The ID of the unique tournament for which the league's total standings are to be retrieved.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/tournament/{tournamentid}/season/{seasonid}/standings/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchh2hduel(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get head-to-head match data for a specific American Football match by providing the match ID."
    id: The ID of the match for which you want to get the head-to-head match data.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/match/{is_id}/duel"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamlogo(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation retrieves the logo image for an American Football team, given its team ID. The image is returned in PNG format."
    id: The ID of the team whose logo you want to retrieve.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/team/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguedetails(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of an American Football league for a specific tournament, including the league's name, logo, country, and other relevant information."
    tournamentid: The unique tournament ID for which you want to retrieve the details.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/tournament/{tournamentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerplayoffsstatistics(seasonid: int, is_id: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves playoffs statistics for a specific player in a given tournament and season."
    seasonid: The ID of the season in which the player participated.
        id: The ID of the player whose playoffs statistics are to be retrieved.
        tournamentid: The ID of the unique tournament in which the player participated.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/statistics/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamtournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team tournaments they are participating."
    id: The ID of the team for which you want to retrieve the tournaments.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/team/{is_id}/tournaments"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguemedia(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the media of an American Football league for a specific tournament, including images and videos related to the league."
    tournamentid: The unique tournament ID for which you want to retrieve the media.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/tournament/{tournamentid}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchplayerstatistics(is_id: int, playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get individual player statistics for a specific American Football match by providing the match ID and player ID."
    id: The ID of the match for which you want to get player statistics.
        playerid: The ID of the player for whom you want to get statistics.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/match/{is_id}/player/{playerid}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchmanagers(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get managers for a specific American Football match by providing its ID."
    id: The ID of the match for which you want to get the managers.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/match/{is_id}/managers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerimage(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the PNG image of an American Football player using their ID."
    id: The ID of the player for which you want to retrieve the image.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/player/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguetopplayersregularseason(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the top players for a specific league in regular season during a given season."
    tournamentid: The ID of the unique tournament for which the league's top players in regular season are to be retrieved.
        seasonid: The ID of the season for which the league's top players in regular season are to be retrieved.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/tournament/{tournamentid}/season/{seasonid}/best-players/regularSeason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information for a specific American Football match by providing the match ID."
    id: The ID of the match for which you want to get the details.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/match/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of an American Football player using their ID."
    id: The ID of the player for which you want to retrieve the details.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/player/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
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
    url = f"https://americanfootballapi.p.rapidapi.com/api/placeholder/team.svg"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamnextmatches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the upcoming matches for a specific American Football team by providing the team ID and page number. You can expect empty responses if no upcoming matches are scheduled."
    id: The ID of the team for which you want to retrieve the next matches.
        page: Zero-based page number.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/team/{is_id}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation searches for players, teams, and tournaments using the provided search term."
    term: The search term to use when searching for American Football players, teams, and tournaments.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/search/{term}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguelogo(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation retrieves the logo image for an American Football league, given its unique tournament ID. The image is returned in PNG format."
    tournamentid: The unique tournament ID of the league whose logo you want to retrieve.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/tournament/{tournamentid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguelastmatches(page: int, tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the last matches of an American Football league for a specific tournament and season, including match timings, teams, and other relevant information."
    page: The zero-based page number.
        tournamentid: The unique tournament ID for which you want to retrieve the last matches.
        seasonid: The season ID for which you want to retrieve the last matches.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/tournament/{tournamentid}/season/{seasonid}/matches/last/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguetopplayerstypes(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the types of best players in the american football league."
    tournamentid: The unique tournament id you want to retrieve the league's best players types.
        seasonid: The season id you want to retrieve the league's best players types.
        
    """
    url = f"https://americanfootballapi.p.rapidapi.com/api/american-football/tournament/{tournamentid}/season/{seasonid}/best-players/types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
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
    url = f"https://americanfootballapi.p.rapidapi.com/api/img/flag/{flag}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "americanfootballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

