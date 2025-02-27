import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tennisteamprevioustournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team or players previous tournaments."
    id: The player or team id you want to retrieve the players seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/team/{is_id}/tournaments/last"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennisteamstandingsseasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team or players standings seasons."
    id: The player or team id you want to retrieve the players seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/team/{is_id}/standings/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennisteamrankings(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player or team rankings."
    id: The player or team id you want to retrieve the rankings.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/team/{is_id}/rankings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennismultipledayscategories(month: int, days: str, year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the categories of tennis matches along with their tournaments for up to 5 specified days in a month."
    month: The month you want to retrieve the categories
        days: The days for which you want to retrieve the categories. Up to 5, separate with comma.
        year: The year you want to retrieve the categories
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/calendar/{month}/{year}/categories"
    querystring = {'days': days, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketballleaguehomestandings(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league home standings."
    seasonid: The season id you want to retrieve the league's home standings.
        tournamentid: The unique tournament id you want to retrieve the league's home standings.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/standings/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketballleagueawaystandings(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league away standings."
    seasonid: The season id you want to retrieve the league's away standings.
        tournamentid: The unique tournament id you want to retrieve the league's away standings.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/standings/away"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketballleagueshotactionsareasregularseason(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the shot actions areas for the basketball league during the regular season."
    seasonid: The season id you want to retrieve the league's shot actions.
        tournamentid: The unique tournament id you want to retrieve the league's shot actions.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/shot-actions/areas/regularSeason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketballleagueshotactionsareasplayoffs(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the shot actions areas for the basketball league during the playoffs."
    tournamentid: The unique tournament id you want to retrieve the league's shot actions.
        seasonid: The season id you want to retrieve the league's shot actions.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/shot-actions/areas/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketballplayerlastmatches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player last matches."
    page: Zero based page.
        id: The player id you want to retrieve the last matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/player/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketballplayershotactionsplayoffs(seasonid: int, tournamentid: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the shot actions of a player in the basketball league during the playoffs."
    seasonid: The season id you want to retrieve the player's shot actions.
        tournamentid: The unique tournament id you want to retrieve the player's shot actions.
        id: The player id you want to retrieve the shot actions.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/shot-actions/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketballplayershotactionsregularseason(is_id: int, seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the shot actions of a player in the basketball league during the regular season."
    id: The player id you want to retrieve the shot actions.
        seasonid: The season id you want to retrieve the player's shot actions.
        tournamentid: The unique tournament id you want to retrieve the player's shot actions.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/shot-actions/regularSeason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketballleagueoverallpergametopplayers(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league per game top players."
    seasonid: The season id you want to retrieve the league's best players.
        tournamentid: The unique tournament id you want to retrieve the league's best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/best-players/per-game"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketballleagueplayoffspergametopplayers(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league per game top players."
    seasonid: The season id you want to retrieve the league's best players.
        tournamentid: The unique tournament id you want to retrieve the league's best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/best-players/per-game/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketballleagueregularseasonpergametopplayers(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league per game top players."
    tournamentid: The unique tournament id you want to retrieve the league's best players.
        seasonid: The season id you want to retrieve the league's best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/best-players/per-game/regularSeason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketballleaguetopplayerstypes(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the types of best players in the basketball league."
    seasonid: The season ID for which you want to get the league's best players types.
        tournamentid: The unique tournament ID for which you want to get the league's best players types.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/best-players/types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketballleagueplayoffstopteams(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the best teams in the basketball league during the playoffs."
    tournamentid: The unique tournament id you want to retrieve the league's best players.
        seasonid: The season id you want to retrieve the league's best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/best-teams/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketballleagueregularseasontopteams(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the best teams in the basketball league during the regular season."
    seasonid: The season id you want to retrieve the league's best players.
        tournamentid: The unique tournament id you want to retrieve the league's best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/best-teams/regularSeason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennisleagueseasoninfo(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about a tennis league's season."
    tournamentid: The unique tournament ID you want to retrieve the tennis league's season information.
        seasonid: The season ID you want to retrieve the tennis league's season information.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/tournament/{tournamentid}/season/{seasonid}/info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennisteamimage(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player image. It generates png image."
    id: The team ID you want to retrieve the image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/team/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennisteamlastevents(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player or team last events."
    id: The player or team id you want to retrieve the last events.
        page: Zero based page.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/team/{is_id}/events/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennisteamnextevents(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player or team next events."
    page: Zero based page.
        id: The team ID you want to retrieve the next events.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/team/{is_id}/events/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennisteamnearevents(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get near events for a specific team."
    id: The team ID you want to retrieve the near events.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/team/{is_id}/events/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennisteamdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The team ID you want to retrieve the details."
    id: The team ID you want to retrieve the details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/team/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugbyleaguenextmatches(page: int, seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league next matches."
    page: Zero based page.
        seasonid: The season id you want to retrieve the league's next matches.
        tournamentid: The unique tournament id you want to retrieve the league's next matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/tournament/{tournamentid}/season/{seasonid}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketballmatchteamshotmap(teamid: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the shotmap of a team in a basketball match."
    teamid: The team id of the match you want to get shotmap.
        id: The id of the match you want to get team shotmap.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/match/{is_id}/team/{teamid}/shotmap"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketballmatchplayershotmap(is_id: int, playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match player shotmap."
    id: The id of the match you want to get player shotmap.
        playerid: The id of the player you want to get shotmap.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/match/{is_id}/player/{playerid}/shotmap"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_prematch_form(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get pre match form."
    id: The id of the match you want to get pre-match form.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/match/{is_id}/form"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugbyleagueawayteamevents(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the last 5 matches for a specific league in a given season for away events."
    seasonid: The ID of the season for which the league's away team events are to be retrieved.
        tournamentid: The unique tournament ID for which you want to retrieve the league's away team events
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/tournament/{tournamentid}/season/{seasonid}/team-events/away"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugbyplayerlastmatches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player last matches."
    page: Zero based page.
        id: The player id you want to retrieve the last matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/player/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugbyleaguetotalteamevents(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the last 5 matches for a specific league in a given season for both home and away events."
    seasonid: The ID of the season for which the league's total team events are to be retrieved.
        tournamentid: The unique tournament ID for which you want to retrieve the league's total team events
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/tournament/{tournamentid}/season/{seasonid}/team-events/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def icehockeyleagueregularseasontopplayerspergame(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league per game top players."
    tournamentid: The unique tournament id you want to retrieve the league's best players.
        seasonid: The season id you want to retrieve the league's best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/best-players/per-game/regularSeason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def icehockeyleagueawayteamevents(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the last 5 matches for a specific league in a given season for away events."
    tournamentid: The unique tournament ID for which you want to retrieve the league's away team events
        seasonid: The ID of the season for which the league's away team events are to be retrieved.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/team-events/away"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def footballuefaclubrankings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the current UEFA club rankings of Football teams."
    
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rankings/uefa/clubs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketballleaguetotalteamevents(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the last 5 matches for a specific league in a given season for both home and away events."
    tournamentid: The unique tournament ID for which you want to retrieve the league's total team events
        seasonid: The ID of the season for which the league's total team events are to be retrieved.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/team-events/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketballleaguerounds(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league team rounds."
    tournamentid: The unique tournament id you want to retrieve the league's rounds.
        seasonid: The season id you want to retrieve the league's rounds.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/rounds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def footballleagueeventsbyround(tournamentid: int, round: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation allows you to retrieve events for a specified football league, season, and round."
    tournamentid: The unique tournament id you want to retrieve the league's round events.
        round: The round
        seasonid: The season id you want to retrieve the league's round events.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/matches/round/{round}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugbymatchodds(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match odds."
    id: The id of the match you want to get odds.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/match/{is_id}/odds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugbymanagerimage(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get manager image. It generates png image."
    id: The manager id you want to retrieve the image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/manager/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugbymanagerdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get manager details."
    id: The manager id you want to retrieve the details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/manager/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugbyunionrankings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the current Rugby Union rankings."
    
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/rankings/rugby-union"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugbyleaguehometeamevents(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the last 5 matches for a specific league in a given season for home events."
    tournamentid: The unique tournament ID for which you want to retrieve the league's home team events
        seasonid: The ID of the season for which you want to retrieve the home events.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/tournament/{tournamentid}/season/{seasonid}/team-events/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
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
    url = f"https://allsportsapi2.p.rapidapi.com/api/match/{is_id}/player/jersey/away"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def footballleagueeventsbyroundandslug(seasonid: int, slug: str, tournamentid: int, round: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation allows you to retrieve events for a specified football league, season, round and slug."
    seasonid: The season id you want to retrieve the league's round events.
        slug: The round slug
        tournamentid: The unique tournament id you want to retrieve the league's round events.
        round: The round
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/matches/round/{round}/slug/{slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def footballrefereenextmatches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of the next matches by a football referee, paginated by the given page number."
    id: The ID of the referee whose next matches you want to retrieve.
        page: Zero based page.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/referee/{is_id}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def footballrefereeimage(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the image of a football referee with the given ID."
    id: The ID of the referee for which you want to retrieve the image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/referee/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def footballrefereestatistics(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the statistics of a specific Football referee."
    id: The ID of the referee  you want to retrieve the statistics for.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/referee/{is_id}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def footballrefereelastmatches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of the last matches by a football referee, paginated by the given page number."
    page: Zero based page.
        id: The ID of the referee whose last matches you want to retrieve.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/referee/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def footballrefereedetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of a specific Football referee."
    id: The ID of the referee  you want to retrieve the details for.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/referee/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def footballleagueawayteamevents(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the last 5 matches for a specific league in a given season for away events."
    tournamentid: The unique tournament ID for which you want to retrieve the league's away team events
        seasonid: The ID of the season for which the league's away team events are to be retrieved.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/team-events/away"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def footballleaguehometeamevents(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the last 5 matches for a specific league in a given season for home events."
    tournamentid: The unique tournament ID for which you want to retrieve the league's home team events
        seasonid: The ID of the season for which you want to retrieve the home events.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/team-events/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def footballleaguetotalteamevents(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the last 5 matches for a specific league in a given season for both home and away events."
    seasonid: The ID of the season for which the league's total team events are to be retrieved.
        tournamentid: The unique tournament ID for which you want to retrieve the league's total team events
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/team-events/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def esportsleaguetotalteamevents(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the last 5 matches for a specific league in a given season for both home and away events."
    tournamentid: The unique tournament ID for which you want to retrieve the league's total team events
        seasonid: The ID of the season for which the league's total team events are to be retrieved.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/tournament/{tournamentid}/season/{seasonid}/team-events/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricketleaguetotalteamevents(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the last 5 matches for a specific league in a given season for both home and away events."
    tournamentid: The unique tournament ID for which you want to retrieve the league's total team events
        seasonid: The ID of the season for which the league's total team events are to be retrieved.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/tournament/{tournamentid}/season/{seasonid}/team-events/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricketplayerlastmatches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player last matches."
    page: Zero based page.
        id: The player id you want to retrieve the last matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/player/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricketteammedia(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team media."
    id: The team id you want to retrieve the media.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/team/{is_id}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricketleaguemedia(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league media."
    tournamentid: The unique tournament id you want to retrieve the league meia.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/tournament/{tournamentid}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketballmanagerlastmatches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get manager last matches."
    id: The manager id you want to retrieve the last matches.
        page: Zero based page.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/manager/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketballleagueoveralltopplayers(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league overall top players."
    seasonid: The season id you want to retrieve the league's best players.
        tournamentid: The unique tournament id you want to retrieve the league's best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/best-players/overall"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketballleaguegroupmatches(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league group matches."
    seasonid: The season id you want to retrieve the league's next matches.
        tournamentid: The tournament id you want to retrieve the league's group matches. This tournament id is returned from the football league groups endpoint.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/group/tournament/{tournamentid}/season/{seasonid}/matches"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketballleaguegroups(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league groups."
    seasonid: The season id you want to retrieve the league's groups.
        tournamentid: The unique tournament id you want to retrieve the league's groups.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/groups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketballleagueoveralltopteams(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league overall top players."
    tournamentid: The unique tournament id you want to retrieve the league's best players.
        seasonid: The season id you want to retrieve the league's best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/best-teams/overall"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketballleagueawayteamevents(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the last 5 matches for a specific league in a given season for away events."
    seasonid: The ID of the season for which the league's away team events are to be retrieved.
        tournamentid: The unique tournament ID for which you want to retrieve the league's away team events
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/team-events/away"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketballleaguehometeamevents(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the last 5 matches for a specific league in a given season for home events."
    seasonid: The ID of the season for which you want to retrieve the home events.
        tournamentid: The unique tournament ID for which you want to retrieve the league's home team events
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/team-events/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketballmatchseries(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get matches in the series for a specific basketball match."
    id: The ID of the match for which you want to get the series
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/match/{is_id}/series"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseballleagueawayteamevents(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the last 5 matches for a specific league in a given season for away events."
    seasonid: The ID of the season for which the league's away team events are to be retrieved.
        tournamentid: The unique tournament ID for which you want to retrieve the league's away team events
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/tournament/{tournamentid}/season/{seasonid}/team-events/away"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseballplayerlastmatches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player last matches."
    id: The player id you want to retrieve the last matches.
        page: Zero based page.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/player/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseballleaguehometeamevents(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the last 5 matches for a specific league in a given season for home events."
    seasonid: The ID of the season for which you want to retrieve the home events.
        tournamentid: The unique tournament ID for which you want to retrieve the league's home team events
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/tournament/{tournamentid}/season/{seasonid}/team-events/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseballleaguetotalteamevents(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the last 5 matches for a specific league in a given season for both home and away events."
    seasonid: The ID of the season for which the league's total team events are to be retrieved.
        tournamentid: The unique tournament ID for which you want to retrieve the league's total team events
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/tournament/{tournamentid}/season/{seasonid}/team-events/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseballleagueawaystandings(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league away standings."
    seasonid: The season id you want to retrieve the league's away standings.
        tournamentid: The unique tournament id you want to retrieve the league's away standings.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/tournament/{tournamentid}/season/{seasonid}/standings/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseballleaguehomestandings(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league home standings."
    tournamentid: The unique tournament id you want to retrieve the league's home standings.
        seasonid: The season id you want to retrieve the league's home standings.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/tournament/{tournamentid}/season/{seasonid}/standings/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def americanfootballplayerlastmatches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the last matches played by a specific American Football player by providing the player ID and page number."
    id: The player id you want to retrieve the last matches.
        page: Zero based page.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/player/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def americanfootballmanagerimage(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get manager image. It generates png image."
    id: The manager id you want to retrieve the image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/manager/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def americanfootballmanagerdetals(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get manager details."
    id: The ID of the manager whose details you want to retrieve.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/manager/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def americanfootballleagueawayteamevents(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the last 5 matches for a specific league in a given season for away events."
    seasonid: The ID of the season for which the league's away team events are to be retrieved.
        tournamentid: The unique tournament ID for which you want to retrieve the league's away team events
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/tournament/{tournamentid}/season/{seasonid}/team-events/away"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def americanfootballleaguehometeamevents(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the last 5 matches for a specific league in a given season for home events."
    tournamentid: The unique tournament ID for which you want to retrieve the league's home team events
        seasonid: The ID of the season for which the league's home team events are to be retrieved.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/tournament/{tournamentid}/season/{seasonid}/team-events/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def americanfootballleaguetotalteamevents(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the last 5 matches for a specific league in a given season for both home and away events."
    seasonid: The ID of the season for which the league's total team events are to be retrieved.
        tournamentid: The unique tournament ID for which you want to retrieve the league's total team events
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/tournament/{tournamentid}/season/{seasonid}/team-events/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def americanfootballleagueawaystandings(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves away standings for a specific league in a given season."
    seasonid: The ID of the season for which the league's away standings are to be retrieved.
        tournamentid: The ID of the unique tournament for which the league's away standings are to be retrieved.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/tournament/{tournamentid}/season/{seasonid}/standings/away"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def americanfootballleaguehomestandings(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves home standings for a specific league in a given season."
    tournamentid: The ID of the unique tournament for which the league's home standings are to be retrieved.
        seasonid: The ID of the season for which the league's home standings are to be retrieved.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/tournament/{tournamentid}/season/{seasonid}/standings/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def americanfootballleaguetopplayerstypes(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the types of best players in the american football league."
    seasonid: The season id you want to retrieve the league's best players types.
        tournamentid: The unique tournament id you want to retrieve the league's best players types.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/tournament/{tournamentid}/season/{seasonid}/best-players/types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
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
    url = f"https://allsportsapi2.p.rapidapi.com/api/placeholder/manager.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
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
    url = f"https://allsportsapi2.p.rapidapi.com/api/placeholder/player.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
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
    url = f"https://allsportsapi2.p.rapidapi.com/api/placeholder/team.svg"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
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
    url = f"https://allsportsapi2.p.rapidapi.com/api/placeholder/tournament.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_league_total_standings(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league total standings."
    seasonid: The season id you want to retrieve the league's total standings.
        tournamentid: The unique tournament id you want to retrieve the league's total standings.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/tournament/{tournamentid}/season/{seasonid}/standings/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_league_group_matches(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league group matches."
    seasonid: The season id you want to retrieve the league's next matches.
        tournamentid: The tournament id you want to retrieve the league's group matches. This tournament id is returned from the football league groups endpoint.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/group/tournament/{tournamentid}/season/{seasonid}/matches"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_player_statistics_overall(is_id: int, tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player regular season statistics overall."
    id: The player id you want to retrieve the overall statistics.
        tournamentid: The unique tournament id you want to retrieve the player's overall statistics.
        seasonid: The season id you want to retrieve the player's overall statistics.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/statistics/overall"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_league_team_performance(tournamentid: int, teamid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league team performance."
    tournamentid: The unique tournament id you want to retrieve the team league's performance.
        teamid: The team id you want to retrieve the team league's performance.
        seasonid: The season id you want to retrieve the team league's performance.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/team/{teamid}/performance"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_league_per_game_top_players(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league per game top players."
    seasonid: The season id you want to retrieve the league's best players.
        tournamentid: The unique tournament id you want to retrieve the league's best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/best-players/per-game"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_league_overall_top_team(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league overall top team."
    tournamentid: The unique tournament id you want to retrieve the league's best teams.
        seasonid: The season id you want to retrieve the league's best teams.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/best-teams"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_player_overall_heatmaps(seasonid: int, is_id: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player overall heatmaps. Check the tutorials to see html example."
    seasonid: The season id you want to retrieve the player's overall heatmaps.
        id: The player id you want to retrieve the overall heatmap.
        tournamentid: The unique tournament id you want to retrieve the player's overall heatmaps.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/heatmap"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_match_players_average_positions(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match players average positions."
    id: The id of the match you want to get average positions.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/match/{is_id}/average-positions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_league_rounds_events(round: int, slug: str, tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league round events."
    round: The round
        slug: The round slug
        tournamentid: The unique tournament id you want to retrieve the league's round events.
        seasonid: The season id you want to retrieve the league's round events.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/tournament/{tournamentid}/season/{seasonid}/events/round/{round}/slug/{slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_league_rounds(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league rounds."
    seasonid: The season id you want to retrieve the league's rounds.
        tournamentid: The unique tournament id you want to retrieve the league's rounds.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/tournament/{tournamentid}/season/{seasonid}/rounds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_player_summary(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player last 12 months summary."
    id: The player id you want to retrieve the summary.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/player/{is_id}/summary"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_league_players_statistics(page: int, limit: int, tournamentid: int, seasonid: int, minapps: str='false', accumulation: str='total', fields: str=None, group: str='summary', filters: str=None, order: str='-rating', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league players statistics."
    page: One based page.
        limit: How many players to include per page?

        tournamentid: The unique tournament id you want to retrieve the league's players statistics.
        seasonid: The season id you want to retrieve the league's players statistics.
        minapps: The statitiscs only includes player with minimum appearences. If set to true nly the players with half of matches of the player with most appearences will be included.
        accumulation: The accumulation. It can be total, perGame or per90.
        fields: The fields to include, up to 6,comma separated. Only use if not specifying the groupt. They can be: successfulDribblesPercentage,goals,blockedShots,penaltyWon,goalsFromOutsideTheBox,hitWoodwork,rating,expectedGoals,totalShots,goalConversionPercentage,shotFromSetPiece,headedGoals, offsides,bigChancesMissed,shotsOnTarget,penaltiesTaken,freeKickGoal,leftFootGoals,penaltyConversion,successfulDribbles,shotsOffTarget,penaltyGoals,goalsFromInsideTheBox,rightFootGoals,setPieceConversion,tackles,errorLeadToGoal,cleanSheet,interceptions,errorLeadToShot,penaltyConceded,ownGoals,clearances,dribbledPast,bigChancesCreated,totalPasses,accurateFinalThirdPasses,accurateLongBalls,assists,accuratePassesPercentage,keyPasses,accurateLongBallsPercentage,accuratePasses,accurateOwnHalfPasses,accurateCrosses,passToAssist,inaccuratePasses,accurateOppositionHalfPasses,accurateCrossesPercentage,saves,savedShotsFromInsideTheBox,punches,crossesNotClaimed,cleanSheet,savedShotsFromOutsideTheBox,runsOut,rating,penaltyFaced,goalsConcededInsideTheBox,successfulRunsOut,penaltySave,goalsConcededOutsideTheBox,highClaims,yellowCards,aerialDuelsWon,minutesPlayed,possessionLost,redCards,aerialDuelsWonPercentage,wasFouled,appearances,groundDuelsWon,totalDuelsWon,fouls,matchesStarted,groundDuelsWonPercentage,totalDuelsWonPercentage,dispossessed.

        group: The statistic group. It can be summary, attack, defence, passing, goalkeeper or leave it to apply special filters.
        filters: The filters to apply, comma separate if more than one. Only use when not specifying the group. Possible filters are:

- **position.in.G~D~M~F** you can omit any position like: **position.in.G~M~F **or even **position.in.F**.

- **type.EQ.home** and **type.EQ.away** to specify if the statistics are for matches played home or away.

- **appearances.GT.5** where GT means greater than and can be replaced by EQ meaning equal and LT  meaning  less than and the 5 is  the number of appearences of the player. 

- **age.GT.25** where GT and 25 is the player's age, logic is the same as appearences.
  
- **preferredFoot.EQ.Both** where Both can be replaced by Right and Left.

  - **team.in.1660~1644** where 1660 and 1644 are team ids, same logic as position.in filter.

- **nationality.in.AO~BR** where AO and BR are country alpha2, same logic as position.in filter.
        order: The sorting order. Based on the response properties. Eg.: For the summary group the default order is -rating. If you add - before the property it sends the descending order, if you let  the property without any sign before it is ascending. Leave it empty to get default order
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/statistics"
    querystring = {'page': page, 'limit': limit, }
    if minapps:
        querystring['minApps'] = minapps
    if accumulation:
        querystring['accumulation'] = accumulation
    if fields:
        querystring['fields'] = fields
    if group:
        querystring['group'] = group
    if filters:
        querystring['filters'] = filters
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_player_last_matches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player last matches."
    page: Zero based page.
        id: The player id you want to retrieve the last matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/player/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_sports_match_tv_channel_details(channid: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match country tv channel details."
    channid: \\\\\\\"The id of the channel you want to get details\\\\\\\"
        id: The id of the match you want to get tv channel details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tv/channel/{channid}/event/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_sports_match_tv_countries(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match tv channel countries."
    id: The id of the match you want to get tv countries.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tv/country/all/event/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_match_player_shotmap(is_id: int, playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match player shotmap."
    id: The id of the match you want to get player shotmap.
        playerid: The id of the player you want to get shotmap.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/match/{is_id}/player/{playerid}/shotmap"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_match_shotmap(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match shotmap."
    id: The id of the match you want to get shotmap.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/match/{is_id}/shotmap"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_sports_category_flag(flag: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get category flag. It can be used for all sports. It generates image/png"
    flag: The flag name.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/img/flag/{flag}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_calendar_daily_categories(day: int, year: int, month: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get daily categories."
    day: The day you want to retrieve the categories
        year: The year you want to retrieve the categories
        month: The month you want to retrieve the categories
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/calendar/{day}/{month}/{year}/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_event_votes(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event votes."
    id: The id of the event you want to get votes.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/event/{is_id}/votes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_player_near_matches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player near matches."
    id: The player id you want to retrieve the near matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/player/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_team_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team details."
    id: The team id you want to retrieve the details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/team/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_team_players(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team players."
    id: The team id you want to retrieve the players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/team/{is_id}/players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_player_image(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player image. It generates png image."
    id: The player id you want to retrieve the image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/player/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_player_or_team_next_events(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player or team next events."
    page: Zero based page.
        id: The player or team id you want to retrieve the next events.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/player/{is_id}/events/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_match_best_players(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match best players."
    id: The id of the match you want to get the best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/match/{is_id}/best-players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_wta_rankings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get WTA rankings."
    
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/rankings/wta"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_league_last_events(page: int, seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league last events."
    page: Zero based page.
        seasonid: The season id you want to retrieve the league's last events.
        tournamentid: The unique tournament id you want to retrieve the league's last events.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/tournament/{tournamentid}/season/{seasonid}/events/last/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_schedules(month: int, day: int, year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get schedules of the day."
    month: The month you want to retrieve the schedules
        day: The day you want to retrieve the schedules
        year: The year you want to retrieve the schedules
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/matches/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_team_next_matches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team next matches."
    id: The team id you want to retrieve the next matches.
        page: Zero based page.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/team/{is_id}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_match_statistics(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match statistics."
    id: The id of the match you want to get statistics.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/match/{is_id}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_team_or_players_standings_seasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team or players standings seasons."
    id: The player or team id you want to retrieve the players seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/player/{is_id}/standings/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_atp_rankings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get ATP rankings."
    
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/rankings/atp"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_team_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team details."
    id: The team id you want to retrieve the details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/team/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_league_groups(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league groups."
    seasonid: The season id you want to retrieve the league's groups.
        tournamentid: The unique tournament id you want to retrieve the league's groups.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/groups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_event(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event by id."
    id: The id of the event you want to get info.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/event/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_live_events(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get live events."
    
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/events/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get categories."
    
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/tournament/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_event_highlights(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event highlights."
    id: The id of the event you want to get highlights.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/event/{is_id}/highlights"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_live_matches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get live matches."
    
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/matches/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_event_point_by_point(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event point by point."
    id: The id of the event you want to get point by point.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/event/{is_id}/point-by-point"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_league_media(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league media."
    tournamentid: The unique tournament id you want to retrieve the league meia.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/tournament/{tournamentid}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_league_overall_top_players(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league overall top players."
    seasonid: The season id you want to retrieve the league's best players.
        tournamentid: The unique tournament id you want to retrieve the league's best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/best-players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_league_home_standings(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league home standings."
    tournamentid: The unique tournament id you want to retrieve the league's home standings.
        seasonid: The season id you want to retrieve the league's home standings.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/standings/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_team_tournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team tournaments."
    id: The team id you want to retrieve the tournaments.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/team/{is_id}/tournaments"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_player_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player details."
    id: The player id you want to retrieve the details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/player/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_league_total_standings(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league total standings."
    seasonid: The season id you want to retrieve the league's total standings.
        tournamentid: The unique tournament id you want to retrieve the league's total standings.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/standings/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_category_schedules(year: int, day: int, month: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get schedules of the day from the category."
    year: The year you want to retrieve the schedules
        day: The day you want to retrieve the schedules
        month: The month you want to retrieve the schedules
        id: The category id you want to retrieve the schedules
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/category/{is_id}/events/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_team_overall_top_players(seasonid: int, is_id: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team overall top players."
    seasonid: The season id you want to retrieve the team's best players.
        id: The team id you want to retrieve the best players.
        tournamentid: The unique tournament id you want to retrieve the team's best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/team/{is_id}/tournament/{tournamentid}/season/{seasonid}/best-players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_schedules_bottom(day: int, month: int, year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get less relevant schedules of the day."
    day: The day you want to retrieve the schedules
        month: The month you want to retrieve the schedules
        year: The year you want to retrieve the schedules
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/matches/bottom/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_tournament_info(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tournament meta."
    id: The unique tournament id you want to get info.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/tournament/{is_id}/info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_team_standings_seasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team standings seasons."
    id: The team id you want to retrieve the standings seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/team/{is_id}/standings/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_manager_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get manager details."
    id: The manager id you want to retrieve the details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/manager/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_team_last_matches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team last matches."
    id: The team id you want to retrieve the last matches.
        page: Zero based page.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/team/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_player_statistics_seasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player statistics seasons."
    id: The player id you want to retrieve the statistics seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/player/{is_id}/statistics/season"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_player_regular_season_statistics(is_id: int, tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player regular season statistics."
    id: The player id you want to retrieve the statistics.
        tournamentid: The unique tournament id you want to retrieve the player's statistics.
        seasonid: The season id you want to retrieve the player's statistics.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/statistics/regularSeason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def icehockeyplayeroverallstatistics(is_id: int, seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player overall statistics."
    id: The player id you want to retrieve the overall statistics.
        seasonid: The season id you want to retrieve the player's overall statistics.
        tournamentid: The unique tournament id you want to retrieve the player's overall statistics.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/statistics/overall"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def icehockeyplayerlastmatches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player last matches."
    page: Zero based page.
        id: The player id you want to retrieve the last matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/player/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def icehockeyleagueregularseasontopteams(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league overall top team."
    seasonid: The season id you want to retrieve the league's best teams.
        tournamentid: The unique tournament id you want to retrieve the league's best teams.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/best-teams/regularSeason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def icehockeyleagueplayoffstopteams(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league overall top team."
    tournamentid: The unique tournament id you want to retrieve the league's best teams.
        seasonid: The season id you want to retrieve the league's best teams.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/best-teams/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def icehockeyleagueplayoffstopplayerspergame(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league per game top players."
    tournamentid: The unique tournament id you want to retrieve the league's best players.
        seasonid: The season id you want to retrieve the league's best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/best-players/per-game/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def icehockeyleagueoveralltopplayerspergame(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league per game top players."
    seasonid: The season id you want to retrieve the league's best players.
        tournamentid: The unique tournament id you want to retrieve the league's best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/best-players/per-game/overall"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def icehockeyleaguetopplayerstypes(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the top players types for a specific Ice Hockey league by providing the tournament ID and season ID."
    seasonid: The season ID for which you want to get the league's best players types.
        tournamentid: The unique tournament ID for which you want to get the league's best players types.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/best-players/types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def icehockeyleagueoveralltopplayers(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league overall top players."
    tournamentid: The unique tournament id you want to retrieve the league's best players.
        seasonid: The season id you want to retrieve the league's best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/best-players/overall"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def icehockeyleaguehometeamevents(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the last 5 matches for a specific league in a given season for home events."
    seasonid: The ID of the season for which you want to retrieve the home events.
        tournamentid: The unique tournament ID for which you want to retrieve the league's home team events
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/team-events/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def icehockeyleaguetotalteamevents(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the last 5 matches for a specific league in a given season for both home and away events."
    seasonid: The ID of the season for which the league's total team events are to be retrieved.
        tournamentid: The unique tournament ID for which you want to retrieve the league's total team events
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/team-events/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def footballuefacountryrankings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the current UEFA country rankings of Football teams."
    
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rankings/uefa/countries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def footballleaguepowerrankingbyroundbyid(round: int, seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation returns the power ranking round for the specified tournament, season and round ID for a football league."
    round: The round
        seasonid: The season ID for which you want to retrieve the power ranking round.
        tournamentid: The unique ID of the tournament for which you want to retrieve the power ranking round.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/power-rankings/round/{round}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def footballleaguepowerrankingrounds(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation returns the power ranking rounds for the specified tournament and season ID for a football league."
    tournamentid: The unique ID of the tournament for which you want to retrieve the power ranking rounds.
        seasonid: The season ID for which you want to retrieve the power ranking rounds.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/power-rankings/rounds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def footballmatchhomegoalkeeperjersey(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the home goalkeeper jersey for a specific Football match."
    id: The ID of the match you want to retrieve the home goalkeeper jersey for.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/match/{is_id}/goalkeeper/jersey/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def footballmatchawaygoalkeeperjersey(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the away goalkeeper jersey for a specific Football match."
    id: The ID of the match you want to retrieve the away goalkeeper jersey for.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/match/{is_id}/goalkeeper/jersey/away"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def footballmatchhomeplayerjersey(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the home player jersey for a specific Football match."
    id: The ID of the match you want to retrieve the home player jersey for.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/match/{is_id}/player/jersey/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_league_old_cup_trees(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league old cup trees structures."
    seasonid: The season id you want to retrieve the league cup trees.
        tournamentid: The unique tournament id you want to retrieve the league cup trees.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/tournament/{tournamentid}/season/{seasonid}/cup-trees/old"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_schedules(day: int, year: int, month: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get schedules of the day."
    day: The day you want to retrieve the schedules
        year: The year you want to retrieve the schedules
        month: The month you want to retrieve the schedules
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/matches/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_player_transfer_history(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player transfer history."
    id: The player id you want to retrieve the transfer history.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/player/{is_id}/transfer"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_calendar(month: int, year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get calendar of the month."
    month: The month you want to retrieve the calendar
        year: The year you want to retrieve the calendar
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/calendar/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_team_top_players_playoffs(is_id: int, seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team playoffs top players."
    id: The team id you want to retrieve the best players.
        seasonid: The season id you want to retrieve the team's best players.
        tournamentid: The unique tournament id you want to retrieve the team's best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/team/{is_id}/tournament/{tournamentid}/season/{seasonid}/best-players/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_match_best_players(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match best players."
    id: The id of the match you want to get the best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/match/{is_id}/best-players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_match_highlights(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match highlights."
    id: The id of the match you want to get highlights.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/match/{is_id}/highlights"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_player_media(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player media."
    id: The player id you want to retrieve the media.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/player/{is_id}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_event_games(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event games."
    id: The id of the event you want to get lineups.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/event/{is_id}/games"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_team_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team details."
    id: The team id you want to retrieve the details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/team/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_schedules(month: int, year: int, day: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get schedules of the day."
    month: The month you want to retrieve the schedules
        year: The year you want to retrieve the schedules
        day: The day you want to retrieve the schedules
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/matches/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_match_votes(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match votes."
    id: The id of the match you want to get votes.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/match/{is_id}/votes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_league_seasons(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league seasons."
    tournamentid: The unique tournament id you want to retrieve the league's seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/tournament/{tournamentid}/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_match_incidents(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match incidents."
    id: The id of the match you want to get incidents.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/match/{is_id}/incidents"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_player_statistics_seasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player statistics seasons."
    id: The player id you want to retrieve the statistics seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/player/{is_id}/statistics/season"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_player_characteristics(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player characteristics."
    id: The player id you want to retrieve the characteristics.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/player/{is_id}/characteristics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_league_details(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league details."
    tournamentid: The unique tournament id you want to retrieve the league's details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/tournament/{tournamentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_team_playoffs_top_players(tournamentid: int, is_id: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team playoffs top players."
    tournamentid: The unique tournament id you want to retrieve the team's best players.
        id: The team id you want to retrieve the best players.
        seasonid: The season id you want to retrieve the team's best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/team/{is_id}/tournament/{tournamentid}/season/{seasonid}/best-players/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_match_managers(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match managers."
    id: The id of the match you want to get managers.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/match/{is_id}/managers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_league_media(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league media."
    tournamentid: The unique tournament id you want to retrieve the league meia.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/tournament/{tournamentid}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_team_near_matches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team near matches."
    id: The team id you want to retrieve the near matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/team/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_player_image(playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player image. It generates png image."
    playerid: The player id you want to retrieve the image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/player/{playerid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_match_graph(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match graph."
    id: The id of the match you want to get graph.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/match/{is_id}/graph"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_team_tournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team tournaments."
    id: The team id you want to retrieve the tournaments.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/team/{is_id}/tournaments"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_search(term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search players, teams and tournaments."
    term: Search term.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/search/{term}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_event_streaks(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event streaks."
    id: The id of the event you want to get event streaks.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/event/{is_id}/streak"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_player_penalty_history(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player penalty history."
    id: The player id you want to retrieve the player penalty history.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/player/{is_id}/penalty"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_team_near_matches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team near matches."
    id: The team id you want to retrieve the near matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/team/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_league_details(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league details."
    tournamentid: The unique tournament id you want to retrieve the league's details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/tournament/{tournamentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_head_to_head_matches(customid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get head-to-head matches."
    customid: The custom id of the event you want to get head-to-head matches. It is obtained by the event.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/event/{customid}/h2h"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_league_seasons(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league seasons."
    tournamentid: The unique tournament id you want to retrieve the league's seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_league_media(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league media."
    tournamentid: The unique tournament id you want to retrieve the league meia.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/tournament/{tournamentid}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_fifa_rankings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get FIFA rankings."
    
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rankings/fifa"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_team_players(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team players."
    id: The team id you want to retrieve the players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/team/{is_id}/players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_team_tournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team tournaments."
    id: The team id you want to retrieve the tournaments.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/team/{is_id}/tournaments"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_player_or_team_last_events(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player or team last events."
    page: Zero based page.
        id: The player or team id you want to retrieve the last events.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/player/{is_id}/events/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_category_tournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all leagues from category."
    id: The category id you want to retrieve all leagues.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/tournament/all/category/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_live_matches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get live matches."
    
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/matches/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get categories."
    
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/tournament/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get categories."
    
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_league_details(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league details."
    tournamentid: The unique tournament id you want to retrieve the league's details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_player_or_team_rankings(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player or team rankings."
    id: The player or team id you want to retrieve the rankings.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/player/{is_id}/rankings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_player_or_team_near_events(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player or team near events."
    id: The player id you want to retrieve the near events.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/player/{is_id}/events/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_league_next_events(page: int, tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league next events."
    page: Zero based page.
        tournamentid: The unique tournament id you want to retrieve the league's next events.
        seasonid: The season id you want to retrieve the league's next events.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/tournament/{tournamentid}/season/{seasonid}/events/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_player_or_team_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player details."
    id: The player or team id you want to retrieve the details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/player/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_match_managers(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match managers."
    id: The id of the match you want to get managers.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/match/{is_id}/managers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_manager_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get manager details."
    id: The manager id you want to retrieve the details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/manager/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_event_statistics(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event statistics."
    id: The id of the event you want to get statistics.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/event/{is_id}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_manager_last_matches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get manager last matches."
    page: Zero based page.
        id: The manager id you want to retrieve the last matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/manager/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_match_odds(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match odds."
    id: The id of the match you want to get odds.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/match/{is_id}/odds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_manager_image(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get manager image. It generates png image."
    id: The manager id you want to retrieve the image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/manager/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_player_near_matches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player near matches."
    id: The player id you want to retrieve the near matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/player/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_league_last_matches(page: int, tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league last matches."
    page: Zero based page.
        tournamentid: The unique tournament id you want to retrieve the league's last matches.
        seasonid: The season id you want to retrieve the league's last matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/tournament/{tournamentid}/season/{seasonid}/matches/last/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_search(term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search players, teams and tournaments."
    term: Search term.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/search/{term}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_league_logo_image(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league logo image. It generates png image."
    tournamentid: The unique tournament id you want to retrieve the league logo image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_prematch_form(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get pre match form."
    id: The id of the match you want to get pre-match form.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/match/{is_id}/form"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_search(term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search players, teams and tournaments."
    term: Search term.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/search/{term}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_category_schedules(is_id: int, year: int, day: int, month: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get schedules of the day from the category."
    id: The category id you want to retrieve the schedules
        year: The year you want to retrieve the schedules
        day: The day you want to retrieve the schedules
        month: The month you want to retrieve the schedules
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/category/{is_id}/events/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_prematch_form(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get pre match form."
    id: The id of the match you want to get pre-match form.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/match/{is_id}/form"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_match_lineups(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match lineups."
    id: The id of the match you want to get lineups.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/match/{is_id}/lineups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_manager_next_matches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get manager next matches."
    page: Zero based page.
        id: The manager id you want to retrieve the next matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/manager/{is_id}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_event_streaks(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event streaks."
    id: The id of the event you want to get event streaks.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/event/{is_id}/streaks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_category_schedules(month: int, is_id: int, day: int, year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get schedules of the day from the category."
    month: The month you want to retrieve the schedules
        id: The category id you want to retrieve the schedules
        day: The day you want to retrieve the schedules
        year: The year you want to retrieve the schedules
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/category/{is_id}/events/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_player_near_matches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player near matches."
    id: The player id you want to retrieve the near matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/player/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_event(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event by id."
    id: The id of the event you want to get info.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/event/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_league_season_info(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league season info."
    seasonid: The season id you want to retrieve the league's season info.
        tournamentid: The unique tournament id you want to retrieve the league's season info.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_league_logo_image(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league logo image. It generates png image."
    tournamentid: The unique tournament id you want to retrieve the league logo image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/tournament/{tournamentid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_team_overall_statistics(seasonid: int, is_id: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team overall statistics."
    seasonid: The season id you want to retrieve the team's overall statistics.
        id: The team id you want to retrieve the overall statistics.
        tournamentid: The unique tournament id you want to retrieve the team's overall statistics.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/team/{is_id}/tournament/{tournamentid}/season/{seasonid}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_league_total_standings(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league total standings."
    seasonid: The season id you want to retrieve the league's total standings.
        tournamentid: The unique tournament id you want to retrieve the league's total standings.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/tournament/{tournamentid}/season/{seasonid}/standings/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_league_cup_trees(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league cup trees."
    seasonid: The season id you want to retrieve the league cup trees.
        tournamentid: The unique tournament id you want to retrieve the league cup trees.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/tournament/{tournamentid}/season/{seasonid}/cuptrees"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_event_odds(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event odds."
    id: The id of the event you want to get odds.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/event/{is_id}/odds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_category_tournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all leagues from category."
    id: The category id you want to retrieve all leagues.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/tournament/all/category/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_league_details(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league details."
    tournamentid: The unique tournament id you want to retrieve the league's details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/tournament/{tournamentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_league_next_matches(seasonid: int, page: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league next matches."
    seasonid: The season id you want to retrieve the league's next matches.
        page: Zero based page.
        tournamentid: The unique tournament id you want to retrieve the league's next matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/tournament/{tournamentid}/season/{seasonid}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_head_to_head_events(customid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get head-to-head events."
    customid: The custom id of the event you want to get head-to-head events.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/event/{customid}/h2h"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_league_media(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league media."
    tournamentid: The unique tournament id you want to retrieve the league meia.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_team_next_matches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team next matches."
    page: Zero based page.
        id: The team id you want to retrieve the next matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/team/{is_id}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_head_to_head_matches(customid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get head-to-head matches."
    customid: The custom id of the match you want to get head-to-head matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/match/{customid}/h2h"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_match_incidents(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match incidents."
    id: The id of the match you want to get incidents.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/match/{is_id}/incidents"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_league_logo_image(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league logo image. It generates png image."
    tournamentid: The unique tournament id you want to retrieve the league logo image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/tournament/{tournamentid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_team_media(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team media."
    id: The team id you want to retrieve the media.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/team/{is_id}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_head_to_head_matches(customid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get head-to-head matches."
    customid: The custom id of the match you want to get head-to-head matches. It is obtained by the match.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/match/{customid}/h2h"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_manager_career_history(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get manager career history."
    id: The manager id you want to retrieve the career history.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/manager/{is_id}/history"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_match(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match by id."
    id: The id of the match you want to get info.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/match/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_league_seasons(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league seasons."
    tournamentid: The unique tournament id you want to retrieve the league's seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/tournament/{tournamentid}/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_league_next_matches(seasonid: int, tournamentid: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league next matches."
    seasonid: The season id you want to retrieve the league's next matches.
        tournamentid: The unique tournament id you want to retrieve the league's next matches.
        page: Zero based page.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_league_seasons(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league seasons."
    tournamentid: The unique tournament id you want to retrieve the league's seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/tournament/{tournamentid}/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_match_votes(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match votes."
    id: The id of the match you want to get votes.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/match/{is_id}/votes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_league_total_standings(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league total standings."
    seasonid: The season id you want to retrieve the league's total standings.
        tournamentid: The unique tournament id you want to retrieve the league's total standings.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/tournament/{tournamentid}/season/{seasonid}/standings/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_team_logo_image(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team logo image. It generates png image."
    id: The team id you want to retrieve the logo image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/team/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_match_votes(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match votes."
    id: The id of the match you want to get votes.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/match/{is_id}/votes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_player_image(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player image. It generates png image."
    id: The player id you want to retrieve the image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/player/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_player_statistics_seasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player statistics seasons."
    id: The player id you want to retrieve the statistics seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/player/{is_id}/statistics/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_team_or_players_previous_tournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team or players previous tournaments."
    id: The player or team id you want to retrieve the players seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/player/{is_id}/tournaments/last"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_game_bans(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get game bans."
    id: The id of the game you want to get bans.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/game/{is_id}/bans"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_search(term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search players, teams and tournaments."
    term: Search term.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/search/{term}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_event_votes(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event votes."
    id: The id of the event you want to get votes.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/event/{is_id}/votes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_match_streaks(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match streaks."
    id: The id of the match you want to get match streaks.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/match/{is_id}/streaks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_player_regular_season_statistics(seasonid: int, is_id: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player regular season statistics."
    seasonid: The season id you want to retrieve the player's statistics.
        id: The player id you want to retrieve the statistics.
        tournamentid: The unique tournament id you want to retrieve the player's statistics.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/statistics/regularSeason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_player_media(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player media."
    id: The player id you want to retrieve the media.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/player/{is_id}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_schedules_top(month: int, day: int, year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get schedules of the day."
    month: The month you want to retrieve the schedules
        day: The day you want to retrieve the schedules
        year: The year you want to retrieve the schedules
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/matches/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_match_odds(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match odds."
    id: The id of the match you want to get odds.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/match/{is_id}/odds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_match_highlights(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match highlights."
    id: The id of the match you want to get highlights.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/match/{is_id}/highlights"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_league_details(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league details."
    tournamentid: The unique tournament id you want to retrieve the league's details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_event_highlights(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event highlights."
    id: The id of the event you want to get highlights.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/event/{is_id}/highlights"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_match_lineups(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match lineups."
    id: The id of the match you want to get lineups.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/match/{is_id}/lineups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_player_last_ratings(seasonid: int, tournamentid: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player last ratings."
    seasonid: The season id you want to retrieve the player's last ratings.
        tournamentid: The unique tournament id you want to retrieve the player's last ratings.
        id: The player id you want to retrieve the last ratings.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/last-ratings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_team_tournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team tournaments."
    id: The team id you want to retrieve the tournaments.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/team/{is_id}/tournaments"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_event_power_graph(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event power graph."
    id: The id of the event you want to get graph.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/event/{is_id}/graph"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_schedules_odds(year: int, day: int, month: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get schedules odds of the day."
    year: The year you want to retrieve the schedules odds
        day: The day you want to retrieve the schedules odds
        month: The month you want to retrieve the schedules odds
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/events/odds/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_player_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player details."
    id: The player id you want to retrieve the details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/player/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_category_schedules(is_id: int, month: int, year: int, day: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get schedules of the day from the category."
    id: The category id you want to retrieve the schedules
        month: The month you want to retrieve the schedules
        year: The year you want to retrieve the schedules
        day: The day you want to retrieve the schedules
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/category/{is_id}/events/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_league_cup_trees(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league cup trees."
    seasonid: The season id you want to retrieve the league cup trees.
        tournamentid: The unique tournament id you want to retrieve the league cup trees.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/cuptrees"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_league_cup_trees(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league cup trees."
    seasonid: The season id you want to retrieve the league cup trees.
        tournamentid: The unique tournament id you want to retrieve the league cup trees.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/tournament/{tournamentid}/season/{seasonid}/cup-trees"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_match_statistics(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match statistics."
    id: The id of the match you want to get statistics.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/match/{is_id}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_match_lineups(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match lineups."
    id: The id of the match you want to get lineups.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/match/{is_id}/lineups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def motorsport_team_races(is_id: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team races."
    id: The team id you want to retrieve the races.
        seasonid: The stage season id you want to retrieve the team's races.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/motorsport/team/{is_id}/stage/season/{seasonid}/races"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_league_logo_image(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league logo image. It generates png image."
    tournamentid: The unique tournament id you want to retrieve the league logo image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/tournament/{tournamentid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_team_media(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team media."
    id: The team id you want to retrieve the media.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/team/{is_id}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_event_h2h_duel(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event h2h duel."
    id: The id of the event you want to get head2head duel.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/event/{is_id}/duel"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_match_incidents(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match incidents."
    id: The id of the match you want to get incidents.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/match/{is_id}/incidents"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_team_next_matches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team next matches."
    page: Zero based page.
        id: The team id you want to retrieve the next matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/team/{is_id}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_league_seasons(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league seasons."
    tournamentid: The unique tournament id you want to retrieve the league's seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/tournament/{tournamentid}/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_match_incidents(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match incidents."
    id: The id of the match you want to get incidents.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/match/{is_id}/incidents"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_team_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team details."
    id: The team id you want to retrieve the details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/team/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_player_transfer_history(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player transfer history."
    id: The player id you want to retrieve the transfer history.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/player/{is_id}/transfers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_team_media(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team media."
    id: The team id you want to retrieve the media.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/team/{is_id}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_match_highlights(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match highlights."
    id: The id of the match you want to get highlights.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/match/{is_id}/highlights"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_category_schedules(is_id: int, year: int, day: int, month: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get schedules of the day from the category."
    id: The category id you want to retrieve the schedules
        year: The year you want to retrieve the schedules
        day: The day you want to retrieve the schedules
        month: The month you want to retrieve the schedules
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/category/{is_id}/events/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_match_lineups(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match lineups."
    id: The id of the match you want to get lineups.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/match/{is_id}/lineups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_league_logo_image(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league logo image. It generates png image."
    tournamentid: The unique tournament id you want to retrieve the league logo image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/tournament/{tournamentid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_team_regular_season_top_players(seasonid: int, tournamentid: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team regular season top players."
    seasonid: The season id you want to retrieve the team's beast players.
        tournamentid: The unique tournament id you want to retrieve the team's best players.
        id: The team id you want to retrieve the best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/team/{is_id}/tournament/{tournamentid}/season/{seasonid}/best-players/regularSeason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_team_logo_image(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team logo image. It generates png image."
    teamid: The team id you want to retrieve the logo image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/team/{teamid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_league_cup_trees(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league cup trees."
    seasonid: The season id you want to retrieve the league's cuptrees.
        tournamentid: The unique tournament id you want to retrieve the league's cuptrees.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/cuptrees"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_match_lineups(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match lineups."
    id: The id of the match you want to get lineups.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/match/{is_id}/lineups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_league_cup_trees(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league cup trees."
    tournamentid: The unique tournament id you want to retrieve the league cup trees.
        seasonid: The season id you want to retrieve the league cup trees.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/tournament/{tournamentid}/season/{seasonid}/cuptrees"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_league_last_matches(tournamentid: int, seasonid: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league last matches."
    tournamentid: The unique tournament id you want to retrieve the league's last matches.
        seasonid: The season id you want to retrieve the league's last matches.
        page: Zero based page.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/matches/last/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def motorsport_team_standings(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team standings."
    id: The stage id you want to retrieve team's standings.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/motorsport/stage/{is_id}/standings/team"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_player_regular_season_statistics(is_id: int, seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player regular season statistics."
    id: The player id you want to retrieve the statistics.
        seasonid: The season id you want to retrieve the player's statistics.
        tournamentid: The unique tournament id you want to retrieve the player's statistics.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/statistics/regularSeason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_live_matches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get live matches."
    
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/matches/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_team_logo_image(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team logo image. It generates png image."
    teamid: The team id you want to retrieve the logo image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/team/{teamid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_player_statistics_seasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player statistics seasons."
    id: The player id you want to retrieve the statistics seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/player/{is_id}/statistics/season"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_manager_last_matches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get manager last matches."
    id: The manager id you want to retrieve the last matches.
        page: Zero based page.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/manager/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_player_media(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player media."
    id: The player id you want to retrieve the media.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/player/{is_id}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_team_top_players_regular_season(seasonid: int, is_id: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team regular season top players."
    seasonid: The season id you want to retrieve the team's best players.
        id: The team id you want to retrieve the best players.
        tournamentid: The unique tournament id you want to retrieve the team's best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/team/{is_id}/tournament/{tournamentid}/season/{seasonid}/best-players/regularseason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def motorsport_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get categories."
    
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/motorsport/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_league_top_players_playoffs(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league playoffs top players."
    seasonid: The season id you want to retrieve the league's best players.
        tournamentid: The unique tournament id you want to retrieve the league's best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/tournament/{tournamentid}/season/{seasonid}/best-players/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_team_players_statistics_seasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team players statistics seasons."
    id: The team id you want to retrieve the players statistics seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/team/{is_id}/players/statistics/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get categories."
    
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/tournament/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_team_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team details."
    id: The team id you want to retrieve the details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/team/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_player_image(playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player image. It generates png image."
    playerid: The player id you want to retrieve the image. This can be obtained from lineups, best players or team players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/player/{playerid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_category_tournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all leagues from category."
    id: The category id you want to retrieve all leagues.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/tournament/all/category/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_league_away_standings(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league away standings."
    seasonid: The season id you want to retrieve the league's away standings.
        tournamentid: The unique tournament id you want to retrieve the league's away standings.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/tournament/{tournamentid}/season/{seasonid}/standings/away"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get categories."
    
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_team_last_matches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team last matches."
    page: Zero based page.
        id: The team id you want to retrieve the last matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/team/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_search(term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search players, teams and tournaments."
    term: Search term.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/search/{term}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_category_tournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all leagues from category."
    id: The category id you want to retrieve all leagues.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/tournament/all/category/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_category_tournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all leagues from category."
    id: The category id you want to retrieve all leagues.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/tournament/all/category/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_player_near_matches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player near matches."
    id: The player id you want to retrieve the near matches. This can be obtained from lineups, best players or team players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/player/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_live_matches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get live matches."
    
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/matches/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_match_lineups(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match lineups."
    id: The id of the match you want to get lineups.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/match/{is_id}/lineups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_schedules(month: int, day: int, year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get schedules of the day."
    month: The month you want to retrieve the schedules
        day: The day you want to retrieve the schedules
        year: The year you want to retrieve the schedules
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/matches/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_league_last_matches(tournamentid: int, seasonid: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league last matches."
    tournamentid: The unique tournament id you want to retrieve the league's last matches.
        seasonid: The season id you want to retrieve the league's last matches.
        page: Zero based page.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/tournament/{tournamentid}/season/{seasonid}/matches/last/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_match_h2h_duel(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match h2h duel."
    id: The id of the match you want to get head2head duel.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/match/{is_id}/duel"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def motorsport_team_image(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team image."
    id: The team id you want to retrieve image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/motorsport/team/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_team_logo_image(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team logo image. It generates png image."
    teamid: The team id you want to retrieve the logo image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/team/{teamid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_player_near_matches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player near matches."
    id: The player id you want to retrieve the near matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/player/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_search(term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search players, teams and tournaments."
    term: Search term.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/search/{term}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_category_tournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all leagues from category."
    id: The category id you want to retrieve all leagues.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/all/category/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_team_next_matches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team next matches."
    page: Zero based page.
        id: The team id you want to retrieve the next matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/team/{is_id}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_team_tournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team tournaments."
    id: The team id you want to retrieve the tournaments.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/team/{is_id}/tournaments"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_match_graph(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match graph."
    id: The id of the match you want to get graph.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/match/{is_id}/graph"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_player_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player details."
    id: The player id you want to retrieve the details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/player/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_player_media(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player media."
    id: The player id you want to retrieve the media.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/player/{is_id}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_league_next_matches(page: int, seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league next matches."
    page: Zero based page.
        seasonid: The season id you want to retrieve the league's next matches.
        tournamentid: The unique tournament id you want to retrieve the league's next matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_player_image(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player image. It generates png image."
    id: The player id you want to retrieve the image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/player/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_player_statistics_regular_season(is_id: int, tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player regular season statistics."
    id: The player id you want to retrieve the regular season statistics.
        tournamentid: The unique tournament id you want to retrieve the player's regular season statistics.
        seasonid: The season id you want to retrieve the player's regular season statistics.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/statistics/regularseason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_category_schedules(year: int, is_id: int, month: int, day: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get schedules of the day from the category."
    year: The year you want to retrieve the schedules
        id: The category id you want to retrieve the schedules
        month: The month you want to retrieve the schedules
        day: The day you want to retrieve the schedules
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/category/{is_id}/events/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_match_statistics(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match statistics."
    id: The id of the match you want to get statistics.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/match/{is_id}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_player_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player details."
    id: The player id you want to retrieve the details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/player/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_team_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team details."
    id: The team id you want to retrieve the details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/team/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_league_seasons(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league seasons."
    tournamentid: The unique tournament id you want to retrieve the league's seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_team_players_statistics_seasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team players statistics seasons."
    id: The team id you want to retrieve the players statistics seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/team/{is_id}/players/statistics/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def motorsport_search(term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search players, teams and tournaments."
    term: Search term.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/motorsport/search/{term}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_league_details(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league details."
    tournamentid: The unique tournament id you want to retrieve the league's details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/tournament/{tournamentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_match_votes(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match votes."
    id: The id of the match you want to get votes.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/match/{is_id}/votes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_head_to_head_matches(customid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get head-to-head matches."
    customid: The custom id of the match you want to get head-to-head matches. It is obtained by the match.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/match/{customid}/h2h"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_league_next_matches(seasonid: int, tournamentid: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league next matches."
    seasonid: The season id you want to retrieve the league's next matches.
        tournamentid: The unique tournament id you want to retrieve the league's next matches.
        page: Zero based page.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/tournament/{tournamentid}/season/{seasonid}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_head_to_head_matches(customid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get head-to-head matches."
    customid: The custom id of the match you want to get head-to-head matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/match/{customid}/h2h"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_league_seasons(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league seasons."
    tournamentid: The unique tournament id you want to retrieve the league's seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/tournament/{tournamentid}/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_match_votes(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match votes."
    id: The id of the match you want to get votes.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/match/{is_id}/votes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_team_standings_seasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team standings seasons."
    id: The team id you want to retrieve the standings seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/team/{is_id}/standings/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_team_last_matches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team last matches."
    page: Zero based page.
        id: The team id you want to retrieve the last matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/team/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_match_highlights(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match highlights."
    id: The id of the match you want to get highlights.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/match/{is_id}/highlights"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_match_highlights(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match highlights."
    id: The id of the match you want to get highlights.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/match/{is_id}/highlights"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_category_schedules(year: int, day: int, month: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get schedules of the day from the category."
    year: The year you want to retrieve the schedules
        day: The day you want to retrieve the schedules
        month: The month you want to retrieve the schedules
        id: The category id you want to retrieve the schedules
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/category/{is_id}/events/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_player_image(playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player image. It generates png image."
    playerid: The player id you want to retrieve the image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/player/{playerid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_schedules(day: int, year: int, month: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get schedules of the day."
    day: The day you want to retrieve the  schedules
        year: The year you want to retrieve the schedules
        month: The month you want to retrieve the schedules
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/events/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_match_h2h_duel(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match h2h duel."
    id: The id of the match you want to get head2head duel.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/baseball/match/{is_id}/duel"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_player_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player details."
    id: The player id you want to retrieve the details. This can be obtained from lineups, best players or team players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/player/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_match_graph(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match graph."
    id: The id of the match you want to get graph.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/match/{is_id}/graph"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_match(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match by id."
    id: The id of the match you want to get info.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/match/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_league_seasons(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league seasons."
    tournamentid: The unique tournament id you want to retrieve the league's seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/tournament/{tournamentid}/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_manager_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get manager details."
    id: The manager id you want to retrieve the details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/manager/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_match_odds(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match odds."
    id: The id of the match you want to get odds.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/match/{is_id}/odds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_league_home_standings(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league home standings."
    tournamentid: The unique tournament id you want to retrieve the league's home standings.
        seasonid: The season id you want to retrieve the league's home standings.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/tournament/{tournamentid}/season/{seasonid}/standings/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_search(term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search players, teams and tournaments."
    term: Search term.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/search/{term}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_match_managers(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match managers."
    id: The id of the match you want to get managers.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/match/{is_id}/managers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_team_media(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team media."
    id: The team id you want to retrieve the media.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/team/{is_id}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_league_next_matches(tournamentid: int, seasonid: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league next matches."
    tournamentid: The unique tournament id you want to retrieve the league's next matches.
        seasonid: The season id you want to retrieve the league's next matches.
        page: Zero based page.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/tournament/{tournamentid}/season/{seasonid}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_live_matches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get live matches."
    
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/matches/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def motorsport_category_stages(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all leagues from category."
    id: The category id you want to retrieve all stages.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/motorsport/category/{is_id}/stages/all"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_team_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team details."
    id: The team id you want to retrieve the details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/team/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_category_schedules(is_id: int, day: int, month: int, year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get schedules of the day from the category."
    id: The category id you want to retrieve the schedules
        day: The day you want to retrieve the schedules
        month: The month you want to retrieve the schedules
        year: The year you want to retrieve the schedules
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/category/{is_id}/events/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def motorsport_stage_substages(stageid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all stage's substages."
    stageid: The stage id you want to retrieve all substages.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/motorsport/stage/{stageid}/substages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_player_playoffs_statistics(tournamentid: int, is_id: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player playoffs statistics."
    tournamentid: The unique tournament id you want to retrieve the player's statistics.
        id: The player id you want to retrieve the statistics.
        seasonid: The season id you want to retrieve the player's statistics.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/statistics/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_team_playoffs_top_players(tournamentid: int, is_id: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team playoffs top players."
    tournamentid: The unique tournament id you want to retrieve the team's best players.
        id: The team id you want to retrieve the best players.
        seasonid: The season id you want to retrieve the team's beast players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/team/{is_id}/tournament/{tournamentid}/season/{seasonid}/best-players/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_match_innings(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match innings."
    id: The id of the match you want to get innings.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/match/{is_id}/innings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_team_near_matches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team near matches."
    id: The team id you want to retrieve the near matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/team/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_player_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player details."
    id: The player id you want to retrieve the details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/player/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_league_top_players_regular_season(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league regular season top players."
    seasonid: The season id you want to retrieve the league's best players.
        tournamentid: The unique tournament id you want to retrieve the league's best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/tournament/{tournamentid}/season/{seasonid}/best-players/regularSeason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def motorsport_stage_seasons(uniquestageid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get seasons of the stage."
    uniquestageid: The unique stage id you want to retrieve the seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/motorsport/unique-stage/{uniquestageid}/season"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_league_logo_image(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league logo image. It generates png image."
    tournamentid: The unique tournament id you want to retrieve the league logo image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/tournament/{tournamentid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_manager_image(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get manager image. It generates png image."
    id: The manager id you want to retrieve the image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/manager/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_team_next_matches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team next matches."
    page: Zero based page.
        id: The team id you want to retrieve the next matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/team/{is_id}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_head_to_head_matches(customid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get head-to-head matches."
    customid: The custom id of the match you want to get head-to-head matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/match/{customid}/h2h"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_live_matches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get live matches."
    
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/matches/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_match_streaks(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match streaks."
    id: The id of the match you want to get match streaks.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/match/{is_id}/team-streaks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_schedules(day: int, month: int, year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get schedules of the day."
    day: The day you want to retrieve the schedules
        month: The month you want to retrieve the schedules
        year: The year you want to retrieve the schedules
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/matches/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_league_logo_image(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league logo image. It generates png image."
    tournamentid: The unique tournament id you want to retrieve the league logo image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_team_next_matches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team next matches."
    page: Zero based page.
        id: The team id you want to retrieve the next matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/team/{is_id}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_player_near_matches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player near matches."
    id: The player id you want to retrieve the near matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/player/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_league_media(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league media."
    tournamentid: The unique tournament id you want to retrieve the league meia.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_league_total_standings(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league total standings."
    tournamentid: The unique tournament id you want to retrieve the league's total standings.
        seasonid: The season id you want to retrieve the league's total standings.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/tournament/{tournamentid}/season/{seasonid}/standings/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_character_image(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get character image. It generates png image."
    tournamentid: The character id you want to retrieve the league logo image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/character/{tournamentid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_league_regular_season_top_players(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league regular season top players."
    seasonid: The season id you want to retrieve the league's best players.
        tournamentid: The unique tournament id you want to retrieve the league's best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/best-players/regularSeason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_match_incidents(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match incidents."
    id: The id of the match you want to get incidents.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/match/{is_id}/incidents"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_match_odds(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match odds."
    id: The id of the match you want to get odds.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/match/{is_id}/odds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_category_tournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all leagues from category."
    id: The category id you want to retrieve all leagues.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/tournament/all/category/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_team_players(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team players."
    id: The team id you want to retrieve the players statistics seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/team/{is_id}/players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_team_regular_season_top_players(is_id: int, tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team regular season top players."
    id: The team id you want to retrieve the best players.
        tournamentid: The unique tournament id you want to retrieve the team's best players.
        seasonid: The season id you want to retrieve the team's best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/team/{is_id}/tournament/{tournamentid}/season/{seasonid}/best-players/regularSeason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_match_player_statistics(playerid: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match player statistics."
    playerid: The id of the player you want to get statistics.
        id: The id of the match you want to get player statistics.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/match/{is_id}/player/{playerid}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_team_standings_seasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team standings seasons."
    id: The team id you want to retrieve the standings seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/team/{is_id}/standings/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_team_logo_image(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team logo image. It generates png image."
    id: The team id you want to retrieve the logo image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/team/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_team_players(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team players."
    id: The team id you want to retrieve the players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/team/{is_id}/players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_match_best_players(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match best players."
    id: The id of the match you want to get the best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/match/{is_id}/best-players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_league_media(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league media."
    tournamentid: The unique tournament id you want to retrieve the league meia.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/tournament/{tournamentid}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get categories."
    
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/tournament/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_player_statistics_playoffs(seasonid: int, is_id: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player playoffs statistics."
    seasonid: The season id you want to retrieve the player's playoffs statistics.
        id: The player id you want to retrieve the playoffs statistics.
        tournamentid: The unique tournament id you want to retrieve the player's playoffs statistics.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/statistics/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_player_image(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player image. It generates png image."
    id: The player id you want to retrieve the image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/player/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_league_last_matches(seasonid: int, tournamentid: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league last matches."
    seasonid: The season id you want to retrieve the league's last matches.
        tournamentid: The unique tournament id you want to retrieve the league's last matches.
        page: Zero based page.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/matches/last/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_match_player_statistics(playerid: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match player statistics."
    playerid: The id of the player you want to get statistics.
        id: The id of the match you want to get player statistics.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/match/{is_id}/player/{playerid}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_team_last_matches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team last matches."
    id: The team id you want to retrieve the last matches.
        page: Zero based page.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/team/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_player_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player details."
    id: The player id you want to retrieve the details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/player/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_league_last_matches(seasonid: int, tournamentid: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league last matches."
    seasonid: The season id you want to retrieve the league's last matches.
        tournamentid: The unique tournament id you want to retrieve the league's last matches.
        page: Zero based page.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/matches/last/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_match_player_statistics(playerid: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match player statistics."
    playerid: The id of the player you want to get statistics.
        id: The id of the match you want to get player statistics.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/match/{is_id}/player/{playerid}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_tournament_meta(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tournament meta."
    id: The unique tournament id you want to get meta.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{is_id}/meta"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_event_h2h_duel(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event h2h duel."
    id: The id of the event you want to get head2head duel.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tennis/event/{is_id}/duel"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_league_away_standings(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league away standings."
    seasonid: The season id you want to retrieve the league's away standings.
        tournamentid: The unique tournament id you want to retrieve the league's away standings.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/standings/away"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_match_streaks(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match streaks."
    id: The id of the match you want to get match streaks.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/match/{is_id}/streaks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_manager_last_matches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get manager last matches."
    id: The manager id you want to retrieve the last matches.
        page: Zero based page.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/manager/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_game_lineups(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get game lineups."
    id: The id of the game you want to get lineups.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/game/{is_id}/lineups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_player_playoffs_statistics(tournamentid: int, is_id: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player playoffs statistics."
    tournamentid: The unique tournament id you want to retrieve the player's statistics.
        id: The player id you want to retrieve the statistics.
        seasonid: The season id you want to retrieve the player's statistics.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/statistics/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_league_rounds(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league team rounds."
    tournamentid: The unique tournament id you want to retrieve the league's rounds.
        seasonid: The season id you want to retrieve the league's rounds.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/rounds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_team_players(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team players."
    id: The team id you want to retrieve the players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/team/{is_id}/players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_league_media(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league media."
    tournamentid: The unique tournament id you want to retrieve the league meia.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/tournament/{tournamentid}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_match_h2h_duel(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match h2h duel."
    id: The id of the match you want to get head2head duel.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/match/{is_id}/duel"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_player_national_team_statistics(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player national team statistics."
    id: The player id you want to retrieve the national team statistics.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/player/{is_id}/statistics/national"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_league_last_matches(seasonid: int, page: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league last matches."
    seasonid: The season id you want to retrieve the league's last matches.
        page: Zero based page.
        tournamentid: The unique tournament id you want to retrieve the league's last matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/tournament/{tournamentid}/season/{seasonid}/matches/last/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_team_next_matches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team next matches."
    page: Zero based page.
        id: The team id you want to retrieve the next matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/team/{is_id}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_team_players_statistics_seasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team players statistics seasons."
    id: The team id you want to retrieve the players statistics seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/team/{is_id}/players/statistics/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_team_transfers(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team transfers."
    id: The team id you want to retrieve the transfers.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/team/{is_id}/transfers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_team_media(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team media."
    id: The team id you want to retrieve the media.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/team/{is_id}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_match_player_heatmap(is_id: int, pid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player heatmap in the  match. Check the tutorials to see html example."
    id: The id of the match you want to get players heatmap.
        pid: The player id you want to retrieve the heatmap.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/match/{is_id}/player/{pid}/heatmap"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_category_tournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all leagues from category."
    id: The category id you want to retrieve all leagues.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/all/category/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_match_h2h_duel(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match h2h duel."
    id: The id of the match you want to get head2head duel.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/match/{is_id}/duel"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_team_players(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team players."
    id: The team id you want to retrieve the players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/team/{is_id}/players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_category_tournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all leagues from category."
    id: The category id you want to retrieve all leagues.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/tournament/all/category/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_league_total_standings(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league total standings."
    tournamentid: The unique tournament id you want to retrieve the league's total standings.
        seasonid: The season id you want to retrieve the league's total standings.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/standings/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_schedules(year: int, day: int, month: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get schedules of the day."
    year: The year you want to retrieve the schedules
        day: The day you want to retrieve the schedules
        month: The month you want to retrieve the schedules
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/matches/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_match_streak_odds(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match streak odds."
    id: The id of the match you want to get match streaks odds.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/match/{is_id}/streaks/odds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_team_last_matches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team last matches."
    page: Zero based page.
        id: The team id you want to retrieve the last matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/team/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_match_incidents(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match incidents."
    id: The id of the match you want to get incidents.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/match/{is_id}/incidents"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_team_statistics_seasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team statistics seasons."
    id: The team id you want to retrieve team statistics seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/team/{is_id}/statistics/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_match_statistics(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match statistics."
    id: The id of the match you want to get statistics.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/match/{is_id}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_league_team_of_the_week(tournamentid: int, seasonid: int, roundid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league team of week."
    tournamentid: The unique tournament id you want to retrieve the league's team of the week.
        seasonid: The season id you want to retrieve the league's team of the week.
        roundid: The round id you want to retrieve the tournament team of the week.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/round/{roundid}/team-of-week"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_match_managers(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match managers."
    id: The id of the match you want to get managers.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/match/{is_id}/managers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def motorsport_stage_image(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stage image."
    id: The stage id you want to retrieve image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/motorsport/stage/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_league_seasons(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league seasons."
    tournamentid: The unique tournament id you want to retrieve the league's seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_team_last_matches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team last matches."
    id: The team id you want to retrieve the last matches.
        page: Zero based page.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/team/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_prematch_form(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get pre match form."
    id: The id of the match you want to get pre-match form.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/match/{is_id}/form"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_league_cup_trees(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league cup trees."
    tournamentid: The unique tournament id you want to retrieve the league cup trees.
        seasonid: The season id you want to retrieve the league cup trees.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/cuptrees"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_league_home_standings(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league home standings."
    seasonid: The season id you want to retrieve the league's home standings.
        tournamentid: The unique tournament id you want to retrieve the league's home standings.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/standings/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_player_near_matches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player near matches."
    id: The player id you want to retrieve the near matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/player/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_player_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player details."
    id: The player id you want to retrieve the details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/player/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_team_media(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team media."
    id: The team id you want to retrieve the media.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/team/{is_id}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_game_rounds(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get game rounds."
    id: The id of the game you want to get rounds.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/game/{is_id}/rounds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_league_logo_image(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league logo image. It generates png image."
    tournamentid: The unique tournament id you want to retrieve the league logo image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/tournament/{tournamentid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_league_last_matches(seasonid: int, page: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league last matches."
    seasonid: The season id you want to retrieve the league's last matches.
        page: Zero based page.
        tournamentid: The unique tournament id you want to retrieve the league's last matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/tournament/{tournamentid}/season/{seasonid}/matches/last/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_league_details(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league details."
    tournamentid: The unique tournament id you want to retrieve the league's details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/tournament/{tournamentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_team_near_matches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team near matches."
    id: The team id you want to retrieve the near matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/team/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_match_h2h_duel(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match h2h duel."
    id: The id of the match you want to get head2head duel.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/match/{is_id}/duel"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_live_matches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get live matches."
    
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/matches/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_schedules(year: int, day: int, month: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get schedules of the day."
    year: The year you want to retrieve the schedules
        day: The day you want to retrieve the schedules
        month: The month you want to retrieve the schedules
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/matches/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_team_near_matches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team near matches."
    id: The team id you want to retrieve the near matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/team/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_search(term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search players, teams and tournaments."
    term: Search term.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/search/{term}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_team_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team details."
    id: The team id you want to retrieve the details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/team/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_league_playoffs_top_players(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league playoffs top players."
    seasonid: The season id you want to retrieve the league's best players.
        tournamentid: The unique tournament id you want to retrieve the league's best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/best-players/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_match_odds(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match odds."
    id: The id of the match you want to get odds.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/match/{is_id}/odds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_match_highlights(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match highlights."
    id: The id of the match you want to get highlights.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/match/{is_id}/highlights"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_league_total_standings(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league total standings."
    seasonid: The season id you want to retrieve the league's total standings.
        tournamentid: The unique tournament id you want to retrieve the league's total standings.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/tournament/{tournamentid}/season/{seasonid}/standings/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_league_details(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league details."
    tournamentid: The unique tournament id you want to retrieve the league's details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/tournament/{tournamentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_team_players_statistics_seasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team players statistics seasons."
    id: The team id you want to retrieve the players statistics seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/team/{is_id}/players/statistics/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_team_last_matches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team last matches."
    id: The team id you want to retrieve the last matches.
        page: Zero based page.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/team/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_match(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match by id."
    id: The id of the match you want to get info.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/match/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_team_standings_seasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team standings seasons."
    id: The team id you want to retrieve team standings seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/team/{is_id}/standings/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_player_image(playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player image. It generates png image."
    playerid: The player id you want to retrieve the image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/player/{playerid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_team_near_matches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team near matches."
    id: The team id you want to retrieve the near matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/team/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_match_statistics(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match statistics."
    id: The id of the match you want to get statistics.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/match/{is_id}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_player_statistics_seasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player statistics seasons."
    id: The player id you want to retrieve the statistics seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/player/{is_id}/statistics/season"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_league_next_matches(page: int, seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league next matches."
    page: Zero based page.
        seasonid: The season id you want to retrieve the league's next matches.
        tournamentid: The unique tournament id you want to retrieve the league's next matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/tournament/{tournamentid}/season/{seasonid}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_league_logo_image(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league logo image. It generates png image."
    tournamentid: The unique tournament id you want to retrieve the league logo image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_head_to_head_matches(customid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get head-to-head matches."
    customid: The custom id of the match you want to get head-to-head matches. It is obtained by the match.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/match/{customid}/h2h"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_match_h2h_duel(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match h2h duel."
    id: The id of the match you want to get head2head duel.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/match/{is_id}/duel"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_team_next_matches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team next matches."
    id: The team id you want to retrieve the next matches.
        page: Zero based page.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/team/{is_id}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get categories."
    
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/tournament/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_player_overall_statistics(is_id: int, tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player overall statistics."
    id: The player id you want to retrieve the overall statistics.
        tournamentid: The unique tournament id you want to retrieve the player's overall statistics.
        seasonid: The season id you want to retrieve the player's overall statistics.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_league_overall_top_players(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league overall top players."
    seasonid: The season id you want to retrieve the league's best players.
        tournamentid: The unique tournament id you want to retrieve the league's best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/tournament/{tournamentid}/season/{seasonid}/top-players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_match_lineups(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match lineups."
    id: The id of the match you want to get lineups.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/match/{is_id}/lineups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_manager_last_matches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get manager last matches."
    id: The manager id you want to retrieve the last matches.
        page: Zero based page.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/manager/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_category_schedules(day: int, year: int, month: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get schedules of the day from the category."
    day: The day you want to retrieve the schedules
        year: The year you want to retrieve the schedules
        month: The month you want to retrieve the schedules
        id: The category id you want to retrieve the schedules
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/category/{is_id}/events/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_match_statistics(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match statistics."
    id: The id of the match you want to get statistics.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/match/{is_id}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_team_media(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team media."
    id: The team id you want to retrieve the media.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/team/{is_id}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_game_statistics(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get game statistics."
    id: The id of the game you want to get statistics.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/game/{is_id}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def motorsport_stage_details(stageid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stage details."
    stageid: The stage id you want to retrieve the details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/motorsport/stage/{stageid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_team_near_matches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team near matches."
    id: The team id you want to retrieve the near matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/team/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_match_managers(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match managers."
    id: The id of the match you want to get managers.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/match/{is_id}/managers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_league_season_info(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league season info."
    seasonid: The season id you want to retrieve the league's season info.
        tournamentid: The unique tournament id you want to retrieve the league's season info.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/tournament/{tournamentid}/season/{seasonid}/info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_prematch_form(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get pre match form."
    id: The id of the match you want to get pre-match form.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/match/{is_id}/form"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_match(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match by id."
    id: The id of the match you want to get info.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/match/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_event_lineups(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event lineups."
    id: The id of the event you want to get lineups.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/event/{is_id}/lineups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_match_votes(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match votes."
    id: The id of the match you want to get votes.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/match/{is_id}/votes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_team_last_matches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team last matches."
    id: The team id you want to retrieve the last matches.
        page: Zero based page.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/team/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_team_tournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team tournaments."
    id: The team id you want to retrieve the tournaments.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/team/{is_id}/tournaments"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def motorsport_team_stage_seasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team stage seasons."
    id: The team id you want to retrieve the stage seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/motorsport/team/{is_id}/stage/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def motorsport_unique_stage_image(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get unique stage image."
    id: The unique stage id you want to retrieve image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/motorsport/unique-stage/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_prematch_form(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get pre match form."
    id: The id of the match you want to get pre-match form.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/match/{is_id}/form"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_head_to_head_matches(customid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get head-to-head matches."
    customid: The custom id of the match you want to get head-to-head matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/match/{customid}/h2h"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_team_logo_image(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team logo image. It generates png image."
    teamid: The team id you want to retrieve the logo image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/team/{teamid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_match(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match by id."
    id: The id of the match you want to get info.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/match/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_match_streak_odds(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match streak odds."
    id: The id of the match you want to get match streaks odds.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/match/{is_id}/streaks/odds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_match_h2h_duel(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match h2h duel."
    id: The id of the match you want to get head2head duel.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/match/{is_id}/duel"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_league_media(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league media."
    tournamentid: The unique tournament id you want to retrieve the league meia.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_prematch_form(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get pre match form."
    id: The id of the match you want to get pre-match form.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/match/{is_id}/form"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get categories."
    
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/tournament/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_player_image(playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player image. It generates png image."
    playerid: The player id you want to retrieve the image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/player/{playerid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_match_best_players(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match best players."
    id: The id of the match you want to get the best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/match/{is_id}/best-players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_league_next_matches(seasonid: int, tournamentid: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league next matches."
    seasonid: The season id you want to retrieve the league's next matches.
        tournamentid: The unique tournament id you want to retrieve the league's next matches.
        page: Zero based page.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ice_hockey_league_away_standings(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league away standings."
    tournamentid: The unique tournament id you want to retrieve the league's away standings.
        seasonid: The season id you want to retrieve the league's away standings.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/ice-hockey/tournament/{tournamentid}/season/{seasonid}/standings/away"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_schedules_top(year: int, month: int, day: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top schedules of the day."
    year: The year you want to retrieve the schedules
        month: The month you want to retrieve the schedules
        day: The day you want to retrieve the schedules
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/matches/top/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_league_team_of_the_week_rounds(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league team rounds."
    tournamentid: The unique tournament id you want to retrieve the league's rounds.
        seasonid: The season id you want to retrieve the league's rounds.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/tournament/{tournamentid}/season/{seasonid}/tow/rounds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_manager_image(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get manager image. It generates png image."
    id: The manager id you want to retrieve the image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/manager/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_player_attributes(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player attributes."
    id: The player id you want to retrieve the attributes.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/player/{is_id}/attribute"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e_sports_map_image(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get map image. It generates png image."
    tournamentid: The unique tournament id you want to retrieve the league logo image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/esport/map/{tournamentid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_player_overall_statistics(is_id: int, seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player overall statistics."
    id: The player id you want to retrieve the overall statistics.
        seasonid: The season id you want to retrieve the player's overall statistics.
        tournamentid: The unique tournament id you want to retrieve the player's overall statistics.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/statistics/overall"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_league_top_players_regular_season(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league regular season top players."
    seasonid: The season id you want to retrieve the league's best players.
        tournamentid: The unique tournament id you want to retrieve the league's best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/best-players/regularseason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_team_transfers(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team transfers."
    id: The team id you want to retrieve the transfers.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/team/{is_id}/transfers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rugby_search(term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search players, teams and tournaments."
    term: Search term.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/rugby/search/{term}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_league_total_standings(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league total standings."
    tournamentid: The unique tournament id you want to retrieve the league's total standings.
        seasonid: The season id you want to retrieve the league's total standings.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/standings/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def motorsport_competitor_standings(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competitor standings."
    id: The stage id you want to retrieve competitor's standings.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/motorsport/stage/{is_id}/standings/competitor"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def motorsport_team_driver_history(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get driver history."
    id: The team id you want to retrieve the driver history.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/motorsport/team/{is_id}/driver/history"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_league_top_players_playoffs(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league playoffs top players."
    tournamentid: The unique tournament id you want to retrieve the league's best players.
        seasonid: The season id you want to retrieve the league's best players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}/season/{seasonid}/best-players/playoffs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_league_total_standings(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league total standings."
    seasonid: The season id you want to retrieve the league's total standings.
        tournamentid: The unique tournament id you want to retrieve the league's total standings.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/tournament/{tournamentid}/season/{seasonid}/standings/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_player_statistics_seasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player statistics seasons."
    id: The player id you want to retrieve the statistics seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/player/{is_id}/statistics/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_league_details(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league details."
    tournamentid: The unique tournament id you want to retrieve the league's details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/tournament/{tournamentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_match(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match by id."
    id: The id of the match you want to get info.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/match/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_match(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match by id."
    id: The id of the match you want to get info.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/match/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_team_logo_image(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team logo image. It generates png image."
    teamid: The team id you want to retrieve the logo image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/team/{teamid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def american_football_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get categories."
    
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/american-football/tournament/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_team_logo_image(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team logo image. It generates png image."
    id: The team id you want to retrieve the logo image.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/basketball/team/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_team_near_matches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team near matches."
    id: The team id you want to retrieve the near matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/team/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_live_matches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get live matches."
    
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/matches/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_team_players(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team players."
    id: The team id you want to retrieve the players.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/team/{is_id}/players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def motorsport_team_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team details."
    id: The team id you want to retrieve the details.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/motorsport/team/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_match_odds(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match odds."
    id: The id of the match you want to get odds.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/match/{is_id}/odds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_team_players_statistics_seasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team players statistics seasons."
    id: The team id you want to retrieve the players statistics seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/team/{is_id}/players/statistics/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_league_last_matches(tournamentid: int, page: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league last matches."
    tournamentid: The unique tournament id you want to retrieve the league's last matches.
        page: Zero based page.
        seasonid: The season id you want to retrieve the league's last matches.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/tournament/{tournamentid}/season/{seasonid}/matches/last/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def motorsport_featured_stage(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get featured stage."
    
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/motorsport/stage/featured"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_team_standings_seasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team standings seasons."
    id: The team id you want to retrieve team standings seasons.
        
    """
    url = f"https://allsportsapi2.p.rapidapi.com/api/cricket/team/{is_id}/standings/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

