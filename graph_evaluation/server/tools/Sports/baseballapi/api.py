import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for baseball-related entities using the provided search term, and filter the results to show only baseball-related entities."
    term: The search term to use for finding baseball-related entities.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/search/{term}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
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
    url = f"https://baseballapi.p.rapidapi.com/api/placeholder/player.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
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
    url = f"https://baseballapi.p.rapidapi.com/api/placeholder/team.svg"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueawaystandings(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the away standings of a specific baseball league for a specific season."
    seasonid: The ID of the season for which you want to retrieve the away standings.
        tournamentid: The unique tournament ID for which you want to retrieve the away standings.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/tournament/{tournamentid}/season/{seasonid}/standings/away"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playernearmatches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the near matches for a specific baseball player using the player ID."
    id: The player ID for which you want to retrieve the near matches.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/player/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
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
    url = f"https://baseballapi.p.rapidapi.com/api/tv/country/all/event/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
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
    url = f"https://baseballapi.p.rapidapi.com/api/img/flag/{flag}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchlineups(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the lineups for a specific baseball match by providing its ID."
    id: The ID of the baseball match for which you want to get the lineups.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/match/{is_id}/lineups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguedetails(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of a specific baseball league using the unique tournament ID."
    tournamentid: The unique tournament ID for which you want to retrieve the league's details.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/tournament/{tournamentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchstatistics(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the statistics for a specific baseball match by providing its ID."
    id: The ID of the baseball match for which you want to get the statistics.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/match/{is_id}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerregularseasonstatistics(tournamentid: int, is_id: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the regular season statistics for a specific baseball player using the player ID, tournament ID, and season ID."
    tournamentid: The unique tournament ID for which you want to retrieve the player's statistics.
        id: The player ID for which you want to retrieve the statistics.
        seasonid: The season ID for which you want to retrieve the player's statistics.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/player/{is_id}/tournament/{tournamentid}/season/{seasonid}/statistics/regularSeason"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguehomestandings(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the home standings of a specific baseball league for a specific season."
    tournamentid: The unique tournament ID for which you want to retrieve the home standings.
        seasonid: The ID of the season for which you want to retrieve the home standings.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/tournament/{tournamentid}/season/{seasonid}/standings/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerstatisticsseasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the statistics seasons for a specific baseball player using the player ID."
    id: The player ID for which you want to retrieve the statistics seasons.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/player/{is_id}/statistics/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerlastmatches(is_id: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the last matches played by a specific Baseball player by providing the player ID and page number."
    id: The ID of the player for which you want to retrieve the last matches.
        page: Zero-based page number.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/player/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguetotalstandings(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the total standings of a specific baseball league for a specific season."
    tournamentid: The unique tournament ID for which you want to retrieve the total standings.
        seasonid: The ID of the season for which you want to retrieve the total standings.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/tournament/{tournamentid}/season/{seasonid}/standings/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamstandingsseasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the team standings for different seasons for a given team by providing its ID."
    id: The ID of the team for which you want to retrieve the team standings for different seasons.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/team/{is_id}/standings/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguenextmatches(page: int, seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the next matches for a specific baseball league using the tournament ID, season ID, and page."
    page: Zero-based page.
        seasonid: The season ID for which you want to retrieve the league's next matches.
        tournamentid: The unique tournament ID for which you want to retrieve the league's next matches.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/tournament/{tournamentid}/season/{seasonid}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguemedia(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the media for a specific baseball league using the unique tournament ID."
    tournamentid: The unique tournament ID for which you want to retrieve the league media.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/tournament/{tournamentid}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerimage(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the image for a specific baseball player using the player ID. Generates a PNG image."
    id: The player ID for which you want to retrieve the image.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/player/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamtournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the tournaments in which a specific baseball team participates."
    id: The ID of the team for which you want to retrieve the tournaments.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/team/{is_id}/tournaments"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchhighlights(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the highlights of a specific baseball match using the match ID."
    id: The ID of the match for which you want to get highlights.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/match/{is_id}/highlights"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguetotalteamevents(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the last 5 matches for a specific league in a given season for both home and away events."
    seasonid: The ID of the season for which you want to retrieve the total team events.
        tournamentid: The unique tournament ID for which you want to retrieve the total team events.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/tournament/{tournamentid}/season/{seasonid}/team-events/total"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguecuptrees(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the cup trees for a specific baseball league using the tournament ID and season ID."
    seasonid: The season ID for which you want to retrieve the league cup trees.
        tournamentid: The unique tournament ID for which you want to retrieve the league cup trees.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/tournament/{tournamentid}/season/{seasonid}/cuptrees"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categorytournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of all leagues from a specific baseball category using the category ID."
    id: The category ID for which you want to retrieve all leagues.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/tournament/all/category/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
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
    url = f"https://baseballapi.p.rapidapi.com/api/placeholder/tournament.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchschedules(month: int, year: int, day: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation returns the baseball match schedules for the given date, including match timings, teams, and other relevant information."
    month: The month for which you want to retrieve the match schedules (1-12).
        year: The year for which you want to retrieve the match schedules (e.g., 2022).
        day: The day of the month for which you want to retrieve the match schedules (1-31).
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/matches/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information for a specific baseball match by providing the match ID."
    id: The ID of the match for which you want to get the details.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/match/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details for a specific baseball player using the player ID."
    id: The player ID for which you want to retrieve the details.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/player/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
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
    seasonid: The ID of the season for which you want to retrieve the away team events.
        tournamentid: The unique tournament ID for which you want to retrieve the away team events.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/tournament/{tournamentid}/season/{seasonid}/team-events/away"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagueseasons(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the seasons for a specific baseball league using the unique tournament ID."
    tournamentid: The unique tournament ID for which you want to retrieve the league's seasons.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/tournament/{tournamentid}/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamlogoimage(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the logo image for a specific baseball team using the team ID. Generates a PNG image."
    id: The team ID for which you want to retrieve the logo image.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/team/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def livematches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get live baseball matches that are currently taking place."
    
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/matches/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamnextmatches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get upcoming baseball matches for a specific team."
    page: The page number (zero-based) of the results you want to retrieve.
        id: The ID of the team for which you want to retrieve upcoming matches.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/team/{is_id}/matches/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categoryschedules(year: int, is_id: int, day: int, month: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the baseball match schedules for a specific day and category using the category ID, day, month, and year."
    year: The year for which you want to retrieve the schedules.
        id: The category ID for which you want to retrieve the schedules.
        day: The day for which you want to retrieve the schedules.
        month: The month for which you want to retrieve the schedules.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/category/{is_id}/events/{day}/{month}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchodds(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the odds for a specific baseball match using the match ID."
    id: The ID of the match for which you want to get the odds.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/match/{is_id}/odds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguelogoimage(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the logo image for a specific baseball league using the unique tournament ID. Generates a PNG image."
    tournamentid: The unique tournament ID for which you want to retrieve the league logo image.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/tournament/{tournamentid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details for a specific baseball team using the team ID."
    id: The team ID for which you want to retrieve the details.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/team/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchh2hduel(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the head-to-head duel for a specific baseball match using the match ID."
    id: The ID of the match for which you want to get the head-to-head duel.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/match/{is_id}/duel"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
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
    url = f"https://baseballapi.p.rapidapi.com/api/tv/channel/{channid}/event/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamnearmatches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the near matches for a specific baseball team using the team ID."
    id: The team ID for which you want to retrieve the near matches.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/team/{is_id}/matches/near"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_manager_placeholder(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the manager placeholder image in PNG format."
    
    """
    url = f"https://baseballapi.p.rapidapi.com/api/placeholder/manager.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamlastmatches(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the last matches for a specific baseball team by providing its ID and page number."
    page: The zero-based page number of the results you want to retrieve.
        id: The ID of the baseball team for which you want to retrieve the last matches.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/team/{is_id}/matches/previous/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguelastmatches(seasonid: int, page: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the last matches for a league by providing the unique tournament ID, season ID, and the page number (0-based)."
    seasonid: The season ID for which you want to retrieve the last matches.
        page: The 0-based page number for which you want to retrieve the last matches.
        tournamentid: The unique tournament ID for which you want to retrieve the last matches.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/tournament/{tournamentid}/season/{seasonid}/matches/last/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchvotes(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the votes for a specific baseball match using the match ID."
    id: The ID of the match for which you want to get the votes.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/match/{is_id}/votes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teammedia(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the media for a specific baseball team using the team ID."
    id: The team ID for which you want to retrieve the media.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/team/{is_id}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def headtoheadmatches(customid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get head-to-head match data for a specific baseball match using its custom ID."
    customid: The custom ID of the match for which you want to get the head-to-head matches.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/match/{customid}/h2h"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def prematchform(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the pre-match form for a specific baseball match using the match ID."
    id: The ID of the match for which you want to get the pre-match form.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/match/{is_id}/form"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamplayers(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the players for a specific baseball team using the team ID."
    id: The team ID for which you want to retrieve the players.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/team/{is_id}/players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguehometeamevents(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the last 5 matches for a specific league in a given season for home events."
    seasonid: The ID of the season for which you want to retrieve the home team events.
        tournamentid: The unique tournament ID for which you want to retrieve the home team events.
        
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/tournament/{tournamentid}/season/{seasonid}/team-events/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of all baseball categories."
    
    """
    url = f"https://baseballapi.p.rapidapi.com/api/baseball/tournament/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseballapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

