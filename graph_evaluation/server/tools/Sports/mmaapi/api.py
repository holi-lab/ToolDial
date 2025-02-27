import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation allows you to search for MMA teams and tournaments based on the search term provided."
    term: The search term to use when searching for teams and tournaments.
        
    """
    url = f"https://mmaapi.p.rapidapi.com/api/mma/search/{term}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmaapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation allows you to retrieve MMA categories."
    
    """
    url = f"https://mmaapi.p.rapidapi.com/api/mma/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmaapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournamentlastmainevents(tournamentid: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get last main events for a unique tournamente"
    tournamentid: This operation allows you to retrieve the next main events for a specified MMA tournament.
        page: Zero based page.
        
    """
    url = f"https://mmaapi.p.rapidapi.com/api/mma/unique-tournament/{tournamentid}/main-events/last/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmaapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def uniquetournamentdetails(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation allows you to retrieve details for a specified MMA tournament."
    tournamentid: The unique tournament ID for which you want to retrieve the details.
        
    """
    url = f"https://mmaapi.p.rapidapi.com/api/mma/unique-tournament/{tournamentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmaapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categorytournaments(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation allows you to retrieve all leagues from a specified MMA category."
    id: The category ID for which you want to retrieve all leagues.
        
    """
    url = f"https://mmaapi.p.rapidapi.com/api/mma/category/{is_id}/unique-tournaments"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmaapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournamentdetails(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation allows you to retrieve details for a specified MMA tournament."
    tournamentid: The tournament ID for which you want to retrieve the details.
        
    """
    url = f"https://mmaapi.p.rapidapi.com/api/mma/tournament/{tournamentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmaapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournamentfeaturedevents(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get featured events for a unique tournament"
    tournamentid: The unique tournament ID for which you want to retrieve the featured events.
        
    """
    url = f"https://mmaapi.p.rapidapi.com/api/mma/unique-tournament/{tournamentid}/featured-events"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmaapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournamentallevents(tournamentid: int, uniquetournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all MMA events for a unique tournament and tournament ID"
    tournamentid: The tournament ID for which you want to retrieve the events.
        uniquetournamentid: The unique tournament ID for which you want to retrieve the events.
        
    """
    url = f"https://mmaapi.p.rapidapi.com/api/mma/unique-tournament/{uniquetournamentid}/tournament/{tournamentid}/mma-events/all"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmaapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def headtoheadevents(customid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about head-to-head tennis events for a specific event."
    customid: The custom ID of the event you want to get head-to-head events.
        
    """
    url = f"https://mmaapi.p.rapidapi.com/api/mma/event/{customid}/h2h/events"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmaapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamcareerstatistics(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the career statistics for a specific team."
    id: The ID of the team you want to retrieve the career statistics for.
        
    """
    url = f"https://mmaapi.p.rapidapi.com/api/mma/team/{is_id}/career-statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmaapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def eventform(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This operation returns the pre game form for a specific MMA event, including team statistics and player data."
    id: The ID of the MMA event for which you want to get pre-match form.
        
    """
    url = f"https://mmaapi.p.rapidapi.com/api/mma/event/{is_id}/pregame-form"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmaapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournamentlogoimage(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tournament logo image in PNG format for a specific tournament."
    id: The unique tournament ID you want to retrieve the logo image.
        
    """
    url = f"https://mmaapi.p.rapidapi.com/api/mma/unique-tournament/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmaapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def femalefighterimageplaceholder(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get female fighter placeholder image"
    
    """
    url = f"https://mmaapi.p.rapidapi.com/api/placeholders/fighter-female.svg"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmaapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamimage(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player image in PNG format for a specific team."
    id: The team ID you want to retrieve the image.
        
    """
    url = f"https://mmaapi.p.rapidapi.com/api/mma/team/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmaapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information about a specific team."
    id: The team ID you want to retrieve the details.
        
    """
    url = f"https://mmaapi.p.rapidapi.com/api/mma/team/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmaapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamnextevents(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about the next events for a specific team."
    page: Zero-based page.
        id: The team ID you want to retrieve the next events.
        
    """
    url = f"https://mmaapi.p.rapidapi.com/api/mma/team/{is_id}/events/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmaapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def eventstatistics(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics for a specific MMA event."
    id: The ID of the event you want to get statistics.
        
    """
    url = f"https://mmaapi.p.rapidapi.com/api/mma/event/{is_id}/statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmaapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournamentnextmainevents(page: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get next main events for a unique tournamente"
    page: Zero based page.
        tournamentid: This operation allows you to retrieve the next main events for a specified MMA tournament.
        
    """
    url = f"https://mmaapi.p.rapidapi.com/api/mma/unique-tournament/{tournamentid}/main-events/next/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmaapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def malefighterimageplaceholder(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get male fighter placeholder image"
    
    """
    url = f"https://mmaapi.p.rapidapi.com/api/placeholders/fighter-male.svg"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmaapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def eventdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about a MMA event based on its ID."
    id: The ID of the event you want to get information about.
        
    """
    url = f"https://mmaapi.p.rapidapi.com/api/mma/event/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmaapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamlastevents(page: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about the last events for a specific team."
    page: Zero-based page.
        id: The team ID you want to retrieve the last events.
        
    """
    url = f"https://mmaapi.p.rapidapi.com/api/mma/team/{is_id}/events/last/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmaapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mmaoctogonimage(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get MMA octagon image"
    
    """
    url = f"https://mmaapi.p.rapidapi.com/api/mma/octogon.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmaapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournamentrnkingsummary(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get ranking summary for a unique tournament"
    tournamentid: This operation allows you to retrieve tournament rankings summary.
        
    """
    url = f"https://mmaapi.p.rapidapi.com/api/mma/rankings/unique-tournament/{tournamentid}/summary"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmaapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def eventstreaks(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get streak information for a specific MMA event."
    id: The ID of the event you want to get event streaks.
        
    """
    url = f"https://mmaapi.p.rapidapi.com/api/mma/event/{is_id}/team-streaks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmaapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

