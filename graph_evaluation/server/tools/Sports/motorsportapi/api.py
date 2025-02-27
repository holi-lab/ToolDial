import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def stagehighlights(stageid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the highlights of a specific Motorsport stage."
    stageid: The stage ID for which to retrieve the highlights.
        
    """
    url = f"https://motorsportapi.p.rapidapi.com/api/motorsport/stage/{stageid}/highlights"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorsportapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Perform a search for Motorsport players, teams, and tournaments using the provided search term."
    term: The search term to use for finding players, teams, and tournaments.
        
    """
    url = f"https://motorsportapi.p.rapidapi.com/api/motorsport/search/{term}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorsportapi.p.rapidapi.com"
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
    url = f"https://motorsportapi.p.rapidapi.com/api/placeholder/team.svg"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorsportapi.p.rapidapi.com"
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
    url = f"https://motorsportapi.p.rapidapi.com/api/placeholder/tournament.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorsportapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamstageseasons(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the stage seasons for a specific Motorsport team."
    id: The team ID for which to retrieve the stage seasons.
        
    """
    url = f"https://motorsportapi.p.rapidapi.com/api/motorsport/team/{is_id}/stage/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorsportapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamstandings(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team standings for a specific Motorsport stage."
    id: The stage ID for which to retrieve team's standings.
        
    """
    url = f"https://motorsportapi.p.rapidapi.com/api/motorsport/stage/{is_id}/standings/team"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorsportapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamdriverhistory(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the driver history for a specific Motorsport team by providing the team ID."
    id: The ID of the team for which you want to get the driver history.
        
    """
    url = f"https://motorsportapi.p.rapidapi.com/api/motorsport/team/{is_id}/driver/history"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorsportapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stageimage(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get image for a specific Motorsport stage."
    id: The stage ID for which to retrieve the image.
        
    """
    url = f"https://motorsportapi.p.rapidapi.com/api/motorsport/stage/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorsportapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamimage(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get image for a specific Motorsport team."
    id: The team ID for which to retrieve the image.
        
    """
    url = f"https://motorsportapi.p.rapidapi.com/api/motorsport/team/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorsportapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def uniquestageseasons(uniquestageid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the seasons of a specific Motorsport stage."
    uniquestageid: The unique stage ID for which to retrieve the seasons.
        
    """
    url = f"https://motorsportapi.p.rapidapi.com/api/motorsport/unique-stage/{uniquestageid}/season"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorsportapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stagedetails(stageid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of a specific Motorsport stage."
    stageid: The stage ID for which to retrieve the details.
        
    """
    url = f"https://motorsportapi.p.rapidapi.com/api/motorsport/stage/{stageid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorsportapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def featuredstage(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of the featured stage in Motorsport."
    
    """
    url = f"https://motorsportapi.p.rapidapi.com/api/motorsport/stage/featured"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorsportapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of all Motorsport categories."
    
    """
    url = f"https://motorsportapi.p.rapidapi.com/api/motorsport/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorsportapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stagesubstages(stageid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all substages of a specific Motorsport stage."
    stageid: The stage ID for which to retrieve all substages.
        
    """
    url = f"https://motorsportapi.p.rapidapi.com/api/motorsport/stage/{stageid}/substages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorsportapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamdetails(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information for a specific Motorsport team by providing the team ID."
    id: The ID of the team for which you want to get the details.
        
    """
    url = f"https://motorsportapi.p.rapidapi.com/api/motorsport/team/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorsportapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stagecompetitorstandings(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the competitor standings for a specific Motorsport stage."
    id: The stage ID for which to retrieve competitor's standings.
        
    """
    url = f"https://motorsportapi.p.rapidapi.com/api/motorsport/stage/{is_id}/standings/competitor"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorsportapi.p.rapidapi.com"
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
    url = f"https://motorsportapi.p.rapidapi.com/api/tv/country/all/event/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorsportapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamraces(is_id: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get races for a specific Motorsport team during a stage season."
    id: The team ID for which to retrieve the races.
        seasonid: The stage season ID for which to retrieve the team's races.
        
    """
    url = f"https://motorsportapi.p.rapidapi.com/api/motorsport/team/{is_id}/stage/season/{seasonid}/races"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorsportapi.p.rapidapi.com"
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
    url = f"https://motorsportapi.p.rapidapi.com/api/img/flag/{flag}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorsportapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def uniquestageimage(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get image for a specific Motorsport unique stage."
    id: The unique stage ID for which to retrieve the image.
        
    """
    url = f"https://motorsportapi.p.rapidapi.com/api/motorsport/unique-stage/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorsportapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categorystages(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all stages for a specific Motorsport category."
    id: The category ID for which to retrieve all stages.
        
    """
    url = f"https://motorsportapi.p.rapidapi.com/api/motorsport/category/{is_id}/stages/all"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorsportapi.p.rapidapi.com"
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
    url = f"https://motorsportapi.p.rapidapi.com/api/tv/channel/{channid}/event/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorsportapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

