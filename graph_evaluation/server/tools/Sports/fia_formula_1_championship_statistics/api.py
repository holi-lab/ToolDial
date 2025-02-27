import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def team_details(team_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to retrieve a detailed information about a specific team. You will need to provide the team's name in the path parameter called team_name. You can specify either full name of the team or partial. If partial name is provided as parameter, still the api will make an attempt to search and return the expected result. The expected response will hold information about team's base location , team chief, chassis & more additional statistical data.
		EXAMPLE: redbull or red, haas or haasf1team, alfaromeo or alfa romeo"
    
    """
    url = f"https://fia-formula-1-championship-statistics.p.rapidapi.com/api/teams/details/{team_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fia-formula-1-championship-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def drivers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to retrieve this season's official F1 line-up. The result from the endpoint will be a breakdown of drivers, points and current positions."
    
    """
    url = f"https://fia-formula-1-championship-statistics.p.rapidapi.com/api/drivers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fia-formula-1-championship-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def race_results(year: int=1987, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to retrieve race results data about a specific F1 championship by specifying a year. If you ommit the ***year*** query parameter, a default value will be set to current year. The response data will contain information about the Grand Prix, Date, Winner, Car (Team), Laps & overall time of each driver in the form of a collection."
    
    """
    url = f"https://fia-formula-1-championship-statistics.p.rapidapi.com/api/standings/race-results"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fia-formula-1-championship-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def race_result(year: int, location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to retrieve information about a specific race result by passing path parameters year & location. The path parameters are required."
    
    """
    url = f"https://fia-formula-1-championship-statistics.p.rapidapi.com/api/races/race-results/{year}/{location}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fia-formula-1-championship-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hall_of_fame(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to retrieve information about all F1 champions also known as the Hall of Fame. The result of the endpoint will be a breakdown of driver's name and in which year the driver was crowned champion."
    
    """
    url = f"https://fia-formula-1-championship-statistics.p.rapidapi.com/api/drivers/hall-of-fame"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fia-formula-1-championship-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def driver_details(driver_fullname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to retrieve a detailed information about a specific driver. You will need to provide the driver's name in the path parameter called team_name. You can specify either full name of the team or partial. If partial name is provided as parameter, still the api will make an attempt to search and return the expected result. Check out the example response for an overview of the result."
    
    """
    url = f"https://fia-formula-1-championship-statistics.p.rapidapi.com/api/drivers/details/{driver_fullname}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fia-formula-1-championship-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to retrieve information about this year's F1 teams. The result will be a breakdown of the Team's Rank (current position in the standings), Points, Team name & Drivers."
    
    """
    url = f"https://fia-formula-1-championship-statistics.p.rapidapi.com/api/teams"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fia-formula-1-championship-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def race_calendar(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to retrieve the current championship schedule and information about the **next round**. Time zone is GMT +00:00 Europe/London"
    
    """
    url = f"https://fia-formula-1-championship-statistics.p.rapidapi.com/api/schedule/race-schedule"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fia-formula-1-championship-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def practice_session_results(year: str, location: str, session_number: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to retrieve information about a specific practice session by passing path parameters for **session number**, **year** &  **location**."
    
    """
    url = f"https://fia-formula-1-championship-statistics.p.rapidapi.com/api/races/practice/{session_number}/{year}/{location}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fia-formula-1-championship-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def qualifying_results(year: int, location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to retrieve the qualifying results for a specific race by specifying the year and location as path parameters. The parameters are required."
    
    """
    url = f"https://fia-formula-1-championship-statistics.p.rapidapi.com/api/races/qualifying/{year}/{location}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fia-formula-1-championship-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def starting_grid(year: int, location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to retrieve information about a specific race starting grid by passing path parameters for year & location. The parameters are required."
    
    """
    url = f"https://fia-formula-1-championship-statistics.p.rapidapi.com/api/races/starting-grid/{year}/{location}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fia-formula-1-championship-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pitstop_summary(year: int, location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to retrieve the pitstop summary about a specific race by passing path parameters year & location. The path parameters are required."
    
    """
    url = f"https://fia-formula-1-championship-statistics.p.rapidapi.com/api/races/pit-stop-summary/{year}/{location}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fia-formula-1-championship-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fastest_laps_result(location: str, year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to retrieve information about fastest laps for a specific race by passing path parameters location & year. The path parameters are required."
    
    """
    url = f"https://fia-formula-1-championship-statistics.p.rapidapi.com/api/races/fastest-laps/{year}/{location}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fia-formula-1-championship-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def constructors_standings(year: int=2005, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to retrieve constructor standings data about a specific F1 championship by specifying a year. If you ommit the ***year *** query parameter, a default value will be set to current year."
    
    """
    url = f"https://fia-formula-1-championship-statistics.p.rapidapi.com/api/standings/constructors-standings"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fia-formula-1-championship-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fastest_laps(year: int=2005, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to retrieve fastest laps data about a specific F1 championship by specifying a year. If you ommit the year query parameter, a default value will be set to current year. The response data will contain information about the Grand Prix, Driver, Car & Fastest Lap Time in the form of a collection."
    
    """
    url = f"https://fia-formula-1-championship-statistics.p.rapidapi.com/api/standings/dhl-fastest-lap"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fia-formula-1-championship-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def drivers_standings(year: int=2020, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to retrieve drivers standings data about a specifit F1 championship by specifying a year. If you ommit the ***year*** query parameter, a default value will be set to current year. The response data will contain information about the position in the rank list, driver's name, nationality, team and total points accumulated."
    
    """
    url = f"https://fia-formula-1-championship-statistics.p.rapidapi.com/api/standings/drivers-standings"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fia-formula-1-championship-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

