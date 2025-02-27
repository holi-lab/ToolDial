import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def basic(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Basic Endpoint: Essential Moon Phase Information**
		
		Obtain the fundamental details about the current phase of the moon with the Basic Endpoint. This endpoint provides a concise JSON response including the name of the moon phase, the stage of the moon's cycle, and the number of days until the next full moon and new moon. Access this endpoint to get a quick overview of the moon's current state."
    
    """
    url = f"https://moon-phase.p.rapidapi.com/basic"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moon-phase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lunar_calendar(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Calendar Endpoint: Lunar Calendar**
		
		Retrieve a comprehensive Lunar Calendar with the Calendar Endpoint. This endpoint generates a markdown table representing the moon phases for each day of the year or a specific month. The table displays the moon phase emoji for each day, allowing you to easily visualise the lunar cycle. Whether you need an overview of the entire year or a specific month, the Calendar Endpoint provides a clear and structured presentation of the moon phases. Access this endpoint to explore the moon's phases in a calendar format."
    
    """
    url = f"https://moon-phase.p.rapidapi.com/calendar"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moon-phase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def advanced(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Advanced Endpoint: Comprehensive Lunar and Solar Details**
		
		Unlock comprehensive insights into the lunar and solar aspects with the Advanced Endpoint. This endpoint delivers an extensive JSON object filled with detailed information about the moon, including its phase, illumination percentage, age in days since the last new moon, lunar cycle completion percentage, current phase name, moon-rise and moon set times, zodiac/star sign association, and much more. Additionally, it provides sun-related data such as sunrise and sunset times, solar noon, day length, and solar positions. Dive deep into lunar and solar analytics with this endpoint to enhance your understanding of the moon's dynamics.
		
		**Filter Data in Advanced Moon API**
		
		The Advanced Moon API allows users to filter the returned data based on their specific requirements. By including the `filters` parameter in the API request, users can customise the data fields they want to retrieve. The `filters` parameter accepts a comma-separated list of keys representing the desired data fields.
		
		**Example Usage**
		
		**Request**
		
		```
		GET /advanced?filters=moon.phase_name,moon.stage,moon_phases.full_moon.next
		```
		
		This example request filters the data to include only the moon's phase name, stage, and the next full moon information.
		
		**Response**
		
		```json
		{
		  "moon": {
		    "phase_name": "First Quarter",
		    "stage": "waxing",
		    "moon_phases": {
		      "full_moon": {
		        "next": {
		          "timestamp": 1671379200,
		          "datestamp": "2023-01-17T00:00:00+00:00",
		          "days_ahead": 258,
		          "name": "Wolf Moon",
		          "description": "Named after the howling wolves often heard during this time."
		        }
		      }
		    }
		  }
		}
		```
		
		In the response, only the filtered data fields are included.
		
		**Additional Filter Examples**
		
		- Filter only the moon's phase name and age in days:
		
		  ```
		  GET /advanced?filters=moon.phase_name,moon.age_days
		  ```
		
		- Filter the moon's phase name and the sunrise time:
		
		  ```
		  GET /advanced?filters=moon.phase_name,sun.sunrise_timestamp
		  ```
		
		- Filter the moon's phase name and the next new moon date:
		
		  ```
		  GET /advanced?filters=moon.phase_name,moon_phases.new_moon.next.datestamp
		  ```
		
		Users can customise the filters according to their specific requirements to retrieve the desired data fields."
    
    """
    url = f"https://moon-phase.p.rapidapi.com/advanced"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moon-phase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def phase(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Plain Text Endpoint: Simple Moon Phase Description**
		
		Retrieve a straightforward, text-based description of the moon's current phase using the Plain Text Endpoint. This endpoint offers a clear and concise explanation of the moon phase, providing you with a brief understanding of its appearance and position in its cycle. Utilise this endpoint when you need a plain text description of the moon phase for easy integration and display."
    
    """
    url = f"https://moon-phase.p.rapidapi.com/plain-text"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moon-phase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def emoji(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Emoji Endpoint: Moon Phase Emoji**
		
		Obtain the relevant emoji representation of the moon's current phase using the Emoji Endpoint. This endpoint returns a single emoji character that represents the specific phase of the moon, allowing you to visually depict the moon's appearance and progression in its cycle. Incorporate this endpoint when you need a concise and expressive emoji representation of the moon phase in your applications or interfaces."
    
    """
    url = f"https://moon-phase.p.rapidapi.com/emoji"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moon-phase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

