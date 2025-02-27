import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def all_divisions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "`/v1.0/divisions`
		Get all divisions of Bangladesh in English and Bangla.
		
		**Response**
		```
		status: object,
		data: [
		  _id: string,
		  division: string
		  divisionbn: string
		]
		```"
    
    """
    url = f"https://bdapi.p.rapidapi.com/v1.0/divisions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bdapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_0(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "`/v1.0`
		Prefix endpoint for version 1.0
		Response provides a list of available endpoints in version 1.0"
    
    """
    url = f"https://bdapi.p.rapidapi.com/v1.0"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bdapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_districts(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "`/v1.0/districts`
		Get all Districts of Bangladesh in English and Bangla.
		
		**Response**
		```
		status: object,
		data: [
		  _id: string,
		  district: string,
		  districtbn: string
		]
		```"
    
    """
    url = f"https://bdapi.p.rapidapi.com/v1.0/districts"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bdapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def district_detail_of_specific_division(division_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "`/v1.0/division/{division_name}`
		Get all Districts and Upazila of any Division.
		
		**Response:**
		```
		status: object,
		data: [
		  _id: string:district,
		  district: string,
		  upazilla: array
		]
		```"
    
    """
    url = f"https://bdapi.p.rapidapi.com/v1.0/division/{division_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bdapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_1(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "`/v1.1`
		Prefix endpoint for version 1.1
		Response provides a list of available endpoints in version 1.1"
    
    """
    url = f"https://bdapi.p.rapidapi.com/v1.1"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bdapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_divisions_with_coordinates(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "`/v1.1/divisions`
		Get all divisions with coordinates in English and Bangla.
		
		**Response**
		```
		status: object,
		data: [
		  _id: string,
		  division: string,
		  divisionbn: string,
		  coordinates: string
		]
		```"
    
    """
    url = f"https://bdapi.p.rapidapi.com/v1.1/divisions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bdapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_districts_with_coordinates(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "`/v1.1/districts`
		Get all Districts with their Coordinates in English and Bangla.
		
		**Response**
		```
		status: object,
		data: [
		  _id: string,
		  district: string,
		  districtbn: string,
		  coordinates: string
		]
		```"
    
    """
    url = f"https://bdapi.p.rapidapi.com/v1.1/districts"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bdapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def district_detail_and_coordinates_of_specific_division(division_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "`/v1.1/division/{division_name}`
		Get all Districts, Coordinates, and Upazila of any Division.
		
		**Response:**
		```
		status: object,
		data: [
		  _id: string:district,
		  district: string,
		  coordinates: string
		  upazilla: array
		]
		```"
    
    """
    url = f"https://bdapi.p.rapidapi.com/v1.1/division/{division_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bdapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

