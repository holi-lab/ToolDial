import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_medications(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides a list of vaccines"
    
    """
    url = f"https://vaccines-availability-data-usa.p.rapidapi.com/api/medications"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vaccines-availability-data-usa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_vaccine_provider_details(providerid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides detailed information for a particular vaccine provider"
    providerid: Provider Id of a Vaccine Provider
        
    """
    url = f"https://vaccines-availability-data-usa.p.rapidapi.com/api/providers"
    querystring = {'providerId': providerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vaccines-availability-data-usa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_vaccine_providers_by_zipcode(radius: int, zipcode: str, medicationids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides a list of vaccine providers in the vicinity of a  zipcode"
    radius: Radius (In Miles) - Must be among the following values
-  radius=1
-  radius=5
-  radius=10
-  radius=25
-  radius=50
-  radius=100
        medicationids: Comma-separated values of medication Ids
        
    """
    url = f"https://vaccines-availability-data-usa.p.rapidapi.com/api/providers/search/by-zipcode"
    querystring = {'radius': radius, 'zipcode': zipcode, 'medicationIds': medicationids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vaccines-availability-data-usa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_vaccine_providers_by_location(longitude: int, medicationids: str, radius: int, latitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides a list of vaccine providers in the vicinity of a  Location (Latitude & Longitude)"
    longitude: Longitude of the location
        medicationids: Comma-separated values of medication Ids
        radius: Radius (In Miles) - Must be among the following values
-  radius=1
-  radius=5
-  radius=10
-  radius=25
-  radius=50
-  radius=100
        latitude: Latitude of the location
        
    """
    url = f"https://vaccines-availability-data-usa.p.rapidapi.com/api/providers/search/by-location"
    querystring = {'longitude': longitude, 'medicationIds': medicationids, 'radius': radius, 'latitude': latitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vaccines-availability-data-usa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

