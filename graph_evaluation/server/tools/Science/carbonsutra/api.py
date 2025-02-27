import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cluster_data(cluster_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Emissions can be grouped into self-defined clusters and retrieved using those labels. This eliminates the need to store the results at developer's end. A key will need to be generated, which is then passed as a Bearer Token (within Authorization) in all calculations, so that the user can be identified and the results are securely delivered.
		
		Use the POST API of RegisterKey to obtain your API_KEY which will be used in all estimation calculations for which clustering is required. Your email address and a secret phrase (password) will be required to generate this key.
		
		The POST API of RetrieveKey will give you the key again if it was misplaced, as long as the correct password is entered.
		
		Note that if clustering is not needed then you do not need API_KEY.
		
		The GET API of ClusterData returns all the estimations that were calculated for a specific cluster_name."
    
    """
    url = f"https://carbonsutra1.p.rapidapi.com/cluster_data"
    querystring = {'cluster_name': cluster_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "carbonsutra1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def distance_between_airports(iata_airport_to: str, iata_airport_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the distance between two IATA airport codes in kilometers."
    
    """
    url = f"https://carbonsutra1.p.rapidapi.com/distance-between-airports"
    querystring = {'iata_airport_to': iata_airport_to, 'iata_airport_from': iata_airport_from, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "carbonsutra1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airports_by_keyword(keyword: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the lists of airport names and IATA code which matches the keyword."
    
    """
    url = f"https://carbonsutra1.p.rapidapi.com/airports-by-keyword"
    querystring = {'keyword': keyword, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "carbonsutra1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nearest_airport_from_postal_code(postal_code: str, country_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is an advanced algorithm which takes a postal code and country code as input and returns the nearest airport its lat-long value. It is important to note that the straight line distance between the lat-long of postal code and airport is measured. The returned value is in kilometers. The source of 1.5 million postal code data is taken as it is from https://public.opendatasoft.com/explore/dataset/geonames-postal-code/. For airports, the publicly available database of IATA codes and airport names is used."
    country_code: Two digit country code
        
    """
    url = f"https://carbonsutra1.p.rapidapi.com/nearest-airport"
    querystring = {'postal_code': postal_code, 'country_code': country_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "carbonsutra1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vehicles_models(vehicle_make: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of Vehicle Models for a Make. A list of every make and model used by this API is at https://www.carbonsutra.com/data-vehicle-make-model.html"
    
    """
    url = f"https://carbonsutra1.p.rapidapi.com/vehicle_makes/{vehicle_make}/vehicle_models"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "carbonsutra1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vehicles_makes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of Vehicle Makes. A list of every make and model used by this API is at https://www.carbonsutra.com/data-vehicle-make-model.html"
    
    """
    url = f"https://carbonsutra1.p.rapidapi.com/vehicle_makes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "carbonsutra1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

