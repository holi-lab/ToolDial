import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get(object_type: str, tripit_object_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A get request is used to retrieve a specific object from the TripIt API given its TripIt object ID. Since get requests do not change data they are all made via an HTTP GET request for a URL that takes the following form:"
    object_type: is one of the following strings:  air activity car cruise directions lodging map note points_program profile rail restaurant transport trip weather
        tripit_object_id: is the object ID of the TripIt object being requested.
        
    """
    url = f"https://community-tripit.p.rapidapi.com/get/{object_type}/id/{TripIt_Object_ID}/format/json"
    querystring = {'tripit_object_id': tripit_object_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-tripit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list(object_type: str, filter_parameter: str, filter_value: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list request is used to retrieve multiple objects given an object type and set of filter parameters. Since list requests do not change data they are all made via an HTTP GET request for a URL that takes the following form:"
    object_type: is one of the following strings:  trip object points_program
        filter_parameter: Valid values for <filter parameter> and <filter value> depend on the <object type>.  See  http://tripit.github.io/api/doc/v1/index.html#resource_section for a table of valid combinations.
        filter_value: Valid values for <filter parameter> and <filter value> depend on the <object type>.  See  http://tripit.github.io/api/doc/v1/index.html#resource_section for a table of valid combinations.
        
    """
    url = f"https://community-tripit.p.rapidapi.com/list/{object_type}/{filter_parameter}/{filter_value}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-tripit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def delete(object_type: str, tripit_object_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A delete request is used to delete existing objects"
    object_type: is one of the following strings:  air activity car cruise directions lodging map note rail restaurant segment transport trip
        tripit_object_id: is the object ID of the TripIt object being requested.
        
    """
    url = f"https://community-tripit.p.rapidapi.com/delete/{object_type}/id/{TripIt_Object_ID}"
    querystring = {'tripit_object_id': tripit_object_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-tripit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

