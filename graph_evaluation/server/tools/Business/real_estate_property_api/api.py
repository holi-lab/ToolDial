import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_property_details_by_id(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will show all property details by its Id.
		Send a URL parameter named "id" with a number.
		Example: ?id=1000"
    
    """
    url = f"https://real-estate-property-api.p.rapidapi.com/get-property-by-id.php"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-estate-property-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_property_details_by_address(assembledaddress: str, city: str='Seattle', state: str='WA', zip: str='98103', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will show all property details by its Address.
		Send a URL parameter named "assembledAddress".
		Optional Parameters for accuracy:
		city
		state
		zip"
    
    """
    url = f"https://real-estate-property-api.p.rapidapi.com/get-property-by-address.php"
    querystring = {'assembledAddress': assembledaddress, }
    if city:
        querystring['city'] = city
    if state:
        querystring['state'] = state
    if zip:
        querystring['zip'] = zip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-estate-property-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

