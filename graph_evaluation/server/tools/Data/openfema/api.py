import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_entity(entity: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can find a full list of supported Entities at http://www.fema.gov/data-feeds#APIs"
    entity: This corresponds to the name of the entity set you are requesting. The entity names can be found in the list of released data sets.
        
    """
    url = f"https://community-openfema.p.rapidapi.com/{entity}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-openfema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_entity_by_id(entity: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a specific record identified by its ID field (_id)"
    entity: This corresponds to the name of the entity set you are requesting. The entity names can be found in the list of released data sets.
        id: This is a _id value of a previously identified record in an entity
        
    """
    url = f"https://community-openfema.p.rapidapi.com/{entity}/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-openfema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

