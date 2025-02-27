import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_location_geourn(matchtype: str, location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This requires an exact match (case insensitive)
		
		The best place to find the locations you are looking for is on Linkedin. You can see them on people profiles as well as location menus in the search functions.  
		
		Here are some examples included:
		
		Seattle, Washington, United States
		Greater Seattle Area
		Washington, United States"
    
    """
    url = f"https://linkedin-geourn-codes.p.rapidapi.com/"
    querystring = {'MatchType': matchtype, 'Location': location, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "linkedin-geourn-codes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

