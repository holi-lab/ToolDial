import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def pixelstix_meta(pixelstix_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will retrieve the meta information for a given PixelStix code and any Maps/Galleries it may be associated with if the PixelStix is of type 'PREMIUM'"
    
    """
    url = f"https://pixelstix.p.rapidapi.com/api/v2/public/object_voices/{pixelstix_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pixelstix.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pixelstix_map_meta(pixelstix_map_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A PixelStix Map is a collection of PREMIUM PixelStix that have been assigned latitude and longitude coordinates. A PREMIUM PixelStix can belong to multiple Maps and is also known as a Gallery.
		
		This endpoint will retrieve information about your particular Map/Gallery. The name used in the endpoint is the same name that you will see in the list of galleries when using the PixelStix app."
    
    """
    url = f"https://pixelstix.p.rapidapi.com/api/v2/public/tags/tag_name/{pixelstix_map_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pixelstix.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_last_n_days_scan_counts_for_tag(days: str, jwt: str, map_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the number of scans over the last N days, per day"
    
    """
    url = f"https://pixelstix.p.rapidapi.com/api/v2/analytics/tags/{map_id}/recent_days_scans"
    querystring = {'days': days, 'jwt': jwt, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pixelstix.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

