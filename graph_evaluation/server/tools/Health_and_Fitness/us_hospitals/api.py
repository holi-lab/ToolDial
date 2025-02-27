import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def gethospitalsbyid(ccn: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "###Find US Hospitals by CMS Certification Number
		Simply add your search string as a parameter to the "ccn" query.
		
		**Note**: The API only returns the first 30 results."
    
    """
    url = f"https://us-hospitals.p.rapidapi.com/"
    querystring = {'ccn': ccn, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-hospitals.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gethospitalsbyname(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "###Find US Hospitals by name.
		Simply add your search string to the "name" parameter in the url.
		Tip:  The API also works if your search for *name='pr'* instead of *name='presbyterian'.* 
		
		**Note**: The API only returns the first 30 results."
    
    """
    url = f"https://us-hospitals.p.rapidapi.com/"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-hospitals.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

