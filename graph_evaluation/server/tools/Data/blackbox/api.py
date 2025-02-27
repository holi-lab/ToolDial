import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def legacy_v1(ips: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a legacy and free IP checked. We provide a simply non-json style response to requests. Responses are cached for 7 days. 
		
		Check out our Documentation for more information.
		
		Responses will be (Y,E,N):
		
		(Y) : Yes, we have determined the IP address to be used by either or multiple of the following: Cloud, Hosting, Tor, Proxy. 
		
		(E) : Error, something has gone wrong with the request.
		
		(N) : No, we have determined the IP address to be NOT used by any of the following: Cloud, Hosting, Tor, Proxy."
    
    """
    url = f"https://blackbox.p.rapidapi.com/v1/{ips}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blackbox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def multiple_array_v2(ips: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using the GET method you can request an IP specific detections, and other IP information.
		
		Check out our documentation for more info on our different detections."
    
    """
    url = f"https://blackbox.p.rapidapi.com/v2/{ips}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blackbox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def singular_object_v2(ip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Note: This will work only with singular IP lookups.
		
		If you are unkeen to lookup multiple IP addresses at the same time, and want a simpler request. This might be for you. 
		
		We will return a singular IP object, that will minimize your request complexity, but with the loss of flexibility."
    
    """
    url = f"https://blackbox.p.rapidapi.com/v2/obj/{ip}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blackbox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

