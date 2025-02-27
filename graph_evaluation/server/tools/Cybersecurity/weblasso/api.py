import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_scan_status(scan_id: str='e04d3e18-bda7-420b-b240-894fd3d4992d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check the status of a scan (ongoing/completed) using a scan ID."
    
    """
    url = f"https://weblasso.p.rapidapi.com/scan-status"
    querystring = {}
    if scan_id:
        querystring['scan_id'] = scan_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weblasso.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_callback_url_status(callback_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check if a callback URL is able to receive scan results (ensure that the callback URL accepts POST requests)."
    
    """
    url = f"https://weblasso.p.rapidapi.com/callback-status"
    querystring = {'callback_url': callback_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weblasso.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_scan_id(target: str, callback_url: str='https://my-callback-url.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Run a scan against a target URL and receive a scan ID to obtain scan results or check scan status at a later time.
		
		Provide an optional callback URL to receive scan results automatically once the scan is completed (ensure that the callback URL accepts POST requests)"
    
    """
    url = f"https://weblasso.p.rapidapi.com/scan-id"
    querystring = {'target': target, }
    if callback_url:
        querystring['callback_url'] = callback_url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weblasso.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_scan_results(scan_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain the scan results from a given scan ID."
    
    """
    url = f"https://weblasso.p.rapidapi.com/scan"
    querystring = {'scan_id': scan_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weblasso.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

