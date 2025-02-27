import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getuser(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets and returns the current user data from the associated userId."
    userid: The Id of the user
        
    """
    url = f"https://endpoint-monitor-tool1.p.rapidapi.com/GetUser"
    querystring = {'userId': userid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "endpoint-monitor-tool1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def removeendpointdata(endpointid: str=None, userid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Remove an endpoint from you user account by Id.
		
		(KNOWN ISSUE with RAPIDAPI)
		Successfully deleting an enpoint will result with a 200, however it MAY provide an ERROR stating "src property must be a valid json object". This is an issue with RAPIDAPI and not our API."
    
    """
    url = f"https://endpoint-monitor-tool1.p.rapidapi.com/RemoveEndpointData"
    querystring = {}
    if endpointid:
        querystring['endpointId'] = endpointid
    if userid:
        querystring['userId'] = userid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "endpoint-monitor-tool1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getendpointdata(userid: str, endpointid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Endpoint retrieves the endpoint data from the specified UserId and EndpointId."
    userid: The Id of the user
        endpointid: The Id of the endpoint requested
        
    """
    url = f"https://endpoint-monitor-tool1.p.rapidapi.com/GetEndpointData"
    querystring = {'UserId': userid, 'EndpointId': endpointid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "endpoint-monitor-tool1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

