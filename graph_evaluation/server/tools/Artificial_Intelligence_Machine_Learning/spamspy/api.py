import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def remember(content: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Functions as a free & unlimited version of the classic reactive spam filtering approach.  Recommended to be used to cut down on "Comprehend" endpoint call usage if the content has a match, there is no need to have SpamSpy recalculate responses it has already reviewed. 
		This endpoint checks the existing spam database for queried content and sends back the latest scanned result if the content exists yet.
		
		Example Responses:
		`{"SpamSpyResult":true}` (content is spam)
		`{"SpamSpyResult":false}`(content is not spam)
		`{"SpamSpyResult":"Content not yet scanned"}` (content has **NOT YET** been scanned)"
    
    """
    url = f"https://spamspy.p.rapidapi.com/Remember"
    querystring = {'content': content, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spamspy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auth_check(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Checks if the API Key is still valid"
    
    """
    url = f"https://spamspy.p.rapidapi.com/Auth"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spamspy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comprehend(content: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Comprehend uses the current AI public modals and submits the content (located in the query string) to be identified as spam or not.
		- Fetch Method Get
		- Returns Boolean: **true** / **false**
		Example Response:
		`{"SpamSpyResult":true}` (content is spam)
		`{"SpamSpyResult":false}`(content is not spam)"
    
    """
    url = f"https://spamspy.p.rapidapi.com/Comprehend"
    querystring = {'content': content, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spamspy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

