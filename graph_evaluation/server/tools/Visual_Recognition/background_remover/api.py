import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getaccount(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a description of your user account information.
		Useful to get the remaining credits on your account
		"
    
    """
    url = f"https://background-remover5.p.rapidapi.com/account"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "background-remover5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getimage(imageid: str, recompute: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retuns the result of an image that was previously handled by autoremove route for the given image Id.
		accepts two parameters:
		  - imageId: The Id of the image to get.
		  - recompute: bollean, if set to true, asks the server to remove the backgorund again for the specified ID (costs credits).
		"
    imageid: ID of the image to get
        recompute: ask the server to remove the background again (cost credits)
        
    """
    url = f"https://background-remover5.p.rapidapi.com/image"
    querystring = {'imageId': imageid, }
    if recompute:
        querystring['recompute'] = recompute
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "background-remover5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

