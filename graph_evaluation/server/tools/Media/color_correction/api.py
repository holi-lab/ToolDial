import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def check_status_and_error(secret: str, identifier: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This returns the job parameters including possible errors. Use this call if your retrieve GET returned 404 to learn more information about the reason."
    secret: This is the secret password that you chose when submitting the image for color correction.
        identifier: This is the numeric identifier assigned to your job when received by the server. It was provided to you in the response to your POST request.
        
    """
    url = f"https://whitebalance.p.rapidapi.com/v1/image/check"
    querystring = {'secret': secret, 'identifier': identifier, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whitebalance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_finished_image(secret: str, identifier: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves an image which was successfully processed. It returns the image with 200 success code, or a 404 error if the image is not ready."
    secret: This is the secret that you submitted your image for color correction.
        identifier: This is the numeric identifier that we assigned to the job upon receiving it. It was returned to you in the response to you API call.
        
    """
    url = f"https://whitebalance.p.rapidapi.com/v1/image/retrieve"
    querystring = {'secret': secret, 'identifier': identifier, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whitebalance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

