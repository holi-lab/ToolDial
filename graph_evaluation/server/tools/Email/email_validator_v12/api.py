import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def email_validate(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The `/email/validate` endpoint is a REST API endpoint that accepts a GET request and expects a query parameter `email` to be passed in the URL. The API endpoint validates the email address using the `email_validator` library and returns a JSON response indicating whether the email address is valid or not.
		
		If the email is valid, the response will contain a JSON object with a key valid set to true. If the email is not valid, the response will contain a JSON object with a key valid set to false, and an additional key error will be present that contains a string describing the validation error.
		
		This endpoint can be used by client applications to validate email addresses before processing or storing them."
    
    """
    url = f"https://email-validator45.p.rapidapi.com/email/validate?email="
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-validator45.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

