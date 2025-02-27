import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def register_new_user(mobile: str, hash: str, uid: int, email: str, vtype: int=3, step: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To initiate the user verification process, please provide us with the relevant user information. Kindly ensure that the email is URL encoded before sending it via GET. For mobile numbers, please include the country code without using either + or 00.
		
		To obtain the hash for your company, please contact us.
		
		There are two verification types available:
		
		Type 3: This type verifies email, mobile, and ID card information.
		Type 4: This type verifies email, mobile, ID card, and landline information via phone call.
		You have the option to skip certain verification steps for specific users. For instance:
		
		1: Skipping email verification
		2: Skipping email and mobile verifications
		3: Skipping email, mobile, and ID card verifications"
    
    """
    url = f"https://kyc-verification1.p.rapidapi.com/registeruser/{uid}/{email}/{mobile}/{hash}/{vtype}/{step}"
    querystring = {}
    if vtype:
        querystring['vtype'] = vtype
    if step:
        querystring['step'] = step
    if vtype:
        querystring['vtype'] = vtype
    if step:
        querystring['step'] = step
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kyc-verification1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check_user_status(is_id: int, hash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "During this step, you can verify the status of your users by sending their user ID and your company hash to our API. The API will process the request and provide the result of the verification as well as information about the steps that the user has completed."
    
    """
    url = f"https://kyc-verification1.p.rapidapi.com/checkstate/{is_id}/{hash}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kyc-verification1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

