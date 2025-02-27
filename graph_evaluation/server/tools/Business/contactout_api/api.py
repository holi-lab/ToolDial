import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def contactinformationbymemberid(member_id: int=750056232, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://contactout-api.p.rapidapi.com/v1/people/linkedin_member_id"
    querystring = {}
    if member_id:
        querystring['member_id'] = member_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "contactout-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def checkforpersonalemail(profile: str='https://www.linkedin.com/in/joualbert', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://contactout-api.p.rapidapi.com/v1/people/linkedin/personal_email_status"
    querystring = {}
    if profile:
        querystring['profile'] = profile
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "contactout-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlinkedinprofileurlforemail(email: str='terry@gmail.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://contactout-api.p.rapidapi.com/v1/people/person"
    querystring = {}
    if email:
        querystring['email'] = email
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "contactout-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bulkquery(bulk_request_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://contactout-api.p.rapidapi.com/v2/people/linkedin/batch/{bulk_request_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "contactout-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def checkforworkemail(profile: str='https://www.linkedin.com/in/joualbert', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://contactout-api.p.rapidapi.com/v1/people/linkedin/work_email_status"
    querystring = {}
    if profile:
        querystring['profile'] = profile
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "contactout-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def checkifphoneexists(profile: str='https://www.linkedin.com/in/joualbert', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://contactout-api.p.rapidapi.com/v1/people/linkedin/phone_status"
    querystring = {}
    if profile:
        querystring['profile'] = profile
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "contactout-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getusagestats(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://contactout-api.p.rapidapi.com/v1/stats"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "contactout-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlinkedinprofiledataforemail(email: str='sample@hotmail.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://contactout-api.p.rapidapi.com/v1/email/enrich"
    querystring = {}
    if email:
        querystring['email'] = email
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "contactout-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def linkedinurltocontactinformation(profile: str='https://www.linkedin.com/in/joualbert', include_phone: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "##### [GET [https://api.contactout.com/v1/people/linkedin{?profile=}]](https://api.contactout.com/v1/people/linkedin{?profile=}]ParametersprofileStringURL)
		
		- Parameters
		    - `profile`
		        - String
		        - Required
		        - Fully formed URL of the linkedin profile. URL must begin with `http` and must contain `linkedin.com/in/` or `linkedin.com/pub/`
		    - `include_phone`
		        - Boolean
		        - Optional
		        - Will return phone data as well if `true`"
    
    """
    url = f"https://contactout-api.p.rapidapi.com/v1/people/linkedin"
    querystring = {}
    if profile:
        querystring['profile'] = profile
    if include_phone:
        querystring['include_phone'] = include_phone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "contactout-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

