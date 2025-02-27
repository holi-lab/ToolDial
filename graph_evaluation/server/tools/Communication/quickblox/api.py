import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_places_json(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieving a list of the created places by application"
    
    """
    url = f"https://community-quickblox.p.rapidapi.com/places.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-quickblox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_users_json(filter: str=None, per_page: str=None, page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Call this API to get a list of current users of you app. By default it returns upto 10 users, but you can change this by adding pagination parameters."
    filter: You can filter the list of users by supplying a filter string. For example, the following filter limits the results to users who's login is either dgem or webdev: string login in dgem, webdev  For more information on filtering, see the filtering documentation
        per_page: The maximum number of users to return per page, if not specified then the default is 10
        page: Used to paginate the results when more than one page of users retrieved
        
    """
    url = f"https://community-quickblox.p.rapidapi.com/users.json"
    querystring = {}
    if filter:
        querystring['filter[]'] = filter
    if per_page:
        querystring['per_page'] = per_page
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-quickblox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_geodata_find_json(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all (by default) geodata for current application. The ID of the application is taken from the token which is specified in the request"
    
    """
    url = f"https://community-quickblox.p.rapidapi.com/geodata/find.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-quickblox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_blobs_uid_xml(uid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download File (Get File as a redirect to the S3 object) by uid. 'uid' is a parameter which should be taken from the response of the request for the creating a file. To have a possibility to download a file you should set a status complete to your file firstly."
    
    """
    url = f"https://community-quickblox.p.rapidapi.com/blobs/{uid}.xml"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-quickblox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_blobs_json(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of files for the current user. The ID of the user is taken from the token specified in the request"
    
    """
    url = f"https://community-quickblox.p.rapidapi.com/blobs.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-quickblox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

