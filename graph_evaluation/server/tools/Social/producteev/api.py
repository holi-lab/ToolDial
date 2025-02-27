import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_a_producteev_file_by_its_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Files are usually attached to Notes. But because they remain an independant object type in the system, you need to upload the file first before being able to attach it to a note."
    
    """
    url = f"https://community-producteev.p.rapidapi.com/files/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-producteev.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_suggested_colors_for_labels(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the list of recommended colors (foreground and background) for the labels. You can create your labels with your own colors, but we strongly recommend to use colors returned by this API call to provide a consistent experience accross all Producteev apps."
    
    """
    url = f"https://community-producteev.p.rapidapi.com/label_colors"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-producteev.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_list_of_pending_invitation_for_the_current_user(type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Network Invitations are used to add people in existing networks. To invite someone you can either use the targetUser attribute if you know that this person is on Producteev and know his id. Otherwise use email and we'll send an invite to this person via email."
    type: requests | authorizations
        
    """
    url = f"https://community-producteev.p.rapidapi.com/network_invitations"
    querystring = {'type': type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-producteev.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_a_list_of_suggestions_for_the_natural_language_processing(query: str, project_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The call will return a collection of NLP suggestion object."
    query: This method provide suggestions based on Keyword + Query. The keywords available are : # : for labels @ : for followers + : for assignees ! : for deadline * : for priority The query parameter must contains both the keyword and the text you wanna get suggestions for. e.g ?query=@Jerome
        
    """
    url = f"https://community-producteev.p.rapidapi.com/nlp_suggestions/{project_id}"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-producteev.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_label_with_the_specified_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Labels are attached to a task and are available across all your projects inside a network. This method returns a Label Object."
    
    """
    url = f"https://community-producteev.p.rapidapi.com/labels/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-producteev.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

