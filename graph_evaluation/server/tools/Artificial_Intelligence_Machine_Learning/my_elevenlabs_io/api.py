import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_voice_v1_voices_voice_id_get(voice_id: str, with_settings: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns metadata about a specific voice."
    voice_id: Voice ID to be used, you can use https://api.elevenlabs.io/v1/voices to list all the available voices.
        with_settings: If set will return settings information corresponding to the voice, requires authorization.
        
    """
    url = f"https://my-elevenlabs-io.p.rapidapi.com/v1/voices/{voice_id}"
    querystring = {}
    if with_settings:
        querystring['with_settings'] = with_settings
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "my-elevenlabs-io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_generated_items_v1_history_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns metadata about all your generated audio."
    
    """
    url = f"https://my-elevenlabs-io.p.rapidapi.com/v1/history"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "my-elevenlabs-io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_audio_from_history_item_v1_history_history_item_id_audio_get(history_item_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the audio of an history item."
    history_item_id: History item ID to be used, you can use GET https://api.elevenlabs.io/v1/history to receive a list of history items and their IDs.
        
    """
    url = f"https://my-elevenlabs-io.p.rapidapi.com/v1/history/{history_item_id}/audio"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "my-elevenlabs-io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_subscription_info_v1_user_subscription_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets extended information about the users subscription"
    
    """
    url = f"https://my-elevenlabs-io.p.rapidapi.com/v1/user/subscription"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "my-elevenlabs-io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_info_v1_user_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets information about the user"
    
    """
    url = f"https://my-elevenlabs-io.p.rapidapi.com/v1/user"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "my-elevenlabs-io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_voices_v1_voices_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a list of all available voices for a user."
    
    """
    url = f"https://my-elevenlabs-io.p.rapidapi.com/v1/voices"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "my-elevenlabs-io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_default_voice_settings_v1_voices_settings_default_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the default settings for voices."
    
    """
    url = f"https://my-elevenlabs-io.p.rapidapi.com/v1/voices/settings/default"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "my-elevenlabs-io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_voice_settings_v1_voices_voice_id_settings_get(voice_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the settings for a specific voice."
    voice_id: Voice ID to be used, you can use https://api.elevenlabs.io/v1/voices to list all the available voices.
        
    """
    url = f"https://my-elevenlabs-io.p.rapidapi.com/v1/voices/{voice_id}/settings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "my-elevenlabs-io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_audio_from_sample_v1_voices_voice_id_samples_sample_id_audio_get(sample_id: str, voice_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the audio corresponding to a sample attached to a voice."
    sample_id: Sample ID to be used, you can use GET https://api.elevenlabs.io/v1/voices/{voice_id} to list all the available samples for a voice.
        voice_id: Voice ID to be used, you can use https://api.elevenlabs.io/v1/voices to list all the available voices.
        
    """
    url = f"https://my-elevenlabs-io.p.rapidapi.com/v1/voices/{voice_id}/samples/{sample_id}/audio"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "my-elevenlabs-io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

