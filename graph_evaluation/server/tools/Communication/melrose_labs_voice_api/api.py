import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def retrieve_text(transactionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve text from previously converted speech-to-text."
    transactionid: ID of speech-to-text transaction
        
    """
    url = f"https://melrose-labs-voice-api1.p.rapidapi.com/speechtotext/{transactionid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "melrose-labs-voice-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_text(transactionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve text from previously converted speech-to-text."
    transactionid: ID of speech-to-text transaction
        
    """
    url = f"https://melrose-labs-voice-api1.p.rapidapi.com/speechtotext/{transactionid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "melrose-labs-voice-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_text(transactionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve text from previously converted speech-to-text."
    transactionid: ID of speech-to-text transaction
        
    """
    url = f"https://melrose-labs-voice-api1.p.rapidapi.com/speechtotext/{transactionid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "melrose-labs-voice-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_available_numbers_for_a_country(country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of available telephone numbers for a country"
    country: Country for which querying for available numbers
        
    """
    url = f"https://melrose-labs-voice-api1.p.rapidapi.com/numbering/available/{country}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "melrose-labs-voice-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_available_numbers_for_a_country_prefix(prefix: str, country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of available telephone numbers for a country and given prefix"
    prefix: Prefix within country for which query for available numbers
        country: Country for which querying for available numbers
        
    """
    url = f"https://melrose-labs-voice-api1.p.rapidapi.com/numbering/available/{country}/{prefix}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "melrose-labs-voice-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_allocated_numbers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of allocated numbers"
    
    """
    url = f"https://melrose-labs-voice-api1.p.rapidapi.com/numbering"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "melrose-labs-voice-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_audio_file(transactionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve audio file from previously converted text-to-speech. File is in MP3 format."
    transactionid: Transaction ID for previously submitted text-to-speech conversion.
        
    """
    url = f"https://melrose-labs-voice-api1.p.rapidapi.com/texttospeech/{transactionid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "melrose-labs-voice-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_audio_file(transactionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve audio file from previously converted text-to-speech. File is in MP3 format."
    transactionid: Transaction ID for previously submitted text-to-speech conversion.
        
    """
    url = f"https://melrose-labs-voice-api1.p.rapidapi.com/texttospeech/{transactionid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "melrose-labs-voice-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_audio_file(transactionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve audio file from previously converted text-to-speech. File is in MP3 format."
    transactionid: Transaction ID for previously submitted text-to-speech conversion.
        
    """
    url = f"https://melrose-labs-voice-api1.p.rapidapi.com/texttospeech/{transactionid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "melrose-labs-voice-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_endpoint(telno: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get endpoint for a number"
    telno: Get current endpoint for telephone number
        
    """
    url = f"https://melrose-labs-voice-api1.p.rapidapi.com/numbering/{telno}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "melrose-labs-voice-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

