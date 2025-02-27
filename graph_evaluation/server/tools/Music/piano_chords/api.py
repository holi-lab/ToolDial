import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_for_a_chord_by_entering_notes_or_midi_keys_v2(n1: str, n2: str, n3: str='68', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The response is an object with a 'result' property, holding an array of chord object(s), or an empty array when no match was found. Only exact matches are returned."
    
    """
    url = f"https://piano-chords.p.rapidapi.com/v2/chords/"
    querystring = {'n1': n1, 'n2': n2, }
    if n3:
        querystring['n3'] = n3
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "piano-chords.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chord_defined_by_root_and_name_v2(root: str, name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The response is an object with a 'result' property, holding an array with one chord object."
    
    """
    url = f"https://piano-chords.p.rapidapi.com/v2/chords/{root}/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "piano-chords.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_chords_for_a_certain_name_v2(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The response is an object with a 'result' property, holding an array of all chords with the specified name."
    
    """
    url = f"https://piano-chords.p.rapidapi.com/v2/chords/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "piano-chords.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_chords_for_a_certain_name(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v1: Returns an object with all chords for a certain name
		v2: Returns an object with an array of all chords for a certain name"
    
    """
    url = f"https://piano-chords.p.rapidapi.com/chords/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "piano-chords.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_chords_for_a_certain_root(root: str='csharp', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v1: Returns an object with all chords for a certain root
		v2: Returns an object with an array of all chords for a certain root"
    
    """
    url = f"https://piano-chords.p.rapidapi.com/chords/{root}"
    querystring = {}
    if root:
        querystring['root'] = root
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "piano-chords.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_chords(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the full list of available chords"
    
    """
    url = f"https://piano-chords.p.rapidapi.com/chords"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "piano-chords.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chord_defined_by_root_and_name(root: str, name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v1: The response is a chord object with name, an array of notes, an array of intervals and an array of midi keys
		v2: The response's result property is an array holding the specified chord object"
    
    """
    url = f"https://piano-chords.p.rapidapi.com/chords/{root}/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "piano-chords.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_a_chord_by_entering_notes_or_midi_keys(n2: str, n1: str, n3: str='68', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an array of chord objects (when a match is found)"
    
    """
    url = f"https://piano-chords.p.rapidapi.com/chords"
    querystring = {'n2': n2, 'n1': n1, }
    if n3:
        querystring['n3'] = n3
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "piano-chords.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_chords_for_a_certain_root_v2(root: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The response is an object, with a 'result' property, holding an array of all chord objects for the specified root note."
    
    """
    url = f"https://piano-chords.p.rapidapi.com/v2/chords/{root}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "piano-chords.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_chords_v2(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The response is an object with a 'result' property, holding an array of all available chord objects."
    
    """
    url = f"https://piano-chords.p.rapidapi.com/v2/chords"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "piano-chords.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

