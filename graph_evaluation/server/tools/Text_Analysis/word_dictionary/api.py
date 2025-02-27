import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def theme(entry: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the themes of a word."
    entry: Type a word to get its themes
        
    """
    url = f"https://twinword-word-graph-dictionary.p.rapidapi.com/theme/"
    querystring = {'entry': entry, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-word-graph-dictionary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def difficulty(entry: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the difficulty level of a word."
    entry: Type a word to get its difficulty level
        
    """
    url = f"https://twinword-word-graph-dictionary.p.rapidapi.com/difficulty/"
    querystring = {'entry': entry, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-word-graph-dictionary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def example(entry: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "See examples of a word used in a sentence"
    entry: Type a word to see examples of it used in a sentence
        
    """
    url = f"https://twinword-word-graph-dictionary.p.rapidapi.com/example/"
    querystring = {'entry': entry, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-word-graph-dictionary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reference(entry: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the broad terms, narrow terms, related terms, evocations, synonyms, associations, and derived terms of a word."
    entry: Type a word to get its broad terms, narrow terms, related terms, evocations, synonyms, associations, and derived terms
        
    """
    url = f"https://twinword-word-graph-dictionary.p.rapidapi.com/reference/"
    querystring = {'entry': entry, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-word-graph-dictionary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def association(entry: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the associations of a word."
    entry: Type a word to get its associations
        
    """
    url = f"https://twinword-word-graph-dictionary.p.rapidapi.com/association/"
    querystring = {'entry': entry, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-word-graph-dictionary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exam_history(entry: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "See which exams a word has been on"
    entry: Type a word to see which exams it has been on
        
    """
    url = f"https://twinword-word-graph-dictionary.p.rapidapi.com/examhistory/"
    querystring = {'entry': entry, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-word-graph-dictionary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def definition(entry: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the definitions of a word."
    entry: a word
        
    """
    url = f"https://twinword-word-graph-dictionary.p.rapidapi.com/definition/"
    querystring = {'entry': entry, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-word-graph-dictionary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

