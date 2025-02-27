import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def first_generation(monstertype: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gathers the monsters from Monster Hunter, Monster Hunter G, and Monster Hunter Freedom. 
		
		Available endpoints are "smallmonsters" and "largemonsters". 
		
		Leaving the monster type blank will get all monsters from this generation."
    monstertype: User \"smallmonsters\" or \"largemonsters\" to get those specific monsters. Leaving this field blank will gather all monsters in this generation. 
You can also search for one monster by doing: \"mh_first/search/{yourTargetMonster}
        
    """
    url = f"https://monster-hunter-basic-information.p.rapidapi.com/mh_first/{monstertype}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monster-hunter-basic-information.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows the gathering of one specific monster from the franchise. Every other endpoint. Search within one generation is also possible through their specific endpoints. Searching for a monster through a specific generation endpoint is likely to yield faster results."
    
    """
    url = f"https://monster-hunter-basic-information.p.rapidapi.com/search"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monster-hunter-basic-information.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def first_generation_search(name: str='velociprey', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a monster from Monster Hunter, Monster Hunter G, and Monster Hunter Freedom."
    
    """
    url = f"https://monster-hunter-basic-information.p.rapidapi.com/mh_first/search"
    querystring = {}
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monster-hunter-basic-information.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fourth_generation_search(name: str='konchu', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a monster from Monster Hunter 4 Ultimate and Monster Hunter Generations Ultimate."
    
    """
    url = f"https://monster-hunter-basic-information.p.rapidapi.com/mh_fourth/search"
    querystring = {}
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monster-hunter-basic-information.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def second_generation_search(name: str='popo', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a monster from Monster Hunter 2, Monster Hunter Freedom 2, and Monster Hunter Freedom Unite."
    
    """
    url = f"https://monster-hunter-basic-information.p.rapidapi.com/mh_second/search"
    querystring = {}
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monster-hunter-basic-information.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fourth_generation(monstertype: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gathers the monsters from Monster Hunter 4, Monster Hunter 4 Ultimate, Monster Hunter Generations, and Monster Hunter Generations Ultimate. 
		
		Available endpoints are "smallmonsters" and "largemonsters". 
		
		Leaving the monster type blank will get all monsters from this generation."
    monstertype: User \"smallmonsters\" or \"largemonsters\" to get those specific monsters. Leaving this field blank will gather all monsters in this generation. 
You can also search for one monster by doing: \"mh_first/search/{yourTargetMonster}
        
    """
    url = f"https://monster-hunter-basic-information.p.rapidapi.com/mh_fourth/{monstertype}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monster-hunter-basic-information.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def third_generation_search(name: str='jaggi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a monster from Monster Hunter 3, Monster Hunter Portable 3rd, and Monster Hunter 3 Ultimate."
    
    """
    url = f"https://monster-hunter-basic-information.p.rapidapi.com/mh_third/search"
    querystring = {}
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monster-hunter-basic-information.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fifth_generation_search(name: str='jagras', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to search for a specific monster that appeared in Monster Hunter Rise: Sunbreak or Monster Hunter World: Iceborne"
    
    """
    url = f"https://monster-hunter-basic-information.p.rapidapi.com/mh_fifth/search"
    querystring = {}
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monster-hunter-basic-information.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fifth_generation(monstertype: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gathers the monsters from Monster Hunter World: Iceborne and Monster Hunter Rise: Sunbreak.
		
		Available endpoints are "smallmonsters" and "largemonsters". 
		
		Leaving the monster type blank will get all monsters from this generation."
    monstertype: User \\\\\\\"smallmonsters\\\\\\\" or \\\\\\\"largemonsters\\\\\\\" to get those specific monsters. Leaving this field blank will gather all monsters in this generation. 
You can also search for one monster by doing: \\\\\\\"mh_first/search/{yourTargetMonster}
        
    """
    url = f"https://monster-hunter-basic-information.p.rapidapi.com/mh_fifth/{monstertype}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monster-hunter-basic-information.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def third_generation(monstertype: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gathers the monsters from Monster Hunter 3, Monster Hunter Portable 3rd, and Monster Hunter 3 Ultimate. 
		
		Available endpoints are "smallmonsters" and "largemonsters". 
		
		Leaving the monster type blank will get all monsters from this generation."
    monstertype: User \"smallmonsters\" or \"largemonsters\" to get those specific monsters. Leaving this field blank will gather all monsters in this generation. 
You can also search for one monster by doing: \"mh_first/search/{yourTargetMonster}
        
    """
    url = f"https://monster-hunter-basic-information.p.rapidapi.com/mh_third/{monstertype}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monster-hunter-basic-information.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def second_generation(monstertype: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gathers the monsters from Monster Hunter 2, Monster Hunter Freedom 2, and Monster Hunter Freedom Unite. 
		
		Available endpoints are "smallmonsters" and "largemonsters". 
		
		Leaving the monster type blank will get all monsters from this generation."
    monstertype: User \"smallmonsters\" or \"largemonsters\" to get those specific monsters. Leaving this field blank will gather all monsters in this generation. 
You can also search for one monster by doing: \"mh_first/search/{yourTargetMonster}
        
    """
    url = f"https://monster-hunter-basic-information.p.rapidapi.com/mh_second/{monstertype}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monster-hunter-basic-information.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

