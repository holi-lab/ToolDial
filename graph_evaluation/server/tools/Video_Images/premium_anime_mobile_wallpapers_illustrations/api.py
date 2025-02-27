import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def random_wallpapers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a random anime wallpaper
		
		Optional GET parameter
		**quality**:0
		
		This API provides two types of wallpaper images
		1. Compress lower size wallpaper.
		2.  Best quality but will result in high size.
		
		if the **quality**  parameter, is not provided then by default it will provide the URL of compressed wallpaper low size if the quality parameter is set to 1 then it will provide a URL of the highest quality wallpaper ( high size ).
		
		**RESPONSE**
		**animename**  ( Depend on subscription )
		if the wallpaper is related to any anime, or game then the Japanese name of the anime will be sent otherwise illustration will be sent.
		
		**arturl**
		URL of wallpaper
		
		**sensitivity** ( Depend on subscription )
		Is wallpaper sensitive, 1 means yes 0 means no
		
		**art_id** ( Depend on subscription )
		Unique wallpaper  id which can be later used to get more info
		
		**premium** (Depend on subscription)
		Whether art is of premium category 1 means yes 0 means no.
		
		**Note: if no wallpaper found response will be**
		`{"status": "empty"}`"
    
    """
    url = f"https://premium-anime-mobile-wallpapers-illustrations.p.rapidapi.com/rapidHandler/random"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premium-anime-mobile-wallpapers-illustrations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def anime_wallpapers_by_pagination(sensitivity: int, page: int, quality: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get anime wallpapers:
		Required GET parameter
		**page**:  1
		**sensitivity**: 1
		On the above **page**, 1 means it will fetch the latest 10 wallpaper if the page value is 2 then It will bring the next 10 latest wallpaper 
		page 1 : 0-10 
		page 2: 10-20 ......
		
		On the above **sensitivity** , 1 means it will not fetch any 18+ or ecchi-related images if the value is 0 then it will bring all without any filter.
		
		Optional GET parameter
		**quality**:0
		
		This API provides two types of wallpaper images
		1. Compress lower size wallpaper.
		2.  Best quality but will result in high size.
		
		if the **quality**  parameter, is not provided then by default it will provide the URL of compressed wallpaper low size if the quality parameter is set to 1 then it will provide a URL of the highest quality wallpaper ( high size ).
		
		**RESPONSE**
		**animename**  ( Depend on subscription )
		if the wallpaper is related to any anime, or game then the Japanese name of the anime will be sent otherwise illustration will be sent.
		
		**arturl**
		URL of wallpaper
		
		**sensitivity** ( Depend on subscription )
		Is wallpaper sensitive, 1 means yes 0 means no
		
		**art_id** ( Depend on subscription )
		Unique wallpaper  id which can be later used to get more info
		
		**premium** (Depend on subscription)
		Whether art is of premium category 1 means yes 0 means no.
		
		**Note: if no wallpaper found response will be**
		`{"status": "empty"}`"
    
    """
    url = f"https://premium-anime-mobile-wallpapers-illustrations.p.rapidapi.com/rapidHandler/anime"
    querystring = {'sensitivity': sensitivity, 'page': page, }
    if quality:
        querystring['quality'] = quality
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premium-anime-mobile-wallpapers-illustrations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def anime_wallpapers_related_to_the_valentine_by_pagination(page: int, sensitivity: int, quality: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get anime wallpapers related to the valentine:
		Required GET parameter
		**page**:  1
		**sensitivity**: 1
		On the above **page**, 1 means it will fetch the latest 10 wallpaper if the page value is 2 then It will bring the next 10 latest wallpaper 
		page 1 : 0-10 
		page 2: 10-20 ......
		
		On the above **sensitivity**, 1 means it will not fetch any 18+ or ecchi-related images if the value is 0 then it will bring all without any filter.
		
		Optional GET parameter
		**quality**:0
		
		This API provides two types of wallpaper images
		1. Compress lower size wallpaper.
		2.  Best quality but will result in high size.
		
		if the **quality**  parameter, is not provided then by default it will provide the URL of compressed wallpaper low size if the quality parameter is set to 1 then it will provide a URL of the highest quality wallpaper ( high size ).
		
		**RESPONSE**
		**animename**  ( Depend on subscription )
		if the wallpaper is related to any anime, or game then the Japanese name of the anime will be sent otherwise illustration will be sent.
		
		**arturl**
		URL of wallpaper
		
		**sensitivity** ( Depend on subscription )
		Is wallpaper sensitive, 1 means yes 0 means no
		
		**art_id** ( Depend on subscription )
		Unique wallpaper  id which can be later used to get more info
		
		**premium** (Depend on subscription)
		Whether art is of premium category 1 means yes 0 means no.
		
		**Note: if no wallpaper found response will be**
		`{"status": "empty"}`"
    
    """
    url = f"https://premium-anime-mobile-wallpapers-illustrations.p.rapidapi.com/rapidHandler/valentine"
    querystring = {'page': page, 'sensitivity': sensitivity, }
    if quality:
        querystring['quality'] = quality
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premium-anime-mobile-wallpapers-illustrations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def anime_wallpapers_related_to_the_wedding_by_pagination(sensitivity: int, page: int, quality: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get anime wallpapers related to the wedding:
		Required GET parameter
		**page**:  1
		**sensitivity**: 1
		On the above **page**, 1 means it will fetch the latest 10 wallpaper if the page value is 2 then It will bring the next 10 latest wallpaper 
		page 1 : 0-10 
		page 2: 10-20 ......
		
		On the above **sensitivity**, 1 means it will not fetch any 18+ or ecchi-related images if the value is 0 then it will bring all without any filter.
		
		Optional GET parameter
		**quality**:0
		
		This API provides two types of wallpaper images
		1. Compress lower size wallpaper.
		2.  Best quality but will result in high size.
		
		if the **quality**  parameter, is not provided then by default it will provide the URL of compressed wallpaper low size if the quality parameter is set to 1 then it will provide a URL of the highest quality wallpaper ( high size ).
		
		**RESPONSE**
		**animename**  ( Depend on subscription )
		if the wallpaper is related to any anime, or game then the Japanese name of the anime will be sent otherwise illustration will be sent.
		
		**arturl**
		URL of wallpaper
		
		**sensitivity** ( Depend on subscription )
		Is wallpaper sensitive, 1 means yes 0 means no
		
		**art_id** ( Depend on subscription )
		Unique wallpaper  id which can be later used to get more info
		
		**premium** (Depend on subscription)
		Whether art is of premium category 1 means yes 0 means no.
		
		**Note: if no wallpaper found response will be**
		`{"status": "empty"}`"
    
    """
    url = f"https://premium-anime-mobile-wallpapers-illustrations.p.rapidapi.com/rapidHandler/wedding"
    querystring = {'sensitivity': sensitivity, 'page': page, }
    if quality:
        querystring['quality'] = quality
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premium-anime-mobile-wallpapers-illustrations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def anime_wallpapers_related_to_the_couple_by_pagination(sensitivity: int, page: int, quality: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get anime wallpapers related to the couple:
		Required GET parameter
		**page**:  1
		**sensitivity**: 1
		On the above **page**, 1 means it will fetch the latest 10 wallpaper if the page value is 2 then It will bring the next 10 latest wallpaper 
		page 1 : 0-10 
		page 2: 10-20 ......
		
		On the above **sensitivity**, 1 means it will not fetch any 18+ or ecchi-related images if the value is 0 then it will bring all without any filter.
		
		Optional GET parameter
		**quality**:0
		
		This API provides two types of wallpaper images
		1. Compress lower size wallpaper.
		2.  Best quality but will result in high size.
		
		if the **quality**  parameter, is not provided then by default it will provide the URL of compressed wallpaper low size if the quality parameter is set to 1 then it will provide a URL of the highest quality wallpaper ( high size ).
		
		**RESPONSE**
		**animename**  ( Depend on subscription )
		if the wallpaper is related to any anime, or game then the Japanese name of the anime will be sent otherwise illustration will be sent.
		
		**arturl**
		URL of wallpaper
		
		**sensitivity** ( Depend on subscription )
		Is wallpaper sensitive, 1 means yes 0 means no
		
		**art_id** ( Depend on subscription )
		Unique wallpaper  id which can be later used to get more info
		
		**premium** (Depend on subscription)
		Whether art is of premium category 1 means yes 0 means no.
		
		**Note: if no wallpaper found response will be**
		`{"status": "empty"}`"
    
    """
    url = f"https://premium-anime-mobile-wallpapers-illustrations.p.rapidapi.com/rapidHandler/couple"
    querystring = {'sensitivity': sensitivity, 'page': page, }
    if quality:
        querystring['quality'] = quality
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premium-anime-mobile-wallpapers-illustrations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def anime_wallpapers_related_to_the_christmas_with_pagination(page: int, sensitivity: int, quality: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get  anime wallpapers related to the christmas:
		Required GET parameter
		**page**:  1
		**sensitivity**: 1
		On the above **page**, 1 means it will fetch the latest 10 wallpaper if the page value is 2 then It will bring the next 10 latest wallpaper 
		page 1 : 0-10 
		page 2: 10-20 ......
		
		On the above **sensitivity**, 1 means it will not fetch any 18+ or ecchi-related images if the value is 0 then it will bring all without any filter.
		
		Optional GET parameter
		**quality**:0
		
		This API provides two types of wallpaper images
		1. Compress lower size wallpaper.
		2.  Best quality but will result in high size.
		
		if the **quality**  parameter, is not provided then by default it will provide the URL of compressed wallpaper low size if the quality parameter is set to 1 then it will provide a URL of the highest quality wallpaper ( high size ).
		
		**RESPONSE**
		**animename**  ( Depend on subscription )
		if the wallpaper is related to any anime, or game then the Japanese name of the anime will be sent otherwise illustration will be sent.
		
		**arturl**
		URL of wallpaper
		
		**sensitivity** ( Depend on subscription )
		Is wallpaper sensitive, 1 means yes 0 means no
		
		**art_id** ( Depend on subscription )
		Unique wallpaper  id which can be later used to get more info
		
		**premium** (Depend on subscription)
		Whether art is of premium category 1 means yes 0 means no.
		
		**Note: if no wallpaper found response will be**
		`{"status": "empty"}`"
    
    """
    url = f"https://premium-anime-mobile-wallpapers-illustrations.p.rapidapi.com/rapidHandler/christmas"
    querystring = {'page': page, 'sensitivity': sensitivity, }
    if quality:
        querystring['quality'] = quality
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premium-anime-mobile-wallpapers-illustrations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def anime_illustrations_by_pagination(page: int, sensitivity: int, quality: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get anime illustrations made by anime fans:
		Required GET parameter
		**page**:  1
		**sensitivity**: 1
		On the above **page**, 1 means it will fetch the latest 10 wallpaper if the page value is 2 then It will bring the next 10 latest wallpaper 
		page 1 : 0-10 
		page 2: 10-20 ......
		
		On the above **sensitivity**, 1 means it will not fetch any 18+ or ecchi-related images if the value is 0 then it will bring all without any filter.
		
		Optional GET parameter
		**quality**:0
		
		This API provides two types of wallpaper images
		1. Compress lower size wallpaper.
		2.  Best quality but will result in high size.
		
		if the **quality**  parameter, is not provided then by default it will provide the URL of compressed wallpaper low size if the quality parameter is set to 1 then it will provide a URL of the highest quality wallpaper ( high size ).
		
		**RESPONSE**
		**animename**  ( Depend on subscription )
		if the wallpaper is related to any anime, or game then the Japanese name of the anime will be sent otherwise illustration will be sent.
		
		**arturl**
		URL of wallpaper
		
		**sensitivity** ( Depend on subscription )
		Is wallpaper sensitive, 1 means yes 0 means no
		
		**art_id** ( Depend on subscription )
		Unique wallpaper  id which can be later used to get more info
		
		**premium** (Depend on subscription)
		Whether art is of premium category 1 means yes 0 means no.
		
		**Note: if no wallpaper found response will be**
		`{"status": "empty"}`"
    
    """
    url = f"https://premium-anime-mobile-wallpapers-illustrations.p.rapidapi.com/rapidHandler/illustration"
    querystring = {'page': page, 'sensitivity': sensitivity, }
    if quality:
        querystring['quality'] = quality
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premium-anime-mobile-wallpapers-illustrations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recent_wallpapers_by_pagination(page: int, sensitivity: int, quality: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get recent anime wallpapers without any categorization:
		Required GET parameter
		**page**:  1
		**sensitivity**: 1
		On the above **page**, 1 means it will fetch the latest 10 wallpaper if the page value is 2 then It will bring the next 10 latest wallpaper 
		page 1: 0-10 
		page 2: 10-20 ......
		
		On the above sensitivity, 1 means it will not fetch any 18+ or ecchi-related images if the value is 0 then it will bring all without any filter.
		
		Optional GET parameter
		**quality**:0
		
		This API provides two types of wallpaper images
		1. Compress lower size wallpaper.
		2.  Best quality but will result in high size.
		
		if the **quality**  parameter, is not provided then by default it will provide the URL of compressed wallpaper low size if the quality parameter is set to 1 then it will provide a URL of the highest quality wallpaper ( high size ).
		
		**RESPONSE**
		**animename**  ( Depend on subscription )
		if the wallpaper is related to any anime, or game then the Japanese name of the anime will be sent otherwise illustration will be sent.
		
		**arturl**
		URL of wallpaper
		
		**sensitivity** ( Depend on subscription )
		Is wallpaper sensitive, 1 means yes 0 means no
		
		**art_id** ( Depend on subscription )
		Unique wallpaper  id which can be later used to get more info
		
		**premium** (Depend on subscription)
		Whether art is of premium category 1 means yes 0 means no.
		
		**Note: if no wallpaper found response will be**
		`{"status": "empty"}`"
    
    """
    url = f"https://premium-anime-mobile-wallpapers-illustrations.p.rapidapi.com/rapidHandler/recent"
    querystring = {'page': page, 'sensitivity': sensitivity, }
    if quality:
        querystring['quality'] = quality
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premium-anime-mobile-wallpapers-illustrations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_animename_art_id(searchtext: str, sensitivity: int, page: int, quality: int=0, exact: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search by animename, art_id
		
		Required GET parameter
		
		**page**: 1
		**sensitivity**: 1
		**searchText**: 'Tokyo'
		**exact**: 0
		
		On the above **page**, 1 means it will fetch the latest 10 wallpaper if the page value is 2 then It will bring the next 10 latest wallpaper
		page 1 : 0-10
		page 2: 10-20 ……
		
		On the above **sensitivity**, 1 means it will not fetch any 18+ or ecchi-related images if the value is 0 then it will bring all without any filter.
		
		On the above **searchText**, means it will try to fetch all arts based on the above text (Japanese anime name)
		OR
		art_id
		
		***Note: searchText param should be min 3 chars length***
		
		Optional GET parameter
		**exact**: 
		
		If the exact parameter is not provided then by default it will be set to 0.
		0 represents fetch all the arts related to any searchText word of animename.
		
		**Note: Exact params value 0 assume that you are passing japanese animename it will search all the wallpapers related to it**
		( So **attack on t** will result in empty response since its english name but **Shingeki no** will not)
		
		1 represents fetch all the arts which have the same animename and searchText.
		
		**Note: Exact params value 1 support both english and japanese animename**
		( So **Attack on titan** which is english will result wallpaper of japanese animename **Shingeki no Kyojin**)
		
		2 represent fetch arts based on art_id ( so If you are sending art_id on searchText make sure you set the exact param to 2 )
		
		**quality**:0
		This API provides two types of wallpaper images
		
		Compress lower size wallpaper.
		Best quality but will result in high size.
		If the **quality** parameter, is not provided then by default it will provide the URL of compressed wallpaper low size if the quality parameter is set to 1 then it will provide a URL of the highest quality wallpaper ( high size ).
		
		RESPONSE
		**animename**( Depend on subscription )
		if the wallpaper is related to any anime, or game then the Japanese name of the anime will be sent otherwise illustration will be sent.
		
		**arturl**
		URL of wallpaper
		
		**sensitivity**( Depend on subscription )
		Is wallpaper sensitive, 1 means yes 0 means no
		
		**art_id**( Depend on subscription )
		Unique wallpaper id which can be later used to get more info
		
		**premium** (Depend on subscription)
		Whether art is of premium category 1 means yes 0 means no.
		
		Note: if no wallpaper found response will be
		
		*{"status": "empty"}*"
    
    """
    url = f"https://premium-anime-mobile-wallpapers-illustrations.p.rapidapi.com/rapidHandler/search"
    querystring = {'searchText': searchtext, 'sensitivity': sensitivity, 'page': page, }
    if quality:
        querystring['quality'] = quality
    if exact:
        querystring['exact'] = exact
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premium-anime-mobile-wallpapers-illustrations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def anime_boy_wallpapers_by_pagination(page: int, sensitivity: int, quality: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get anime boy wallpapers:
		Required GET parameter
		**page**:  1
		**sensitivity**: 1
		On the above **page**, 1 means it will fetch the latest 10 wallpaper if the page value is 2 then It will bring the next 10 latest wallpaper 
		page 1 : 0-10 
		page 2: 10-20 ......
		
		On the above **sensitivity**, 1 means it will not fetch any 18+ or ecchi-related images if the value is 0 then it will bring all without any filter.
		
		Optional GET parameter
		**quality**:0
		
		This API provides two types of wallpaper images
		1. Compress lower size wallpaper.
		2.  Best quality but will result in high size.
		
		if the **quality**  parameter, is not provided then by default it will provide the URL of compressed wallpaper low size if the quality parameter is set to 1 then it will provide a URL of the highest quality wallpaper ( high size ).
		
		**RESPONSE**
		**animename**  ( Depend on subscription )
		if the wallpaper is related to any anime, or game then the Japanese name of the anime will be sent otherwise illustration will be sent.
		
		**arturl**
		URL of wallpaper
		
		**sensitivity** ( Depend on subscription )
		Is wallpaper sensitive, 1 means yes 0 means no
		
		**art_id** ( Depend on subscription )
		Unique wallpaper  id which can be later used to get more info
		
		**premium** (Depend on subscription)
		Whether art is of premium category 1 means yes 0 means no.
		
		**Note: if no wallpaper found response will be**
		`{"status": "empty"}`"
    
    """
    url = f"https://premium-anime-mobile-wallpapers-illustrations.p.rapidapi.com/rapidHandler/boy"
    querystring = {'page': page, 'sensitivity': sensitivity, }
    if quality:
        querystring['quality'] = quality
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premium-anime-mobile-wallpapers-illustrations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def anime_wallpapers_related_to_the_family_by_pagination(sensitivity: int, page: int, quality: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get anime wallpapers related to the family:
		Required GET parameter
		**page**:  1
		**sensitivity**: 1
		On the above **page**, 1 means it will fetch the latest 10 wallpaper if the page value is 2 then It will bring the next 10 latest wallpaper 
		page 1 : 0-10 
		page 2: 10-20 ......
		
		On the above **sensitivity**, 1 means it will not fetch any 18+ or ecchi-related images if the value is 0 then it will bring all without any filter.
		
		Optional GET parameter
		**quality**:0
		
		This API provides two types of wallpaper images
		1. Compress lower size wallpaper.
		2.  Best quality but will result in high size.
		
		if the **quality**  parameter, is not provided then by default it will provide the URL of compressed wallpaper low size if the quality parameter is set to 1 then it will provide a URL of the highest quality wallpaper ( high size ).
		
		**RESPONSE**
		**animename**  ( Depend on subscription )
		if the wallpaper is related to any anime, or game then the Japanese name of the anime will be sent otherwise illustration will be sent.
		
		**arturl**
		URL of wallpaper
		
		**sensitivity** ( Depend on subscription )
		Is wallpaper sensitive, 1 means yes 0 means no
		
		**art_id** ( Depend on subscription )
		Unique wallpaper  id which can be later used to get more info
		
		**premium** (Depend on subscription)
		Whether art is of premium category 1 means yes 0 means no.
		
		**Note: if no wallpaper found response will be**
		`{"status": "empty"}`"
    
    """
    url = f"https://premium-anime-mobile-wallpapers-illustrations.p.rapidapi.com/rapidHandler/family"
    querystring = {'sensitivity': sensitivity, 'page': page, }
    if quality:
        querystring['quality'] = quality
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premium-anime-mobile-wallpapers-illustrations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def anime_wallpapers_related_to_the_halloween_by_pagination(sensitivity: int, page: int, quality: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get anime wallpapers related to the halloween:
		Required GET parameter
		**page**:  1
		**sensitivity**: 1
		On the above **page**, 1 means it will fetch the latest 10 wallpaper if the page value is 2 then It will bring the next 10 latest wallpaper 
		page 1 : 0-10 
		page 2: 10-20 ......
		
		On the above **sensitivity**, 1 means it will not fetch any 18+ or ecchi-related images if the value is 0 then it will bring all without any filter.
		
		Optional GET parameter
		**quality**:0
		
		This API provides two types of wallpaper images
		1. Compress lower size wallpaper.
		2.  Best quality but will result in high size.
		
		if the **quality**  parameter, is not provided then by default it will provide the URL of compressed wallpaper low size if the quality parameter is set to 1 then it will provide a URL of the highest quality wallpaper ( high size ).
		
		**RESPONSE**
		**animename**  ( Depend on subscription )
		if the wallpaper is related to any anime, or game then the Japanese name of the anime will be sent otherwise illustration will be sent.
		
		**arturl**
		URL of wallpaper
		
		**sensitivity** ( Depend on subscription )
		Is wallpaper sensitive, 1 means yes 0 means no
		
		**art_id** ( Depend on subscription )
		Unique wallpaper  id which can be later used to get more info
		
		**premium** (Depend on subscription)
		Whether art is of premium category 1 means yes 0 means no.
		
		**Note: if no wallpaper found response will be**
		`{"status": "empty"}`"
    
    """
    url = f"https://premium-anime-mobile-wallpapers-illustrations.p.rapidapi.com/rapidHandler/halloween"
    querystring = {'sensitivity': sensitivity, 'page': page, }
    if quality:
        querystring['quality'] = quality
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premium-anime-mobile-wallpapers-illustrations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def premium_anime_wallpapers_by_pagination(sensitivity: int, page: int, quality: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Premium anime arts are wallpapers that look best of the best on a mobile device or contain high-quality digital art:
		Required GET parameter
		**page**:  0
		**sensitivity**: 1
		On the above **page**, 0 means it will fetch the latest 10 wallpaper if the page value is 1 then It will bring the next 10 latest wallpaper 
		page 0 : 0-10 
		page 1: 10-20 ......
		
		On the above sensitivity, 1 means it will not fetch any 18+ or ecchi-related images if the value is 0 then it will bring all without any filter.
		
		Optional GET parameter
		**quality**:0
		
		This API provides two types of wallpaper images
		1. Compress lower size wallpaper.
		2.  Best quality but will result in high size.
		
		if the **quality**  parameter, is not provided then by default it will provide the URL of compressed wallpaper low size if the quality parameter is set to 1 then it will provide a URL of the highest quality wallpaper ( high size ).
		
		**RESPONSE**
		**animename**  ( Depend on subscription )
		if the wallpaper is related to any anime, or game then the Japanese name of the anime will be sent otherwise illustration will be sent.
		
		**arturl**
		URL of wallpaper
		
		**sensitivity** ( Depend on subscription )
		Is wallpaper sensitive, 1 means yes 0 means no
		
		**art_id** ( Depend on subscription )
		Unique wallpaper  id which can be later used to get more info
		
		**premium** (Depend on subscription)
		Whether art is of premium category 1 means yes 0 means no.
		
		**Note: if no wallpaper found response will be**
		`{"status": "empty"}`"
    
    """
    url = f"https://premium-anime-mobile-wallpapers-illustrations.p.rapidapi.com/rapidHandler/premium"
    querystring = {'sensitivity': sensitivity, 'page': page, }
    if quality:
        querystring['quality'] = quality
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premium-anime-mobile-wallpapers-illustrations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

