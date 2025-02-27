import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def bolivia_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Bolivia Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/bolivia"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def slovakia_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Slovakia Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/slovakia"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def romania_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Romania Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/romania"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def austria_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Austria Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/austria"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chile_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Chile Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/chile"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def iceland_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Iceland Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/iceland"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hungary_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Hungary Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/hungary"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def brazil_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Brazil Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/brazil"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def norway_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Norway Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/norway"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def croatia_czech_republic_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Croatia Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/czech-republic"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def turkey_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Turkey Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/turkey"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def switzerland_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Switzerland Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/switzerland"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def south_africa_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "South Africa Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/south-africa"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def philippines_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Philippines Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/philippines"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def france_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "France Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/france"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mexico_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mexico Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/mexico"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def croatia_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Croatia Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/croatia"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def singapore_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Singapore Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/singapore"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def u_k_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "U.K Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/u-k"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def taiwan_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Taiwan Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/taiwan"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sweden_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sweden Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/sweden"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def spain_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Spain Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/spain"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def philippines_south_korea_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "South Korea Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/south-korea"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def portugal_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Portugal Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/portugal"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def poland_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Poland Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/poland"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def malaysia_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Malaysia Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/malaysia"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ireland_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ireland Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/ireland"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greece_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greece Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/greece"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def peru_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Peru Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/peru"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def new_zealand_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "New Zealand Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/new-zealand"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def netherlands_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Netherlands Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/netherlands"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def luxembourg_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Luxembourg Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/luxembourg"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def indonesia_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Indonesia Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/indonesia"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def india_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "India Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/india"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hong_kong_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Hong Kong Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/hong-kong"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def germany_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Germany Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/germany"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def finland_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Finland Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/finland"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ecuador_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ecuador Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/ecuador"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def colombia_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Colombia Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/colombia"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def denmark_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Denmark Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/denmark"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def belgium_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Belgium Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/belgium"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def australia_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Australia Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/country-songs/australia"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def billboard_u_s_afrobeats_songs(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Billboard U.S. Afrobeats Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/billboard-u-s-afrobeats-songs"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def billboard_global_200(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Billboard Global 200 chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/billboard-global-200"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def billboard_200(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Billboard 200 chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/billboard-200"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_mainstream_rock_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Mainstream Rock Artists chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-of-all-time-mainstream-rock-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_mainstream_rock_songs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Mainstream Rock Songs chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-of-all-time-mainstream-rock-songs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_adult_alternative_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Adult Alternative Artists chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-of-all-time-adult-alternative-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_adult_alternative_songs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Adult Alternative Songs chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-of-all-time-adult-alternative-songs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_alternative_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Alternative Artists chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-alternative-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_alternative_songs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Alternative Songs chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-alternative-songs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_top_r_b_hip_hop_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Top R&B/Hip-Hop Artists chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-r-b-hip-hop-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_top_r_b_hip_hop_albums(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Top R&B/Hip-Hop Albums chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-r-b-hip-hop-albums"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_hot_r_b_hip_hop_songs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Hot R&B/Hip-Hop Songs chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-r-b-hip-hop-songs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_top_dance_club_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Top Dance Club Artists chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-top-dance-club-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_hot_latin_songs_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Hot Latin Songs Artists chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-hot-latin-songs-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_hot_latin_songs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Hot Latin Songs chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-hot-latin-songs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_latin_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Latin Artists chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-of-all-time-latin-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_top_country_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Top Country Artists chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-country-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_top_country_albums(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Top Country Albums chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-country-albums"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_hot_country_songs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Hot Country Songs chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-country-songs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_adult_pop_songs_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Adult Pop Songs Artists chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-adult-pop-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_adult_pop_songs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Adult Pop Songs chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-adult-pop-songs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_pop_songs_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Pop Songs Artists chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-of-all-time-pop-songs-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_pop_songs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Pop Songs chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-of-all-time-pop-songs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_top_holiday_albums(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Top Holiday Albums chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-of-all-time-top-holiday-albums"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_holiday_100_songs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Holiday 100 Songs chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-of-all-time-holiday-100-songs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_songs_of_the_summer(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Songs of the Summer chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-of-all-time-songs-of-the-summer"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_songs_of_the_90s(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Songs of the '90s chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-billboards-top-songs-90s"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_songs_of_the_80s(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Songs of the '80s chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-billboards-top-songs-80s"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_billboard_200_albums_by_women(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Billboard 200 Albums by Women chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-billboard-200-albums-by-women"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_hot_100_women_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Hot 100 Women Artists chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-hot-100-women-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_hot_100_songs_by_women(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Hot 100 Songs by Women chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-hot-100-songs-by-women"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_billboard_200_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Billboard 200 Artists chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-billboard-200-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_billboard_200_albums(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Billboard 200 Albums chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-billboard-200-albums"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_hot_100_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Hot 100 Artists chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-hot-100-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_hot_100_songs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Hot 100 Songs chart.
		See response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-hot-100-singles"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greatest of All Time Artists chart see response example"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/all-time/greatest-of-all-time-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def year_end_top_labels(year: str='2020', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Year end Top Labels
		is yearly based. 
		
		If `year` is not supplied, will default to last year."
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/year-end/labels"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def year_end_top_new_artists(year: str='2020', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Year end Top New Artists chart
		is yearly based. 
		
		If `year` is not supplied, will default to last year."
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/year-end/top-new-artists"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def year_end_top_artists_male(year: str='2020', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Year end Top Artists – Male chart
		is yearly based. 
		
		If `year` is not supplied, will default to last year."
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/year-end/top-artists-male"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def year_end_top_artists_female(year: str='2020', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Year end Top Top Artists – Female chart
		is yearly based. 
		
		If `year` is not supplied, will default to last year."
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/year-end/top-artists-female"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def year_end_top_artists_duo_group(year: str='2020', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Year end Top Artists – Duo/Group chart
		is yearly based. 
		
		If `year` is not supplied, will default to last year."
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/year-end/top-artists-duo-group"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def year_end_top_artists(year: str='2020', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Year end Top Artists
		is yearly based. 
		
		If `year` is not supplied, will default to last year."
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/year-end/top-artists"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def year_end_billboard_global_200(year: str='2020', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Year end Billboard Global 200chart 
		is yearly based. 
		
		If `year` is not supplied, will default to last year."
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/year-end/billboard-global-200"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def year_end_billboard_200_albums(year: str='2020', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Year end Billboard 200 Albums chart 
		is yearly based. 
		
		If `year` is not supplied, will default to last year."
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/year-end/top-billboard-200-albums"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def year_end_hot_100_songs(year: str='2020', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Year end Hot 100 Songs chart 
		is yearly based. 
		
		If `year` is not supplied, will default to last year."
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/year-end/hot-100-songs"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def independent_albums(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Independent Albums chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/independent-albums"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def catalog_albums(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Billboard U.S. Afrobeats Songs chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/catalog-albums"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def artist_100(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Artist 100 chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/artist-100"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def billboard_hot_100(week: str='2022-10-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Billboard Hot 100 chart 
		is weekly based. 
		
		If `week` is not supplied, will default to last week. 
		if the week date is not Saturday, will default to Saturday of that week"
    
    """
    url = f"https://billboard-api5.p.rapidapi.com/api/charts/hot-100"
    querystring = {}
    if week:
        querystring['week'] = week
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

