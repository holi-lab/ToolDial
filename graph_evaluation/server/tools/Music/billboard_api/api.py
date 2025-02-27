import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def new_zealand_songs(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the New Zealand Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/new-zealand"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def billboard_japan_hot_100(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the BILLBOARD JAPAN HOT 100 chart information"
    date: format(YYYY-MM-DD)
        range: max range(1-100)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/billboard-japan-hot-100"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def u_k_songs(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the U.K. Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/u-k"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def turkey_songs(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Turkey Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/turkey"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def taiwan_songs(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Taiwan Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/taiwan"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def switzerland_songs(range: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Switzerland Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/switzerland"
    querystring = {'range': range, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sweden_songs(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Sweden Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/sweden"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def spain_songs(range: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Spain Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/spain"
    querystring = {'range': range, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def south_korea_songs(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the South Korea Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/south-korea"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def south_africa_songs(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the South Africa Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/south-africa"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def slovakia_songs(range: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Slovakia Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/slovakia"
    querystring = {'range': range, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def singapore_songs(range: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Singapore Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/singapore"
    querystring = {'range': range, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def romania_songs(range: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Romania Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/romania"
    querystring = {'range': range, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def portugal_songs(range: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Portugal Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/portugal"
    querystring = {'range': range, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def poland_songs(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Poland Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/poland"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def philippines_songs(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Philippines Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/philippines"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def peru_songs(range: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Peru Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/peru"
    querystring = {'range': range, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def norway_songs(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Norway Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/norway"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def netherlands_songs(range: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Netherlands Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/netherlands"
    querystring = {'range': range, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mexico_songs(range: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Mexico Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/mexico"
    querystring = {'range': range, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def malaysia_songs(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Malaysia Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/malaysia"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def luxembourg_songs(range: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Luxembourg Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/luxembourg"
    querystring = {'range': range, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ireland_songs(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Ireland Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/ireland"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def indonesia_songs(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Indonesia Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/indonesia"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def india_songs(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the India Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/india"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def iceland_songs(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Iceland Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/iceland"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hungary_songs(range: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Hungary Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/hungary"
    querystring = {'range': range, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hong_kong_songs(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Hong Kong Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/hong-kong"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greece_songs(range: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greece Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/greece"
    querystring = {'range': range, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def germany_songs(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Germany Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/germany"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def france_songs(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the France Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/france"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def finland_songs(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Finland Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/finland"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ecuador_songs(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Ecuador Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/ecuador"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def denmark_songs(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Denmark Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/denmark"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def czech_republic_songs(range: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Czech Republic Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/czech-republic"
    querystring = {'range': range, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def croatia_songs(range: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Croatia Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/croatia"
    querystring = {'range': range, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def colombia_songs(range: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Colombia Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/colombia"
    querystring = {'range': range, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chile_songs(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Chile Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/chile"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def brazil_songs(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Brazil Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/brazil"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bolivia_songs(range: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Bolivia Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/bolivia"
    querystring = {'range': range, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def belgium_songs(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Belgium Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/belgium"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def austria_songs(range: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Austria Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/austria"
    querystring = {'range': range, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def australia_songs(range: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Australia Songs chart information"
    date: date format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/country-songs-chart/australia"
    querystring = {'range': range, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def adult_alternative_songs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Adult Alternative Songs chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-of-all-time-adult-alternative-songs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_r_b_hip_hop_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Top R&B/Hip-Hop Artists chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-r-b-hip-hop-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hot_latin_songs_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Hot Latin Songs Artists chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-hot-latin-songs-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def songs_of_the_summer(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Songs of the Summer chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-of-all-time-songs-of-the-summer"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hot_100_women_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Hot 100 Women Artists chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-hot-100-women-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mainstream_rock_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Mainstream Rock Artists chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-of-all-time-mainstream-rock-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mainstream_rock_songs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Mainstream Rock Songs chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-of-all-time-mainstream-rock-songs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def adult_alternative_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Adult Alternative Artists chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-of-all-time-adult-alternative-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alternative_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Alternative Artists chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-alternative-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alternative_songs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Alternative Songs chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-alternative-songs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_r_b_hip_hop_albums(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Top R&B/Hip-Hop Albums chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-r-b-hip-hop-albums"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hot_r_b_hip_hop_songs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Hot R&B/Hip-Hop Songs chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-r-b-hip-hop-songs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_dance_club_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Top Dance Club Artists chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-top-dance-club-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hot_latin_songs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Hot Latin Songs chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-hot-latin-songs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latin_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Latin Artists chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-of-all-time-latin-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_country_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Top Country Artists chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-country-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_country_albums(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Top Country Albums chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-country-albums"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hot_country_songs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Hot Country Songs chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-country-songs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def adult_pop_songs_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Adult Pop Songs Artists chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-adult-pop-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def adult_pop_songs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Adult Pop Songs chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-adult-pop-songs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pop_songs_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Pop Songs Artists chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-of-all-time-pop-songs-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pop_songs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Pop Songs chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-of-all-time-pop-songs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_holiday_albums(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Top Holiday Albums chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-of-all-time-top-holiday-albums"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def holiday_100_songs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Holiday 100 Songs chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-of-all-time-holiday-100-songs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def songs_of_the_90s(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Songs of the '90s chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-billboards-top-songs-90s"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def songs_of_the_80s(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Songs of the '80s chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-billboards-top-songs-80s"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def billboard_200_albums_by_women(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Billboard 200 Albums by Women chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-billboard-200-albums-by-women"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hot_100_songs_by_women(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Hot 100 Songs by Women chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-hot-100-songs-by-women"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def billboard_200_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Billboard 200 Artists chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-billboard-200-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def billboard_200_albums_greatest_of_all_time(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Billboard 200 Albums chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-billboard-200-albums"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hot_100_artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Hot 100 Artists chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-hot-100-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hot_100_songs_greatest_of_all_time(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Hot 100 Songs chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-hot-100-singles"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def artists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the Greatest of All Time Artists chart information"
    
    """
    url = f"https://billboard-api2.p.rapidapi.com/goat-chart/greatest-of-all-time-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hot_trending_songs_powered_by_twitter(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the HOT TRENDING SONGS POWERED BY TWITTER chart information"
    date: format(YYYY-MM-DD)
        range: max range(1-20)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/hot-trending-songs-powered-by-twitter"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def billboard_u_s_afrobeats_songs(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the BILLBOARD U.S. AFROBEATS SONGS chart information"
    date: format(YYYY-MM-DD)
        range: max range(1-50)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/billboard-us-afrobeats-songs"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def artist_100(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the ARTIST 100 chart information"
    date: format(YYYY-MM-DD)
        range: max range(1-100)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/artist-100"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hot_100(range: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the HOT 100 chart information"
    range: max range(1-100)
        date: format(YYYY-MM-DD)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/hot-100"
    querystring = {'range': range, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_artists_male(year: str='2022', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the YEAR-END Top Artists - Male chart information
		
		If `year` is not supplied, will default to last year."
    year: date format(YYYY)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/year-end-chart/top-artists-male"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_artists_female(year: str='2022', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the YEAR-END Top Artists - Female chart information
		
		If `year` is not supplied, will default to last year."
    year: date format(YYYY)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/year-end-chart/top-artists-female"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def billboard_200_albums(year: str='2022', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the YEAR-END Billboard 200 Albums chart information
		
		If `year` is not supplied, will default to last year."
    year: date format(YYYY)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/year-end-chart/top-billboard-200-albums"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_labels(year: str='2022', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the YEAR-END Top Labels chart information
		
		If `year` is not supplied, will default to last year."
    year: date format(YYYY)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/year-end-chart/labels"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_new_artists(year: str='2022', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the YEAR-END Top New Artists chart information
		
		If `year` is not supplied, will default to last year."
    year: date format(YYYY)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/year-end-chart/top-new-artists"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_artists_duo_group(year: str='2022', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the YEAR-END Top Artists - Duo/Group chart information
		
		If `year` is not supplied, will default to last year."
    year: date format(YYYY)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/year-end-chart/top-artists-duo-group"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def billboard_global_200_year_end(year: str='2022', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the YEAR-END Billboard Global 200 chart information
		
		If `year` is not supplied, will default to last year."
    year: date format(YYYY)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/year-end-chart/billboard-global-200"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_artists(year: str='2022', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the YEAR-END Top Artists chart information
		
		If `year` is not supplied, will default to last year."
    year: date format(YYYY)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/year-end-chart/top-artists"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hot_100_songs(year: str='2022', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the YEAR-END Hot 100 Songs chart information
		
		If `year` is not supplied, will default to last year."
    year: date format(YYYY)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/year-end-chart/hot-100-songs"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def billboard_global_200(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the BILLBOARD GLOBAL 200 chart information"
    date: format(YYYY-MM-DD)
        range: max range(1-200)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/billboard-global-200"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def billboard_200(date: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the BILLBOARD 200 chart information"
    date: format(YYYY-MM-DD)
        range: max range(1-200)
        
    """
    url = f"https://billboard-api2.p.rapidapi.com/billboard-200"
    querystring = {'date': date, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

