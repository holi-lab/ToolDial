import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def hin_decoder_return_json(hin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns json decoded data"
    
    """
    url = f"https://hull-id-boat-hin-decoder.p.rapidapi.com/api-decoder.php"
    querystring = {'HIN': hin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hull-id-boat-hin-decoder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hin_decoder_return_json(hin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns json decoded data"
    
    """
    url = f"https://hull-id-boat-hin-decoder.p.rapidapi.com/api-decoder.php"
    querystring = {'HIN': hin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hull-id-boat-hin-decoder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def year_make_lookup_returns_json(year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup the boat manufacturers (makes) for a given year. This API can be used to create selection drop down menu for year and make. It will return json results"
    year: The year you want to look for the boat builders. format YYYY.  From 1970 to present year.
        
    """
    url = f"https://hull-id-boat-hin-decoder.p.rapidapi.com/API-year-make-MIC-lookup.php"
    querystring = {'year': year, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hull-id-boat-hin-decoder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def year_make_lookup_returns_json(year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup the boat manufacturers (makes) for a given year. This API can be used to create selection drop down menu for year and make. It will return json results"
    year: The year you want to look for the boat builders. format YYYY.  From 1970 to present year.
        
    """
    url = f"https://hull-id-boat-hin-decoder.p.rapidapi.com/API-year-make-MIC-lookup.php"
    querystring = {'year': year, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hull-id-boat-hin-decoder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mic_lookup(mic: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter a boat manuf. code (3 letter MIC) to find info about the company"
    
    """
    url = f"https://hull-id-boat-hin-decoder.p.rapidapi.com/mic-api-lookup.php"
    querystring = {'MIC': mic, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hull-id-boat-hin-decoder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hin_decoder_return_html(hin: str, usertable: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "setting usertable=1 will return an html table instead of json"
    usertable: setting usertable to 1 will give an html output instead of json
        
    """
    url = f"https://hull-id-boat-hin-decoder.p.rapidapi.com/api-decoder.php"
    querystring = {'HIN': hin, }
    if usertable:
        querystring['usertable'] = usertable
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hull-id-boat-hin-decoder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

