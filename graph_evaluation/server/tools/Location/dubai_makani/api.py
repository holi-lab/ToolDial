import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getmakaniinfofromcoordination(latitude: int, longitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "When entering the coordinate (latitude & longitude) for certain
		location, this method/function will return building details (in case
		this location falls inside a building), Makani details for the building
		and latitude & longitude for each Makani Number. Makani
		entrance(s) for the building can be plotted / pinned on an
		electronic map by using the latitude & longitude for each Makani
		Number."
    
    """
    url = f"https://dubai-makani.p.rapidapi.com/mi2tech/public/api/v1/makaniInfoFromCoord"
    querystring = {'latitude': latitude, 'longitude': longitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dubai-makani.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmakanidetails(makanono: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "When entering Makani Number for certain building entrance, this
		method/function will return building details and Makani details for
		the building. In case Makani Number is available in more than
		one emirate, this method/function will list the emirates and their
		communities"
    
    """
    url = f"https://dubai-makani.p.rapidapi.com/mi2tech/public/api/v1/makaniDetails/{makanono}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dubai-makani.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def isvalidmakani(makanino: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Makani Number is generated to uniquely identify each main entrance of
		an existing building. When entering Makani Number for certain
		entrance, this method/function will verify if the entry is valid"
    
    """
    url = f"https://dubai-makani.p.rapidapi.com/mi2tech/public/api/v1/isValidMakani/{makanino}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dubai-makani.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getqrcodeformakanino(makanino: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "When entering Makani Number, this method/function returns the
		QR Code download path in .jpg file which includes URL for
		entrance’s location link."
    
    """
    url = f"https://dubai-makani.p.rapidapi.com/mi2tech/public/api/v1/makaniQRCode/{makanino}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dubai-makani.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

