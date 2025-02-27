import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getpowerplantbyradiusandaddress(address: str, radius_km: int=None, page_number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get powerplants which have address falling into a specific radius"
    radius_km: default value is 100 km (100)
        
    """
    url = f"https://rankiteo-climate-risk-assessment.p.rapidapi.com/GetPowerplantByRadiusAndAddress"
    querystring = {'address': address, }
    if radius_km:
        querystring['radius_km'] = radius_km
    if page_number:
        querystring['page_number'] = page_number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rankiteo-climate-risk-assessment.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpowerplantbyradiusandgps(latitude: int, longitude: int, page_number: int=None, radius_km: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get powerplants which have coordinate points falling into a specific radius"
    radius_km: default value is 100 km (100)
        
    """
    url = f"https://rankiteo-climate-risk-assessment.p.rapidapi.com/GetPowerplantByRadiusAndGps"
    querystring = {'latitude': latitude, 'longitude': longitude, }
    if page_number:
        querystring['page_number'] = page_number
    if radius_km:
        querystring['radius_km'] = radius_km
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rankiteo-climate-risk-assessment.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdisasterbyradiusandaddress(end_date: str, start_date: str, address: str, radius_km: str='100', page_number: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get disasters which have address falling into a specific radius for a specific date range"
    radius_km: Default value: 100
        
    """
    url = f"https://rankiteo-climate-risk-assessment.p.rapidapi.com/GetDisasterByRadiusAndAddress"
    querystring = {'end_date': end_date, 'start_date': start_date, 'address': address, }
    if radius_km:
        querystring['radius_km'] = radius_km
    if page_number:
        querystring['page_number'] = page_number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rankiteo-climate-risk-assessment.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdisasterbyradiusandgps(start_date: str, latitude: int, longitude: int, end_date: str, page_number: str=None, radius_km: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get disasters which have coordinate points falling into a specific radius for a specific date range"
    radius_km: Default value: 100
        
    """
    url = f"https://rankiteo-climate-risk-assessment.p.rapidapi.com/GetDisasterByRadiusAndGps"
    querystring = {'start_date': start_date, 'latitude': latitude, 'longitude': longitude, 'end_date': end_date, }
    if page_number:
        querystring['page_number'] = page_number
    if radius_km:
        querystring['radius_km'] = radius_km
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rankiteo-climate-risk-assessment.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcityexposedbydisasterid(disaster_id: str, page_number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get affected city along with informations by disaster id"
    
    """
    url = f"https://rankiteo-climate-risk-assessment.p.rapidapi.com/GetCityExposedByDisasterId"
    querystring = {'disaster_id': disaster_id, }
    if page_number:
        querystring['page_number'] = page_number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rankiteo-climate-risk-assessment.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdisastertypebydateandstatus(disaster_status: int, start_date: str, end_date: str, page_number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get disaster by status within a time range"
    disaster_status: 0 : Past Disaster
1 : Ongoing Disaster
        
    """
    url = f"https://rankiteo-climate-risk-assessment.p.rapidapi.com/GetDisasterTypeByDateAndStatus"
    querystring = {'disaster_status': disaster_status, 'start_date': start_date, 'end_date': end_date, }
    if page_number:
        querystring['page_number'] = page_number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rankiteo-climate-risk-assessment.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdisastertypebydate(disaster_type_number: int, end_date: str, start_date: str, page_number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get disaster by type within a time range"
    disaster_type_number: 1 : Earthquake
2 : Tropical Cyclone
3 : Floods
4 : Volcano
5 : Drought
6 : WildFire
        
    """
    url = f"https://rankiteo-climate-risk-assessment.p.rapidapi.com/GetDisasterTypeByDate"
    querystring = {'disaster_type_number': disaster_type_number, 'end_date': end_date, 'start_date': start_date, }
    if page_number:
        querystring['page_number'] = page_number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rankiteo-climate-risk-assessment.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getclimatescorebygps(latitude: int, longitude: int, disaster_type_number: int, activity_type_number: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get climate score for a specific location mapped to some industries/disasters with a given gps position."
    disaster_type_number: 0 : **ALL**
1 : Air Quality
2 : Earthquake
3 : Floods
4 : Heat Stress
5 : Sea Level Rise 
6 : Tropical
7 : Tsunami
8 : Volcano
9 : Water Stress
10 : Wildfire
        activity_type_number: 0 : **ALL**
1 : Agriculture
2 : Nuclear Plants
3 : Telecommunications
4 : Education
5 : Automobile
6 : Chemical Industry
7 : Market
8 : Finance Industry
9 : Aerospace, 
10 : Construction
11 : Mining
12 : News Media
13 : Pharmaceutical
14 : Datacenter
15 : Entertainment complex
16 : Freight logistics
17 : Hospital
18 : Hotel
19 : Manufacturing
20 : Office
21 : Shopping center
22 : Residential complex
23 : Storage Ware houses
24 : Oil and Gas Industry
25 : Transportation
        
    """
    url = f"https://rankiteo-climate-risk-assessment.p.rapidapi.com/GetClimateScoreByGps"
    querystring = {'latitude': latitude, 'longitude': longitude, 'disaster_type_number': disaster_type_number, 'activity_type_number': activity_type_number, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rankiteo-climate-risk-assessment.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getclimatescorebyaddress(address: str, activity_type_number: int, disaster_type_number: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get climate score for a specific location mapped to some industries/disasters with a given address."
    activity_type_number: 0 : **ALL**
1 : Agriculture
2 : Nuclear Plants
3 : Telecommunications
4 : Education
5 : Automobile
6 : Chemical Industry
7 : Market
8 : Finance Industry
9 : Aerospace, 
10 : Construction
11 : Mining
12 : News Media
13 : Pharmaceutical
14 : Datacenter
15 : Entertainment complex
16 : Freight logistics
17 : Hospital
18 : Hotel
19 : Manufacturing
20 : Office
21 : Shopping center
22 : Residential complex
23 : Storage Ware houses
24 : Oil and Gas Industry
25 : Transportation
        disaster_type_number: 0 : **ALL**
1 : Air Quality
2 : Earthquake
3 : Floods
4 : Heat Stress
5 : Sea Level Rise 
6 : Tropical
7 : Tsunami
8 : Volcano
9 : Water Stress
10 : Wildfire
        
    """
    url = f"https://rankiteo-climate-risk-assessment.p.rapidapi.com/GetClimateScoreByAddress"
    querystring = {'address': address, 'activity_type_number': activity_type_number, 'disaster_type_number': disaster_type_number, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rankiteo-climate-risk-assessment.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

