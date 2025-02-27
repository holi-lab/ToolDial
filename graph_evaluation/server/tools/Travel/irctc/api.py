import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def checkseatavailability(quota: str, trainno: int, tostationcode: str, fromstationcode: str, date: str, classtype: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    
    """
    url = f"https://irctc1.p.rapidapi.com/api/v1/checkSeatAvailability"
    querystring = {'quota': quota, 'trainNo': trainno, 'toStationCode': tostationcode, 'fromStationCode': fromstationcode, 'date': date, 'classType': classtype, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "irctc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_live_station(hours: str, fromstationcode: str, tostationcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    
    """
    url = f"https://irctc1.p.rapidapi.com/api/v3/getLiveStation"
    querystring = {'hours': hours, 'fromStationCode': fromstationcode, 'toStationCode': tostationcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "irctc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettrainclasses(trainno: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://irctc1.p.rapidapi.com/api/v1/getTrainClasses"
    querystring = {'trainNo': trainno, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "irctc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_pnr_status_v3(pnrnumber: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    
    """
    url = f"https://irctc1.p.rapidapi.com/api/v3/getPNRStatus"
    querystring = {'pnrNumber': pnrnumber, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "irctc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchstation(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://irctc1.p.rapidapi.com/api/v1/searchStation"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "irctc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchtrain(query: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://irctc1.p.rapidapi.com/api/v1/searchTrain"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "irctc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_train_live_status(trainno: str, startday: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API will take train no and day of journey, and will return current status of train with other informations."
    startday: Optional File
start day range from 0-4
0 = Day 1
1 = 1 Day Ago
2 = 2 Day Ago
3 = 3 Day Ago
4 = 4 Day Ago
        
    """
    url = f"https://irctc1.p.rapidapi.com/api/v1/liveTrainStatus"
    querystring = {'trainNo': trainno, }
    if startday:
        querystring['startDay'] = startday
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "irctc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trainsbetweenstations_v3(dateofjourney: str, tostationcode: str, fromstationcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    
    """
    url = f"https://irctc1.p.rapidapi.com/api/v3/trainBetweenStations"
    querystring = {'dateOfJourney': dateofjourney, 'toStationCode': tostationcode, 'fromStationCode': fromstationcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "irctc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_train_schedule(trainno: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    
    """
    url = f"https://irctc1.p.rapidapi.com/api/v1/getTrainSchedule"
    querystring = {'trainNo': trainno, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "irctc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fare(trainno: int, tostationcode: str, fromstationcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    
    """
    url = f"https://irctc1.p.rapidapi.com/api/v2/getFare"
    querystring = {'trainNo': trainno, 'toStationCode': tostationcode, 'fromStationCode': fromstationcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "irctc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_trains_by_station(stationcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    
    """
    url = f"https://irctc1.p.rapidapi.com/api/v3/getTrainsByStation"
    querystring = {'stationCode': stationcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "irctc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trainsbetweenstations(tostationcode: str, fromstationcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    
    """
    url = f"https://irctc1.p.rapidapi.com/api/v2/trainBetweenStations"
    querystring = {'toStationCode': tostationcode, 'fromStationCode': fromstationcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "irctc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def checkpnrstatus(pnrnumber: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    
    """
    url = f"https://irctc1.p.rapidapi.com/api/v2/getPNRStatus"
    querystring = {'pnrNumber': pnrnumber, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "irctc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfare(trainno: int, fromstationcode: str, tostationcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://irctc1.p.rapidapi.com/api/v1/getFare"
    querystring = {'trainNo': trainno, 'fromStationCode': fromstationcode, 'toStationCode': tostationcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "irctc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

