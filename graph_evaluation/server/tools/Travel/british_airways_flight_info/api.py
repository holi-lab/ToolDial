import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_departures_by_flight(flightnumber_scheduleddeparturedate_scheduleddeparturedate: str, flightnumber: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-british-airways-flight-info.p.rapidapi.com/flights;flightNumber={flightnumber};scheduledDepartureDate={scheduleddeparturedate}.json"
    querystring = {'flightnumber-scheduleddeparturedate-scheduleddeparturedate': flightnumber_scheduleddeparturedate_scheduleddeparturedate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-british-airways-flight-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_today_s_arrivals_by_time(arrivallocation_starttime_starttime_endtime_endtime: str, arrivallocation: str, starttime: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-british-airways-flight-info.p.rapidapi.com/flights;arrivalLocation={arrivallocation};startTime={starttime};endTime={endtime}.json"
    querystring = {'arrivallocation-starttime-starttime-endtime-endtime': arrivallocation_starttime_starttime_endtime_endtime, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-british-airways-flight-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_arrivals_by_flight(flightnumber_scheduledarrivaldate_scheduledarrivaldate: str, flightnumber: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-british-airways-flight-info.p.rapidapi.com/flights;flightNumber={flightnumber};scheduledArrivalDate={scheduledarrivaldate}.json"
    querystring = {'flightnumber-scheduledarrivaldate-scheduledarrivaldate': flightnumber_scheduledarrivaldate_scheduledarrivaldate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-british-airways-flight-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_departures_by_route(departurelocation_arrivallocation_arrivallocation_scheduleddeparturedate_scheduleddeparturedate: str, departurelocation: str, arrivallocation: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-british-airways-flight-info.p.rapidapi.com/flights;departureLocation={departurelocation};arrivalLocation={arrivallocation};scheduledDepartureDate={scheduleddeparturedate}.json"
    querystring = {'departurelocation-arrivallocation-arrivallocation-scheduleddeparturedate-scheduleddeparturedate': departurelocation_arrivallocation_arrivallocation_scheduleddeparturedate_scheduleddeparturedate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-british-airways-flight-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_arrivals_by_route(departurelocation_arrivallocation_arrivallocation_scheduledarrivaldate_scheduledarrivaldate: str, departurelocation: str, arrivallocation: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-british-airways-flight-info.p.rapidapi.com/flights;departureLocation={departurelocation};arrivalLocation={arrivallocation};scheduledArrivalDate={scheduledarrivaldate}.json"
    querystring = {'departurelocation-arrivallocation-arrivallocation-scheduledarrivaldate-scheduledarrivaldate': departurelocation_arrivallocation_arrivallocation_scheduledarrivaldate_scheduledarrivaldate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-british-airways-flight-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_today_s_departures_by_time(departurelocation_starttime_starttime_endtime_endtime: str, departurelocation: str, starttime: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-british-airways-flight-info.p.rapidapi.com/flights;departureLocation={departurelocation};startTime={starttime};endTime={endtime}.json"
    querystring = {'departurelocation-starttime-starttime-endtime-endtime': departurelocation_starttime_starttime_endtime_endtime, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-british-airways-flight-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

