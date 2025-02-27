import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def decode_route_string(routestring: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide an ICAO Format route string. Return the coordinates associated with each of the waypoints."
    
    """
    url = f"https://fliteroute.p.rapidapi.com/api/decoderoutestring/{routestring}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fliteroute.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fuel_time_and_co2_estimates_route_string_distance_calculator(model: str, routestring: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Input a valid aircraft ICAO Code ( e.g. **A320**) and a valid Route String. The Route string must contain a valid Origin Airport ICAO Code ( e.g. **KJFK**), at least one waypoint ( e.g. **WAVEY**), and a valid Destination Airport ICAO Code ( e.g. **MKJP**). The origin and destination airport codes must be distinct. Once a request is made, a line will be drawn starting from the origin airport, moving along each waypoint, and then terminating at the destination airport. During this time, the FIR intersections will be calculated. Using Aircraft information, the distance, time, fuel, FIR traversal, and CO2 emissions will be calculated for the climb, cruise, and descent of the flight. 
		Example Route String: **KJFK..WAVEY..EMJAY.J174.DIW..DIZNY.AR3.ZQA.Y307.ENAMO..NEFTU.UP525.EMABU.UA301.IMADI..SAVEM..MKJP**
		
		The response will produce a set of features containing information about each FIR that has been intersected. Additionally, the response will produce data related to the climb (**climbData**) and descent (**descentData**) of the aircraft. 
		Each feature will contain:
		- The name of the FIR ( e.g. **KZNY**)
		- The country name ( e.g. **United States**)
		- Distance in km and nm ( e.g. **415.3 km** / **224.2 nm**)
		- Cruise Time in minutes ( e.g. **22**)
		- Fuel consumption in kilograms ( e.g. **797.48**)
		- A numerical sequence number along the route ( e.g. **1**)
		- The geographic coordinates of the line segment  in [longitude, latitude] format ( e.g. **[ -73.78, 40.64], [-73.57, 40.42]** )
		
		The Climb Data will contain:
		- Fuel used during the climb in kilograms ( e.g. **2937**)
		- CO2 emissions used during the climb in kilograms ( e.g. **9279**)
		- Distance of the climb in km and nm (nautical miles) ( e.g. **624 km** / **337 nm**)
		- Time of the climb in minutes ( e.g. **65**)
		- The FIR in which the climb occurred ( e.g. **KZDC**)
		
		The Descent Data will contain:
		- Fuel used during the descent in kilograms ( e.g. **74**)
		- CO2 emissions used during the descent in kilograms ( e.g. **233**)
		- Distance of the descent in km and nm (nautical miles) ( e.g. **126 km** / **68 nm**)
		- Time of the descent in minutes ( e.g. **14**)
		- The FIR in which the descent occurred ( e.g. **MKJK**)
		- The FIR in which the cruise finished and the descent began ( e.g. **MUFH**)"
    
    """
    url = f"https://fliteroute.p.rapidapi.com/api/rsfuelandtime/model/{model}/routestring/{routestring}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fliteroute.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fuel_time_and_co2_estimates_great_circle_distance_calculator(model: str, origin: str, dest: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Input an Origin Airport IATA code ( e.g. **GVA**), a Destination Airport IATA code ( e.g. **MIA**), and a valid aircraft ICAO code ( e.g. **A320**). The airport codes MUST be distinct and valid 3-letter IATA codes. Once the request is made, a great circle will be drawn between the two airports, while taking the aircraft into account. During this time, the FIR intersections will be calculated. Using Aircraft information, the distance, time, fuel, FIR traversal, and CO2 emissions will be calculated for the climb, cruise, and descent of the flight.
		
		The response will produce a set of features containing information about each FIR that has been intersected. Additionally, the response will produce data related to the climb (**climbData**) and descent (**descentData**) of the aircraft. 
		Each feature will contain:
		- The name of the FIR ( e.g. **KZNY**)
		- The country name ( e.g. **United States**)
		- Distance in km and nm ( e.g. **415.3 km** / **224.2 nm**)
		- Cruise Time in minutes ( e.g. **22**)
		- Fuel consumption in kilograms ( e.g. **797.48**)
		- A numerical sequence number along the route ( e.g. **1**)
		- The geographic coordinates of the line segment  in [longitude, latitude] format ( e.g. **[ -73.78, 40.64], [-73.57, 40.42]** )
		
		The Climb Data will contain:
		- Fuel used during the climb in kilograms ( e.g. **2937**)
		- CO2 emissions used during the climb in kilograms ( e.g. **9279**)
		- Distance of the climb in km and nm (nautical miles) ( e.g. **624 km** / **337 nm**)
		- Time of the climb in minutes ( e.g. **65**)
		- The FIR in which the climb occurred ( e.g. **KZDC**)
		
		The Descent Data will contain:
		- Fuel used during the descent in kilograms ( e.g. **74**)
		- CO2 emissions used during the descent in kilograms ( e.g. **233**)
		- Distance of the descent in km and nm (nautical miles) ( e.g. **126 km** / **68 nm**)
		- Time of the descent in minutes ( e.g. **14**)
		- The FIR in which the descent occurred ( e.g. **MKJK**)
		- The FIR in which the cruise finished and the descent began ( e.g. **MUFH**)"
    
    """
    url = f"https://fliteroute.p.rapidapi.com/api/gcfuelandtime/origin/{origin}/dest/{dest}/model/{model}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fliteroute.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def route_string_distance_calculator(routestring: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Input a valid Route String. The Route string must contain a valid Origin Airport ICAO Code ( e.g. **KJFK**), at least one waypoint ( e.g. **WAVEY**), and a valid Destination Airport ICAO Code ( e.g. **MKJP**). The origin and destination airport codes must be distinct. Once a request is made, a line will be drawn starting from the origin airport, moving along each waypoint, and then terminating at the destination airport. During this time, the FIR intersections will be calculated. 
		Example Route String: **KJFK..WAVEY..EMJAY.J174.DIW..DIZNY.AR3.ZQA.Y307.ENAMO..NEFTU.UP525.EMABU.UA301.IMADI..SAVEM..MKJP**
		
		The response will produce a set of features containing information about each FIR that has been intersected. Each feature will contain:
		- The name of the FIR ( e.g. **KZNY**)
		- The country name ( e.g. **United States**)
		- Distance in km and nm ( e.g. **415.3 km** / **224.2 nm**)
		- A numerical sequence number along the route ( e.g. **1**)
		- The geographic coordinates of the line segment  in [longitude, latitude] format ( e.g. **[ -73.78, 40.64], [-73.57, 40.42]** )
		
		Note: The geographic coordinates are in Decimal Degrees (DD) format. South of the Equator and West of the Prime Meridian are denoted with a minus (-) sign.
		
		Please note that this route string feature does not yet support SID/STAR waypoints. Please ensure that the non-airport waypoints are either 3 or 5 letters in length, and contain only alphabetic uppercase characters."
    
    """
    url = f"https://fliteroute.p.rapidapi.com/api/routestring/{routestring}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fliteroute.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def great_circle_distance_calculator(dest: str, origin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Input an Origin Airport IATA code ( e.g. **GVA**) and Destination Airport IATA code ( e.g. **MIA**). These airport codes MUST be distinct and valid 3-letter IATA codes. 
		Once the request is made, a great circle will be drawn between the two airports, and the FIR intersections will be calculated. 
		
		The response will produce a set of features containing information about each FIR that has been intersected. Each feature will contain:
		- The name of the FIR ( e.g. **LSAS**)
		- The country name ( e.g. **Switzerland**)
		- Distance in km and nm ( e.g. **4.21 km** / **2.27 nm**)
		- A numerical sequence number along the route ( e.g. **1**)
		- The geographic coordinates of the line segment  in [longitude, latitude] format ( e.g. **[ 6.10, 46.23], [6.07, 46.26]** )
		
		Note: The geographic coordinates are in Decimal Degrees (DD) format. South of the Equator and West of the Prime Meridian are denoted with a minus (-) sign."
    
    """
    url = f"https://fliteroute.p.rapidapi.com/api/greatcircle/origin/{origin}/dest/{dest}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fliteroute.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

