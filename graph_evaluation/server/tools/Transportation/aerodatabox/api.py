import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def beta_aircraft_image_by_registration(reg: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**What is the aircraft photo?  **
		
		Aircraft images are being searched externally in Flickr. Only images with licenses approved for commercial use are returned. Please be advised that you may be required to mention author attribution before using the image.  
		
		Returns: Image data with medium-sized direct image URL and licence approved for commercial use is returned."
    reg: Registration of the aircraft (full, stripped and any case formats are acceptable).
        
    """
    url = f"https://aerodatabox.p.rapidapi.com/aircrafts/reg/{reg}/image/beta"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def beta_weather_forecast_at_the_airport(code: str, codetype: str, tolocal: str='2023-04-03T14:00', fromlocal: str='2023-04-03T12:00', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "- What is the current weather at the airport?
		- What is the weather forecast for the airport? (up to 48 hours ahead)"
    code: If `codeType` is:
* `icao`, then this field must be a 4-character ICAO-code of the airport (e.g.: EHAM, KLAX, UUEE, etc.);
* `iata`, then this field must be a 3-character IATA-code of the airport (e.g.: AMS, SFO, LAX, etc.).
        codetype: Type of code to search airport by (`iata` or `icao`)
        tolocal: (Optional) End of the search range (local time, format: `YYYY-MM-DDTHH:mm`). Must be equal to or more than the beginning of the search range specified in `fromLocal`, up to 48 hours in the future. Default equal to `fromLocal`.
        fromlocal: (Optional) Beginning of the search range (local time, format: `YYYY-MM-DDTHH:mm`). Must be in range from current time up to 48 hours in the future. Default - current time.
        
    """
    url = f"https://aerodatabox.p.rapidapi.com/airports/{codetype}/{code}/weather/{fromlocal}/{tolocal}"
    querystring = {}
    if tolocal:
        querystring['toLocal'] = tolocal
    if fromlocal:
        querystring['fromLocal'] = fromlocal
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def solar_and_day_time_at_the_airport(codetype: str, code: str, datelocal: str='2023-04-03T19:00', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "- What is the sun position in the sky now at a specific time at the airport? 
		- When does the sun rise and set at the airport today or on the other day at the airport? 
		- Is it dark now or is it day at the airport?"
    codetype: Type of code to search airport by (`iata` or `icao`)
        code: If `codeType` is:
* `icao`, then this field must be a 4-character ICAO-code of the airport (e.g.: EHAM, KLAX, UUEE, etc.);
* `iata`, then this field must be a 3-character IATA-code of the airport (e.g.: AMS, SFO, LAX, etc.).
        datelocal: The moment of time which solar data is request for (local time, format: YYYY-MM-DDTHH:mm). Default - current time.
        
    """
    url = f"https://aerodatabox.p.rapidapi.com/airports/{codetype}/{code}/time/solar/{datelocal}"
    querystring = {}
    if datelocal:
        querystring['dateLocal'] = datelocal
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def current_local_time_at_the_airport(code: str, codetype: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "What is the current local time at the airport?"
    code: If `codeType` is:
 * `icao`, then this field must be a 4-character ICAO-code of the airport (e.g.: EHAM, KLAX, UUEE, etc.);
* `iata`, then this field must be a 3-character IATA-code of the airport (e.g.: AMS, SFO, LAX, etc.).
        codetype: Type of code to search airport by (iata or icao)
        
    """
    url = f"https://aerodatabox.p.rapidapi.com/airports/{codetype}/{code}/time/local"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airport_delays_current_historical(codetype: str, code: str, datelocal: str='2022-06-03T12:00', datetolocal: str='2022-06-04T00:00', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**What is the current or historical average delay in the airport?** or **What is the delay index of the airport right now or at a moment in past?** or **What were the delays within a specific period of time?** or **How the delays changed within a specific period of time?**
		
		Statistics are calculated against the flight data we aggregated. Hencerforth, it is available only for those airports and those periods of time when delay information could have been reasonably derived. This applies to flights which qualify for this calculation (actual time and status of the flight must be available). This normally applies to flights arriving to / departing from the airports with Live or Radar + Scheduled quality. Learn more about coverage here: https://aerodatabox.com/data-coverage/
		
		Median delay and delay index are available if there were at least 2 flights per hour per departures or arrivals at the time requested and at least 3 total qualifying flights within the coverage period.
		
		*Returns*: Statistical delay information about delays (median delay, delay index, cancelled flights) of arrivals and departures for the requested airport, represented by:
		* a single `AirportDelayContract` item displaying the delay information based on flight movements within the 2 hours prior to the current moment, if no `dateLocal` is specified;
		* a single `AirportDelayContract` item displaying the delay information based on flight movements within the 2 hours prior to the moment requested in `dateLocal`, if `dateLocal` is specified and `dateToLocal` - is not;
		* a collection of `AiportDelayContract` items displaying the delay information at multiple moments within the period between `dateLocal` and `dateToLocal`, if both are specified."
    codetype: Type of code to search airport by (`iata` or `icao`)
        code: If `codeType` is:
* `icao`, then this field must be a 4-character ICAO-code of the airport (e.g.: EHAM, KLAX, UUEE, etc.);
* `iata`, then this field must be a 3-character IATA-code of the airport (e.g.: AMS, SFO, LAX, etc.).
        datelocal: The moment of time for / from which delay data is requested (local time, format: YYYY-MM-DDTHH:mm).
        datetolocal: The moment of time until which delay data is requested (local time, format: YYYY-MM-DDTHH:mm).
        
    """
    url = f"https://aerodatabox.p.rapidapi.com/airports/{codetype}/{code}/delays/{datelocal}/{datetolocal}"
    querystring = {}
    if datelocal:
        querystring['dateLocal'] = datelocal
    if datetolocal:
        querystring['dateToLocal'] = datetolocal
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fids_schedules_airport_departures_and_arrivals(fromlocal: str, tolocal: str, codetype: str, code: str, withleg: bool=None, direction: str=None, withcancelled: bool=None, withcodeshared: bool=None, withcargo: bool=None, withlocation: bool=None, withprivate: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**What are current departures or arrivals at the airport?** or **What is the flight schedule at the airport?** or **What is flight history at the airport?**
		
		Flights may contain live updates with corresponding information related to the actual progress of the flight (including actual/estimated arrival/departure times). In this case this endpoint serves as a FIDS endpoint. Presense of live updates is subject to data coverage: not all airports have this coverage in our system.
		
		Otherwise the flight information will be limited to schedule only and will not be updated with time. Much more airports have this coverage.
		
		To check if airport is tracked and on which level, use `/health/services/airports/{icao}/feeds` endpoint. You can also use `/health/services/feeds/{service}/airports` to get the list of supported airports for this or that layer of coverage.
		
		To learn more about the data coverage, refer to https://www.aerodatabox.com/data-coverage.
		
		Returns: the list of arriving and/or departing flights scheduled and/or planned and/or commenced within a specified time range for a specified airport."
    fromlocal: Beginning of the search range (local time, format: YYYY-MM-DDTHH:mm)
        tolocal: End of the search range (local time, format: YYYY-MM-DDTHH:mm). Should be more than beginning of the search range, but not more than by 12 hours.
        codetype: Type of code to search airport by (`iata` or `icao`)
        code: If `codeType` is:
* `icao`, then this field must be a 4-character ICAO-code of the airport (e.g.: EHAM, KLAX, UUEE, etc.);
* `iata`, then this field must be a 3-character IATA-code of the airport (e.g.: AMS, SFO, LAX, etc.).
        withleg: If set to true, will include movement information from airport opposite in this flight leg (airport of origin for arriving flight or airport of destination for departing flight). In this case, Movement property will be replaced with Departure and Arrival properties for each flight. Default: false.
        direction: Direction of flights: Arrival, Departure or Both (default)
        withcancelled: If set to true, result will include cancelled, divered, likely cancelled (CanceledUncertain) flights. Default: true.
        withcodeshared: If set to true, result will include flights with all code-shared statuses. Otherwise, code-sharing flights will be excluded. For airports, where no information about code-share statuses of flights are supplied (all flights are CodeshareStatus=Unknown), complex filtering will be applied to determine which flights are likely to be operational (caution: false results are possible).
        withcargo: If set to true, result will include cargo flights (subject to availability).
        withlocation: If set to true, each currently active flight within the result will be populated with its present real-time location, altitude, speed and track (subject to availability).
        withprivate: If set to true, result will include private flights (subject to availability).
        
    """
    url = f"https://aerodatabox.p.rapidapi.com/flights/airports/{codetype}/{code}/{fromlocal}/{tolocal}"
    querystring = {}
    if withleg:
        querystring['withLeg'] = withleg
    if direction:
        querystring['direction'] = direction
    if withcancelled:
        querystring['withCancelled'] = withcancelled
    if withcodeshared:
        querystring['withCodeshared'] = withcodeshared
    if withcargo:
        querystring['withCargo'] = withcargo
    if withlocation:
        querystring['withLocation'] = withlocation
    if withprivate:
        querystring['withPrivate'] = withprivate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aiport_by_code(codetype: str, code: str, withrunways: bool=None, withtime: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets airport by code (IATA or ICAO)."
    codetype: Type of code to search airport by (`iata` or `icao`)
        code: If `codeType` is:
* `icao`, then this field must be a 4-character ICAO-code of the airport (e.g.: EHAM, KLAX, UUEE, etc.);
* `iata`, then this field must be a 3-character IATA-code of the airport (e.g.: AMS, SFO, LAX, etc.).
        withrunways: Should include runway information
        withtime: Should include current local time information
        
    """
    url = f"https://aerodatabox.p.rapidapi.com/airports/{codetype}/{code}"
    querystring = {}
    if withrunways:
        querystring['withRunways'] = withrunways
    if withtime:
        querystring['withTime'] = withtime
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airport_runways(codetype: str, code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets information about runways at specified airport."
    codetype: Type of code to search airport by (`iata` or `icao`)
        code: If `codeType` is:
* `icao`, then this field must be a 4-character ICAO-code of the airport (e.g.: EHAM, KLAX, UUEE, etc.);
* `iata`, then this field must be a 3-character IATA-code of the airport (e.g.: AMS, SFO, LAX, etc.).
        
    """
    url = f"https://aerodatabox.p.rapidapi.com/airports/{codetype}/{code}/runways"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def flight_delay_statistics_by_flight_number(number: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**By how much the flight is delayed in average?** 
		
		Gets delay statistics for the flight with specified number."
    number: Flight number (with or without spaces, IATA or ICAO, any case formats are acceptable, e.g. KL1395, Klm 1395)
        
    """
    url = f"https://aerodatabox.p.rapidapi.com/flights/{number}/delays"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_airports_by_ip_address_geolocation(q: str, radiuskm: int, limit: int, withflightinfoonly: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**What are the airports closest to the customer, based on their IP address?**
		** What are the airports closest to the location determined (geo-located) by a IP address?**
		
		Gets the list of airports found within the specified radius around the location approximated (geo-located) from the specified IP address."
    q: A valid public IP v4 address
        radiuskm: Radius of search around specified location in kilometers (max. 1000 km)
        limit: Maximum number of airports to be returned (max. 250)
        withflightinfoonly: If set to true, will only return airports which have flight data (scheduled or live) available. Default = false.
        
    """
    url = f"https://aerodatabox.p.rapidapi.com/airports/search/ip"
    querystring = {'q': q, 'radiusKm': radiuskm, 'limit': limit, }
    if withflightinfoonly:
        querystring['withFlightInfoOnly'] = withflightinfoonly
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def deprecated_search_airports_by_ip_address_geolocation_path_style(ip: str, limit: int, radiuskm: int, withflightinfoonly: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User `Search airports by IP address geolocation` endpoint instead."
    ip: A valid public IP v4 address
        limit: Maximum number of airports to be returned
        radiuskm: Radius of search around specified location in kilometers (maximum 1000 km)
        withflightinfoonly: If set to true, will only return airports which have flight data (scheduled or live) available. Default = false.
        
    """
    url = f"https://aerodatabox.p.rapidapi.com/airports/search/ip/{ip}/km/{radiuskm}/{limit}"
    querystring = {}
    if withflightinfoonly:
        querystring['withFlightInfoOnly'] = withflightinfoonly
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def deprecated_search_airports_by_location_path_style(lon: int, lat: int, limit: int, radiuskm: int, withflightinfoonly: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use `Search airports by location` endpoint instead."
    lon: Longitude location coordinate in decimal format (between -180 and 180)
        lat: Latitude location coordinate in decimal format (between -90 and 90)
        limit: Maximum number of airports to be returned
        radiuskm: Radius of search around specified location in kilometers (maximum 1000 km)
        withflightinfoonly: If set to true, will only return airports which have flight data (scheduled or live) available. Default = false.
        
    """
    url = f"https://aerodatabox.p.rapidapi.com/airports/search/location/{lat}/{lon}/km/{radiuskm}/{limit}"
    querystring = {}
    if withflightinfoonly:
        querystring['withFlightInfoOnly'] = withflightinfoonly
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_airports_by_free_text(q: str, limit: int=10, withflightinfoonly: bool=None, withsearchbycode: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the list of airports with ICAO or IATA code matching the search term case-insensitively and airports with airport or city name containing the search term case-insensitively."
    q: Search query (min. 3 non whitespace characters length)
        limit: Maximum number of airports to be returned (max. 250, defaut = 10)
        withflightinfoonly: If set to true, will only return airports which have flight data (scheduled or live) available. Default = false.
        withsearchbycode: If set to true, will attempt to interpret short words within the search query as IATA or ICAO code and prioritize exact matches by these codes (they will appear higher than others). Otherwise, the search by code will be completely excluded (only the name of an airport or its city will be searched). Default = true.
        
    """
    url = f"https://aerodatabox.p.rapidapi.com/airports/search/term"
    querystring = {'q': q, }
    if limit:
        querystring['limit'] = limit
    if withflightinfoonly:
        querystring['withFlightInfoOnly'] = withflightinfoonly
    if withsearchbycode:
        querystring['withSearchByCode'] = withsearchbycode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_airports_by_location(radiuskm: int, lon: int, lat: int, limit: int, withflightinfoonly: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the list of airports found within specified radius around specified location."
    radiuskm: Radius of search around specified location in kilometers (max. 1000 km)
        lon: Longitude location coordinate in decimal format (between -180 and 180)
        lat: Latitude location coordinate in decimal format (between -90 and 90)
        limit: Maximum number of airports to be returned (max. 250)
        withflightinfoonly: If set to true, will only return airports which have flight data (scheduled or live) available. Default = false.
        
    """
    url = f"https://aerodatabox.p.rapidapi.com/airports/search/location"
    querystring = {'radiusKm': radiuskm, 'lon': lon, 'lat': lat, 'limit': limit, }
    if withflightinfoonly:
        querystring['withFlightInfoOnly'] = withflightinfoonly
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airport_routes_and_daily_flights_statistics(codetype: str, code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**What are the most popular routes from an airport?** or **Where I can fly from an airport?** or **How many daily flights to different destinations from an airport?**
		
		Gets list of routes and daily flights amount departing from an airport."
    codetype: Type of code to search airport by (`iata` or `icao`)
        code: If `codeType` is:
* `icao`, then this field must be a 4-character ICAO-code of the airport (e.g.: EHAM, KLAX, UUEE, etc.);
* `iata`, then this field must be a 3-character IATA-code of the airport (e.g.: AMS, SFO, LAX, etc.).
        
    """
    url = f"https://aerodatabox.p.rapidapi.com/airports/{codetype}/{code}/stats/routes/daily"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def global_delays_current_historical(datetimeutc: str='2022-01-21T12:00', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**What is the current or historical delay situation in all airports?** or **What is the delay index of all airports right now or at a moment in past?**
		
		*Returns:* Statistical delay information about delays (median delay, delay index, cancelled flights) of arrivals and departures for all known airports, represented by a collection of items sorted in departure index descending order (from worst to best)."
    datetimeutc: The moment of time for / from which delay data is requested (UTC time, format: YYYY-MM-DDTHH:mm). Default - current time.
        
    """
    url = f"https://aerodatabox.p.rapidapi.com/airports/delays/{datetimeutc}"
    querystring = {}
    if datetimeutc:
        querystring['dateTimeUtc'] = datetimeutc
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def distance_flight_time_between_airports(codetype: str, codefrom: str, codeto: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**What is distance between airports?** and **What is approximate flight time between airports?**
		
		Returns: Distance and approximate flight time between airports, if both airports found."
    codetype: Type of code to search airport by (IATA or ICAO)
        codefrom: If `codeType` is:
* `icao`, then this field must be a 4-character ICAO-code of the origin airport (e.g.: EHAM, KLAX, UUEE, etc.);
* `iata`, then this field must be a 3-character IATA-code of the origin airport (e.g.: AMS, SFO, LAX, etc.).
        codeto: If `codeType` is:
* `icao`, then this field must be a 4-character ICAO-code of the destination airport (e.g.: EHAM, KLAX, UUEE, etc.);
* `iata`, then this field must be a 3-character IATA-code of the destination  airport (e.g.: AMS, SFO, LAX, etc.).
        
    """
    url = f"https://aerodatabox.p.rapidapi.com/airports/{codetype}/{codefrom}/distance-time/{codeto}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airports_supporting_data_feed_service(service: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Which airports support flight schedules? **or **Which airports support live flight updates?**
		
		Returns: Collection ICAO codes of airports supporting specified airport data feed service."
    service: Airport data feed service name
        
    """
    url = f"https://aerodatabox.p.rapidapi.com/health/services/feeds/{service}/airports"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airport_data_feed_services_status_by_icao_code(icao: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets current status of airport data feed services (live flight updates, flight schedules, etc.) for requested airport."
    icao: 4-digit ICAO-code of the airport (e.g.: EHAM, KLAX, UUEE, etc.). Full, stripped and any case formats are acceptable.
        
    """
    url = f"https://aerodatabox.p.rapidapi.com/health/services/airports/{icao}/feeds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def general_status_of_data_feed_services(service: str, withhttpcode: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Which is the general health of the data feed service?**
		
		Returns: Status of the service in general, regardless of the airports"
    service: Data feed service name
        withhttpcode: If true, reflect status of the service in the HTTP code of the response (if the service is down, HTTP code will be 503).

 Default: false
        
    """
    url = f"https://aerodatabox.p.rapidapi.com/health/services/feeds/{service}"
    querystring = {}
    if withhttpcode:
        querystring['withHttpCode'] = withhttpcode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def flight_status_nearest(searchby: str, searchparameter: str, withaircraftimage: bool=None, withlocation: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets information about the status of the nearest (either in past or in future) flight or about flight departing or arriving on the day specified (local time), operated:
		- under specified flight number; or
		- by an aircraft with specified registration; or
		- under specified ATC-callsign; or
		- by an aircraft with specified Mode-S 24-bit ICAO Transponder address."
    searchby: Criteria to search by: 
- **number**: search flight operated under specified flight number;
- **callsign**: search flight operated under specified ATC call-sign;
- **reg**: search flight operated on an aircraft with specified registration;
- **icao24**: search flight operated on an aircraft with specified ICAO 24-bit Mode-S transponder address.
        searchparameter: If **searchBy** is: 
- **number**, then then this field should be Flight number (with or without spaces, IATA or ICAO, any case formats are acceptable, e.g. KL1395, Klm 1395)
- **callsign**, then this field should be ATC call-sign of the flight (with or without spaces, any case formats are acceptable, e.g. AFL1482, nca 008X);
- **reg**: then this field should be Aircraft registration (with or without spaces or dashes, any case formats are acceptable, e.g. PH-BXO, DeMhJ);
- **icao24**, then this field should be Aircraft ICAO 24-bit Mode-S address specified in hexadecimal format (e.g. 484161, 483EFD).
        withaircraftimage: Should include aircraft image.
        withlocation: Should include real-time positional data, e.g.: location, speed, altitude, etc., if available 
        
    """
    url = f"https://aerodatabox.p.rapidapi.com/flights/{searchby}/{searchparameter}"
    querystring = {}
    if withaircraftimage:
        querystring['withAircraftImage'] = withaircraftimage
    if withlocation:
        querystring['withLocation'] = withlocation
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def flight_status_with_date(searchby: str, searchparameter: str, datelocal: str, withlocation: bool=None, withaircraftimage: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets information about the status of the nearest (either in past or in future) flight or about flight departing or arriving on the day specified (local time), operated:
		- under specified flight number; or
		- by an aircraft with specified registration; or
		- under specified ATC-callsign; or
		- by an aircraft with specified Mode-S 24-bit ICAO Transponder address."
    searchby: Criteria to search by: 
- **number**: search flight operated under specified flight number;
- **callsign**: search flight operated under specified ATC call-sign;
- **reg**: search flight operated on an aircraft with specified registration;
- **icao24**: search flight operated on an aircraft with specified ICAO 24-bit Mode-S transponder address.
        searchparameter: If **searchBy** is: 
- **number**, then then this field should be Flight number (with or without spaces, IATA or ICAO, any case formats are acceptable, e.g. KL1395, Klm 1395)
- **callsign**, then this field should be ATC call-sign of the flight (with or without spaces, any case formats are acceptable, e.g. AFL1482, nca 008X);
- **reg**: then this field should be Aircraft registration (with or without spaces or dashes, any case formats are acceptable, e.g. PH-BXO, DeMhJ);
- **icao24**, then this field should be Aircraft ICAO 24-bit Mode-S address specified in hexadecimal format (e.g. 484161, 483EFD).
        datelocal: Optional parameter: Local date of departure or arrival (in format: YYYY-MM-DD, e.g.: 2019-08-29). If specified, endpoint gets information about flight departing or arriving on the day specified (local time).
        withlocation: Should include real-time positional data, e.g.: location, speed, altitude, etc., if available 
        withaircraftimage: Should include aircraft image.
        
    """
    url = f"https://aerodatabox.p.rapidapi.com/flights/{searchby}/{searchparameter}/{datelocal}"
    querystring = {}
    if withlocation:
        querystring['withLocation'] = withlocation
    if withaircraftimage:
        querystring['withAircraftImage'] = withaircraftimage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def flight_departure_dates(searchparameter: str, searchby: str, tolocal: str='2020-06-15', fromlocal: str='2020-06-01', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**On which days the flight operates?** or **What is the flight schedule?**
		
		Get flight departure dates for a flight operated:
		, operated:
		- under specified flight number; or
		- by an aircraft with specified registration; or
		- under specified ATC-callsign; or
		- by an aircraft with specified Mode-S 24-bit ICAO Transponder address.
		
		Returns: Array of local departure dates in (YYYY-MM-DD)."
    searchparameter: If **searchBy** is: 
- **number**, then then this field should be Flight number (with or without spaces, IATA or ICAO, any case formats are acceptable, e.g. KL1395, Klm 1395)
- **callsign**, then this field should be ATC call-sign of the flight (with or without spaces, any case formats are acceptable, e.g. AFL1482, nca 008X);
- **reg**: then this field should be Aircraft registration (with or without spaces or dashes, any case formats are acceptable, e.g. PH-BXO, DeMhJ);
- **icao24**, then this field should be Aircraft ICAO 24-bit Mode-S address specified in hexadecimal format (e.g. 484161, 483EFD).
        searchby: Criteria to search by: 
- **number**: search flight operated under specified flight number;
- **callsign**: search flight operated under specified ATC call-sign;
- **reg**: search flight operated on an aircraft with specified registration;
- **icao24**: search flight operated on an aircraft with specified ICAO 24-bit Mode-S transponder address.
        tolocal: End of the search range (local time, format: YYYY-MM-DD)
        fromlocal: Beginning of the search range (local time, format: YYYY-MM-DD)
        
    """
    url = f"https://aerodatabox.p.rapidapi.com/flights/{searchby}/{searchparameter}/dates/{fromlocal}/{tolocal}"
    querystring = {}
    if tolocal:
        querystring['toLocal'] = tolocal
    if fromlocal:
        querystring['fromLocal'] = fromlocal
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aircraft_registration_history_by_tail_number_mode_s_or_id(searchby: str, searchparam: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns: A list of all known registrations of a single aircraft, best matching specified search criteria, if found"
    searchby: Criteria to search aircraft by
        searchparam: Value of the search criteria. If *searchBy* is:
  - **reg**: then this field should be aircraft registration (with or without spaces or dashes, any case formats are acceptable, e.g.PH-BXO, DeMhJ);
-  **icao24**, then this field should be aircraft ICAO 24-bit Mode-S address specified in hexadecimal format (e.g. 484161, 483EFD).

        
    """
    url = f"https://aerodatabox.p.rapidapi.com/aircrafts/{searchby}/{searchparam}/registrations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_single_aircraft_by_tail_number_mode_s_or_id(searchparam: str, searchby: str, withregistrations: bool=None, withimage: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns: Single aircraft, best matching specified search criteria, if found"
    searchparam: Value of the search criteria. If *searchBy* is:
  - **reg**: then this field should be aircraft registration (with or without spaces or dashes, any case formats are acceptable, e.g.PH-BXO, DeMhJ);
-  **icao24**, then this field should be aircraft ICAO 24-bit Mode-S address specified in hexadecimal format (e.g. 484161, 483EFD).

        searchby: Criteria to search aircraft by
        withregistrations: Should include the history of aircraft registrations (default: false).
        withimage: Should include aircraft image (default: false).
        
    """
    url = f"https://aerodatabox.p.rapidapi.com/aircrafts/{searchby}/{searchparam}"
    querystring = {}
    if withregistrations:
        querystring['withRegistrations'] = withregistrations
    if withimage:
        querystring['withImage'] = withimage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_aircraft_by_tail_number_mode_s_or_id(searchby: str, searchparam: str, withimage: bool=None, withregistrations: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns: A list of all aircraft matching the search criteria"
    searchby: Criteria to search aircraft by
        searchparam: Value of the search criteria. If *searchBy* is:
  - **reg**: then this field should be aircraft registration (with or without spaces or dashes, any case formats are acceptable, e.g.PH-BXO, DeMhJ);
-  **icao24**, then this field should be aircraft ICAO 24-bit Mode-S address specified in hexadecimal format (e.g. 484161, 483EFD).

        withimage: Should include aircraft image (default: false).
        withregistrations: Should include the history of aircraft registrations (default: false).
        
    """
    url = f"https://aerodatabox.p.rapidapi.com/aircrafts/{searchby}/{searchparam}/all"
    querystring = {}
    if withimage:
        querystring['withImage'] = withimage
    if withregistrations:
        querystring['withRegistrations'] = withregistrations
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_web_hook_subscriptions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of existing active web-hook subscriptions"
    
    """
    url = f"https://aerodatabox.p.rapidapi.com/subscriptions/webhook"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_web_hook_subscription(subscriptionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets information about an existing web-hook subscription."
    subscriptionid: Subscription ID
        
    """
    url = f"https://aerodatabox.p.rapidapi.com/subscriptions/webhook/{subscriptionid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

