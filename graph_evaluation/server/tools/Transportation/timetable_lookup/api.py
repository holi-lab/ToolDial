import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def codes_entertainment_codes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a list of entertainment codes used by the airlines"
    
    """
    url = f"https://timetable-lookup.p.rapidapi.com/codes/entertainment/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timetable-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def codes_equipment_codes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a list of equipment codes used by the airlines"
    
    """
    url = f"https://timetable-lookup.p.rapidapi.com/codes/equipment/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timetable-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def codes_meal_codes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a list of meal code used by the airlines"
    
    """
    url = f"https://timetable-lookup.p.rapidapi.com/codes/meal/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timetable-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def flight_schedules(is_from: str, date: str, to: str, time: str=None, compression: str=None, get_7day: str=None, interline: str=None, results: str=None, connection: str=None, sort: str=None, maxconnect: str=None, max_results: str=None, nofilter: str=None, airline: str=None, flightnumber: str=None, expandresults: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "FlightLookup Web Services, powered by FlightLookup Routing Engine.  This API provides a powerful information source for developing sophisticated applications requiring flight related schedule data.  Whether you are a software company developing products for sale or an in-house developer, the FlightLookup Web Services APIs are unmatched in terms of breadth and richness of information. Our development tools and documentation will get you up to speed quickly enabling short development cycles."
    from: 3 letter IATA code for the departure airport
        date: Departure date (YYYYMMDD)
        to: 3 letter IATA code for the destination airport
        time: Return flights that happen during the specified time period (ANY, AM, PM, NIGHT)
        compression: Displays non-stops, directs and logical single and multiple connections up to the maximum number of results requested (AUTO, NONSTOP, DIRECT, 1STOP, MORE)
        get_7day: Setting to "N" results in 1 days of results. Setting to "Y" results in 7 days of results. (N, Y)
        interline: Please see documentation
        results: Range = 1 to 500, Maximum number of results to return
        connection: Displays non-stops, directs and logical single and multiple connections up to the maximum number of results requested
        sort: Specify the sort order for the results (Departure, Optional - Arrival, Duration, Flights)
        maxconnect: Range = 120-1200. Maximum time between flight in minutes. Default = 240 which is the IATA standard.
        max_results: Override the results count.
        nofilter: Disable Traffic Restriction Code Filters
        airline: Restrict by specific airline 2 or 3 letter airline IATA code
        flightnumber: Restrict by specific flight number must be used with Airline
        expandresults: Y = Override the flight efficiency filter, showing more results. N = Do not override the flight efficiency filter, showing fewer results.
        
    """
    url = f"https://timetable-lookup.p.rapidapi.com/TimeTable/{is_from}/{to}/{date}/"
    querystring = {}
    if time:
        querystring['Time'] = time
    if compression:
        querystring['Compression'] = compression
    if get_7day:
        querystring['7Day'] = get_7day
    if interline:
        querystring['Interline'] = interline
    if results:
        querystring['Results'] = results
    if connection:
        querystring['Connection'] = connection
    if sort:
        querystring['Sort'] = sort
    if maxconnect:
        querystring['MaxConnect'] = maxconnect
    if max_results:
        querystring['Max_Results'] = max_results
    if nofilter:
        querystring['NoFilter'] = nofilter
    if airline:
        querystring['Airline'] = airline
    if flightnumber:
        querystring['FlightNumber'] = flightnumber
    if expandresults:
        querystring['ExpandResults'] = expandresults
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timetable-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airports_nonstop_routes_for_an_airport_by_airline(airlineiatacode: str, airportiatacode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of nonstop routes for an airport restricted to an airline"
    airlineiatacode: Airline IATA Code
        airportiatacode: Airport IATA Code
        
    """
    url = f"https://timetable-lookup.p.rapidapi.com/airports/{airportiatacode}/routes/nonstops/{airlineiatacode}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timetable-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airlines_airports_in_a_country_an_airline_operates_in(countryiatacode: str, airlineiatacode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a list of airports in a country an airline operates in"
    countryiatacode: Country IATA code
        airlineiatacode: Airline IATA code
        
    """
    url = f"https://timetable-lookup.p.rapidapi.com/airlines/{airlineiatacode}/countries/{countryiatacode}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timetable-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airlines_countries_airline_operates_in(airlineiatacode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a list of the countries that an airline operates in"
    
    """
    url = f"https://timetable-lookup.p.rapidapi.com/airlines/{airlineiatacode}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timetable-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airlines_direct_flights_for_an_airline_from_an_airport(airlineiatacode: str, airportiatacode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a list of direct flights for an airline from an airport"
    airlineiatacode: Airline IATA code
        airportiatacode: Airport IATA code
        
    """
    url = f"https://timetable-lookup.p.rapidapi.com/airlines/{airlineiatacode}/routes/directs/{airportiatacode}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timetable-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airlines_nonstop_and_direct_flights_for_an_airline(airlineiatacode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a list of nonstop and direct flights for an airline"
    airlineiatacode: Airline Iata Code
        
    """
    url = f"https://timetable-lookup.p.rapidapi.com/airlines/{airlineiatacode}/routes/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timetable-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airlines_nonstop_flights_for_an_airline_from_an_airport(airlineiatacode: str, airportiatacode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a list of nonstop flights for an airline from an airport"
    airlineiatacode: Airline IATA Code
        airportiatacode: Airport IATA Code
        
    """
    url = f"https://timetable-lookup.p.rapidapi.com/airlines/{airlineiatacode}/routes/nonstops/{airportiatacode}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timetable-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airports_airport_information(airportiatacode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return an airport's information"
    airportiatacode: Airport IATA Code
        
    """
    url = f"https://timetable-lookup.p.rapidapi.com/airports/{airportiatacode}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timetable-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airports_airports_in_a_country(countryiatacode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of airports in a country"
    countryiatacode: Country IATA code
        
    """
    url = f"https://timetable-lookup.p.rapidapi.com/airports/countries/{countryiatacode}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timetable-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airports_airports_in_a_metro(metroiatacode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of airports in a metro"
    metroiatacode: Metro IATA Code
        
    """
    url = f"https://timetable-lookup.p.rapidapi.com/airports/metros/{metroiatacode}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timetable-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airports_carriers_operating_out_of_an_airport(airportiatacode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a list of carriers operating out of an airport"
    airportiatacode: Airport IATA Code
        
    """
    url = f"https://timetable-lookup.p.rapidapi.com/airports/{airportiatacode}/airlines/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timetable-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airports_country_iata_codes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of country IATA codes"
    
    """
    url = f"https://timetable-lookup.p.rapidapi.com/airports/countries/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timetable-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airports_direct_routes_for_an_airport(airportiatacode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of direct routes for an airport"
    airportiatacode: Airport IATA Code
        
    """
    url = f"https://timetable-lookup.p.rapidapi.com/airports/{airportiatacode}/routes/directs/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timetable-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airports_direct_routes_for_an_airport_by_airline(airportiatacode: str, airlineiatacode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of direct routes for an airport restricted to an airline"
    airportiatacode: Airport IATA Code
        airlineiatacode: Airline IATA Code
        
    """
    url = f"https://timetable-lookup.p.rapidapi.com/airports/{airportiatacode}/routes/directs/{airlineiatacode}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timetable-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airports_latitude_and_longitude_for_location_in_country(location: str, countryiatacode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the latitude and longitude for location in country"
    location: A location in the country
        countryiatacode: Country IATA Code
        
    """
    url = f"https://timetable-lookup.p.rapidapi.com/airports/nearest/{countryiatacode}/{location}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timetable-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airports_list_of_airports(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a list of the airports worldwide"
    
    """
    url = f"https://timetable-lookup.p.rapidapi.com/airports/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timetable-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airports_metro_iata_codes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of metro IATA codes"
    
    """
    url = f"https://timetable-lookup.p.rapidapi.com/airports/metros/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timetable-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airports_nearest_airports_for_a_given_latitude_and_longitude(lon: str, lat: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the nearest airports for a given latitude and longitude"
    lon: Specify longitude
        lat: Specify latitude
        
    """
    url = f"https://timetable-lookup.p.rapidapi.com/airports/nearest/{lat}/{lon}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timetable-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airports_nonstop_and_direct_routes_for_an_airport(airportiatacode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of nonstop and direct routes for an airport"
    airportiatacode: Airport IATA Code
        
    """
    url = f"https://timetable-lookup.p.rapidapi.com/airports/{airportiatacode}/routes/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timetable-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airports_nonstop_and_direct_routes_for_an_airport_by_airline(airlineiatacode: str, airportiatacode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of nonstop and direct routes for an airport restricted to an airline"
    airlineiatacode: Airline IATA Code
        airportiatacode: Airport IATA Code
        
    """
    url = f"https://timetable-lookup.p.rapidapi.com/airports/{airportiatacode}/routes/{airlineiatacode}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timetable-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airports_nonstop_routes_for_an_airport(airportiatacode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of nonstop routes for an airport"
    airportiatacode: Airport IATA Code
        
    """
    url = f"https://timetable-lookup.p.rapidapi.com/airports/{airportiatacode}/routes/nonstops/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timetable-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airlines_airlines_and_the_countries_they_operate_in(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a list of airlines and the countries they operate in"
    
    """
    url = f"https://timetable-lookup.p.rapidapi.com/airlines/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timetable-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

