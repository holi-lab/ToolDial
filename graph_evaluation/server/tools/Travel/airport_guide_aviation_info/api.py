import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def airport_details_by_use_country_type_and_city(city: str, type: str, country: str, airport_use: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of airports by entering the country and one or more of the following optional fields: airport use (public, private, military), facility type and city name. The successful call returns the following data: Airport Name, IATA ID, ICAO ID, City, State, Country, Latitude, Longitude. Airportguide provides Country Data List: https://airportguide.com/search/browse-airports-by-country/
		Note: The results are limited to 5000 records."
    city: Name of the city
        type: Valid landing facility types are: Airport, Balloonport, Gliderport, Heliport, Seaplane Base, or Ultralight
        country: Enter the code of the country. e.g. country=CA
        airport_use: Enter the landing facility use (public, private, military).
        
    """
    url = f"https://forteweb-airportguide-airport-basic-info-v1.p.rapidapi.com/getairport_by_use"
    querystring = {'city': city, 'type': type, 'country': country, 'airport_use': airport_use, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forteweb-airportguide-airport-basic-info-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airport_details_given_an_airport_id(airport_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter an airport ID (FAA, IATA, ICAO, local) and get general information regarding that airport."
    airport_id: Enter the airport ID. This can be an IATA ID, ICAO ID, FAA ID, TC ID, or any locally used ID for that airport.
        
    """
    url = f"https://forteweb-airportguide-airport-basic-info-v1.p.rapidapi.com/get_airport"
    querystring = {'airport_id': airport_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forteweb-airportguide-airport-basic-info-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airport_count_by_type_and_country(country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter the Country Code to get the total number of airports in the country and the totals by Landing Facility Type (Airport, Balloonport, Gliderport, Heliport, Seaplane Base, Ultralight). Airportguide provides Country Data List: https://airportguide.com/search/browse-airports-by-country/"
    country: Enter the Code of the country. e.g. country=CA
        
    """
    url = f"https://forteweb-airportguide-airport-basic-info-v1.p.rapidapi.com/count_airporttypebycountry"
    querystring = {'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forteweb-airportguide-airport-basic-info-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airport_details_by_iata(airport_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter an IATA airport ID and get general information regarding that airport."
    airport_id: Enter the IATA airport ID to receive the airport's details.
        
    """
    url = f"https://forteweb-airportguide-airport-basic-info-v1.p.rapidapi.com/get_airport_by_iata"
    querystring = {'airport_id': airport_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forteweb-airportguide-airport-basic-info-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airport_count_by_use_and_country(country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "An Api Call which returns: Enter the Country Name to get the total number of airports in the country and the totals by Landing Facility Use. Airportguide provides Country Data List: https://airportguide.com/search/browse-airports-by-country/"
    country: Enter the CODE of the country. e.g. Country=CA
        
    """
    url = f"https://forteweb-airportguide-airport-basic-info-v1.p.rapidapi.com/count_airportusebycountry"
    querystring = {'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forteweb-airportguide-airport-basic-info-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def offline_fbo_fuel(code: str, param: str='{business_name:Signature Flight Support,services:,fuel_brand:,fuel_grade:,offset:0}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The FBO & FUEL API will search airport Fixed Based Operators by airport, fuel, services, and more. Review Parameter List."
    code: Get FBO & Fuel Search by Code, this is mandatory Parameter when you request for detail. You need to pass AIRPORT code LAX
        param: Airportguide provides accurate Data, Be specific what you searched by this saves your time and Query returns more accurate result. An optional Parameter List: business_name, services, fuel_brand, fuel_grade, offset. You need to pass json format parameters similar to above example.
        
    """
    url = f"https://forteweb-airportguide-airport-basic-info-v1.p.rapidapi.com/getfbo"
    querystring = {'code': code, }
    if param:
        querystring['param'] = param
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forteweb-airportguide-airport-basic-info-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def offline_all_delays_us(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve real-time airport ground delays for US and Southern Canadian airports."
    
    """
    url = f"https://forteweb-airportguide-airport-basic-info-v1.p.rapidapi.com/all_delay"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forteweb-airportguide-airport-basic-info-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def offline_get_metar_detail(long: str, lat: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current METAR details given any latitude/longitude combination. The API finds METAR reports within a 100 mile radius."
    long: Pass longitude -104.65
        lat: Pass latitude 39.83
        
    """
    url = f"https://forteweb-airportguide-airport-basic-info-v1.p.rapidapi.com/getmetar"
    querystring = {'long': long, 'lat': lat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forteweb-airportguide-airport-basic-info-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def offline_get_taf_detail(long: str, lat: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current Terminal Area Forecast (TAF) details given any latitude/longitude combination. The API finds METAR reports within a 20 mile radius or returns an error."
    long: Pass Longitude -104.65
        lat: Pass Latitude 39.83
        
    """
    url = f"https://forteweb-airportguide-airport-basic-info-v1.p.rapidapi.com/gettaf"
    querystring = {'long': long, 'lat': lat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forteweb-airportguide-airport-basic-info-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airline_by_country(country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of airlines by entering the country code. The successful call returns the following data: Airline Name, IATA ID, ICAO ID, Country, Airline Url, Call Signs.Airportguide provides Country Data List: https://airportguide.com/search/browse-airports-by-country/ Condition Apply: API accepts offset till 5 only. Contact us for more!"
    country: Pass the Country code, Don't have Country code list? Refer AirportGuide's Country List here: https://airportguide.com/search/browse-airports-by-country
        
    """
    url = f"https://forteweb-airportguide-airport-basic-info-v1.p.rapidapi.com/airline_bycountry"
    querystring = {'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forteweb-airportguide-airport-basic-info-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

