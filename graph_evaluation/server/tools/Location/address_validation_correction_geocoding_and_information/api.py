import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def address(zip: str=None, add1: str=None, address: str='Herslev-Kirkevej-26-34-7000-Fredericia-Denmark', add2: str=None, country: str=None, region: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this Endpoint, you can validate the given address and receive it in standard form as a response.
		
		Optional Arguments:
		address: one line full address spaces should be replaced with “-”
		add1,add2, region, zip, country: these arguments must be used in combination otherwise the API mechanism will look up for address argument.
		***Note: address argument can not be used with a combination of other arguments. ***
		add1, add2, region, zip, and country arguments can be used when better precision is required."
    
    """
    url = f"https://address-validation-correction-geocoding-and-information.p.rapidapi.com/api/v1/ask/address"
    querystring = {}
    if zip:
        querystring['zip'] = zip
    if add1:
        querystring['add1'] = add1
    if address:
        querystring['address'] = address
    if add2:
        querystring['add2'] = add2
    if country:
        querystring['country'] = country
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "address-validation-correction-geocoding-and-information.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_information(country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint exposes general information like the country's phone code, time zone, etc..
		Required Argument:
		country: country name"
    
    """
    url = f"https://address-validation-correction-geocoding-and-information.p.rapidapi.com/api/v1/ask/country/info"
    querystring = {'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "address-validation-correction-geocoding-and-information.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def geo_coordinates_timezone_information(lat: str, lon: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "From this endpoint you can retrieve the timezone offset and the timezone name with Geo-coordinates.
		Required Arguments:
		lat : Latitude
		lon : Longitude"
    
    """
    url = f"https://address-validation-correction-geocoding-and-information.p.rapidapi.com/api/v1/ask/timezone"
    querystring = {'lat': lat, 'lon': lon, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "address-validation-correction-geocoding-and-information.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def city_information(country: str, state: str, city: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "City Info Endpoint gives you informations like state, state code, timezone etc. about the given city.
		Required Arguments:
		city: City name
		country:  Country Name
		state: name of the state where the given city is located"
    
    """
    url = f"https://address-validation-correction-geocoding-and-information.p.rapidapi.com/api/v1/ask/city/info"
    querystring = {'country': country, 'state': state, 'city': city, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "address-validation-correction-geocoding-and-information.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download(filename: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "From this endpoint, you can download your processed CSV file.
		The processed CSV files are not available for download after 24 hours of their creation."
    
    """
    url = f"https://address-validation-correction-geocoding-and-information.p.rapidapi.com/api/v1/getfile/{filename}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "address-validation-correction-geocoding-and-information.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

