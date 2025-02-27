import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getcityfromipaddress(ip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the city name associated with the provided IP address. By utilizing the geoip library, it performs a lookup based on the IP address and returns the city name if it is found. If no city is associated with the IP address or an error occurs during the process, an appropriate error message will be returned."
    
    """
    url = f"https://ip-based-geolocation-and-city-lookup-api.p.rapidapi.com/city-from-ip/{ip}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ip-based-geolocation-and-city-lookup-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcoordinatesfromcity(city: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the geographical coordinates (latitude and longitude) of a city based on its name. This API uses Nominatim OpenStreetMap to search for the city and returns the corresponding coordinates. If the city is not found, the API will return an error message indicating that the city was not found. This endpoint is useful for applications that require mapping or location-based services, where converting a city name into its geographical coordinates is needed."
    
    """
    url = f"https://ip-based-geolocation-and-city-lookup-api.p.rapidapi.com/coords-from-city/{city}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ip-based-geolocation-and-city-lookup-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcityfromcoordinates(lon: str, lat: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is an API endpoint that retrieves the city name associated with the given latitude and longitude coordinates. The endpoint requires both latitude and longitude as query parameters. It uses the Nominatim API from OpenStreetMap to perform a reverse geocoding lookup to obtain the city information. If a city, town, or village is found for the given coordinates, the name is returned in the response. In case of errors or if the city is not found, appropriate error messages are returned. This endpoint is useful for applications that need to display location-based information or provide services specific to a user's geographic location, such as weather services, local news, or nearby points of interest."
    
    """
    url = f"https://ip-based-geolocation-and-city-lookup-api.p.rapidapi.com/city-from-coords/{lat}/{lon}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ip-based-geolocation-and-city-lookup-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcityfromproviderip(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is an API endpoint that retrieves the city associated with the client's IP address, as provided by their internet service provider (ISP). It first obtains the client's IP address, converts it to IPv4 if necessary, and then uses the geoip-lite library to look up the location information. If a city is found for the given IP address, the city name is returned in the response. In case of errors or if the city is not found, appropriate error messages are returned. This endpoint is useful for obtaining location-based information for users based on their IP addresses, such as for personalizing content or providing targeted services."
    
    """
    url = f"https://ip-based-geolocation-and-city-lookup-api.p.rapidapi.com/city-from-provider-ip"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ip-based-geolocation-and-city-lookup-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcoordinatesfromproviderip(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The 'GetCoordinatesFromProviderIP' endpoint retrieves the latitude and longitude coordinates of a user based on their IP address. The IP address is automatically detected, and the corresponding geolocation data is returned in the response. If the coordinates cannot be found or an error occurs, an appropriate error message is returned."
    
    """
    url = f"https://ip-based-geolocation-and-city-lookup-api.p.rapidapi.com/coords-from-provider-ip"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ip-based-geolocation-and-city-lookup-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

