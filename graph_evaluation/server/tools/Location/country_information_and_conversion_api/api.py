import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def bycountryname(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Country Information and Conversion API
		
		The Country Information and Conversion API provides comprehensive country-related data and conversion capabilities, allowing developers to access detailed information about countries worldwide and seamlessly convert between country names and country codes.
		
		## Endpoints
		
		### Get Country Code by Name
		
		Retrieve the country code based on the country name.
		
		- **URL:** `/api/country/code/:name`
		- **Method:** `GET`
		- **Parameters:**
		  - `:name` - The name of the country.
		
		Example Request:
		```
		GET /api/country/code/United States
		```
		
		Example Response:
		```json
		{
		  "code": "US",
		  "name": "United States",
		  "capital": "Washington, D.C.",
		  "population": 331002651,
		  "language": "English",
		  "currency": "USD",
		  "timezone": "UTC-04:00, UTC-05:00"
		}
		```
		
		### Get Country Name by Code
		
		Retrieve the country name based on the country code.
		
		- **URL:** `/api/country/name/:code`
		- **Method:** `GET`
		- **Parameters:**
		  - `:code` - The country code.
		
		Example Request:
		```
		GET /api/country/name/US
		```
		
		Example Response:
		```json
		{
		  "code": "US",
		  "name": "United States",
		  "capital": "Washington, D.C.",
		  "population": 331002651,
		  "language": "English",
		  "currency": "USD",
		  "timezone": "UTC-04:00, UTC-05:00"
		}
		```
		
		## Additional Country Information
		
		In addition to the country code and name, both endpoints provide the following additional information about each country:
		
		- `capital`: The capital city of the country.
		- `population`: The population count of the country.
		- `language`: The official language(s) spoken in the country.
		- `currency`: The currency used in the country.
		- `timezone`: The time zone(s) observed in the country.
		
		Please note that the additional country information is included in the example responses for both endpoints.
		
		## Error Handling
		
		The API returns appropriate HTTP status codes and error responses to indicate various scenarios:
		
		- `200 OK`: Successful request and response.
		- `400 Bad Request`: Invalid request parameters or missing required parameters.
		- `404 Not Found`: Country not found or invalid country name/code.
		- `500 Internal Server Error`: An unexpected error occurred.
		
		Please make sure to handle these status codes appropriately in your application."
    
    """
    url = f"https://country-information-and-conversion-api1.p.rapidapi.com/api/v1/country/name/United%20States"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "country-information-and-conversion-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bycountrycode(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Country Information and Conversion API
		
		The Country Information and Conversion API provides comprehensive country-related data and conversion capabilities, allowing developers to access detailed information about countries worldwide and seamlessly convert between country names and country codes.
		
		## Endpoints
		
		### Get Country Code by Name
		
		Retrieve the country code based on the country name.
		
		- **URL:** `/api/country/code/:name`
		- **Method:** `GET`
		- **Parameters:**
		  - `:name` - The name of the country.
		
		Example Request:
		```
		GET /api/country/code/United States
		```
		
		Example Response:
		```json
		{
		  "code": "US",
		  "name": "United States",
		  "capital": "Washington, D.C.",
		  "population": 331002651,
		  "language": "English",
		  "currency": "USD",
		  "timezone": "UTC-04:00, UTC-05:00"
		}
		```
		
		### Get Country Name by Code
		
		Retrieve the country name based on the country code.
		
		- **URL:** `/api/country/name/:code`
		- **Method:** `GET`
		- **Parameters:**
		  - `:code` - The country code.
		
		Example Request:
		```
		GET /api/country/name/US
		```
		
		Example Response:
		```json
		{
		  "code": "US",
		  "name": "United States",
		  "capital": "Washington, D.C.",
		  "population": 331002651,
		  "language": "English",
		  "currency": "USD",
		  "timezone": "UTC-04:00, UTC-05:00"
		}
		```
		
		## Additional Country Information
		
		In addition to the country code and name, both endpoints provide the following additional information about each country:
		
		- `capital`: The capital city of the country.
		- `population`: The population count of the country.
		- `language`: The official language(s) spoken in the country.
		- `currency`: The currency used in the country.
		- `timezone`: The time zone(s) observed in the country.
		
		Please note that the additional country information is included in the example responses for both endpoints.
		
		## Error Handling
		
		The API returns appropriate HTTP status codes and error responses to indicate various scenarios:
		
		- `200 OK`: Successful request and response.
		- `400 Bad Request`: Invalid request parameters or missing required parameters.
		- `404 Not Found`: Country not found or invalid country name/code.
		- `500 Internal Server Error`: An unexpected error occurred.
		
		Please make sure to handle these status codes appropriately in your application."
    
    """
    url = f"https://country-information-and-conversion-api1.p.rapidapi.com/api/v1/country/code/US"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "country-information-and-conversion-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

