import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def transactions(zipcode: str, page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Show transactions given a zipcode. The API returns 50 records each time in descending order by date and supports pagination. This API requires a subscription cost.
		
		The returned data shows the summary of information. To get further detail (such as the unit of a building that posts the most recent transaction, use summary API.
		
		Sample data of the API when query zipcode 10019 is:
		 "data": [
		    {
		      "_id": {
		        "STREET_NUMBER": "310",
		        "STREET": "W 56th St",
		        "PROPERTY_TYPE": "SINGLE RESIDENTIAL COOP UNIT",
		        "ZIPCODE": "10019",
		        "STATE": "NY"
		      },
		      "lastSalesDate": "2021-07-21T00:00:00.000Z",
		      "lastSalesAmount": 514000,
		      "maxSalesAmount": 1359000,
		      "TotalRecords": 74
		    },"
    
    """
    url = f"https://real-estate-records.p.rapidapi.com/search/zipcode"
    querystring = {'zipcode': zipcode, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-estate-records.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def detail(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all detailed historical transactions of a given address. To call this API, it requires to call summary API first using address information. Then use the Id value returned from the summary API and call this API with the same Id. This API requires a subscription cost."
    
    """
    url = f"https://real-estate-records.p.rapidapi.com/search/detail"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-estate-records.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def summary(number: str, street: str, zipcode: str='10019', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a summary of property records by units given an address (if it's a multi-unit building). Street number, name and zipcode are needed to match records. The street names are normalized using Here.com conventions. This API requires a subscription cost. 
		
		This API provides details a level further than the "address" endpoint. For instance,  for 220 Central Park S, New York, NY 10019 it returns
		  "data": [
		    {
		      "_id": {
		        "id": "6144a17b3afc5ca06ea0dd95",
		        "Loc": {
		          "coordinates": [
		            -73.9802,
		            40.76719
		          ],
		          "type": "Point"
		        },
		        "STREET_NUMBER": "220",
		        "STREET": "Central Park S",
		        "UNIT": "67",
		        "PROPERTY_TYPE": "SINGLE RESIDENTIAL CONDO UNIT",
		        "ZIPCODE": "10019",
		        "STATE": "NY"
		      },
		      "lastSalesDate": "2021-05-24T00:00:00.000Z",
		      "lastSalesAmount": 59500000,
		      "maxSalesAmount": 59500000,
		      "SQFT": 0,
		      "TotalRecords": 1
		    },
		
		The API is available for public records for New York City 5 boroughs, all New Jersey Counties and DC. 
		Sources of information are:
		NJ: http://tax1.co.monmouth.nj.us/cgi-bin/prc6.cgi?&ms_user=monm&passwd=&srch_type=0&adv=0&out_type=0&district=0200
		NYC Acris: https://a836-acris.nyc.gov/DS/DocumentSearch/Index"
    
    """
    url = f"https://real-estate-records.p.rapidapi.com/search/summary"
    querystring = {'number': number, 'street': street, }
    if zipcode:
        querystring['zipcode'] = zipcode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-estate-records.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def address(street: str, number: str, zipcode: str='10019', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a high level of summary of property records given an address. Street number, name and zipcode are needed to match records. The street names are normalized using Here.com conventions. **This API is free to use.**
		
		If it's an apartment building, it will return total records available for every property type. For instance,  for 220 Central Park S, New York, NY 10019,  it returns:
		  "data": [
		    {
		      "_id": {
		        "STREET_NUMBER": "220",
		        "STREET": "Central Park S",
		        "PROPERTY_TYPE": "SINGLE RESIDENTIAL CONDO UNIT",
		        "ZIPCODE": "10019",
		        "STATE": "NY"
		      },
		      "Location": {
		        "coordinates": [
		          -73.9802,
		          40.76719
		        ],
		        "type": "Point"
		      },
		      "lastSalesDate": "2021-05-24T00:00:00.000Z",
		      "lastSalesAmount": 59500000,
		      "maxSalesAmount": 239958219.15,
		      "TotalRecords": 100
		    },
		
		The API is available for public records for New York City 5 boroughs, all New Jersey Counties and DC. 
		Sources of information are:
		NJ: http://tax1.co.monmouth.nj.us/cgi-bin/prc6.cgi?&ms_user=monm&passwd=&srch_type=0&adv=0&out_type=0&district=0200
		NYC Acris: https://a836-acris.nyc.gov/DS/DocumentSearch/Index"
    
    """
    url = f"https://real-estate-records.p.rapidapi.com/search/address"
    querystring = {'street': street, 'number': number, }
    if zipcode:
        querystring['zipcode'] = zipcode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-estate-records.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

