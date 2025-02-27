import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_rent_a_car(pickuptime: str, returndate: str, returntime: str, pickupdate: str, pickupid: str, market: str='UK', driverage: int=None, locale: str='en-GB', currency: str='EUR', returnid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Start and poll rent-a-car search with prices"
    pickuptime: Pickup time
        returndate: Return date
        returntime: Return time
        pickupdate: Pickup date
        pickupid: Pickup location entity id
        market: User country, default UK
        driverage: Driver's age
Enter if age < 25 or age > 75
        locale: User locale, default en-GB
        currency: Price currency
        returnid: Return location entity id
        
    """
    url = f"https://skyscanner44.p.rapidapi.com/search-rentacar"
    querystring = {'pickupTime': pickuptime, 'returnDate': returndate, 'returnTime': returntime, 'pickupDate': pickupdate, 'pickupId': pickupid, }
    if market:
        querystring['market'] = market
    if driverage:
        querystring['driverAge'] = driverage
    if locale:
        querystring['locale'] = locale
    if currency:
        querystring['currency'] = currency
    if returnid:
        querystring['returnId'] = returnid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner44.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_all_flights(adults: int, departuredate: str, origin: str, destination: str, childage8: int=None, returnarriveto: str='23:59', startfrom: str='00:00', childage7: int=None, returnstartfrom: str='00:00', stops: str='0,1,2', arriveto: str='23:59', market: str='UK', locale: str='en-GB', duration: int=50, childage3: int=None, childage1: int=None, childage6: int=None, childage2: int=None, childage5: int=None, cabinclass: str=None, childage4: int=None, currency: str='EUR', returndate: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Start and poll all flights search with prices"
    adults: Number of adult passengers (16 and over)
Allowed values: 1-8
        departuredate: Departure date 
Format: YYYY-MM-DD
        origin: Departure airport IATA code
        destination: Destination airport IATA code
        childage8: Age of eighth child
Allowed values: 0- 15
        returnarriveto: Filter by return flight arrival time (arrival before)
        startfrom: Filter by outbound flight start time (depart after)
        childage7: Age of seventh child
Allowed values: 0- 15
        returnstartfrom: Filter by return flight start time (depart after)
        stops: Filter by number of stops per leg, separated by comma. Possible values 0, 1 and 2 (2 means two or more stops)
        arriveto: Filter by outbound flight arrival time (arrival before)
        market: User country, default UK
        locale: User locale, default en-GB
        duration: Filter by maximum duration of journeys in hours
        childage3: Age of third child
Allowed values: 0- 15
        childage1: Age of first child
Allowed values: 0- 15
        childage6: Age of sixth child
Allowed values: 0- 15
        childage2: Age of second child
Allowed values: 0- 15
        childage5: Age of fifth child
Allowed values: 0- 15
        childage4: Age of fourth child
Allowed values: 0- 15
        returndate: Return date
Format: YYYY-MM-DD
        
    """
    url = f"https://skyscanner44.p.rapidapi.com/search-extended"
    querystring = {'adults': adults, 'departureDate': departuredate, 'origin': origin, 'destination': destination, }
    if childage8:
        querystring['childAge8'] = childage8
    if returnarriveto:
        querystring['returnArriveTo'] = returnarriveto
    if startfrom:
        querystring['startFrom'] = startfrom
    if childage7:
        querystring['childAge7'] = childage7
    if returnstartfrom:
        querystring['returnStartFrom'] = returnstartfrom
    if stops:
        querystring['stops'] = stops
    if arriveto:
        querystring['arriveTo'] = arriveto
    if market:
        querystring['market'] = market
    if locale:
        querystring['locale'] = locale
    if duration:
        querystring['duration'] = duration
    if childage3:
        querystring['childAge3'] = childage3
    if childage1:
        querystring['childAge1'] = childage1
    if childage6:
        querystring['childAge6'] = childage6
    if childage2:
        querystring['childAge2'] = childage2
    if childage5:
        querystring['childAge5'] = childage5
    if cabinclass:
        querystring['cabinClass'] = cabinclass
    if childage4:
        querystring['childAge4'] = childage4
    if currency:
        querystring['currency'] = currency
    if returndate:
        querystring['returnDate'] = returndate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner44.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_best_flights(origin: str, departuredate: str, adults: int, destination: str, childage7: int=None, market: str='UK', childage5: int=None, childage3: int=None, childage6: int=None, childage8: int=None, locale: str='en-GB', childage4: int=None, childage2: int=None, returndate: str=None, cabinclass: str=None, childage1: int=None, currency: str='EUR', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Start and poll best flights search with prices"
    origin: Departure airport IATA code
        departuredate: Departure date 
Format: YYYY-MM-DD
        adults: Number of adult passengers (16 and over)
Allowed values: 1-8
        destination: Destination airport IATA code
        childage7: Age of seventh child
Allowed values: 0- 15
        market: User country, default UK
        childage5: Age of fifth child
Allowed values: 0- 15
        childage3: Age of third child
Allowed values: 0- 15
        childage6: Age of sixth child
Allowed values: 0- 15
        childage8: Age of eighth child
Allowed values: 0- 15
        locale: User locale, default en-GB
        childage4: Age of fourth child
Allowed values: 0- 15
        childage2: Age of second child
Allowed values: 0- 15
        returndate: Return date
Format: YYYY-MM-DD
        childage1: Age of first child
Allowed values: 0- 15
        
    """
    url = f"https://skyscanner44.p.rapidapi.com/search"
    querystring = {'origin': origin, 'departureDate': departuredate, 'adults': adults, 'destination': destination, }
    if childage7:
        querystring['childAge7'] = childage7
    if market:
        querystring['market'] = market
    if childage5:
        querystring['childAge5'] = childage5
    if childage3:
        querystring['childAge3'] = childage3
    if childage6:
        querystring['childAge6'] = childage6
    if childage8:
        querystring['childAge8'] = childage8
    if locale:
        querystring['locale'] = locale
    if childage4:
        querystring['childAge4'] = childage4
    if childage2:
        querystring['childAge2'] = childage2
    if returndate:
        querystring['returnDate'] = returndate
    if cabinclass:
        querystring['cabinClass'] = cabinclass
    if childage1:
        querystring['childAge1'] = childage1
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner44.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def flights_to_country(departuredate: str, origin: str, destination: str, market: str='UK', locale: str='en-GB', returndate: str='2023-07-21', currency: str='EUR', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search flights from an airport to a country. API returns estimated lowest prices only, found in the last 8 days."
    departuredate: Departure date 
Format: YYYY-MM-DD
        origin: Departure airport IATA code
        destination: Destination country ISO 3166-2 code
        market: User country, default UK
        locale: User locale, default en-GB
        returndate: Return date
Format: YYYY-MM-DD
        currency: Price currency, default EUR
        
    """
    url = f"https://skyscanner44.p.rapidapi.com/fly-to-country"
    querystring = {'departureDate': departuredate, 'origin': origin, 'destination': destination, }
    if market:
        querystring['market'] = market
    if locale:
        querystring['locale'] = locale
    if returndate:
        querystring['returnDate'] = returndate
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner44.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def flights_anywhere(origin: str, departuredate: str, market: str='UK', locale: str='en-GB', currency: str='EUR', returndate: str='2023-07-21', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search flights from an airport to anywhere. API returns estimated lowest prices only, found in the last 8 days."
    origin: Departure airport IATA code
        departuredate: Departure date 
Format: YYYY-MM-DD
        market: User country, default UK
        locale: User locale, default en-GB
        currency: Price currency, default EUR
        returndate: Return date
Format: YYYY-MM-DD
        
    """
    url = f"https://skyscanner44.p.rapidapi.com/fly-anywhere"
    querystring = {'origin': origin, 'departureDate': departuredate, }
    if market:
        querystring['market'] = market
    if locale:
        querystring['locale'] = locale
    if currency:
        querystring['currency'] = currency
    if returndate:
        querystring['returnDate'] = returndate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner44.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_location_data_for_hotel_search(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get location data (entity id, class, name, location)
		Entity_id is used as a locationId in hotels search"
    query: Query string
        
    """
    url = f"https://skyscanner44.p.rapidapi.com/autocomplete-hotel"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner44.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_hotel_rooms(rooms: int, checkin: str, locationid: str, adults: int, checkout: str, childage4: int=None, locale: str='en-GB', market: str='UK', page: int=None, currency: str='EUR', childage1: int=None, chidlage3: int=None, childage2: int=None, childage5: int=None, sortby: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Start and poll hotel room search with prices"
    rooms: Number of rooms
Allowed values: 1-5
        checkin: Checkin date
Format YYYY-MM-DD
        locationid: Location entity id
        adults: Number of adult guests (18 and over)
Allowed values: 1-10
        checkout: Checkout date
Format YYYY-MM-DD
        childage4: Age of fourth child
Allowed values: 0-17
        locale: User locale, default en-GB
        market: User country, default UK
        page: Number of result page
        currency: Price currency
        childage1: Age of first child
Allowed values: 0-17
        chidlage3: Age of third child
Allowed values: 0-17
        childage2: Age of second child
Allowed values: 0-17
        childage5: Age of fifth child
Allowed values: 0-17
        
    """
    url = f"https://skyscanner44.p.rapidapi.com/search-hotel"
    querystring = {'rooms': rooms, 'checkin': checkin, 'locationId': locationid, 'adults': adults, 'checkout': checkout, }
    if childage4:
        querystring['childAge4'] = childage4
    if locale:
        querystring['locale'] = locale
    if market:
        querystring['market'] = market
    if page:
        querystring['page'] = page
    if currency:
        querystring['currency'] = currency
    if childage1:
        querystring['childAge1'] = childage1
    if chidlage3:
        querystring['chidlAge3'] = chidlage3
    if childage2:
        querystring['childAge2'] = childage2
    if childage5:
        querystring['childAge5'] = childage5
    if sortby:
        querystring['sortBy'] = sortby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner44.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_location_data_for_rent_a_car_search(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get location data (entity id, class, name, location)
		Entity_id is used as a pickupId and returnId in car rental search"
    query: Query string
        
    """
    url = f"https://skyscanner44.p.rapidapi.com/autocomplete-rentacar"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner44.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_airport_data(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get airport data (IATA code, name, location)"
    query: Query string
        
    """
    url = f"https://skyscanner44.p.rapidapi.com/autocomplete"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner44.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

