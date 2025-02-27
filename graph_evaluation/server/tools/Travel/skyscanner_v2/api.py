import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_similar_hotels(hotelid: str, adults: int, checkin: str, checkout: str, price: str=None, waittime: int=2000, sorting: str=None, stars: str=None, maxprice: int=None, minprice: int=None, mealplan: str=None, searchid: str=None, guesttype: str=None, chain: str=None, amenities: str=None, discounts: str=None, propertytype: str=None, rating: str=None, cancellation: str=None, market: str='en-US', countrycode: str='US', currency: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the similar hotels based on the selected hotel."
    hotelid: **Hotel ID** can be retrieved from **Search Hotel** or **Get Hotel Details** API in **Hotels** collection.
        adults: Number of adult guests (with age 18 and over)
        checkin: Checkin Date
Format: YYYY-MM-DD
        checkout: Checkout Date
Format: YYYY-MM-DD
        price: **total** - Total price for the stay.
**per**   - Price per night
        waittime: Wait time in milliseconds. Sometimes the API needs **wait times** or more than **one** API call to get **complete data**. 
        stars: The Hotel **star** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        maxprice: Maximum cost filter to stay in the hotel.
        minprice: Minimum cost filter to stay in the hotel.
        mealplan: The Hotel **mealPlan** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        searchid: **searchId** can be retrieved from **Search Hotel** API in **Hotels** collection.
        guesttype: The Hotel **guestType** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        chain: The Hotel **chain** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        amenities: The Hotel **amenities** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        discounts: The Hotel **discounts** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        propertytype: The Hotel **propertyType** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        rating: The Hotel **rating** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        cancellation: The Hotel **cancellation** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        market: The list of available **market** can be retrieved through the **Get Config** API in the **Config** collection.
        countrycode: The list of available **countryCode** can be retrieved through the **Get Config** API in the **Config** collection.
        currency: The list of available **currencies** can be retrieved through the **Get Config** API in the **Config** collection.
        
    """
    url = f"https://skyscanner50.p.rapidapi.com/api/v1/getSimilarHotels"
    querystring = {'hotelId': hotelid, 'adults': adults, 'checkin': checkin, 'checkout': checkout, }
    if price:
        querystring['price'] = price
    if waittime:
        querystring['waitTime'] = waittime
    if sorting:
        querystring['sorting'] = sorting
    if stars:
        querystring['stars'] = stars
    if maxprice:
        querystring['maxPrice'] = maxprice
    if minprice:
        querystring['minPrice'] = minprice
    if mealplan:
        querystring['mealPlan'] = mealplan
    if searchid:
        querystring['searchId'] = searchid
    if guesttype:
        querystring['guestType'] = guesttype
    if chain:
        querystring['chain'] = chain
    if amenities:
        querystring['amenities'] = amenities
    if discounts:
        querystring['discounts'] = discounts
    if propertytype:
        querystring['propertyType'] = propertytype
    if rating:
        querystring['rating'] = rating
    if cancellation:
        querystring['cancellation'] = cancellation
    if market:
        querystring['market'] = market
    if countrycode:
        querystring['countryCode'] = countrycode
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner50.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_flights_filter_by_agent_id(destination: str, origin: str, date: str, agentid: str, returndate: str=None, cabinclass: str=None, children: int=None, countrycode: str='US', adults: int=1, infants: int=None, currency: str='USD', filter: str=None, market: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get and poll all flights, search between origin and destination with prices, filter, sorting for specific agent id."
    destination: Destination airport **IATA** code. The **IATA** code can be extracted from the **Search Airport** API in the **Flights** collection.
        origin: Origin airport **IATA** code. The **IATA** code can be extracted from the **Search Airport** API in the **Flights** collection.
        date: Departure or travel date.
Format: YYYY-MM-DD
        returndate: Return date.
Format: YYYY-MM-DD
        children: Number of Childrens (with age between 2-12 years)
        countrycode: The list of available **countryCode** can be retrieved through the **Get Config** API in the **Config** collection.
        adults: Number of Adults (with age 18 and over)
Default Value: 1
        infants: Number of Infants (with age < 2 years)
        currency: The list of available **currencies** can be retrieved through the **Get Config** API in the **Config** collection.
        market: The list of available **market** can be retrieved through the **Get Config** API in the **Config** collection.
        
    """
    url = f"https://skyscanner50.p.rapidapi.com/api/v1/searchFlightsByAgent"
    querystring = {'destination': destination, 'origin': origin, 'date': date, 'agentId': agentid, }
    if returndate:
        querystring['returnDate'] = returndate
    if cabinclass:
        querystring['cabinClass'] = cabinclass
    if children:
        querystring['children'] = children
    if countrycode:
        querystring['countryCode'] = countrycode
    if adults:
        querystring['adults'] = adults
    if infants:
        querystring['infants'] = infants
    if currency:
        querystring['currency'] = currency
    if filter:
        querystring['filter'] = filter
    if market:
        querystring['market'] = market
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner50.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_flights_multi_stops(legs: str, infants: int=None, waittime: int=5000, filter: str=None, market: str='en-US', children: int=None, cabinclass: str=None, countrycode: str='US', adults: int=1, currency: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get and poll all flights. Searching the flights between multiple locations with prices, filter, sorting etc."
    legs: The legs must contain the **origin**, **destination** and **date** in object format and must be passed in an array.
EXAMPLE:
[
  {
    'origin': 'LOND',
    'destination': 'NYCA',
    'date': '2023-02-07'
  },
  ...
]
**Note:** If there are multiple stops, there should be more leg objects on the board.
        infants: Number of Infants (with age < 2 years)
        waittime: Wait time in milliseconds. Sometimes the API needs **wait times** or more than **one** API call to get **complete data**. 
        market: The list of available **market** can be retrieved through the **Get Config** API in the **Config** collection.
        children: Number of Childrens (with age between 2-12 years)
        countrycode: The list of available **countryCode** can be retrieved through the **Get Config** API in the **Config** collection.
        adults: Number of Adults (with age 18 and over)
Default Value: 1
        currency: The list of available **currencies** can be retrieved through the **Get Config** API in the **Config** collection.
        
    """
    url = f"https://skyscanner50.p.rapidapi.com/api/v1/searchFlightsMultiStops"
    querystring = {'legs': legs, }
    if infants:
        querystring['infants'] = infants
    if waittime:
        querystring['waitTime'] = waittime
    if filter:
        querystring['filter'] = filter
    if market:
        querystring['market'] = market
    if children:
        querystring['children'] = children
    if cabinclass:
        querystring['cabinClass'] = cabinclass
    if countrycode:
        querystring['countryCode'] = countrycode
    if adults:
        querystring['adults'] = adults
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner50.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_cars(pickupentityid: int, pickupdate: str, pickuptime: str, dropoffentityid: int=None, dropofftime: str=None, currency: str=None, driverage: int=None, market: str=None, countrycode: str=None, dropoffdate: str=None, waittime: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this API for searching the cars for hire based on location."
    pickupentityid: **pickUpEntityId** can be retrieved from **Search Location** API in **Cars** collection as **entity_id**
        pickupdate: Pick Up Date
        pickuptime: Pick Up Time
        dropoffentityid: **dropOffEntityId** can be retrieved from **Search Location** API  in **Cars** collection **entity_id**
        dropofftime: Drop Off Time
        driverage: Driver's age (21 years and over)
        dropoffdate: Drop Off Date
        waittime: Wait time in milliseconds. Sometimes the API needs **wait times** or more than **one** API call to get **complete data**.
        
    """
    url = f"https://skyscanner50.p.rapidapi.com/api/v1/searchCars"
    querystring = {'pickUpEntityId': pickupentityid, 'pickUpDate': pickupdate, 'pickUpTime': pickuptime, }
    if dropoffentityid:
        querystring['dropOffEntityId'] = dropoffentityid
    if dropofftime:
        querystring['dropOffTime'] = dropofftime
    if currency:
        querystring['currency'] = currency
    if driverage:
        querystring['driverAge'] = driverage
    if market:
        querystring['market'] = market
    if countrycode:
        querystring['countryCode'] = countrycode
    if dropoffdate:
        querystring['dropOffDate'] = dropoffdate
    if waittime:
        querystring['waitTime'] = waittime
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner50.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_location(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this API to get EntityId for the pickup and drop location."
    query: Location name for pickup/drop off.
        
    """
    url = f"https://skyscanner50.p.rapidapi.com/api/v1/searchLocation"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner50.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_recommend_hotels(searchid: str, checkout: str, hotelid: str, entityid: str, checkin: str, adults: int, market: str='en-US', currency: str='USD', waittime: int=2000, countrycode: str='US', maxprice: int=None, chain: str=None, rating: str=None, minprice: int=None, stars: str=None, guesttype: str=None, cancellation: str=None, mealplan: str=None, discounts: str=None, propertytype: str=None, amenities: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this API to get recommend hotel based on the hotel selected."
    searchid: **searchId** can be retrieved from **Search Hotel** API in **Hotels** collection.
        checkout: Checkout Date
Format: YYYY-MM-DD
        hotelid: **Hotel ID** can be retrieved from **Search Hotel** or **Get Hotel Details** API in **Hotels** collection.
        entityid: **entityId** can be retrieved from **Search Hotel** API in **Hotels** collection.
        checkin: Checkin Date
Format: YYYY-MM-DD
        adults: Number of adult guests (With age 18 and over)
        market: The list of available **market** can be retrieved through the **Get Config** API in the **Config** collection.
        currency: The list of available **currencies** can be retrieved through the **Get Config** API in the **Config** collection.
        waittime: Wait time in milliseconds. Sometimes the API needs **wait times** or more than **one** API call to get **complete data**. 
        countrycode: The list of available **countryCode** can be retrieved through the **Get Config** API in the **Config** collection.
        maxprice: Maximum cost filter to stay in the hotel.
        chain: The Hotel **chain** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        rating: The Hotel **rating** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        minprice: Minimum cost filter to stay in the hotel.
        stars: The Hotel **star** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        guesttype: The Hotel **guestType** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        cancellation: The Hotel **cancellation** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        mealplan: The Hotel **mealPlan** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        discounts: The Hotel **discounts** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        propertytype: The Hotel **propertyType** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        amenities: The Hotel **amenities** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        
    """
    url = f"https://skyscanner50.p.rapidapi.com/api/v1/getRecommendHotels"
    querystring = {'searchId': searchid, 'checkout': checkout, 'hotelId': hotelid, 'entityId': entityid, 'checkin': checkin, 'adults': adults, }
    if market:
        querystring['market'] = market
    if currency:
        querystring['currency'] = currency
    if waittime:
        querystring['waitTime'] = waittime
    if countrycode:
        querystring['countryCode'] = countrycode
    if maxprice:
        querystring['maxPrice'] = maxprice
    if chain:
        querystring['chain'] = chain
    if rating:
        querystring['rating'] = rating
    if minprice:
        querystring['minPrice'] = minprice
    if stars:
        querystring['stars'] = stars
    if guesttype:
        querystring['guestType'] = guesttype
    if cancellation:
        querystring['cancellation'] = cancellation
    if mealplan:
        querystring['mealPlan'] = mealplan
    if discounts:
        querystring['discounts'] = discounts
    if propertytype:
        querystring['propertyType'] = propertytype
    if amenities:
        querystring['amenities'] = amenities
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner50.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_things_to_do(entityid: str, lat: str, lng: str, market: str='en-US', currency: str='USD', countrycode: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this API to get Thing  to do nearby the hotel selected."
    entityid: **entityId** can be retrieved from **Search Hotel** API in **Hotels** collection.
        lat: **lat** can be retrieved from **Hotel Details** or **Search Hotel** API in **Hotels** collection as **latitude**.
        lng: **lng** can be retrieved from **Hotel Details** or **Search Hotel** API in **Hotels** collection as **longitude**.
        market: The list of available **market** can be retrieved through the **Get Config** API in the **Config** collection.
        currency: The list of available **currencies** can be retrieved through the **Get Config** API in the **Config** collection.
        countrycode: The list of available **countryCode** can be retrieved through the **Get Config** API in the **Config** collection.
        
    """
    url = f"https://skyscanner50.p.rapidapi.com/api/v1/getThingsToDo"
    querystring = {'entityId': entityid, 'lat': lat, 'lng': lng, }
    if market:
        querystring['market'] = market
    if currency:
        querystring['currency'] = currency
    if countrycode:
        querystring['countryCode'] = countrycode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner50.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_hotel_reviews(hotelid: str, searchid: str=None, currency: str='USD', countrycode: str='US', offset: int=None, market: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the review related to the hotel selected."
    hotelid: **Hotel ID** can be retrieved from **Search Hotel** or **Get Hotel Details** API in **Hotels** collection.
        searchid: **searchId** can be retrieved from **Search Hotel** API in **Hotels** collection.
        currency: The list of available **currencies** can be retrieved through the **Get Config** API in the **Config** collection.
        countrycode: The list of available **countryCode** can be retrieved through the **Get Config** API in the **Config** collection.
        market: The list of available **market** can be retrieved through the **Get Config** API in the **Config** collection.
        
    """
    url = f"https://skyscanner50.p.rapidapi.com/api/v1/getHotelReviews"
    querystring = {'hotelId': hotelid, }
    if searchid:
        querystring['searchId'] = searchid
    if currency:
        querystring['currency'] = currency
    if countrycode:
        querystring['countryCode'] = countrycode
    if offset:
        querystring['offset'] = offset
    if market:
        querystring['market'] = market
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner50.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_hotel_price(checkin: str, checkout: str, adults: int, entityid: str, hotelid: str, childrenages: str=None, market: str='en-US', currency: str='USD', countrycode: str='US', searchid: str=None, minprice: int=None, stars: str=None, guesttype: str=None, discounts: str=None, maxprice: int=None, mealplan: str=None, propertytype: str=None, rating: str=None, cancellation: str=None, amenities: str=None, chain: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the detail price for the hotel and room selected."
    checkin: Checkin Date
Format: YYYY-MM-DD
        checkout: Checkout Date
Format: YYYY-MM-DD
        adults: Number of adult guests (with age 18 and over)
        entityid: **entityId** can be retrieved from **Search Hotel** API in **Hotels** collection.
        hotelid: **Hotel ID** can be retrieved from **Search Hotel** or **Get Hotel Details** API in **Hotels** collection.
        childrenages: Number of children (with age between 0 and 17)
Example:
If 1st child's age is 11 months and 2nd child's age is 10 years, then the parameter should be passed as [0,10]
        market: The list of available **market** can be retrieved through the **Get Config** API in the **Config** collection.
        currency: The list of available **currencies** can be retrieved through the **Get Config** API in the **Config** collection.
        countrycode: The list of available **countryCode** can be retrieved through the **Get Config** API in the **Config** collection.
        searchid: **searchId** can be retrieved from **Search Hotel** API in **Hotels** collection.
        minprice: Minimum cost filter to stay in the hotel.
        stars: The Hotel **star** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        guesttype: The Hotel **guestType** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        discounts: The Hotel **discounts** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        maxprice: Maximum cost filter to stay in the hotel.
        mealplan: The Hotel **mealPlan** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        propertytype: The Hotel **propertyType** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        rating: The Hotel **rating** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        cancellation: The Hotel **cancellation** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        amenities: The Hotel **amenities** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        chain: The Hotel **chain** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        
    """
    url = f"https://skyscanner50.p.rapidapi.com/api/v1/getHotelPrice"
    querystring = {'checkin': checkin, 'checkout': checkout, 'adults': adults, 'entityId': entityid, 'hotelId': hotelid, }
    if childrenages:
        querystring['childrenAges'] = childrenages
    if market:
        querystring['market'] = market
    if currency:
        querystring['currency'] = currency
    if countrycode:
        querystring['countryCode'] = countrycode
    if searchid:
        querystring['searchId'] = searchid
    if minprice:
        querystring['minPrice'] = minprice
    if stars:
        querystring['stars'] = stars
    if guesttype:
        querystring['guestType'] = guesttype
    if discounts:
        querystring['discounts'] = discounts
    if maxprice:
        querystring['maxPrice'] = maxprice
    if mealplan:
        querystring['mealPlan'] = mealplan
    if propertytype:
        querystring['propertyType'] = propertytype
    if rating:
        querystring['rating'] = rating
    if cancellation:
        querystring['cancellation'] = cancellation
    if amenities:
        querystring['amenities'] = amenities
    if chain:
        querystring['chain'] = chain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner50.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_hotel_details(hotelid: str, countrycode: str='US', currency: str='USD', market: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the details of the hotel, room, amenities, reviews etc. based on a unique Id selected."
    hotelid: **Hotel ID** can be retrieved from **Search Hotel** API in **Hotels** collection.
        countrycode: The list of available **countryCode** can be retrieved through the **Get Config** API in the **Config** collection.
        currency: The list of available **currencies** can be retrieved through the **Get Config** API in the **Config** collection.
        market: The list of available **market** can be retrieved through the **Get Config** API in the **Config** collection.
        
    """
    url = f"https://skyscanner50.p.rapidapi.com/api/v1/getHotelDetails"
    querystring = {'hotelId': hotelid, }
    if countrycode:
        querystring['countryCode'] = countrycode
    if currency:
        querystring['currency'] = currency
    if market:
        querystring['market'] = market
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner50.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_hotel(entityid: str, checkin: str, checkout: str, currency: str='USD', countrycode: str='US', market: str='en-US', waittime: int=2000, stars: str=None, adults: int=None, price: str=None, maxprice: int=None, minprice: int=None, cancellation: str=None, amenities: str=None, guesttype: str=None, propertytype: str=None, chain: str=None, discounts: str=None, sorting: str=None, mealplan: str=None, rating: str=None, childrenages: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get and poll hotel room search with prices, filter etc."
    entityid: **entityId** can be retrieved from **Search Hotel** API in **Hotels** collection.
        checkin: Checkin Date
Format: YYYY-MM-DD
        checkout: Checkout Date
Format: YYYY-MM-DD
        currency: The list of available **currencies** can be retrieved through the **Get Config** API in the **Config** collection.
        countrycode: The list of available **countryCode** can be retrieved through the **Get Config** API in the **Config** collection.
        market: The list of available **market** can be retrieved through the **Get Config** API in the **Config** collection.
        waittime: Wait time in milliseconds. Sometimes the API needs **wait times** or more than **one** API call to get **complete data**. 
        stars: The Hotel **star** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        adults: Number of adult guests (with age 18 and over)
        price: **total** - Total price for the stay.
**per**   - Price per night
        maxprice: Maximum cost filter to stay in the hotel.
        minprice: Minimum cost filter to stay in the hotel.
        cancellation: The Hotel **cancellation** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        amenities: The Hotel **amenities** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        guesttype: The Hotel **guestType** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        propertytype: The Hotel **propertyType** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        chain: The Hotel **chain** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        discounts: The Hotel **discounts** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        mealplan: The Hotel **mealPlan** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        rating: The Hotel **rating** filter range is retrievable from the **Get Hotel Filters** API in **Hotels** collection.
        childrenages: Number of children (with age between 0 and 17)
Example:
If 1st child's age is 11 months and 2nd child's age is 10 years, then the parameter should be passed as [0,10]
        
    """
    url = f"https://skyscanner50.p.rapidapi.com/api/v1/searchHotel"
    querystring = {'entityId': entityid, 'checkin': checkin, 'checkout': checkout, }
    if currency:
        querystring['currency'] = currency
    if countrycode:
        querystring['countryCode'] = countrycode
    if market:
        querystring['market'] = market
    if waittime:
        querystring['waitTime'] = waittime
    if stars:
        querystring['stars'] = stars
    if adults:
        querystring['adults'] = adults
    if price:
        querystring['price'] = price
    if maxprice:
        querystring['maxPrice'] = maxprice
    if minprice:
        querystring['minPrice'] = minprice
    if cancellation:
        querystring['cancellation'] = cancellation
    if amenities:
        querystring['amenities'] = amenities
    if guesttype:
        querystring['guestType'] = guesttype
    if propertytype:
        querystring['propertyType'] = propertytype
    if chain:
        querystring['chain'] = chain
    if discounts:
        querystring['discounts'] = discounts
    if sorting:
        querystring['sorting'] = sorting
    if mealplan:
        querystring['mealPlan'] = mealplan
    if rating:
        querystring['rating'] = rating
    if childrenages:
        querystring['childrenAges'] = childrenages
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner50.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_hotel_filters(adults: int, checkout: str, entityid: str, checkin: str, childrenages: str=None, waittime: int=2000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API is used to get all the available filters for the place to visit."
    adults: Number of adult guests (with age 18 and over)
        checkout: Checkout Date
Format: YYYY-MM-DD
        entityid: **entityId** can be retrieved from **Search Hotel** API in **Hotels** collection.
        checkin: Checkin Date
Format: YYYY-MM-DD
        childrenages: Number of children (with age between 0 and 17)
Example:
If 1st child's age is 11 months and 2nd child's age is 10 years, then the parameter should be passed as [0,10]
        waittime: Wait time in milliseconds. Sometimes the API needs **wait times** or more than **one** API call to get **complete data**. 
        
    """
    url = f"https://skyscanner50.p.rapidapi.com/api/v1/getHotelFilters"
    querystring = {'adults': adults, 'checkout': checkout, 'entityId': entityid, 'checkin': checkin, }
    if childrenages:
        querystring['childrenAges'] = childrenages
    if waittime:
        querystring['waitTime'] = waittime
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner50.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_place(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a place to get the **entityId** needed in searching the hotel API."
    query: Location Name in which you want to visit.
        
    """
    url = f"https://skyscanner50.p.rapidapi.com/api/v1/searchPlace"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner50.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_flights(date: str, origin: str, destination: str, infants: int=None, cabinclass: str=None, returndate: str=None, adults: int=1, currency: str='USD', children: int=None, filter: str=None, market: str='en-US', countrycode: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get and poll all flights, search between origin and destination with prices, filter, sorting."
    date: Departure or travel date.
Format: YYYY-MM-DD
        origin: Origin airport **IATA** code. The **IATA** code can be extracted from the **Search Airport** API in the **Flights** collection.
        destination: Destination airport **IATA** code. The **IATA** code can be extracted from the **Search Airport** API in the **Flights** collection.
        infants: Number of Infants (with age < 2 years)
        returndate: Return date.
Format: YYYY-MM-DD
        adults: Number of Adults (with age 18 and over)
Default Value: 1
        currency: The list of available **currencies** can be retrieved through the **Get Config** API in the **Config** collection.
        children: Number of Childrens (with age between 2-12 years)
        market: The list of available **market** can be retrieved through the **Get Config** API in the **Config** collection.
        countrycode: The list of available **countryCode** can be retrieved through the **Get Config** API in the **Config** collection.
        
    """
    url = f"https://skyscanner50.p.rapidapi.com/api/v1/searchFlights"
    querystring = {'date': date, 'origin': origin, 'destination': destination, }
    if infants:
        querystring['infants'] = infants
    if cabinclass:
        querystring['cabinClass'] = cabinclass
    if returndate:
        querystring['returnDate'] = returndate
    if adults:
        querystring['adults'] = adults
    if currency:
        querystring['currency'] = currency
    if children:
        querystring['children'] = children
    if filter:
        querystring['filter'] = filter
    if market:
        querystring['market'] = market
    if countrycode:
        querystring['countryCode'] = countrycode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner50.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_flight_details(legs: str, itineraryid: str, children: int=None, market: str='en-US', adults: int=1, countrycode: str='US', currency: str='USD', infants: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the details of the fights based on a unique Id."
    legs: The legs must contain the **origin**, **destination** and **date** in object format and must be passed in an array.
EXAMPLE:
[
  {
    'origin': 'LOND',
    'destination': 'NYCA',
    'date': '2023-02-07'
  },
  ...
]
**Note:** If there are multiple stops, there should be more leg objects on the board. And the legs have to be the same as the **Search Flights** API in **Flight** collection.
        itineraryid: **itineraryId** can be retrieved from **Search Flights** API in **Flight** collection as **id**.
        children: Number of Childrens (with age between 2-12 years)
        market: The list of available **market** can be retrieved through the **Get Config** API in the **Config** collection.
        adults: Number of Adults (with age 18 and over)
Default Value: 1
        countrycode: The list of available **countryCode** can be retrieved through the **Get Config** API in the **Config** collection.
        currency: The list of available **currencies** can be retrieved through the **Get Config** API in the **Config** collection.
        infants: Number of Infants (with age < 2 years)
        
    """
    url = f"https://skyscanner50.p.rapidapi.com/api/v1/getFlightDetails"
    querystring = {'legs': legs, 'itineraryId': itineraryid, }
    if children:
        querystring['children'] = children
    if market:
        querystring['market'] = market
    if adults:
        querystring['adults'] = adults
    if countrycode:
        querystring['countryCode'] = countrycode
    if currency:
        querystring['currency'] = currency
    if infants:
        querystring['infants'] = infants
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner50.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_flight_everywhere(origin: str, oneway: bool=None, anytime: bool=None, returndate: str=None, currency: str='USD', market: str='en-US', countrycode: str='US', traveldate: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this API to get flights when the destination is everywhere or unsure."
    origin: Origin airport **IATA** code. The **IATA** code can be extracted from the **Search Airport** API in the **Flights** collection.
        oneway: Send **oneWay** as **true** when you are unsure of the return date.
        anytime: Send **anytime** as **true** when you are unsure of the trip date.
        returndate: Return date.
Format: YYYY-MM-DD

Note: If **anytime** is **true**, you do not need to pass **returnDate**.
        currency: The list of available **currencies** can be retrieved through the **Get Config** API in the **Config** collection.
        market: The list of available **market** can be retrieved through the **Get Config** API in the **Config** collection.
        countrycode: The list of available **countryCode** can be retrieved through the **Get Config** API in the **Config** collection.
        traveldate: Departure or travel date.
Format: YYYY-MM-DD

Note: If **anytime** is **true**, you do not need to pass **travelDate**.
        
    """
    url = f"https://skyscanner50.p.rapidapi.com/api/v1/searchFlightEverywhere"
    querystring = {'origin': origin, }
    if oneway:
        querystring['oneWay'] = oneway
    if anytime:
        querystring['anytime'] = anytime
    if returndate:
        querystring['returnDate'] = returndate
    if currency:
        querystring['currency'] = currency
    if market:
        querystring['market'] = market
    if countrycode:
        querystring['countryCode'] = countrycode
    if traveldate:
        querystring['travelDate'] = traveldate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner50.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_flight_everywhere_details(origin: str, countryid: str, returndate: str=None, oneway: bool=None, market: str='en-US', anytime: bool=None, traveldate: str=None, currency: str='USD', countrycode: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the details of the fights based on a unique Id."
    origin: Origin airport **IATA** code. The **IATA** code can be extracted from the **Search Airport** API in the **Flights** collection.
        countryid: The **CountryId** can be extracted from the **Search Flight Everywhere** API in the **Flights** collection.
        returndate: Return date.
Format: YYYY-MM-DD

Note: If **anytime** is **true**, you do not need to pass **returnDate**.
        oneway: Send **oneWay** as **true** when you are unsure of the return date.
        market: The list of available **market** can be retrieved through the **Get Config** API in the **Config** collection.
        anytime: Send **anytime** as **true** when you are unsure of the trip date.
        traveldate: Departure or travel date.
Format: YYYY-MM-DD

Note: If **anytime** is **true**, you do not need to pass **travelDate**.
        currency: The list of available **currencies** can be retrieved through the **Get Config** API in the **Config** collection.
        countrycode: The list of available **countryCode** can be retrieved through the **Get Config** API in the **Config** collection.
        
    """
    url = f"https://skyscanner50.p.rapidapi.com/api/v1/searchFlightEverywhereDetails"
    querystring = {'origin': origin, 'CountryId': countryid, }
    if returndate:
        querystring['returnDate'] = returndate
    if oneway:
        querystring['oneWay'] = oneway
    if market:
        querystring['market'] = market
    if anytime:
        querystring['anytime'] = anytime
    if traveldate:
        querystring['travelDate'] = traveldate
    if currency:
        querystring['currency'] = currency
    if countrycode:
        querystring['countryCode'] = countrycode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner50.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def test_api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API is to test if server is up and running"
    
    """
    url = f"https://skyscanner50.p.rapidapi.com/api/v1/test"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner50.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_config(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint is responsible for providing  some of the parameters require for most of the API's."
    
    """
    url = f"https://skyscanner50.p.rapidapi.com/api/v1/getConfig"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner50.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_airport(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint is responsible for providing a list of Airports for the location."
    query: Name of the location where the **Airport** is situated.
        
    """
    url = f"https://skyscanner50.p.rapidapi.com/api/v1/searchAirport"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner50.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

