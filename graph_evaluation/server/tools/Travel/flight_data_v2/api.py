import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def the_prices_for_the_alternative_directions(origin: str, destination: str, flexibility: str='0', currency: str='RUB', depart_date: str=None, return_date: str=None, show_to_affiliates: str='True', limit: str='10', distance: str='100', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Brings the prices for the directions between the nearest to the target cities back."
    origin: the point of departure. The IATA city code or the country code. The length - from 2 to 3 symbols
        destination: the point of destination. The IATA city code or the country code. The length - from 2 to 3 symbols
        flexibility: expansion of the range of dates upward or downward. The value may vary from 0 to 7, where 0 shall show the variants for the dates specified, 7 – all the variants found for a week prior to the specified dates and a week after
        currency: the airline tickets currency
        depart_date: day or month of departure
        return_date: day or month of return
        show_to_affiliates: false - all the prices, true - just the prices, found using the partner marker (recommended)
        limit: the number of variants entered, from 1 to 20. Where 1 – is just the variant with the specified points of departure and the points of destination
        distance: the number of variants entered, from 1 to 20. Where 1 – is just the variant with the specified points of departure and the points of destination
        
    """
    url = f"https://travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com/v2/prices/nearest-places-matrix"
    querystring = {'origin': origin, 'destination': destination, }
    if flexibility:
        querystring['flexibility'] = flexibility
    if currency:
        querystring['currency'] = currency
    if depart_date:
        querystring['depart_date'] = depart_date
    if return_date:
        querystring['return_date'] = return_date
    if show_to_affiliates:
        querystring['show_to_affiliates'] = show_to_affiliates
    if limit:
        querystring['limit'] = limit
    if distance:
        querystring['distance'] = distance
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def non_stop_tickets(destination: str, origin: str, return_date: str=None, depart_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the cheapest non-stop tickets for the selected route with departure/return date filters."
    destination: IATA code of the destination city (for all routes, enter “-”). The IATA code is shown in uppercase letters, for example
        origin: IATA code of the departure city. The IATA code is shown in uppercase letters, for example
        return_date: a month of return (yyyy-mm)
        depart_date: a month of departure (yyyy-mm)
        
    """
    url = f"https://travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com/v1/prices/direct/"
    querystring = {'destination': destination, 'origin': origin, }
    if return_date:
        querystring['return_date'] = return_date
    if depart_date:
        querystring['depart_date'] = depart_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def the_popular_directions_from_a_city(currency: str, origin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Brings the most popular directions from a specified city back."
    currency: the airline tickets currency
        origin: the point of departure. The IATA city code or the country code. The length - from 2 to 3 symbols.
        
    """
    url = f"https://travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com/v1/city-directions"
    querystring = {'currency': currency, 'origin': origin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airline_data_in_json_format(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API returns a file with a list of airlines from the database"
    
    """
    url = f"https://travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com/data/en-GB/airlines.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def the_calendar_of_prices_for_a_month(destination: str, origin: str, currency: str=None, month: str=None, show_to_affiliates: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Brings the prices for each day of a month, grouped together by the number of transfers, back"
    destination: the point of destination. The IATA city code or the country code. The length - from 2 to 3 symbols. Note! If the point of departure and the point of destination are not specified, then the API shall bring 30 the most cheapest tickets, which have been found during the recent 48 hours, back.
        origin: the point of departure. The IATA city code or the country code. The length - from 2 to 3 symbols
        currency: the airline tickets currency
        month: the beginning of the month in the YYYY-MM-DD format
        show_to_affiliates: false - all the prices, true - just the prices, found using the partner marker (recommended). The default value - true.
        
    """
    url = f"https://travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com/v2/prices/month-matrix"
    querystring = {'destination': destination, 'origin': origin, }
    if currency:
        querystring['currency'] = currency
    if month:
        querystring['month'] = month
    if show_to_affiliates:
        querystring['show_to_affiliates'] = show_to_affiliates
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airplane_data_in_json_format(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API returns a file with a list of airplanes from the database"
    
    """
    url = f"https://travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com/data/planes.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def popular_airline_routes(airline_code: str, limit: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns routes for which an airline operates flights, sorted by popularity."
    airline_code: IATA code of airline
        limit: records limit per page. Default value is 100. Not less than 1000
        
    """
    url = f"https://travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com/v1/airline-directions"
    querystring = {'airline_code': airline_code, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def city_data_in_json_format(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API returns a file with a list of cities from the database"
    
    """
    url = f"https://travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com/data/en-GB/cities.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tickets_for_each_day_of_month(calendar_type: str, destination: str, origin: str, depart_date: str, currency: str='RUB', return_date: str=None, length: str='None', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the cheapest non-stop, one-stop, and two-stop flights for the selected route for each day of the selected month"
    calendar_type: field used to build the calendar. Equal to either: departure_date or return_date
        destination: IATA code of destination city. IATA code is shown by uppercase letters, for example LED
        origin: IATA code of departure city. IATA code is shown by uppercase letters, for example MOW
        depart_date: day or month of departure
        currency: currency of prices
        return_date: day or month of return. Pay attention! If the return_date is not specified, you will get the "One way" flights
        length: length of stay in the destination city
        
    """
    url = f"https://travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com/v1/prices/calendar"
    querystring = {'calendar_type': calendar_type, 'destination': destination, 'origin': origin, 'depart_date': depart_date, }
    if currency:
        querystring['currency'] = currency
    if return_date:
        querystring['return_date'] = return_date
    if length:
        querystring['length'] = length
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alliance_data_in_json_format(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API returns a file with a list of alliances from the database"
    
    """
    url = f"https://travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com/data/airlines_alliances.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def special_offers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Brings the recent special offers from the airline companies back in the .XML format."
    
    """
    url = f"https://travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com/v2/prices/special-offers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def the_prices_for_the_airline_tickets(destination: str, origin: str, period_type: str, one_way: str=' ', show_to_affiliates: int=True, trip_duration: int=None, trip_class: int=0, beginning_of_period: str=None, currency: str='RUB', page: int=1, sorting: str='price', limit: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Brings back to the list of prices, found by our users during the recent 48 hours according to the filters used."
    destination: the point of destination. The IATA city code or the country code
        origin: the point of departure. The IATA city code or the country code
        period_type: the period, for which the tickets have been found (the required parameter)
        one_way: true — one way, false — back-to-back.
        show_to_affiliates: false — all the prices, true — just the prices, found using the partner marker (recommended)
        trip_duration: the length of stay in weeks or days (for period_type=day)
        trip_class: trip_class. 0 — The economy class (the default value); 1 — The business class; 2 — The first class.
        beginning_of_period: the beginning of the period, within which the dates of departure fall (in the YYYY-MM-DD format, for example, 2016-05-01). Must be specified if period_type is equal to month
        currency: the airline tickets currency
        page: a page number
        sorting: the assorting of prices. price — by the price (the default value). For the directions city — city assorting by the price is possible only; route — by the popularity of a route; distance_unit_price — by the price for 1 km.
        limit: the total number of records on a page. The maximum value — 1000
        
    """
    url = f"https://travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com/v2/prices/latest"
    querystring = {'destination': destination, 'origin': origin, 'period_type': period_type, }
    if one_way:
        querystring['one_way'] = one_way
    if show_to_affiliates:
        querystring['show_to_affiliates'] = show_to_affiliates
    if trip_duration:
        querystring['trip_duration'] = trip_duration
    if trip_class:
        querystring['trip_class'] = trip_class
    if beginning_of_period:
        querystring['beginning_of_period'] = beginning_of_period
    if currency:
        querystring['currency'] = currency
    if page:
        querystring['page'] = page
    if sorting:
        querystring['sorting'] = sorting
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airport_data_in_json_format(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API returns a file with a list of airports from the database"
    
    """
    url = f"https://travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com/data/en-GB/airports.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def the_calendar_of_prices_for_a_week(origin: str, destination: str, return_date: str=None, depart_date: str=None, show_to_affiliates: str=None, currency: str='RUB', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Brings the prices for the nearest dates to the target ones back"
    origin: the point of departure. The IATA city code or the country code. The length - from 2 to 3 symbols
        destination: the point of destination. The IATA city code or the country code. The length - from 2 to 3 symbols
        return_date: day or month of return
        depart_date: day or month of departure
        show_to_affiliates: false - all the prices, true - just the prices, found using the partner marker (recommended).
        currency: the airline tickets currency
        
    """
    url = f"https://travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com/v2/prices/week-matrix"
    querystring = {'origin': origin, 'destination': destination, }
    if return_date:
        querystring['return_date'] = return_date
    if depart_date:
        querystring['depart_date'] = depart_date
    if show_to_affiliates:
        querystring['show_to_affiliates'] = show_to_affiliates
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cheapest_tickets(origin: str, page: str, currency: str, depart_date: str=None, return_date: str=None, destination: str='-', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the cheapest non-stop tickets, as well as tickets with 1 or 2 stops, for the selected route with departure/return date filters. Important: Old dates may be specified in a query. No error will be generated, but no data will be returned."
    origin: IATA code of the departure city. IATA code is shown by uppercase letters,
        page: Is used to display the found data (by default the page displays 100 found prices. If the destination isn't selected, there can be more data. In this case, use the page, to display the next set of data, for example, page=2).
        currency: Currency of prices
        depart_date: Day or month of departure
        return_date: Day or month of return
        destination: IATA code of the destination city (for all routes, enter "-"). IATA code is shown by uppercase letters
        
    """
    url = f"https://travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com/v1/prices/cheap"
    querystring = {'origin': origin, 'page': page, 'currency': currency, }
    if depart_date:
        querystring['depart_date'] = depart_date
    if return_date:
        querystring['return_date'] = return_date
    if destination:
        querystring['destination'] = destination
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def data_of_countries_in_json_format(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API returns a file with a list of countrys from the database"
    
    """
    url = f"https://travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com/data/en-GB/countries.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cheapest_tickets_grouped_by_month(origin: str, destination: str, currency: str='RUB', length: str='3', depart_date: str=None, return_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the cheapest non-stop tickets, as well as tickets with 1 or 2 stops, for the selected route grouped by month."
    origin: IATA code of departure city. IATA code is shown by uppercase letters, for example MOW
        destination: IATA code of destination city. IATA code is shown by uppercase letters, for example LED
        currency: currency of prices
        length: length of stay in the destination city
        depart_date: day or month of departure
        return_date: day or month of return. Pay attention! If the return_date is not specified, you will get the "One way" flights
        
    """
    url = f"https://travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com/v1/prices/monthly"
    querystring = {'origin': origin, 'destination': destination, }
    if currency:
        querystring['currency'] = currency
    if length:
        querystring['length'] = length
    if depart_date:
        querystring['depart_date'] = depart_date
    if return_date:
        querystring['return_date'] = return_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def data_on_the_routes_in_json_format(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The query returns a file with a list of routes from the database."
    
    """
    url = f"https://travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com/data/routes.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

