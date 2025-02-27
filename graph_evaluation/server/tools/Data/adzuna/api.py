import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def average_salary(app_id: str, app_key: str, country: str, what: str=None, location0: str='location0=UK&location1=South East England&location2=Surrey', location1: str=None, location2: str=None, location3: str=None, location4: str=None, location5: str=None, location6: str=None, location7: str=None, where: str=None, category: str=None, months: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides historical average salaries"
    app_id: Application ID, supplied by Adzuna
        app_key: Application key, supplied by Adzuna
        country: ISO 8601 country code of the relevant country.
        what: The keywords to search for. Use space or comma characters to separate multiple keywords.
        location0: The locationN fields may be used to describe a location, in a similar form to that returned in a Adzuna::API::Response::Location object.
        location1: See location0
        location2: See location0
        location3: See location0
        location4: See location0
        location5: See location0
        location6: See location0
        location7: See location0
        where: The geographic centre of the search. Place names or postal codes may be used.
        category: The category tag, as returned by the "category" endpoint.
        months: The number of months back for which to retrieve data.
        
    """
    url = f"https://baskarm28-adzuna-v1.p.rapidapi.com/jobs/{country}/history"
    querystring = {'app_id': app_id, 'app_key': app_key, }
    if what:
        querystring['what'] = what
    if location0:
        querystring['location0'] = location0
    if location1:
        querystring['location1'] = location1
    if location2:
        querystring['location2'] = location2
    if location3:
        querystring['location3'] = location3
    if location4:
        querystring['location4'] = location4
    if location5:
        querystring['location5'] = location5
    if location6:
        querystring['location6'] = location6
    if location7:
        querystring['location7'] = location7
    if where:
        querystring['where'] = where
    if category:
        querystring['category'] = category
    if months:
        querystring['months'] = months
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-adzuna-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_employers(app_id: str, app_key: str, country: str, what: str=None, location0: str='location0=UK&location1=South East England&location2=Surrey', location1: str=None, location2: str=None, location3: str=None, location4: str=None, location5: str=None, location6: str=None, location7: str=None, where: str=None, category: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the employers with most jobs advertised."
    app_id: Application ID, supplied by Adzuna
        app_key: Application key, supplied by Adzuna
        country: ISO 8601 country code of the relevant country.
        what: The keywords to search for. Use space or comma characters to separate multiple keywords.
        location0: The locationN fields may be used to describe a location, in a similar form to that returned in a Adzuna::API::Response::Location object
        location1: See location0
        location2: See location0
        location3: See location0
        location4: See location0
        location5: See location0
        location6: See location0
        location7: See location0
        where: The geographic centre of the search. Place names or postal codes may be used.
        category: The category tag, as returned by the "category" endpoint.
        
    """
    url = f"https://baskarm28-adzuna-v1.p.rapidapi.com/jobs/{country}/top_companies"
    querystring = {'app_id': app_id, 'app_key': app_key, }
    if what:
        querystring['what'] = what
    if location0:
        querystring['location0'] = location0
    if location1:
        querystring['location1'] = location1
    if location2:
        querystring['location2'] = location2
    if location3:
        querystring['location3'] = location3
    if location4:
        querystring['location4'] = location4
    if location5:
        querystring['location5'] = location5
    if location6:
        querystring['location6'] = location6
    if location7:
        querystring['location7'] = location7
    if where:
        querystring['where'] = where
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-adzuna-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def salary_data(app_id: str, country: str, app_key: str=None, what: str=None, location0: str='location0=UK&location1=South East England&location2=Surrey', location1: str=None, location2: str=None, location3: str=None, location4: str=None, location5: str=None, location6: str=None, location7: str=None, where: str=None, category: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides salary data for locations"
    app_id: Application ID, supplied by Adzuna
        country: ISO 8601 country code of the relevant country.
        app_key: Application key, supplied by Adzuna
        what: The keywords to search for. Use space or comma characters to separate multiple keywords.
        location0: The locationN fields may be used to describe a location, in a similar form to that returned in a Adzuna::API::Response::Location object.
        location1: See location0
        location2: See location0
        location3: See location0
        location4: See location0
        location5: See location0
        location6: See location0
        location7: See location0
        where: The geographic centre of the search. Place names or postal codes may be used.
        category: The category tag, as returned by the "category" endpoint.
        
    """
    url = f"https://baskarm28-adzuna-v1.p.rapidapi.com/jobs/{country}/geodata"
    querystring = {'app_id': app_id, }
    if app_key:
        querystring['app_key'] = app_key
    if what:
        querystring['what'] = what
    if location0:
        querystring['location0'] = location0
    if location1:
        querystring['location1'] = location1
    if location2:
        querystring['location2'] = location2
    if location3:
        querystring['location3'] = location3
    if location4:
        querystring['location4'] = location4
    if location5:
        querystring['location5'] = location5
    if location6:
        querystring['location6'] = location6
    if location7:
        querystring['location7'] = location7
    if where:
        querystring['where'] = where
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-adzuna-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def salary_estimate(app_id: str, app_key: str, title: str=None, description: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an estimate of the salary of a job."
    app_id: Application ID, supplied by Adzuna
        app_key: Application key, supplied by Adzuna
        title: The title of the job advertisement.
        description: The job description.
        
    """
    url = f"https://baskarm28-adzuna-v1.p.rapidapi.com/jobs/gb/jobsworth"
    querystring = {'app_id': app_id, 'app_key': app_key, }
    if title:
        querystring['title'] = title
    if description:
        querystring['description'] = description
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-adzuna-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_version(app_id: str, app_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the current version of this API"
    app_id: Application ID, supplied by Adzuna
        app_key: Application key, supplied by Adzuna
        
    """
    url = f"https://baskarm28-adzuna-v1.p.rapidapi.com/version"
    querystring = {'app_id': app_id, 'app_key': app_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-adzuna-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_category(vertical: str, app_id: str, app_key: str, country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List available categories."
    app_id: Application ID, supplied by Adzuna
        app_key: Application key, supplied by Adzuna
        country: ISO 8601 country code of the relevant country.
        
    """
    url = f"https://baskarm28-adzuna-v1.p.rapidapi.com/{vertical}/{country}/categories"
    querystring = {'app_id': app_id, 'app_key': app_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-adzuna-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_cars(app_id: str, app_key: str, country: str, page: int, location5: str=None, results_per_page: int=None, what: str=None, what_and: str=None, what_phrase: str=None, what_or: str=None, what_exclude: str=None, title_only: str=None, location0: str='location0=UK&location1=South East England&location2=Surrey', location1: str=None, location2: str=None, location3: str=None, location4: str=None, location6: str=None, location7: str=None, where: str=None, distance: int=None, max_days_old: int=None, category: str=None, sort_direction: str=None, sort_by: str=None, fuel_type: str=None, transmission: str=None, body_type: str=None, engine_min: int=None, engine_max: int=None, colour: str=None, mileage_min: int=None, mileage_max: int=None, year_min: int=None, year_max: int=None, seller_type: str=None, price_min: int=None, price_include_unknown: str=None, price_max: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Adzuna cars."
    app_id: Application ID, supplied by Adzuna
        app_key: Application key, supplied by Adzuna
        country: ISO 8601 country code of the relevant country.
        page: Page number
        location5: See location0
        results_per_page: The number of results to include on each page of search results.
        what: The keywords to search for. Use space or comma characters to separate multiple keywords.
        what_and: The keywords to search for, all keywords must be found.
        what_phrase: An entire phrase which must be found in the description or title.
        what_or: Keywords to search for. Use space or comma characters to separate multiple keywords.
        what_exclude: Keywords to exclude from the search. Use space or comma characters to separate multiple keywords.
        title_only: Keywords to find, but only in the title. Use space or comma characters to separate multiple keywords.
        location0: The locationN fields may be used to describe a location, in a similar form to that returned in a Adzuna::API::Response::Location object.
        location1: See location0
        location2: See location0
        location3: See location0
        location4: See location0
        location6: See location0
        location7: See location0
        where: The geographic centre of the search. Place names or postal codes may be used.
        distance: The distance in kilometres from the centre of the place described by the 'where' parameter.
        max_days_old: The age of the oldest advertisment in days that will be returned.
        category: The category tag, as returned by the "category" endpoint.
        sort_direction: The order of search results (ascending or descending).
        sort_by: The ordering of the search results.
        fuel_type: The type of fuel the vehicle requires.
        transmission: The type of transmission the vehicle has.
        body_type: The type of body a vehicle has.
        engine_min: The minimum engine capacity in cubic centimetres.
        engine_max: The maximum engine capacity in cubic centimetres.
        colour: The colour of the car
        mileage_min: The minimum mileage of vehicles to be considered.
        mileage_max: The maximum mileage of vehicles to be considered.
        year_min: The earliest year to be considered.
        year_max: The most recent year to be considered.
        seller_type: The type of seller.
        price_min: The minimum price we wish to get results for.
        price_include_unknown: When using price_min and/or price_max set to "1", to include cars with unknown price in results.
        price_max: The maximum price we wish to get results for.
        
    """
    url = f"https://baskarm28-adzuna-v1.p.rapidapi.com/cars/{country}/search/{page}"
    querystring = {'app_id': app_id, 'app_key': app_key, }
    if location5:
        querystring['location5'] = location5
    if results_per_page:
        querystring['results_per_page'] = results_per_page
    if what:
        querystring['what'] = what
    if what_and:
        querystring['what_and'] = what_and
    if what_phrase:
        querystring['what_phrase'] = what_phrase
    if what_or:
        querystring['what_or'] = what_or
    if what_exclude:
        querystring['what_exclude'] = what_exclude
    if title_only:
        querystring['title_only'] = title_only
    if location0:
        querystring['location0'] = location0
    if location1:
        querystring['location1'] = location1
    if location2:
        querystring['location2'] = location2
    if location3:
        querystring['location3'] = location3
    if location4:
        querystring['location4'] = location4
    if location6:
        querystring['location6'] = location6
    if location7:
        querystring['location7'] = location7
    if where:
        querystring['where'] = where
    if distance:
        querystring['distance'] = distance
    if max_days_old:
        querystring['max_days_old'] = max_days_old
    if category:
        querystring['category'] = category
    if sort_direction:
        querystring['sort_direction'] = sort_direction
    if sort_by:
        querystring['sort_by'] = sort_by
    if fuel_type:
        querystring['fuel_type'] = fuel_type
    if transmission:
        querystring['transmission'] = transmission
    if body_type:
        querystring['body_type'] = body_type
    if engine_min:
        querystring['engine_min'] = engine_min
    if engine_max:
        querystring['engine_max'] = engine_max
    if colour:
        querystring['colour'] = colour
    if mileage_min:
        querystring['mileage_min'] = mileage_min
    if mileage_max:
        querystring['mileage_max'] = mileage_max
    if year_min:
        querystring['year_min'] = year_min
    if year_max:
        querystring['year_max'] = year_max
    if seller_type:
        querystring['seller_type'] = seller_type
    if price_min:
        querystring['price_min'] = price_min
    if price_include_unknown:
        querystring['price_include_unknown'] = price_include_unknown
    if price_max:
        querystring['price_max'] = price_max
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-adzuna-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_properties(app_id: str, app_key: str, country: str, page: int, results_per_page: int=None, what: str=None, what_and: str=None, what_phrase: str=None, what_or: str=None, what_exclude: str=None, title_only: str=None, location0: str='location0=UK&location1=South East England&location2=Surrey', location1: str=None, location2: str=None, location3: str=None, location4: str=None, location6: str=None, location7: str=None, where: str=None, distance: int=None, max_days_old: int=None, category: str=None, sort_direction: str=None, sort_by: str=None, beds: int=None, is_furnished: str=None, price_min: int=None, price_max: int=None, price_include_unknown: str=None, property_type: str=None, location5: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Adzuna properties"
    app_id: Application ID, supplied by Adzuna
        app_key: Application key, supplied by Adzuna
        country: ISO 8601 country code of the relevant country.
        page: Page number
        results_per_page: The number of results to include on each page of search results.
        what: The keywords to search for. Use space or comma characters to separate multiple keywords.
        what_and: The keywords to search for, all keywords must be found.
        what_phrase: An entire phrase which must be found in the description or title.
        what_or: Keywords to search for. Use space or comma characters to separate multiple keywords.
        what_exclude: Keywords to exclude from the search. Use space or comma characters to separate multiple keywords.
        title_only: Keywords to find, but only in the title. Use space or comma characters to separate multiple keywords.
        location0: The locationN fields may be used to describe a location, in a similar form to that returned in a Adzuna::API::Response::Location object.
        location1: See location0
        location2: See location0
        location3: See location0
        location4: See location0
        location6: See location0
        location7: See location0
        where: The geographic centre of the search. Place names or postal codes may be used.
        distance: The distance in kilometres from the centre of the place described by the 'where' parameter.
        max_days_old: The age of the oldest advertisment in days that will be returned.
        category: The category tag, as returned by the "category" endpoint.
        sort_direction: The order of search results (ascending or descending).
        sort_by: The ordering of the search results.
        beds: The number of beds to search for. For values greater than five, this is considered to be minimum value, rather than the exact value to find.
        is_furnished: One of "0" or "1", to indicate no or yes. If omitted, both types of property are found.
        price_min: The minimum price we wish to get results for (for rental properties price is monthly).
        price_max: The maximum price we wish to get results for (for rental properties price is monthly).
        price_include_unknown: When using price_min or price_max set to "1", to include properties with an unknown price in results.
        property_type: The type of property of interest.
        location5: See location0
        
    """
    url = f"https://baskarm28-adzuna-v1.p.rapidapi.com/property/{country}/search/{page}"
    querystring = {'app_id': app_id, 'app_key': app_key, }
    if results_per_page:
        querystring['results_per_page'] = results_per_page
    if what:
        querystring['what'] = what
    if what_and:
        querystring['what_and'] = what_and
    if what_phrase:
        querystring['what_phrase'] = what_phrase
    if what_or:
        querystring['what_or'] = what_or
    if what_exclude:
        querystring['what_exclude'] = what_exclude
    if title_only:
        querystring['title_only'] = title_only
    if location0:
        querystring['location0'] = location0
    if location1:
        querystring['location1'] = location1
    if location2:
        querystring['location2'] = location2
    if location3:
        querystring['location3'] = location3
    if location4:
        querystring['location4'] = location4
    if location6:
        querystring['location6'] = location6
    if location7:
        querystring['location7'] = location7
    if where:
        querystring['where'] = where
    if distance:
        querystring['distance'] = distance
    if max_days_old:
        querystring['max_days_old'] = max_days_old
    if category:
        querystring['category'] = category
    if sort_direction:
        querystring['sort_direction'] = sort_direction
    if sort_by:
        querystring['sort_by'] = sort_by
    if beds:
        querystring['beds'] = beds
    if is_furnished:
        querystring['is_furnished'] = is_furnished
    if price_min:
        querystring['price_min'] = price_min
    if price_max:
        querystring['price_max'] = price_max
    if price_include_unknown:
        querystring['price_include_unknown'] = price_include_unknown
    if property_type:
        querystring['property_type'] = property_type
    if location5:
        querystring['location5'] = location5
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-adzuna-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def histogram_data(country: str, what: str=None, location1: str=None, location2: str=None, location3: str=None, location4: str=None, location5: str=None, location6: str=None, location7: str=None, where: str=None, category: str=None, app_id: str=None, app_key: str=None, location0: str='location0=UK&location1=South East England&location2=Surrey', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide histogram data of salary data"
    country: ISO 8601 country code of the relevant country.
        what: The keywords to search for. Use space or comma characters to separate multiple keywords.
        location1: See location0
        location2: See location0
        location3: See location0
        location4: See location0
        location5: See location0
        location6: See location0
        location7: See location0
        where: The geographic centre of the search. Place names or postal codes may be used.
        category: The category tag, as returned by the "category" endpoint.
        app_id: Application ID, supplied by Adzuna.
        app_key: Application key, supplied by Adzuna.
        location0: The locationN fields may be used to describe a location, in a similar form to that returned in a Adzuna::API::Response::Location object.
        
    """
    url = f"https://baskarm28-adzuna-v1.p.rapidapi.com/jobs/{country}/histogram"
    querystring = {}
    if what:
        querystring['what'] = what
    if location1:
        querystring['location1'] = location1
    if location2:
        querystring['location2'] = location2
    if location3:
        querystring['location3'] = location3
    if location4:
        querystring['location4'] = location4
    if location5:
        querystring['location5'] = location5
    if location6:
        querystring['location6'] = location6
    if location7:
        querystring['location7'] = location7
    if where:
        querystring['where'] = where
    if category:
        querystring['category'] = category
    if app_id:
        querystring['app_id'] = app_id
    if app_key:
        querystring['app_key'] = app_key
    if location0:
        querystring['location0'] = location0
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-adzuna-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_jobs(app_id: str, app_key: str, country: str, page: int, results_per_page: int=None, what: str=None, what_and: str=None, what_phrase: str=None, what_or: str=None, what_exclude: str=None, title_only: str=None, location1: str=None, location2: str=None, location3: str=None, location4: str=None, location5: str=None, location6: str=None, location7: str=None, where: str=None, distance: str=None, max_days_old: str=None, category: str=None, sort_direction: str=None, sort_by: str=None, salary_min: str=None, salary_max: str=None, salary_include_unknown: str=None, full_time: str=None, part_time: str=None, contract: str=None, permanent: str=None, company: str=None, location0: str='location0=UK&location1=South East England&location2=Surrey', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search the Adzuna jobs"
    app_id: Application ID, supplied by Adzuna
        app_key: Application key, supplied by Adzuna
        country: ISO 8601 country code of the relevant country.
        page: Page number
        results_per_page: The number of results to include on each page of search results.
        what: The keywords to search for. Use space or comma characters to separate multiple keywords.
        what_and: The keywords to search for, all keywords must be found.
        what_phrase: An entire phrase which must be found in the description or title.
        what_or: Keywords to search for. Use space or comma characters to separate multiple keywords.
        what_exclude: Keywords to exclude from the search. Use space or comma characters to separate multiple keywords.
        title_only: Keywords to find, but only in the title. Use space or comma characters to separate multiple keywords.
        location1: See location0
        location2: See location0
        location3: See location0
        location4: See location0
        location5: See location0
        location6: See location0
        location7: See location0
        where: The geographic centre of the search. Place names or postal codes may be used.
        distance: The distance in kilometres from the centre of the place described by the 'where' parameter. Defaults to 10km.
        max_days_old: The age of the oldest advertisment in days that will be returned.
        category: The category tag, as returned by the "category" endpoint.
        sort_direction: The order of search results (ascending or descending).
        sort_by: The ordering of the search results.
        salary_min: The minimum salary we wish to get results for.
        salary_max: The maximum salary we wish to get results for.
        salary_include_unknown: When using salary_min and/or salary_max set this to "1", to include jobs with unknown salaries in results.
        full_time: If set to "1", only full time jobs will be returned.
        part_time: If set to "1", only part time jobs will be returned.
        contract: If set to "1", only contract jobs will be returned.
        permanent: If set to "1", only permanent jobs will be returned.
        company: The canonical company name. This may be returned in a Adzuna::API::Response::Company object when a job is returned. A full list of allowed terms in not available through the API.
        location0: The locationN fields may be used to describe a location, in a similar form to that returned in a Adzuna::API::Response::Location object.
        
    """
    url = f"https://baskarm28-adzuna-v1.p.rapidapi.com/jobs/{country}/search/{page}"
    querystring = {'app_id': app_id, 'app_key': app_key, }
    if results_per_page:
        querystring['results_per_page'] = results_per_page
    if what:
        querystring['what'] = what
    if what_and:
        querystring['what_and'] = what_and
    if what_phrase:
        querystring['what_phrase'] = what_phrase
    if what_or:
        querystring['what_or'] = what_or
    if what_exclude:
        querystring['what_exclude'] = what_exclude
    if title_only:
        querystring['title_only'] = title_only
    if location1:
        querystring['location1'] = location1
    if location2:
        querystring['location2'] = location2
    if location3:
        querystring['location3'] = location3
    if location4:
        querystring['location4'] = location4
    if location5:
        querystring['location5'] = location5
    if location6:
        querystring['location6'] = location6
    if location7:
        querystring['location7'] = location7
    if where:
        querystring['where'] = where
    if distance:
        querystring['distance'] = distance
    if max_days_old:
        querystring['max_days_old'] = max_days_old
    if category:
        querystring['category'] = category
    if sort_direction:
        querystring['sort_direction'] = sort_direction
    if sort_by:
        querystring['sort_by'] = sort_by
    if salary_min:
        querystring['salary_min'] = salary_min
    if salary_max:
        querystring['salary_max'] = salary_max
    if salary_include_unknown:
        querystring['salary_include_unknown'] = salary_include_unknown
    if full_time:
        querystring['full_time'] = full_time
    if part_time:
        querystring['part_time'] = part_time
    if contract:
        querystring['contract'] = contract
    if permanent:
        querystring['permanent'] = permanent
    if company:
        querystring['company'] = company
    if location0:
        querystring['location0'] = location0
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-adzuna-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

