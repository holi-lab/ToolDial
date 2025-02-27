import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def properties_get_parcel_bounds(propertyid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get parcel bounds info. This endpoint is used to get fipsCode and apn to pass in .../properties/get-flood-info endpoint."
    propertyid: The value of propertyId field returned in .../properties/list endpoint.
        
    """
    url = f"https://unofficial-redfin.p.rapidapi.com/properties/get-parcel-bounds"
    querystring = {'propertyId': propertyid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-redfin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_list(region_type: int, region_id: int, uipt: str, status: int, min_year_built: int=None, wd: bool=None, green: bool=None, financing_type: int=None, elevator: bool=None, time_on_market_range: str=None, min_parcel_size: int=None, excl_ss: bool=None, min_price_per_sqft: int=None, pool_types: int=None, primary_bed_on_main: bool=None, excl_ll: bool=None, include_outdoor_parking: bool=None, unrated_schools: bool=None, walk_score: str=None, guest_house: bool=None, max_property_tax: int=None, school_rating: str=None, wf: bool=None, accessible: bool=None, transit_score: str=None, rv_parking: bool=None, bike_score: str=None, pets_allowed: bool=None, fixer: bool=None, basement_types: str=None, ac: bool=None, virtual_tour: bool=None, excl_ar: bool=None, open_house: int=None, max_stories: int=None, min_price: int=None, school_types: str=None, max_listing_approx_size: int=None, num_beds: int=None, sold_within_days: int=None, max_price: int=None, hoa: int=None, min_stories: int=None, view: bool=None, rd: str=None, max_year_built: int=None, sf: str='1,2,3,5,6,7', fireplace: bool=None, pkg: str=None, max_price_per_sqft: int=None, min_listing_approx_size: int=None, max_parcel_size: int=None, num_baths: int=None, num_homes: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List properties for sales, sold, etc... with options and filters"
    region_type: One of the following : -1:Unknowns|1:Neighborhood|2:Zip Code|4:State|5:County|6:City|7:School|8:School District|9:Service Region|10:Minor Civil Division|11:Country|30:CA Postal Code|31:CA Province|32:CA Provincial Division|33:CA Municipality|34:CA Forward Sortation Area
You need to use regex to examine the value of url field returned in .../auto-complete for suitable value. Ex : '/city/30749/...', it is city so it is 6 in this case.
        region_id: You need to use regex to extract the value of id field returned in .../auto-complete endpoint. Ex : '2_30749', 30749 is the value you need to extract from the string.
        uipt: Property types, uses of the following : 1-Home|2-Condo|3-Townhouse|4-Multi-family|5-Land|6-Other|7-Manufactured|8-Co-op
Separated by comma for multiple options. Ex : 1,2,3,4,7,8
        status: One of the following : 1-Active listings|8-Coming soon listings|9-Active + coming soon listings|130-Only under contract/pending|131-Active + under contract/pending
        min_year_built: Filter by year built
        wd: true : has washer/dryer hookup
        green: true : Green Home
        financing_type: Accepted Financing, one of the following : 1-FHA|2-VA
        elevator: true : has elevator
        time_on_market_range: Time on Redfin, you need to put the minus before/after every numbers. Ex :
1-:Less than 1 day|3-:Less than 3 days|7-|14-|30-|-30:More than 30 days|-60|-120|-:For any time
        min_parcel_size: Lot size
        excl_ss: true : Exclude short sales
        min_price_per_sqft: Filter by price per sqft
        pool_types: One of the following : 1-Private pool|2-Community pool|3-Private or Community pool|4-No private pool
        primary_bed_on_main: true : has primary bedroom on main floor
        excl_ll: true : Exclude land leases
        include_outdoor_parking: true : Include outdoor parking
        unrated_schools: true : Include unrated schools. Use along with school_rating parameter.
        walk_score: Filter by walk score, you need to put the minus after every numbers. Ex : 10-|20-|30-|etc...
        guest_house: true : has guest house
        max_property_tax: Max property tax. Ex : 750
        school_rating: Filter by school rate, you need to put the minus after every numbers. Ex : 1-|2-|3-|...|10-
        wf: true : has water front
        accessible: true : Accessible Home
        transit_score: Filter by transit score, you need to put the minus after every numbers. Ex : 10-|20-|30-|etc...
        rv_parking: true : Include RV parking
        bike_score: Filter by bike score, you need to put the minus after every numbers. Ex : 10-|20-|30-|etc...
        pets_allowed: true : allow pets
        fixer: true : Fixer Uppers Only
        basement_types: One of the following : 1-Finished|3-Unfinished. Separated by comma for multiple options. Ex : 1,3
        ac: true : has air conditioning
        virtual_tour: true|false
        excl_ar: true : Exclude 55+ communities
        open_house: 1-Any time|2-This weekend
        max_stories: Filter by stories
        min_price: Filter by price
        school_types: Use along with school_rating parameter. One of the following : 1-Elementary schools|2-Middle schools|3-High schools. Separated by comma for multiple options. Ex : 1,2,3
        max_listing_approx_size: Square Feet
        num_beds: Filter by number of beds room at least. Ex : 2
        sold_within_days: Use this parameter to list SOLD properties. If you pass this parameter, leave sf parameter empty.
        max_price: Filter by price
        hoa: HOA fees. Ex : 0|25|50|75|...|2000
        min_stories: Filter by stories
        view: true : Must have view
        rd: Price reduce, you need to put the minus before/after every numbers. Ex :
1-:Less than 1 day|3-:Less than 3 days|7-|14-|30-|-30:More than 30 days|-60|-120|-:For any time
        max_year_built: Filter by year built
        sf: MLS listings, uses of the following : 1,7-Agent listed Homes|2-MLS listed foreclosures|3-For sale by owner|4-Foreclosures|5,6-New construction
Separated by comma for multiple options. Ex : 1,2,3,5,6,7
        fireplace: true : has fireplace
        pkg: Garage spots 1-|2-|3-|4-|5- (You need to put the minus behind every numbers)
        max_price_per_sqft: Filter by price per sqft
        min_listing_approx_size: Square Feet
        max_parcel_size: Lot size
        num_baths: Filter by number of baths at least. Ex : 2.5
        num_homes: Number of items per response (Max 400). Redfin always return max 400 properties for every search, there is no paging feature.
        
    """
    url = f"https://unofficial-redfin.p.rapidapi.com/properties/list"
    querystring = {'region_type': region_type, 'region_id': region_id, 'uipt': uipt, 'status': status, }
    if min_year_built:
        querystring['min_year_built'] = min_year_built
    if wd:
        querystring['wd'] = wd
    if green:
        querystring['green'] = green
    if financing_type:
        querystring['financing_type'] = financing_type
    if elevator:
        querystring['elevator'] = elevator
    if time_on_market_range:
        querystring['time_on_market_range'] = time_on_market_range
    if min_parcel_size:
        querystring['min_parcel_size'] = min_parcel_size
    if excl_ss:
        querystring['excl_ss'] = excl_ss
    if min_price_per_sqft:
        querystring['min_price_per_sqft'] = min_price_per_sqft
    if pool_types:
        querystring['pool_types'] = pool_types
    if primary_bed_on_main:
        querystring['primary_bed_on_main'] = primary_bed_on_main
    if excl_ll:
        querystring['excl_ll'] = excl_ll
    if include_outdoor_parking:
        querystring['include_outdoor_parking'] = include_outdoor_parking
    if unrated_schools:
        querystring['unrated_schools'] = unrated_schools
    if walk_score:
        querystring['walk_score'] = walk_score
    if guest_house:
        querystring['guest_house'] = guest_house
    if max_property_tax:
        querystring['max_property_tax'] = max_property_tax
    if school_rating:
        querystring['school_rating'] = school_rating
    if wf:
        querystring['wf'] = wf
    if accessible:
        querystring['accessible'] = accessible
    if transit_score:
        querystring['transit_score'] = transit_score
    if rv_parking:
        querystring['rv_parking'] = rv_parking
    if bike_score:
        querystring['bike_score'] = bike_score
    if pets_allowed:
        querystring['pets_allowed'] = pets_allowed
    if fixer:
        querystring['fixer'] = fixer
    if basement_types:
        querystring['basement_types'] = basement_types
    if ac:
        querystring['ac'] = ac
    if virtual_tour:
        querystring['virtual_tour'] = virtual_tour
    if excl_ar:
        querystring['excl_ar'] = excl_ar
    if open_house:
        querystring['open_house'] = open_house
    if max_stories:
        querystring['max_stories'] = max_stories
    if min_price:
        querystring['min_price'] = min_price
    if school_types:
        querystring['school_types'] = school_types
    if max_listing_approx_size:
        querystring['max_listing_approx_size'] = max_listing_approx_size
    if num_beds:
        querystring['num_beds'] = num_beds
    if sold_within_days:
        querystring['sold_within_days'] = sold_within_days
    if max_price:
        querystring['max_price'] = max_price
    if hoa:
        querystring['hoa'] = hoa
    if min_stories:
        querystring['min_stories'] = min_stories
    if view:
        querystring['view'] = view
    if rd:
        querystring['rd'] = rd
    if max_year_built:
        querystring['max_year_built'] = max_year_built
    if sf:
        querystring['sf'] = sf
    if fireplace:
        querystring['fireplace'] = fireplace
    if pkg:
        querystring['pkg'] = pkg
    if max_price_per_sqft:
        querystring['max_price_per_sqft'] = max_price_per_sqft
    if min_listing_approx_size:
        querystring['min_listing_approx_size'] = min_listing_approx_size
    if max_parcel_size:
        querystring['max_parcel_size'] = max_parcel_size
    if num_baths:
        querystring['num_baths'] = num_baths
    if num_homes:
        querystring['num_homes'] = num_homes
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-redfin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mortgage_check_rates(loanbalance: int, locationid: int, homeprice: int, locationtype: int, homevalue: int, loantype: str='purchase', points: str='zero', veterantype: str='Non-Military', percentagedown: int=20, creditscore: int=760, loanterms: str='30yr', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Finance tools to check mortgage rates"
    loanbalance: Loan balance
        locationid: You need to use regex to extract the value of id field returned in .../auto-complete endpoint. Ex : '2_30749', 30749 is the value you need to extract from the string.
        homeprice: Home price
        locationtype: The value of type field returned in .../auto-complete
        homevalue: Home value
        loantype: One of the following : purchase|refinance
        points: One of the following : zero|zerotoone|onetotwo|all
        veterantype: One of the following : Non-Military|RegularMilitary|NationalGuard|Reserves|SpouseOfRegularMilitary|SpouseOfReserveMilitary
        percentagedown: Percentage down payment
        creditscore: From 670 to 760
        loanterms: One of the following : 30yr|20yr|15yr|10yr|3-1arm|5-1arm|7-1arm
        
    """
    url = f"https://unofficial-redfin.p.rapidapi.com/mortgage/check-rates"
    querystring = {'loanBalance': loanbalance, 'locationId': locationid, 'homePrice': homeprice, 'locationType': locationtype, 'homeValue': homevalue, }
    if loantype:
        querystring['loanType'] = loantype
    if points:
        querystring['points'] = points
    if veterantype:
        querystring['veteranType'] = veterantype
    if percentagedown:
        querystring['percentageDown'] = percentagedown
    if creditscore:
        querystring['creditScore'] = creditscore
    if loanterms:
        querystring['loanTerms'] = loanterms
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-redfin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_get_popularity_info(listingid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get popularity info relating to a property"
    listingid: The value of listingId field returned in .../properties/list endpoint.
        
    """
    url = f"https://unofficial-redfin.p.rapidapi.com/properties/get-popularity-info"
    querystring = {'listingId': listingid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-redfin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_get_tour_insights(propertyid: str, listingid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tour insights of a property"
    propertyid: The value of propertyId field returned in .../properties/list endpoint.
        listingid: The value of listingId field returned in .../properties/list endpoint.
        
    """
    url = f"https://unofficial-redfin.p.rapidapi.com/properties/get-tour-insights"
    querystring = {'propertyId': propertyid, 'listingId': listingid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-redfin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_get_estimate(listingid: str, propertyid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Redfin's estimate of a property"
    listingid: The value of listingId field returned in .../properties/list endpoint.
        propertyid: The value of propertyId field returned in .../properties/list endpoint.
        
    """
    url = f"https://unofficial-redfin.p.rapidapi.com/properties/get-estimate"
    querystring = {'listingId': listingid, 'propertyId': propertyid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-redfin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_get_mortgage_calculator(listingid: str, propertyid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get mortgage calculator of a property"
    listingid: The value of listingId field returned in .../properties/list endpoint.
        propertyid: The value of propertyId field returned in .../properties/list endpoint.
        
    """
    url = f"https://unofficial-redfin.p.rapidapi.com/properties/get-mortgage-calculator"
    querystring = {'listingId': listingid, 'propertyId': propertyid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-redfin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_get_amenities(propertyid: str, listingid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all amenities of a property"
    propertyid: The value of propertyId field returned in .../properties/list endpoint.
        listingid: The value of listingId field returned in .../properties/list endpoint.
        
    """
    url = f"https://unofficial-redfin.p.rapidapi.com/properties/get-amenities"
    querystring = {'propertyId': propertyid, 'listingId': listingid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-redfin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_get_agent(propertyid: str, listingid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get agent info relating to a property"
    propertyid: The value of propertyId field returned in .../properties/list endpoint.
        listingid: The value of listingId field returned in .../properties/list endpoint.
        
    """
    url = f"https://unofficial-redfin.p.rapidapi.com/properties/get-agent"
    querystring = {'propertyId': propertyid, 'listingId': listingid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-redfin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_get_flood_info(apn: str, lng: int, lat: int, fipscode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get flood info"
    apn: The value of apn field returned in .../properties/get-parcel-bounds endpoint.
        lng: The longitude of GEO location
        lat: The latitude of GEO location
        fipscode: The value of fipsCode field returned in .../properties/get-parcel-bounds endpoint.
        
    """
    url = f"https://unofficial-redfin.p.rapidapi.com/properties/get-flood-info"
    querystring = {'apn': apn, 'lng': lng, 'lat': lat, 'fipsCode': fipscode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-redfin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_get_walk_score(propertyid: str, listingid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get walk score of a property"
    propertyid: The value of propertyId field returned in .../properties/list endpoint.
        listingid: The value of listingId field returned in .../properties/list endpoint.
        
    """
    url = f"https://unofficial-redfin.p.rapidapi.com/properties/get-walk-score"
    querystring = {'propertyId': propertyid, 'listingId': listingid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-redfin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_get_extra_info(listingid: str, propertyid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get extra info of a property"
    listingid: The value of listingId field returned in .../properties/list endpoint.
        propertyid: The value of propertyId field returned in .../properties/list endpoint.
        
    """
    url = f"https://unofficial-redfin.p.rapidapi.com/properties/get-extra-info"
    querystring = {'listingId': listingid, 'propertyId': propertyid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-redfin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_get_main_info(propertyid: str, listingid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get main info of a property"
    propertyid: The value of propertyId field returned in .../properties/list endpoint.
        listingid: The value of listingId field returned in .../properties/list endpoint.
        
    """
    url = f"https://unofficial-redfin.p.rapidapi.com/properties/get-main-info"
    querystring = {'propertyId': propertyid, 'listingId': listingid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-redfin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_complete(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get auto complete suggestions by term or phrase."
    location: Any terms or phrases that you are familiar with.
        
    """
    url = f"https://unofficial-redfin.p.rapidapi.com/auto-complete"
    querystring = {'location': location, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-redfin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

