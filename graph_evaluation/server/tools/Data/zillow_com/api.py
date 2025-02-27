import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def build_web_url(location: str=None, beds: int=None, baths: int=None, home_type: str=None, isauction: bool=None, sqft: int=None, max_price: int=None, min_price: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Beta** 
		Build Web Url."
    location: You can pass: state, city,  zipcode or county.
        home_type: Property type comma-separated or empty for all types:

- Multi-family
- Apartments
- Houses
- Manufactured
- Condos
- LotsLand
- Townhomes
        
    """
    url = f"https://zillow-com1.p.rapidapi.com/buildWebUrl"
    querystring = {}
    if location:
        querystring['location'] = location
    if beds:
        querystring['beds'] = beds
    if baths:
        querystring['baths'] = baths
    if home_type:
        querystring['home_type'] = home_type
    if isauction:
        querystring['isAuction'] = isauction
    if sqft:
        querystring['sqft'] = sqft
    if max_price:
        querystring['max_price'] = max_price
    if min_price:
        querystring['min_price'] = min_price
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ping(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ping"
    
    """
    url = f"https://zillow-com1.p.rapidapi.com/ping"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def building_building_details(lotid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Building details."
    
    """
    url = f"https://zillow-com1.p.rapidapi.com/building"
    querystring = {'lotId': lotid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def zestimatehistory_zestimate_history(property_url: str=None, zpid: int=13172523, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Zestimate History for property by `zpid` or `url`"
    property_url: Property page full URL - `https://www.zillow.com/homedetails/7646-S-Cook-Way-Centennial-CO-80122/13172523_zpid/`
        zpid: Unique ID that Zillow gives to each property.
        
    """
    url = f"https://zillow-com1.p.rapidapi.com/zestimateHistory"
    querystring = {}
    if property_url:
        querystring['property_url'] = property_url
    if zpid:
        querystring['zpid'] = zpid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def walkandtransitscore_walk_bike_and_transit_scores(zpid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Data about walk, bike and transit scores by zpid."
    zpid: Unique ID that Zillow gives to each property.
        
    """
    url = f"https://zillow-com1.p.rapidapi.com/walkAndTransitScore"
    querystring = {'zpid': zpid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locationsuggestions_search_for_location(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a location by name."
    q: State, county, neighborhood, city, street name
        
    """
    url = f"https://zillow-com1.p.rapidapi.com/locationSuggestions"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def propertybycoordinates_search_by_coordinates(long: int, lat: int, d: int=0, includesold: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search the property by coordinates.
		*Note.* The endpoint will return only an array of zpid. To get more information use `/property` endpoint. If you need additional filters, you can use `/propertyExtendedSearch` with a parameter `coordinates` or `polygon`."
    long: Longitude
        lat: Latitude
        d: Diameter in miles. The max value is 0.5, and the low value is 0.05. The default value is 0.1
        includesold: Include to results sold properties.
true or 1 to include (default).
false or 0 to exclude.
        
    """
    url = f"https://zillow-com1.p.rapidapi.com/propertyByCoordinates"
    querystring = {'long': long, 'lat': lat, }
    if d:
        querystring['d'] = d
    if includesold:
        querystring['includeSold'] = includesold
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def propertycomps_property_comps(zpid: int=None, property_url: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get property comps by `zpid` or `property_url`."
    
    """
    url = f"https://zillow-com1.p.rapidapi.com/propertyComps"
    querystring = {}
    if zpid:
        querystring['zpid'] = zpid
    if property_url:
        querystring['property_url'] = property_url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def propertybymls_search_by_mls(mls: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for the property by MLS #.
		*Note.* The endpoint will return an array of `zpid`. To get more information, use `/property` endpoint.
		If we find more than one address with the given MLS, we will send them in `otherAddress` key. You can check them for additional."
    mls: MLS #
        
    """
    url = f"https://zillow-com1.p.rapidapi.com/propertyByMls"
    querystring = {'mls': mls, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def priceandtaxhistory_price_tax_history(property_url: str=None, zpid: str='49000475', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "History of Property Taxes and Prices.
		`zpid` or `property_url` is required."
    
    """
    url = f"https://zillow-com1.p.rapidapi.com/priceAndTaxHistory"
    querystring = {}
    if property_url:
        querystring['property_url'] = property_url
    if zpid:
        querystring['zpid'] = zpid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchbyurl_list_of_properties_by_url(url: str, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of properties by providing Zillow's search results URL"
    url: This URL you can get from browser address bar after you apply all parameters on Zillow site.
        
    """
    url = f"https://zillow-com1.p.rapidapi.com/searchByUrl"
    querystring = {'url': url, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def images_property_images_and_videos(zpid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Property images and videos."
    
    """
    url = f"https://zillow-com1.p.rapidapi.com/images"
    querystring = {'zpid': zpid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_property_details(property_url: str=None, zpid: int=2080998890, address: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Property details by zpid or url or address. Zestimate value."
    property_url: Full page URL - `https://www.zillow.com/homedetails/101-California-Ave-UNIT-506-Santa-Monica-CA-90403/20485717_zpid/`
        zpid: Unique ID that Zillow gives to each property.
        
    """
    url = f"https://zillow-com1.p.rapidapi.com/property"
    querystring = {}
    if property_url:
        querystring['property_url'] = property_url
    if zpid:
        querystring['zpid'] = zpid
    if address:
        querystring['address'] = address
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rentestimate_rent_estimate(propertytype: str, address: str='1093 County Route 60, Newton Falls', lat: int=None, long: int=None, sqftmax: int=None, beds: int=None, baths: int=None, d: int=0, sqftmin: int=None, includecomps: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**BETA**
		*`address` or `long` and `lat` is required!*
		
		For more accuracy use  `beds` and `baths` parameters.
		Rent estimates and comparable rentals."
    d: Diameter in miles. The max and value is 0.5, and the low value is 0.05. The default value is 0.5
        
    """
    url = f"https://zillow-com1.p.rapidapi.com/rentEstimate"
    querystring = {'propertyType': propertytype, }
    if address:
        querystring['address'] = address
    if lat:
        querystring['lat'] = lat
    if long:
        querystring['long'] = long
    if sqftmax:
        querystring['sqftMax'] = sqftmax
    if beds:
        querystring['beds'] = beds
    if baths:
        querystring['baths'] = baths
    if d:
        querystring['d'] = d
    if sqftmin:
        querystring['sqftMin'] = sqftmin
    if includecomps:
        querystring['includeComps'] = includecomps
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def marketlocation(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**beta version**
		
		Get `resourceId` for city you want."
    location: Search by city or ZIP
        
    """
    url = f"https://zillow-com1.p.rapidapi.com/marketLocation"
    querystring = {'location': location, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def marketdata(resourceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**beta version**
		
		Rental market summary and year trends. Use zip code as `resourceId` or get `resourceId` for your city in response data from `/marketLocation`."
    resourceid: Get it from the endpoint `/marketLocation` response.
        
    """
    url = f"https://zillow-com1.p.rapidapi.com/marketData"
    querystring = {'resourceId': resourceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def propertyextendedsearch_extended_search(ismountainview: bool=None, sort: str=None, lotsizemax: str=None, iswaterview: bool=None, sqftmax: int=None, minprice: int=None, salebyowner: str=None, haspool: bool=None, is3dhome: bool=None, polygon: str=None, status_type: str=None, home_type: str='Houses', location: str='santa monica, ca', rentminprice: int=None, maxprice: int=None, rentmaxprice: int=None, page: int=None, sqftmin: int=None, bedsmax: int=None, dayson: str=None, isacceptingbackupoffers: int=None, ispendingundercontract: int=None, soldinlast: str=None, iscomingsoon: bool=None, isnewconstruction: bool=None, lotsizemin: str=None, keywords: str=None, otherlistings: int=None, isforsaleforeclosure: bool=None, coordinates: str=None, isparkview: bool=None, schools: str=None, isauction: bool=None, schoolsrating: str=None, parkingspots: int=None, iswaterfront: bool=None, bathsmax: int=None, bathsmin: int=None, includehomeswithnohoadata: bool=None, hasgarage: bool=None, buildyearmin: int=None, bedsmin: int=None, buildyearmax: int=None, isbasementunfinished: int=None, isbasementfinished: int=None, salebyagent: str=None, iscityview: bool=None, hasairconditioning: bool=None, includeunratedschools: bool=None, isopenhousesonly: bool=None, hoa: int=None, isforeclosed: bool=None, ispreforeclosure: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for properties by parameters.
		*Note.* If you search by exact address, the endpoint will return only zpid, or list of `zpid`s if it's a building with many units or `lotId`s for building.
		**To get more information about a property by `zpid`, use a `/property` endpoint. For `lotId` use a `/building` endpoint.**
		`location` is required if `polygon` and `coordinates` are empty."
    sort: For `status_type` = `ForSale` OR `RecentlySold` are available:

- `Homes_for_You`
- `Price_High_Low`
- `Price_Low_High`
- `Newest`
- `Bedrooms`
- `Bathrooms`
- `Square_Feet`
- `Lot_Size`

default `Homes_for_You`

For `status_type` = `ForRent` are available:
- `Verified_Source`
- `Payment_High_Low`
- `Payment_Low_High`
- `Newest`
- `Bedrooms`
- `Bathrooms`
- `Square_Feet`
- `Lot_Size`

default `Verified_Source`
        lotsizemax: Available values:

- 1,000 sqft
- 2,000 sqft
- 3,000 sqft
- 4,000 sqft
- 5,000 sqft
- 7,500 sqft
- 1/4 acre/10,890 sqft
- 1/2 acre/21,780 sqft
- 1 acre/43,560 sqft
- 2 acres/87,120 sqft
- 5 acres/217,800 sqft
- 10 acres/435,600 sqft
- 20 acres/871,200 sqft
- 50 acres/2,178,000 sqft
- 100 acres/4,356,000 sqft
        sqftmax: Square Feet max value.
        minprice: If `status_type` = `ForSale` OR `RecentlySold` you can filter by min price.
        salebyowner: Default - `true`. If you only need to get `FSBA` set to false or 0.
        is3dhome: Must have 3D Tour
        polygon: Format: `lon lat,lon1 lat1,lon2 lat2`
It is required if the `location` or `coordinates` is empty.
*The last pair must be the same as the first pair.*
        home_type: Property type comma-separated or empty for all types:
**For Rent** 

- `Townhomes`
- `Houses`
- `Apartments_Condos_Co-ops`

**For others:**

- `Multi-family`
- `Apartments`
- `Houses`
- `Manufactured`
- `Condos`
- `LotsLand`
- `Townhomes`
        location: Location details, address, county, neighborhood or Zip code.
It is required if the `polygon` or `coordinates` is empty.
Max length: 100
        rentminprice: If `status_type` = `ForRent` you can filter by min rent price.
        maxprice: If `status_type` = `ForSale` OR `RecentlySold` you can filter by max price.
        rentmaxprice: If `status_type` = `ForRent` you can filter by max rent price.
        page: Page number if at the previous response `totalPages` > 1.
Max value is 20.

*To be able to access more data, you can break down your request by dividing it into price groups using the minPrice and maxPrice parameters.
For example 0 - 100,000, 100,001 - 500,000, 500,001 - 800,000 and so on.
This trick will help you to get more data.*
        sqftmin: Square Feet min value
        bedsmax: Bedrooms max count
        dayson: Days on Z. Use with `status_type`='ForSale' or `status_type`='ForRent'
        isacceptingbackupoffers: Accepting Backup Offers filter. **Set it to 1 if needed.**
        ispendingundercontract: Pending & Under Contract filter. **Set it to 1 if needed.**
        soldinlast: 'Sold In Last' on Z. Use with `status_type`='RecentlySold'.
        iscomingsoon: Coming Soon listings are homes that will soon be on the market.. **Set it to 1 if needed.**
        isnewconstruction: New Construction filter. Set it to 1 if you only need properties with `New Construction` status.
        lotsizemin: Available values: 

- 1,000 sqft
- 2,000 sqft
- 3,000 sqft
- 4,000 sqft
- 5,000 sqft
- 7,500 sqft
- 1/4 acre/10,890 sqft
- 1/2 acre/21,780 sqft
- 1 acre/43,560 sqft
        keywords: Filter with keywords.
        otherlistings: If set to 1, the results will only include data from the `Other Listings` tab.
        isforsaleforeclosure: If you only need to get ForSaleForeclosure set to true or 1.
        coordinates: It is required if the `location` or `polygon` is empty.

Format: `lon lat,diameter`. Diameter in miles from 1 to 99
**-118.51750373840332 34.007063913440916**,*20*
        schools: Available values: `elementary, public, private, middle, charter, high`
For multiple selection, separate with comma: `middle,high`
        isauction: Auctions. Default `true`.
        schoolsrating: Min school ratings. From 1 to 10
        parkingspots: Parking Spots. Max value - 4
        bathsmax: Bathrooms max count
        bathsmin: Bathrooms min count
        includehomeswithnohoadata: Default - `true`. 
        hasgarage: Must have a garage. Default value `false`
        buildyearmin: Year Built min value.

        bedsmin: Bedrooms min count
        buildyearmax: Year Built max value.
        isbasementunfinished: Basement filter. **Set it to 1 if needed.**
        isbasementfinished: Basement filter. **Set it to 1 if needed.**
        salebyagent: Default - `true`. If you only need to get `FSBO` set to false or 0.
        includeunratedschools: Include schools with no rating
        isopenhousesonly: Must have open house
        hoa: Max HOA.
        
    """
    url = f"https://zillow-com1.p.rapidapi.com/propertyExtendedSearch"
    querystring = {}
    if ismountainview:
        querystring['isMountainView'] = ismountainview
    if sort:
        querystring['sort'] = sort
    if lotsizemax:
        querystring['lotSizeMax'] = lotsizemax
    if iswaterview:
        querystring['isWaterView'] = iswaterview
    if sqftmax:
        querystring['sqftMax'] = sqftmax
    if minprice:
        querystring['minPrice'] = minprice
    if salebyowner:
        querystring['saleByOwner'] = salebyowner
    if haspool:
        querystring['hasPool'] = haspool
    if is3dhome:
        querystring['is3dHome'] = is3dhome
    if polygon:
        querystring['polygon'] = polygon
    if status_type:
        querystring['status_type'] = status_type
    if home_type:
        querystring['home_type'] = home_type
    if location:
        querystring['location'] = location
    if rentminprice:
        querystring['rentMinPrice'] = rentminprice
    if maxprice:
        querystring['maxPrice'] = maxprice
    if rentmaxprice:
        querystring['rentMaxPrice'] = rentmaxprice
    if page:
        querystring['page'] = page
    if sqftmin:
        querystring['sqftMin'] = sqftmin
    if bedsmax:
        querystring['bedsMax'] = bedsmax
    if dayson:
        querystring['daysOn'] = dayson
    if isacceptingbackupoffers:
        querystring['isAcceptingBackupOffers'] = isacceptingbackupoffers
    if ispendingundercontract:
        querystring['isPendingUnderContract'] = ispendingundercontract
    if soldinlast:
        querystring['soldInLast'] = soldinlast
    if iscomingsoon:
        querystring['isComingSoon'] = iscomingsoon
    if isnewconstruction:
        querystring['isNewConstruction'] = isnewconstruction
    if lotsizemin:
        querystring['lotSizeMin'] = lotsizemin
    if keywords:
        querystring['keywords'] = keywords
    if otherlistings:
        querystring['otherListings'] = otherlistings
    if isforsaleforeclosure:
        querystring['isForSaleForeclosure'] = isforsaleforeclosure
    if coordinates:
        querystring['coordinates'] = coordinates
    if isparkview:
        querystring['isParkView'] = isparkview
    if schools:
        querystring['schools'] = schools
    if isauction:
        querystring['isAuction'] = isauction
    if schoolsrating:
        querystring['schoolsRating'] = schoolsrating
    if parkingspots:
        querystring['parkingSpots'] = parkingspots
    if iswaterfront:
        querystring['isWaterfront'] = iswaterfront
    if bathsmax:
        querystring['bathsMax'] = bathsmax
    if bathsmin:
        querystring['bathsMin'] = bathsmin
    if includehomeswithnohoadata:
        querystring['includeHomesWithNoHoaData'] = includehomeswithnohoadata
    if hasgarage:
        querystring['hasGarage'] = hasgarage
    if buildyearmin:
        querystring['buildYearMin'] = buildyearmin
    if bedsmin:
        querystring['bedsMin'] = bedsmin
    if buildyearmax:
        querystring['buildYearMax'] = buildyearmax
    if isbasementunfinished:
        querystring['isBasementUnfinished'] = isbasementunfinished
    if isbasementfinished:
        querystring['isBasementFinished'] = isbasementfinished
    if salebyagent:
        querystring['saleByAgent'] = salebyagent
    if iscityview:
        querystring['isCityView'] = iscityview
    if hasairconditioning:
        querystring['hasAirConditioning'] = hasairconditioning
    if includeunratedschools:
        querystring['includeUnratedSchools'] = includeunratedschools
    if isopenhousesonly:
        querystring['isOpenHousesOnly'] = isopenhousesonly
    if hoa:
        querystring['hoa'] = hoa
    if isforeclosed:
        querystring['isForeclosed'] = isforeclosed
    if ispreforeclosure:
        querystring['isPreForeclosure'] = ispreforeclosure
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lender_details(screenname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get lender details"
    
    """
    url = f"https://zillow-com1.p.rapidapi.com/lender/details"
    querystring = {'screenName': screenname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lender_reviews(lenderid: str, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reviews"
    
    """
    url = f"https://zillow-com1.p.rapidapi.com/lender/reviews"
    querystring = {'lenderId': lenderid, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lender_search(location: str, lendername: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for lender"
    location: City, State or Zip.
Only lenders licensed in the state will be displayed.
        
    """
    url = f"https://zillow-com1.p.rapidapi.com/lender/search"
    querystring = {'location': location, }
    if lendername:
        querystring['lenderName'] = lendername
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def otherprofessionals_reviews(zuid: str, page: int=None, size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get reviews"
    zuid: `zuid` from search response.
        size: Max value - 20
        
    """
    url = f"https://zillow-com1.p.rapidapi.com/otherProfessionals/reviews"
    querystring = {'zuid': zuid, }
    if page:
        querystring['page'] = page
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def otherprofessionals_search(type: str, location: str, page: int=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for other professionals. Check the `type` parameter."
    location: Neighborhood /City /Zip
        
    """
    url = f"https://zillow-com1.p.rapidapi.com/otherProfessionals/search"
    querystring = {'type': type, 'location': location, }
    if page:
        querystring['page'] = page
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agentsoldlistings_agent_s_sold_listings(zuid: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Listings of sold property by agent (`zuid`)"
    zuid: Agent unique id - `zuid`
        
    """
    url = f"https://zillow-com1.p.rapidapi.com/agentSoldListings"
    querystring = {'zuid': zuid, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agentactivelistings_agent_s_active_listings(zuid: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Agent's active listings.
		The endpoint will return a list of the property. To get a property details use `/property (Zillow property details)` endpoint."
    zuid: Agent unique id - `zuid`
        
    """
    url = f"https://zillow-com1.p.rapidapi.com/agentActiveListings"
    querystring = {'zuid': zuid, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agentreviews_agent_reviews(zuid: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Agent reviews"
    zuid: Agent unique id - `zuid`
        
    """
    url = f"https://zillow-com1.p.rapidapi.com/agentReviews"
    querystring = {'zuid': zuid, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findagent(lat: str=None, name: str='Regina Vannicola', locationtext: str=None, page: int=1, lng: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find agent by `name` or `location name` or `lat` and `lng`"
    lat: e.g. `34.010116`
        name: Search by name
e.g. `Regina Vannicola`
        locationtext: e.g. `Newport Beach` or zip code `90278`
        lng: e.g. `-118.498786`
        
    """
    url = f"https://zillow-com1.p.rapidapi.com/findAgent"
    querystring = {}
    if lat:
        querystring['lat'] = lat
    if name:
        querystring['name'] = name
    if locationtext:
        querystring['locationText'] = locationtext
    if page:
        querystring['page'] = page
    if lng:
        querystring['lng'] = lng
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agentdetails_agent_details_by_username(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get agent details by username (contact details, active listings, reviews, etc)."
    
    """
    url = f"https://zillow-com1.p.rapidapi.com/agentDetails"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def similarforrent_get_similar_properties_for_rent(property_url: str=None, zpid: int=77396, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get similar properties for rent. `zpid` or `property_url` is required parameter."
    property_url: Full page URL - `https://www.zillow.com/homedetails/102-Griffith-Ave-Prattville-AL-36066/77224_zpid/`
        zpid: You can get it from `/propertyExtendedSearch` or `/propertyByCoordinates` endpoints, or extract it from a full URL.
        
    """
    url = f"https://zillow-com1.p.rapidapi.com/similarForRent"
    querystring = {}
    if property_url:
        querystring['property_url'] = property_url
    if zpid:
        querystring['zpid'] = zpid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getjobresults_get_the_result_data_by_jobnumber(jobnumber: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the result data by `jobNumber`."
    
    """
    url = f"https://zillow-com1.p.rapidapi.com/getJobResults"
    querystring = {'jobNumber': jobnumber, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def removejob_use_for_remove_job_from_queue(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use for remove job from queue by id."
    id: Job id
        
    """
    url = f"https://zillow-com1.p.rapidapi.com/removeJob"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agentrentallistings_agent_s_rental_listings(zuid: str, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Listings of rental property by agent (`zuid`)"
    zuid: Agent unique id - `zuid`
        
    """
    url = f"https://zillow-com1.p.rapidapi.com/agentRentalListings"
    querystring = {'zuid': zuid, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def similarproperty_get_similar_properties_for_sale(zpid: int=2080998890, property_url: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get similar properties for sale. `zpid` or `property_url` is required parameter."
    zpid: You can get it from `/propertyExtendedSearch` or `/propertyByCoordinates` endpoints, or extract it from a full URL.
        property_url: Full page URL - `https://www.zillow.com/homedetails/101-California-Ave-UNIT-506-Santa-Monica-CA-90403/20485717_zpid/`
        
    """
    url = f"https://zillow-com1.p.rapidapi.com/similarProperty"
    querystring = {}
    if zpid:
        querystring['zpid'] = zpid
    if property_url:
        querystring['property_url'] = property_url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def similarsales_recently_sold_homes_with_similar_features(property_url: str=None, zpid: int=19959099, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Recently sold homes with similar features to those passed by zpid home, such as bedrooms, bathrooms, location and square footage. `zpid` or `property_url` is required parameter."
    property_url: Full page URL - `https://www.zillow.com/homedetails/7301-Lennox-Ave-UNIT-D06-Los-Angeles-CA-91405/19959099_zpid/`
        zpid: You can get it from `/propertyExtendedSearch` or `/propertyByCoordinates` endpoints, or extract it from a full URL.
        
    """
    url = f"https://zillow-com1.p.rapidapi.com/similarSales"
    querystring = {}
    if property_url:
        querystring['property_url'] = property_url
    if zpid:
        querystring['zpid'] = zpid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

