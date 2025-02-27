import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_for_properties_by_coordinates(d: int, lat: int, long: int, keywords: str=None, greatschoolsrating_min: int=None, iselementaryschool: bool=None, isparkview: bool=None, onlyrentalsmalldogsallowed: bool=None, baths_min: int=None, sort: str=None, built_min: int=None, issinglefamily: bool=None, page: int=None, ispublicschool: bool=None, ismultifamily: bool=None, isauction: bool=None, onlyrentallargedogsallowed: bool=None, iswaterview: bool=None, ismountainview: bool=None, isnewconstruction: bool=None, onlyrentalpetsallowed: bool=None, onlyrentalcatsallowed: bool=None, onlyrentalincomerestricted: bool=None, iscomingsoon: bool=None, ismiddleschool: bool=None, isforsaleforeclosure: bool=None, onlywithphotos: bool=None, hasairconditioning: bool=None, isprivateschool: bool=None, hoa_max: int=None, iszillowownedonly: bool=None, built_max: str=None, onlyrentalacceptsapplications: bool=None, hasgarage: bool=None, iscityview: bool=None, onlypricereduction: bool=None, sqft_min: int=None, isforsalebyowner: bool=None, singlestory: bool=None, isforsalebyagent: bool=None, monthlypayment_min: int=None, haspool: bool=None, hoa_min: int=None, monthlypayment_max: int=None, baths_max: int=None, beds_min: int=None, listing_type: str=None, beds_max: int=None, isapartment: bool=None, ismanufactured: bool=None, istownhouse: bool=None, doz: str=None, price_max: int=None, price_min: int=None, islotland: bool=None, iscondo: bool=None, status: str=None, output: str=None, iswaterfront: bool=None, enableschools: bool=None, is3dhome: bool=None, sqft_max: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for filtered properties by coordinates.
		You can select the output format (JSON , CSV , XLSX) using the optional "output" parameter."
    d: Diameter in miles.
        lat: Latitude
        long: Longitude 
        sort: Sorting possible values :
priorityscore: Default,
saved: Date Saved,
listingstatus: Listing Status,
mostrecentchange: Most Recent Change,
globalrelevanceex: Homes for You,
featured: Verified Source,
priced: Price (High to Low),
pricea: Price (Low to High),
paymentd: Payment (High to Low),
paymenta: Payment (Low to High),
days: Newest,
beds: Bedrooms,
baths: Bathrooms,
size: Square Feet,
lot: Lot Size,
zest: Zestimate (High to Low),
zesta: Zestimate (Low to High),
        listing_type: Listing Type possible values :
By agent (Default value)
By owner & other (for off market properties)
        doz: Days on Zillow possible values :
any: Any,
1: 1 day,
7: 7 days,
14: 14 days,
30: 30 days,
90: 90 days,
6m: 6 months,
12m: 12 months,
24m: 24 months,
36m: 36 months
        status: Status type of the properties
Default : forSale
-forSale
-forRent
-recentlySold
        output: Output format possible values :
json (Default value) :Data in a JSON format
csv : URL to the generated CSV file
xlsx : URL to the generated XLSX (excel) file
        
    """
    url = f"https://zillow56.p.rapidapi.com/search_coordinates"
    querystring = {'d': d, 'lat': lat, 'long': long, }
    if keywords:
        querystring['keywords'] = keywords
    if greatschoolsrating_min:
        querystring['greatSchoolsRating_min'] = greatschoolsrating_min
    if iselementaryschool:
        querystring['isElementarySchool'] = iselementaryschool
    if isparkview:
        querystring['isParkView'] = isparkview
    if onlyrentalsmalldogsallowed:
        querystring['onlyRentalSmallDogsAllowed'] = onlyrentalsmalldogsallowed
    if baths_min:
        querystring['baths_min'] = baths_min
    if sort:
        querystring['sort'] = sort
    if built_min:
        querystring['built_min'] = built_min
    if issinglefamily:
        querystring['isSingleFamily'] = issinglefamily
    if page:
        querystring['page'] = page
    if ispublicschool:
        querystring['isPublicSchool'] = ispublicschool
    if ismultifamily:
        querystring['isMultiFamily'] = ismultifamily
    if isauction:
        querystring['isAuction'] = isauction
    if onlyrentallargedogsallowed:
        querystring['onlyRentalLargeDogsAllowed'] = onlyrentallargedogsallowed
    if iswaterview:
        querystring['isWaterView'] = iswaterview
    if ismountainview:
        querystring['isMountainView'] = ismountainview
    if isnewconstruction:
        querystring['isNewConstruction'] = isnewconstruction
    if onlyrentalpetsallowed:
        querystring['onlyRentalPetsAllowed'] = onlyrentalpetsallowed
    if onlyrentalcatsallowed:
        querystring['onlyRentalCatsAllowed'] = onlyrentalcatsallowed
    if onlyrentalincomerestricted:
        querystring['onlyRentalIncomeRestricted'] = onlyrentalincomerestricted
    if iscomingsoon:
        querystring['isComingSoon'] = iscomingsoon
    if ismiddleschool:
        querystring['isMiddleSchool'] = ismiddleschool
    if isforsaleforeclosure:
        querystring['isForSaleForeclosure'] = isforsaleforeclosure
    if onlywithphotos:
        querystring['onlyWithPhotos'] = onlywithphotos
    if hasairconditioning:
        querystring['hasAirConditioning'] = hasairconditioning
    if isprivateschool:
        querystring['isPrivateSchool'] = isprivateschool
    if hoa_max:
        querystring['hoa_max'] = hoa_max
    if iszillowownedonly:
        querystring['isZillowOwnedOnly'] = iszillowownedonly
    if built_max:
        querystring['built_max'] = built_max
    if onlyrentalacceptsapplications:
        querystring['onlyRentalAcceptsApplications'] = onlyrentalacceptsapplications
    if hasgarage:
        querystring['hasGarage'] = hasgarage
    if iscityview:
        querystring['isCityView'] = iscityview
    if onlypricereduction:
        querystring['onlyPriceReduction'] = onlypricereduction
    if sqft_min:
        querystring['sqft_min'] = sqft_min
    if isforsalebyowner:
        querystring['isForSaleByOwner'] = isforsalebyowner
    if singlestory:
        querystring['singleStory'] = singlestory
    if isforsalebyagent:
        querystring['isForSaleByAgent'] = isforsalebyagent
    if monthlypayment_min:
        querystring['monthlyPayment_min'] = monthlypayment_min
    if haspool:
        querystring['hasPool'] = haspool
    if hoa_min:
        querystring['hoa_min'] = hoa_min
    if monthlypayment_max:
        querystring['monthlyPayment_max'] = monthlypayment_max
    if baths_max:
        querystring['baths_max'] = baths_max
    if beds_min:
        querystring['beds_min'] = beds_min
    if listing_type:
        querystring['listing_type'] = listing_type
    if beds_max:
        querystring['beds_max'] = beds_max
    if isapartment:
        querystring['isApartment'] = isapartment
    if ismanufactured:
        querystring['isManufactured'] = ismanufactured
    if istownhouse:
        querystring['isTownhouse'] = istownhouse
    if doz:
        querystring['doz'] = doz
    if price_max:
        querystring['price_max'] = price_max
    if price_min:
        querystring['price_min'] = price_min
    if islotland:
        querystring['isLotLand'] = islotland
    if iscondo:
        querystring['isCondo'] = iscondo
    if status:
        querystring['status'] = status
    if output:
        querystring['output'] = output
    if iswaterfront:
        querystring['isWaterfront'] = iswaterfront
    if enableschools:
        querystring['enableSchools'] = enableschools
    if is3dhome:
        querystring['is3dHome'] = is3dhome
    if sqft_max:
        querystring['sqft_max'] = sqft_max
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow56.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_properties_by_url(url: str, listing_type: str=None, output: str=None, page: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of properties by providing the zillow search results URL
		You can select the output format (JSON , CSV , XLSX) using the optional "output" parameter."
    listing_type: Listing Type possible values :
By agent (Default value)
By owner & other (for off market properties)
        output: Output format possible values :
json (Default value) :Data in a JSON format
csv : URL to the generated CSV file
xlsx : URL to the generated XLSX (excel) file
        
    """
    url = f"https://zillow56.p.rapidapi.com/search_url"
    querystring = {'url': url, }
    if listing_type:
        querystring['listing_type'] = listing_type
    if output:
        querystring['output'] = output
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow56.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_properties(location: str, isforsalebyagent: bool=None, istownhouse: bool=None, iscondo: bool=None, greatschoolsrating_min: int=None, isnewconstruction: bool=None, isauction: bool=None, keywords: str=None, iszillowownedonly: bool=None, onlyrentalincomerestricted: bool=None, onlyrentallargedogsallowed: bool=None, is3dhome: bool=None, iscityview: bool=None, doz: str=None, sortselection: str=None, price_min: int=None, lotsize_min: int=None, enableschools: bool=None, iselementaryschool: bool=None, lotsize_max: int=None, isparkview: bool=None, onlyrentalacceptsapplications: bool=None, onlyrentalpetsallowed: bool=None, onlyrentalsmalldogsallowed: bool=None, onlyrentalcatsallowed: bool=None, onlypricereduction: bool=None, iswaterview: bool=None, ismiddleschool: bool=None, ismountainview: bool=None, ispublicschool: bool=None, hasairconditioning: bool=None, singlestory: bool=None, iscomingsoon: bool=None, isapartment: bool=None, islotland: bool=None, baths_max: int=None, listing_type: str=None, hasgarage: bool=None, issinglefamily: bool=None, beds_min: int=None, haspool: bool=None, price_max: int=None, sqft_min: int=None, monthlypayment_max: int=None, status: str=None, monthlypayment_min: int=None, output: str=None, sqft_max: int=None, ismanufactured: bool=None, isforsalebyowner: bool=None, iswaterfront: bool=None, hoa_max: int=None, beds_max: int=None, ismultifamily: bool=None, built_min: int=None, baths_min: int=None, page: int=None, isprivateschool: bool=None, isforsaleforeclosure: bool=None, built_max: str=None, onlywithphotos: bool=None, hoa_min: int=None, parkingspots_min: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for filtered properties by address, neighborhood, city, or ZIP code.
		PS : Searching by address would return the details of the property instead of a list
		For a list of properties, you can select the output format (JSON , CSV , XLSX) using the optional "output" parameter."
    location: Location can be an address, neighborhood, city, or ZIP code.
        doz: Days on Zillow (For Sale/Rent listings)/ Sold In Last (Sold listings) 
Possible values :
any: Any,
1: 1 day,
7: 7 days,
14: 14 days,
30: 30 days,
90: 90 days,
6m: 6 months,
12m: 12 months,
24m: 24 months,
36m: 36 months
        sortselection: Sorting possible values :
days: Newest (Default value),
saved: Date Saved,
listingstatus: Listing Status,
mostrecentchange: Most Recent Change,
globalrelevanceex: Homes for You,
featured: Verified Source,
priced: Price (High to Low),
pricea: Price (Low to High),
paymentd: Payment (High to Low),
paymenta: Payment (Low to High),
beds: Bedrooms,
baths: Bathrooms,
size: Square Feet,
lot: Lot Size,
zest: Zestimate (High to Low),
zesta: Zestimate (Low to High),
        lotsize_min: in sqft
        lotsize_max: in sqft
        listing_type: Listing Type possible values :
By agent (Default value)
By owner & other (for off market properties)
        status: Status type of the properties
Default : forSale
-forSale
-forRent
-recentlySold
        output: Output format possible values :
json (Default value) :Data in a JSON format
csv : URL to the generated CSV file
xlsx : URL to the generated XLSX (excel) file
        
    """
    url = f"https://zillow56.p.rapidapi.com/search"
    querystring = {'location': location, }
    if isforsalebyagent:
        querystring['isForSaleByAgent'] = isforsalebyagent
    if istownhouse:
        querystring['isTownhouse'] = istownhouse
    if iscondo:
        querystring['isCondo'] = iscondo
    if greatschoolsrating_min:
        querystring['greatSchoolsRating_min'] = greatschoolsrating_min
    if isnewconstruction:
        querystring['isNewConstruction'] = isnewconstruction
    if isauction:
        querystring['isAuction'] = isauction
    if keywords:
        querystring['keywords'] = keywords
    if iszillowownedonly:
        querystring['isZillowOwnedOnly'] = iszillowownedonly
    if onlyrentalincomerestricted:
        querystring['onlyRentalIncomeRestricted'] = onlyrentalincomerestricted
    if onlyrentallargedogsallowed:
        querystring['onlyRentalLargeDogsAllowed'] = onlyrentallargedogsallowed
    if is3dhome:
        querystring['is3dHome'] = is3dhome
    if iscityview:
        querystring['isCityView'] = iscityview
    if doz:
        querystring['doz'] = doz
    if sortselection:
        querystring['sortSelection'] = sortselection
    if price_min:
        querystring['price_min'] = price_min
    if lotsize_min:
        querystring['lotSize_min'] = lotsize_min
    if enableschools:
        querystring['enableSchools'] = enableschools
    if iselementaryschool:
        querystring['isElementarySchool'] = iselementaryschool
    if lotsize_max:
        querystring['lotSize_max'] = lotsize_max
    if isparkview:
        querystring['isParkView'] = isparkview
    if onlyrentalacceptsapplications:
        querystring['onlyRentalAcceptsApplications'] = onlyrentalacceptsapplications
    if onlyrentalpetsallowed:
        querystring['onlyRentalPetsAllowed'] = onlyrentalpetsallowed
    if onlyrentalsmalldogsallowed:
        querystring['onlyRentalSmallDogsAllowed'] = onlyrentalsmalldogsallowed
    if onlyrentalcatsallowed:
        querystring['onlyRentalCatsAllowed'] = onlyrentalcatsallowed
    if onlypricereduction:
        querystring['onlyPriceReduction'] = onlypricereduction
    if iswaterview:
        querystring['isWaterView'] = iswaterview
    if ismiddleschool:
        querystring['isMiddleSchool'] = ismiddleschool
    if ismountainview:
        querystring['isMountainView'] = ismountainview
    if ispublicschool:
        querystring['isPublicSchool'] = ispublicschool
    if hasairconditioning:
        querystring['hasAirConditioning'] = hasairconditioning
    if singlestory:
        querystring['singleStory'] = singlestory
    if iscomingsoon:
        querystring['isComingSoon'] = iscomingsoon
    if isapartment:
        querystring['isApartment'] = isapartment
    if islotland:
        querystring['isLotLand'] = islotland
    if baths_max:
        querystring['baths_max'] = baths_max
    if listing_type:
        querystring['listing_type'] = listing_type
    if hasgarage:
        querystring['hasGarage'] = hasgarage
    if issinglefamily:
        querystring['isSingleFamily'] = issinglefamily
    if beds_min:
        querystring['beds_min'] = beds_min
    if haspool:
        querystring['hasPool'] = haspool
    if price_max:
        querystring['price_max'] = price_max
    if sqft_min:
        querystring['sqft_min'] = sqft_min
    if monthlypayment_max:
        querystring['monthlyPayment_max'] = monthlypayment_max
    if status:
        querystring['status'] = status
    if monthlypayment_min:
        querystring['monthlyPayment_min'] = monthlypayment_min
    if output:
        querystring['output'] = output
    if sqft_max:
        querystring['sqft_max'] = sqft_max
    if ismanufactured:
        querystring['isManufactured'] = ismanufactured
    if isforsalebyowner:
        querystring['isForSaleByOwner'] = isforsalebyowner
    if iswaterfront:
        querystring['isWaterfront'] = iswaterfront
    if hoa_max:
        querystring['hoa_max'] = hoa_max
    if beds_max:
        querystring['beds_max'] = beds_max
    if ismultifamily:
        querystring['isMultiFamily'] = ismultifamily
    if built_min:
        querystring['built_min'] = built_min
    if baths_min:
        querystring['baths_min'] = baths_min
    if page:
        querystring['page'] = page
    if isprivateschool:
        querystring['isPrivateSchool'] = isprivateschool
    if isforsaleforeclosure:
        querystring['isForSaleForeclosure'] = isforsaleforeclosure
    if built_max:
        querystring['built_max'] = built_max
    if onlywithphotos:
        querystring['onlyWithPhotos'] = onlywithphotos
    if hoa_min:
        querystring['hoa_min'] = hoa_min
    if parkingspots_min:
        querystring['parkingSpots_min'] = parkingspots_min
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow56.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def zestimate_history(zpid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Zestimate history by zpid"
    
    """
    url = f"https://zillow56.p.rapidapi.com/zestimate_history"
    querystring = {'zpid': zpid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow56.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def photos(zpid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a property's photos with different sizes and types."
    
    """
    url = f"https://zillow56.p.rapidapi.com/photos"
    querystring = {'zpid': zpid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow56.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_details(zpid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a property's details by its zpid"
    
    """
    url = f"https://zillow56.p.rapidapi.com/property"
    querystring = {'zpid': zpid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow56.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def walk_transit_and_bike_score(zpid: int=20485700, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Walk, Transit and Bike Score of a property by zpid"
    
    """
    url = f"https://zillow56.p.rapidapi.com/walk_transit_bike_score"
    querystring = {}
    if zpid:
        querystring['zpid'] = zpid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow56.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rent_zestimate_and_comparable_properties(address: str, bedrooms: str=None, pets: str=None, amenities: str=None, laundry: str=None, propertytypes: str=None, activetypes: str=None, deactivateddays: str=None, activateddays: str=None, distanceinmiles: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a property's rent zestimate and it's comparable properties in the same area."
    bedrooms: [SIMILARFLOORPLANS] 
Filter for number of bedrooms: (To choose multiple values separate with comma eg : 0,1,2)
Possible values:
**0
1
2
3
4plus**

        pets: [SIMILARFLOORPLANS] 
Filter for Pets: (To choose multiple values separate with comma eg : dogs,cats)
Possible values:
**any (Default)
dogs
cats**

        amenities: [SIMILARFLOORPLANS] 
Filter for amenities: (To choose multiple values separate with comma eg : cooling,parking)
Possible values:
**any (Default)
cooling
heating
parking**

        laundry: [SIMILARFLOORPLANS] 
Filter for Laundry: (To choose multiple values separate with comma eg : inUnit,shared)
Possible values:
**any (Default)
inUnit
shared**

        propertytypes: [SIMILARFLOORPLANS] 
Filter for Property Types: (To choose multiple values separate with comma eg : house,condo)
Possible values:
**any (Default)
apartment
house
townhouse
condo**

        activetypes: SimilarFloorPlans filter:
Possible values:
any (Default)
active (Active Rentals)
inactive (Inactive Rentals)

        deactivateddays: [SIMILARFLOORPLANS]
Filter for inactive rentals within X days:
Possible values:
30 (Within 30 days (max))
15 (Within 15 days)
7 (Within 7 days)

        activateddays: [SIMILARFLOORPLANS]
Filter for Active Rentals within X days:
Possible values:
any (Default)
30 (Within 30 days)
15 (Within 15 days)
7 (Within 7 days)

        distanceinmiles: [SIMILARFLOORPLANS] 
Filter for distance in Miles: 
Possible values:
**any
1
2
3
4
5**

        
    """
    url = f"https://zillow56.p.rapidapi.com/rent_estimate"
    querystring = {'address': address, }
    if bedrooms:
        querystring['bedrooms'] = bedrooms
    if pets:
        querystring['pets'] = pets
    if amenities:
        querystring['amenities'] = amenities
    if laundry:
        querystring['laundry'] = laundry
    if propertytypes:
        querystring['propertyTypes'] = propertytypes
    if activetypes:
        querystring['activeTypes'] = activetypes
    if deactivateddays:
        querystring['deactivatedDays'] = deactivateddays
    if activateddays:
        querystring['activatedDays'] = activateddays
    if distanceinmiles:
        querystring['distanceInMiles'] = distanceinmiles
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow56.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agent_s_active_listings(zuid: str, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get agent's active listings by zuid"
    
    """
    url = f"https://zillow56.p.rapidapi.com/agent_active_listings"
    querystring = {'zuid': zuid, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow56.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agent_reviews(zuid: str, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get agent reviews by the agent's zuid"
    
    """
    url = f"https://zillow56.p.rapidapi.com/agent_reviews"
    querystring = {'zuid': zuid, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow56.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agent_s_rental_listings(zuid: str, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get agent's rental listings by zuid"
    
    """
    url = f"https://zillow56.p.rapidapi.com/agent_rental_listings"
    querystring = {'zuid': zuid, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow56.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agent_details_by_username(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get agent's details by username(contact infos, active listings and reviews etc).
		PS : username is the profile link 
		Example : 
		username :  Pardee-Properties
		for https://www.zillow.com/profile/Pardee-Properties/"
    
    """
    url = f"https://zillow56.p.rapidapi.com/agent"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow56.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_agents(location: str, language: str=None, specialties: str=None, name: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for agents by location and name"
    
    """
    url = f"https://zillow56.p.rapidapi.com/search_agents"
    querystring = {'location': location, }
    if language:
        querystring['language'] = language
    if specialties:
        querystring['specialties'] = specialties
    if name:
        querystring['name'] = name
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow56.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

