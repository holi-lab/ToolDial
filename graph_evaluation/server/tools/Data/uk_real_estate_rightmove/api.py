import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def commercial_property_to_sale(identifier: str, sort_by: str=None, page: int=None, max_size: int=None, property_type: str=None, max_price: int=None, min_price: int=None, size_unit: str=None, search_radius: str='0.0', has_parking: bool=None, has_auction_property: bool=None, min_size: int=None, has_business_for_sale: bool=None, has_include_under_offer_sold_stc: bool=None, do_not_show_business_for_sale: bool=None, added_to_site: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Commercial property for sale"
    identifier: `identifier` get from  **auto-complete**  endpoint,  data[index]->`locationIdentifier`
        property_type: Property type comma-separated
Ex: Office,BusinessPark

`Office`: Office
`ServicedOfficeFlexibleWorkspace`: Serviced Office / Flexible Workspace
`BusinessPark`: Business Park
`RetailShoppingCentres`: Retail (Shopping centres)
`RetailRetailParks`: Retail (Retail parks)
`RetailHighStreet`: Retail (High street)
`RetailOutOfTown`: Retail (Out of town)
`RetailPopUpShops`: Retail (Pop up shops)
`ConvenienceStore`: Convenience Store
`Garage`: Garage
`HairdresserBarbers`: Hairdresser / barbers
`PostOffice`: Post Office
`Shop`: Shop
`Workshop`: Workshop
`Hospitality`: Hospitality
`LeisureFacility`: Leisure facility
`BarNightclub`: Bar / Nightclub
`Cafe`: Cafe
`GuestHouseBB`: Guest House / B&B
`Hotel`: Hotel
`Pub`: Pub
`Restaurant`: Restaurant
`Takeaway`: Takeaway
`DistributionWarehouse`: Distribution Warehouse
`Factory`: Factory
`HeavyIndustrial`: Heavy Industrial
`IndustrialPark`: Industrial Park
`LightIndustrial`: Light Industrial
`Showroom`: Showroom
`Storage`: Storage
`TradeCounter`: Trade Counter
`Warehouse`: Warehouse
`Land`: Land
`CommercialDevelopment`: Commercial Development
`IndustrialDevelopment`: Industrial Development
`ResidentialDevelopment`: Residential Development
`Farm`: Farm
`ChildcareFacility`: Childcare Facility
`HealthcareFacility`: Healthcare Facility
`MixedUse`: Mixed Use
`PetrolStation`: Petrol Station
`PlaceOfWorship`: Place of Worship
`CommercialProperty`: Commercial Property
`Other`: Other
        search_radius: 0.0
0.25
0.5
1.0
3.0
5.0
10.0
15.0
20.0
30.0
40.0
        
    """
    url = f"https://uk-real-estate-rightmove.p.rapidapi.com/commercial/property-to-sale"
    querystring = {'identifier': identifier, }
    if sort_by:
        querystring['sort_by'] = sort_by
    if page:
        querystring['page'] = page
    if max_size:
        querystring['max_size'] = max_size
    if property_type:
        querystring['property_type'] = property_type
    if max_price:
        querystring['max_price'] = max_price
    if min_price:
        querystring['min_price'] = min_price
    if size_unit:
        querystring['size_unit'] = size_unit
    if search_radius:
        querystring['search_radius'] = search_radius
    if has_parking:
        querystring['has_parking'] = has_parking
    if has_auction_property:
        querystring['has_auction_property'] = has_auction_property
    if min_size:
        querystring['min_size'] = min_size
    if has_business_for_sale:
        querystring['has_business_for_sale'] = has_business_for_sale
    if has_include_under_offer_sold_stc:
        querystring['has_include_under_offer_sold_stc'] = has_include_under_offer_sold_stc
    if do_not_show_business_for_sale:
        querystring['do_not_show_business_for_sale'] = do_not_show_business_for_sale
    if added_to_site:
        querystring['added_to_site'] = added_to_site
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-real-estate-rightmove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sold_house_prices(identifier: str, search_radius: str='0.0', last_year: str=None, property_type: str=None, tenure_type: str=None, sort_by: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sold house prices"
    identifier: `identifier` get from  **/house-prices/auto-complete**  endpoint,  data[index]->`locationIdentifier`
        search_radius: 0.0
0.25
0.5
1.0
3.0
5.0
10.0
15.0
        property_type: Property type comma-separated
Ex: Terraced,Other

`Detached`
`Flat`
`SemiDetached`
`Terraced`
`Other`
        
    """
    url = f"https://uk-real-estate-rightmove.p.rapidapi.com/house-prices/sold-house-prices"
    querystring = {'identifier': identifier, }
    if search_radius:
        querystring['search_radius'] = search_radius
    if last_year:
        querystring['last_year'] = last_year
    if property_type:
        querystring['property_type'] = property_type
    if tenure_type:
        querystring['tenure_type'] = tenure_type
    if sort_by:
        querystring['sort_by'] = sort_by
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-real-estate-rightmove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def student_property_to_rent(identifier: str, sort_by: str=None, page: int=None, max_price: int=None, search_radius: str='0.0', min_price: int=None, max_bedroom: int=None, furnished_type: str=None, min_bedroom: int=None, has_garden: bool=None, added_to_site: str=None, property_type: str=None, keywords: str=None, has_house_share: bool=None, has_include_let_agreed: bool=None, has_parking: bool=None, do_not_show_house_share: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Student property to rent"
    identifier: `identifier` get from  **auto-complete**  endpoint,  data[index]->`locationIdentifier`
        search_radius: 0.0
0.25
0.5
1.0
3.0
5.0
10.0
15.0
20.0
30.0
40.0
        furnished_type: Furnished type comma-separated
Ex: PartFurnished,Unfurnished

Furnished,PartFurnished,Unfurnished
        property_type: Property type comma-separated
Ex: Bungalow,StudentHalls
Detached,SemiDetached,Terraced,Flat,Bungalow,StudentHalls
        keywords: Keywords comma-separated
Ex: pool, garden
        
    """
    url = f"https://uk-real-estate-rightmove.p.rapidapi.com/rent/student-property-to-rent"
    querystring = {'identifier': identifier, }
    if sort_by:
        querystring['sort_by'] = sort_by
    if page:
        querystring['page'] = page
    if max_price:
        querystring['max_price'] = max_price
    if search_radius:
        querystring['search_radius'] = search_radius
    if min_price:
        querystring['min_price'] = min_price
    if max_bedroom:
        querystring['max_bedroom'] = max_bedroom
    if furnished_type:
        querystring['furnished_type'] = furnished_type
    if min_bedroom:
        querystring['min_bedroom'] = min_bedroom
    if has_garden:
        querystring['has_garden'] = has_garden
    if added_to_site:
        querystring['added_to_site'] = added_to_site
    if property_type:
        querystring['property_type'] = property_type
    if keywords:
        querystring['keywords'] = keywords
    if has_house_share:
        querystring['has_house_share'] = has_house_share
    if has_include_let_agreed:
        querystring['has_include_let_agreed'] = has_include_let_agreed
    if has_parking:
        querystring['has_parking'] = has_parking
    if do_not_show_house_share:
        querystring['do_not_show_house_share'] = do_not_show_house_share
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-real-estate-rightmove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def new_homes_for_sale(identifier: str, sort_by: str=None, page: int=None, search_radius: str='0.0', min_bedroom: int=None, added_to_site: str=None, min_price: int=None, max_price: int=None, max_bedroom: int=None, property_type: str=None, has_include_under_offer_sold_stc: bool=None, has_garden: bool=None, has_parking: bool=None, has_retirement_home: bool=None, do_not_show_retirement_home: bool=None, has_auction_property: bool=None, has_buying_schemes: bool=None, do_not_show_buying_schemes: bool=None, keywords: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "New homes for sale"
    identifier: `identifier` get from  **auto-complete**  endpoint,  data[index]->`locationIdentifier`
        search_radius: 0.0
0.25
0.5
1.0
3.0
5.0
10.0
15.0
20.0
30.0
40.0
        property_type: Property type comma-separated:
Ex: Detached,SemiDetached

Detached
SemiDetached
Terraced
Flat
Bungalow
Land
ParkHome
        keywords: Keywords comma-separated
Ex: pool, garden
        
    """
    url = f"https://uk-real-estate-rightmove.p.rapidapi.com/buy/new-homes-for-sale"
    querystring = {'identifier': identifier, }
    if sort_by:
        querystring['sort_by'] = sort_by
    if page:
        querystring['page'] = page
    if search_radius:
        querystring['search_radius'] = search_radius
    if min_bedroom:
        querystring['min_bedroom'] = min_bedroom
    if added_to_site:
        querystring['added_to_site'] = added_to_site
    if min_price:
        querystring['min_price'] = min_price
    if max_price:
        querystring['max_price'] = max_price
    if max_bedroom:
        querystring['max_bedroom'] = max_bedroom
    if property_type:
        querystring['property_type'] = property_type
    if has_include_under_offer_sold_stc:
        querystring['has_include_under_offer_sold_stc'] = has_include_under_offer_sold_stc
    if has_garden:
        querystring['has_garden'] = has_garden
    if has_parking:
        querystring['has_parking'] = has_parking
    if has_retirement_home:
        querystring['has_retirement_home'] = has_retirement_home
    if do_not_show_retirement_home:
        querystring['do_not_show_retirement_home'] = do_not_show_retirement_home
    if has_auction_property:
        querystring['has_auction_property'] = has_auction_property
    if has_buying_schemes:
        querystring['has_buying_schemes'] = has_buying_schemes
    if do_not_show_buying_schemes:
        querystring['do_not_show_buying_schemes'] = do_not_show_buying_schemes
    if keywords:
        querystring['keywords'] = keywords
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-real-estate-rightmove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_for_sale(identifier: str, page: int=None, min_price: int=None, has_garden: bool=None, sort_by: str=None, search_radius: str='0.0', max_price: int=None, max_bedroom: int=None, property_type: str=None, added_to_site: str=None, has_parking: bool=None, keywords: str=None, min_bedroom: int=None, do_not_show_buying_schemes: bool=None, has_new_home: bool=None, has_retirement_home: bool=None, do_not_show_retirement_home: bool=None, has_buying_schemes: bool=None, has_auction_property: bool=None, do_not_show_new_home: bool=None, has_include_under_offer_sold_stc: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Property for sale"
    identifier: `identifier` get from  **auto-complete**  endpoint,  data[index]->`locationIdentifier`
        search_radius: 0.0
0.25
0.5
1.0
3.0
5.0
10.0
15.0
20.0
30.0
40.0
        property_type: Property type comma-separated 
Ex:  Detached,SemiDetached,Flat

Detached
SemiDetached
Terraced
Flat
Bungalow
Land
ParkHome
        keywords: Keywords comma-separated
Ex: pool, garden
        
    """
    url = f"https://uk-real-estate-rightmove.p.rapidapi.com/buy/property-for-sale"
    querystring = {'identifier': identifier, }
    if page:
        querystring['page'] = page
    if min_price:
        querystring['min_price'] = min_price
    if has_garden:
        querystring['has_garden'] = has_garden
    if sort_by:
        querystring['sort_by'] = sort_by
    if search_radius:
        querystring['search_radius'] = search_radius
    if max_price:
        querystring['max_price'] = max_price
    if max_bedroom:
        querystring['max_bedroom'] = max_bedroom
    if property_type:
        querystring['property_type'] = property_type
    if added_to_site:
        querystring['added_to_site'] = added_to_site
    if has_parking:
        querystring['has_parking'] = has_parking
    if keywords:
        querystring['keywords'] = keywords
    if min_bedroom:
        querystring['min_bedroom'] = min_bedroom
    if do_not_show_buying_schemes:
        querystring['do_not_show_buying_schemes'] = do_not_show_buying_schemes
    if has_new_home:
        querystring['has_new_home'] = has_new_home
    if has_retirement_home:
        querystring['has_retirement_home'] = has_retirement_home
    if do_not_show_retirement_home:
        querystring['do_not_show_retirement_home'] = do_not_show_retirement_home
    if has_buying_schemes:
        querystring['has_buying_schemes'] = has_buying_schemes
    if has_auction_property:
        querystring['has_auction_property'] = has_auction_property
    if do_not_show_new_home:
        querystring['do_not_show_new_home'] = do_not_show_new_home
    if has_include_under_offer_sold_stc:
        querystring['has_include_under_offer_sold_stc'] = has_include_under_offer_sold_stc
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-real-estate-rightmove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def commercial_property_to_rent_detail(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Commercial property to rent for detail"
    id: id from commercial/property-to-rent endpoint 
        
    """
    url = f"https://uk-real-estate-rightmove.p.rapidapi.com/commercial/property-to-rent/detail"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-real-estate-rightmove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def commercial_property_to_sale_detail(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Commercial property for sale for detail"
    id: id from commercial/property-to-sale endpoint 
        
    """
    url = f"https://uk-real-estate-rightmove.p.rapidapi.com/commercial/property-to-sale/detail"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-real-estate-rightmove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def commercial_property_to_rent(identifier: str, page: int=None, min_size: int=None, search_radius: str='0.0', sort_by: str=None, max_price: int=None, price_type: str=None, min_price: int=None, max_size: int=None, property_type: str=None, size_unit: str=None, has_include_let_agreed: bool=None, added_to_site: str=None, has_parking: bool=None, type_of_let: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Commercial property to rent"
    identifier: `identifier` get from  **auto-complete**  endpoint,  data[index]->`locationIdentifier`
        search_radius: 0.0
0.25
0.5
1.0
3.0
5.0
10.0
15.0
20.0
30.0
40.0
        property_type: Property type comma-separated
Ex: Office,BusinessPark

`Office`: Office
`ServicedOfficeFlexibleWorkspace`: Serviced Office / Flexible Workspace
`BusinessPark`: Business Park
`RetailShoppingCentres`: Retail (Shopping centres)
`RetailRetailParks`: Retail (Retail parks)
`RetailHighStreet`: Retail (High street)
`RetailOutOfTown`: Retail (Out of town)
`RetailPopUpShops`: Retail (Pop up shops)
`ConvenienceStore`: Convenience Store
`Garage`: Garage
`HairdresserBarbers`: Hairdresser / barbers
`PostOffice`: Post Office
`Shop`: Shop
`Workshop`: Workshop
`Hospitality`: Hospitality
`LeisureFacility`: Leisure facility
`BarNightclub`: Bar / Nightclub
`Cafe`: Cafe
`GuestHouseBB`: Guest House / B&B
`Hotel`: Hotel
`Pub`: Pub
`Restaurant`: Restaurant
`Takeaway`: Takeaway
`DistributionWarehouse`: Distribution Warehouse
`Factory`: Factory
`HeavyIndustrial`: Heavy Industrial
`IndustrialPark`: Industrial Park
`LightIndustrial`: Light Industrial
`Showroom`: Showroom
`Storage`: Storage
`TradeCounter`: Trade Counter
`Warehouse`: Warehouse
`Land`: Land
`CommercialDevelopment`: Commercial Development
`IndustrialDevelopment`: Industrial Development
`ResidentialDevelopment`: Residential Development
`Farm`: Farm
`ChildcareFacility`: Childcare Facility
`HealthcareFacility`: Healthcare Facility
`MixedUse`: Mixed Use
`PetrolStation`: Petrol Station
`PlaceOfWorship`: Place of Worship
`CommercialProperty`: Commercial Property
`Other`: Other
        
    """
    url = f"https://uk-real-estate-rightmove.p.rapidapi.com/commercial/property-to-rent"
    querystring = {'identifier': identifier, }
    if page:
        querystring['page'] = page
    if min_size:
        querystring['min_size'] = min_size
    if search_radius:
        querystring['search_radius'] = search_radius
    if sort_by:
        querystring['sort_by'] = sort_by
    if max_price:
        querystring['max_price'] = max_price
    if price_type:
        querystring['price_type'] = price_type
    if min_price:
        querystring['min_price'] = min_price
    if max_size:
        querystring['max_size'] = max_size
    if property_type:
        querystring['property_type'] = property_type
    if size_unit:
        querystring['size_unit'] = size_unit
    if has_include_let_agreed:
        querystring['has_include_let_agreed'] = has_include_let_agreed
    if added_to_site:
        querystring['added_to_site'] = added_to_site
    if has_parking:
        querystring['has_parking'] = has_parking
    if type_of_let:
        querystring['type_of_let'] = type_of_let
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-real-estate-rightmove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def estate_agents_list(identifier: str, page: int=None, search_radius: str='0.0', branch_type: str=None, agent_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find estate agents"
    identifier: `identifier` get from  **auto-complete**  endpoint,  data[index]->`locationIdentifier`
        search_radius: 0.0
0.25
0.5
1.0
3.0
5.0
10.0
15.0
20.0
30.0
40.0
        
    """
    url = f"https://uk-real-estate-rightmove.p.rapidapi.com/estate-agents/list"
    querystring = {'identifier': identifier, }
    if page:
        querystring['page'] = page
    if search_radius:
        querystring['search_radius'] = search_radius
    if branch_type:
        querystring['branch_type'] = branch_type
    if agent_name:
        querystring['agent_name'] = agent_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-real-estate-rightmove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def house_prices_detail(detail_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "house-prices/detail"
    detail_url: detailUrl from sold-house-prices/nearby-properties endpoint
        
    """
    url = f"https://uk-real-estate-rightmove.p.rapidapi.com/house-prices/detail"
    querystring = {'detail_url': detail_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-real-estate-rightmove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def house_prices_auto_complete(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "house-prices/auto-complete"
    
    """
    url = f"https://uk-real-estate-rightmove.p.rapidapi.com/house-prices/auto-complete"
    querystring = {'location': location, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-real-estate-rightmove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def student_property_to_rent_detail(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Student property to rent for detail"
    id: id from student-property-to-rent endpoint 
        
    """
    url = f"https://uk-real-estate-rightmove.p.rapidapi.com/rent/student-property-to-rent/detail"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-real-estate-rightmove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_to_rent_detail(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Property to rent for detail"
    id: id from property-to-rent endpoint 
        
    """
    url = f"https://uk-real-estate-rightmove.p.rapidapi.com/rent/property-to-rent/detail"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-real-estate-rightmove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_to_rent(identifier: str, sort_by: str=None, page: int=None, search_radius: str='0.0', do_not_show_retirement_home: bool=None, added_to_site: str=None, max_price: int=None, min_price: int=None, furnished_type: str=None, min_bedroom: int=None, max_bedroom: int=None, type_of_let: str=None, has_garden: bool=None, has_house_share: bool=None, has_student_accommodation: bool=None, keywords: str=None, do_not_show_house_share: bool=None, has_parking: bool=None, do_not_show_student_accommodation: bool=None, has_retirement_home: bool=None, has_include_let_agreed: bool=None, property_type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Property to rent"
    identifier: `identifier` get from  **auto-complete**  endpoint,  data[index]->`locationIdentifier`
        search_radius: 0.0
0.25
0.5
1.0
3.0
5.0
10.0
15.0
20.0
30.0
40.0
        furnished_type: Furnished type comma-separated
Ex: Furnished,PartFurnished

Furnished,PartFurnished,Unfurnished
        keywords: Keywords comma-separated
Ex: pool, garden
        property_type: Property type comma-separated
Ex: Detached,SemiDetached

Detached
SemiDetached
Terraced
Flat
Bungalow
Land
ParkHome
StudentHalls
        
    """
    url = f"https://uk-real-estate-rightmove.p.rapidapi.com/rent/property-to-rent"
    querystring = {'identifier': identifier, }
    if sort_by:
        querystring['sort_by'] = sort_by
    if page:
        querystring['page'] = page
    if search_radius:
        querystring['search_radius'] = search_radius
    if do_not_show_retirement_home:
        querystring['do_not_show_retirement_home'] = do_not_show_retirement_home
    if added_to_site:
        querystring['added_to_site'] = added_to_site
    if max_price:
        querystring['max_price'] = max_price
    if min_price:
        querystring['min_price'] = min_price
    if furnished_type:
        querystring['furnished_type'] = furnished_type
    if min_bedroom:
        querystring['min_bedroom'] = min_bedroom
    if max_bedroom:
        querystring['max_bedroom'] = max_bedroom
    if type_of_let:
        querystring['type_of_let'] = type_of_let
    if has_garden:
        querystring['has_garden'] = has_garden
    if has_house_share:
        querystring['has_house_share'] = has_house_share
    if has_student_accommodation:
        querystring['has_student_accommodation'] = has_student_accommodation
    if keywords:
        querystring['keywords'] = keywords
    if do_not_show_house_share:
        querystring['do_not_show_house_share'] = do_not_show_house_share
    if has_parking:
        querystring['has_parking'] = has_parking
    if do_not_show_student_accommodation:
        querystring['do_not_show_student_accommodation'] = do_not_show_student_accommodation
    if has_retirement_home:
        querystring['has_retirement_home'] = has_retirement_home
    if has_include_let_agreed:
        querystring['has_include_let_agreed'] = has_include_let_agreed
    if property_type:
        querystring['property_type'] = property_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-real-estate-rightmove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def similar_to_property(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Similar to this property"
    id: id from property-for-sale/new-homes-for-sale endpoint 
        
    """
    url = f"https://uk-real-estate-rightmove.p.rapidapi.com/buy/similar-to-property"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-real-estate-rightmove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def new_homes_for_sale_detail(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "New homes for sale for detail"
    id: id from new-homes-for-sale endpoint
        
    """
    url = f"https://uk-real-estate-rightmove.p.rapidapi.com/buy/new-homes-for-sale/detail"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-real-estate-rightmove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def similar_to_property_detail(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Similar to this property for detail"
    id: id from similar-to-property endpoint 
        
    """
    url = f"https://uk-real-estate-rightmove.p.rapidapi.com/buy/similar-to-property/detail"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-real-estate-rightmove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_for_sale_detail(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Property for sale for detail"
    id: id from property-for-sale endpoint
        
    """
    url = f"https://uk-real-estate-rightmove.p.rapidapi.com/buy/property-for-sale/detail"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-real-estate-rightmove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_complete(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get auto complete suggestion by term or phrase"
    
    """
    url = f"https://uk-real-estate-rightmove.p.rapidapi.com/auto-complete"
    querystring = {'location': location, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-real-estate-rightmove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

