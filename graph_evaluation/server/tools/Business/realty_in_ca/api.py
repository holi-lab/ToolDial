import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def locations_v2_auto_complete(query: str, includelocations: bool=None, cultureid: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get auto complete suggestions by city, ward, street name or an actual address"
    query: City, ward, street name, etc... or an actual address
        includelocations: Whether or not include locations
        cultureid: 1 - English|2 - French
        
    """
    url = f"https://realty-in-ca1.p.rapidapi.com/locations/v2/auto-complete"
    querystring = {'Query': query, }
    if includelocations:
        querystring['IncludeLocations'] = includelocations
    if cultureid:
        querystring['CultureId'] = cultureid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-ca1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keywords_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all supported tags/keywords for filtering"
    
    """
    url = f"https://realty-in-ca1.p.rapidapi.com/keywords/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-ca1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locations_auto_complete(area: str, cultureid: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get auto complete suggestions by city, ward, street name"
    area: City, ward, street name, etc...
        cultureid: 1 - English|2 - French
        
    """
    url = f"https://realty-in-ca1.p.rapidapi.com/locations/auto-complete"
    querystring = {'Area': area, }
    if cultureid:
        querystring['CultureId'] = cultureid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-ca1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agents_detail(is_id: int, cultureid: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail information of an agent"
    id: The value of IndividualID field returned in .../agents/list endpoint
        cultureid: 1 - English|2 - French
        
    """
    url = f"https://realty-in-ca1.p.rapidapi.com/agents/detail"
    querystring = {'id': is_id, }
    if cultureid:
        querystring['CultureId'] = cultureid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-ca1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agents_get_listings(organizationid: int, sortorder: str='D', currentpage: int=1, recordsperpage: int=10, cultureid: int=1, sortby: int=6, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get properties listed by agent"
    organizationid: The value of OrganizationID field returned in .../agents/list or .../agents/detail endpoint
        sortorder: A - ascending | D - descending
        currentpage: For paging purpose
        recordsperpage: Number items returned per request, max 50
        cultureid: 1 - English|2 - French
        sortby: One of the following : 1-Price($)|6-Date|11-Virtual Tour|12-Open Houses|13-More Photos
        
    """
    url = f"https://realty-in-ca1.p.rapidapi.com/agents/get-listings"
    querystring = {'OrganizationId': organizationid, }
    if sortorder:
        querystring['SortOrder'] = sortorder
    if currentpage:
        querystring['CurrentPage'] = currentpage
    if recordsperpage:
        querystring['RecordsPerPage'] = recordsperpage
    if cultureid:
        querystring['CultureId'] = cultureid
    if sortby:
        querystring['SortBy'] = sortby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-ca1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agents_list(recordsperpage: int=10, specialties: int=None, provinceids: int=None, designations: int=None, sortorder: str='A', companyname: str=None, iscccmember: bool=None, currentpage: int=1, firstname: str=None, languages: int=None, city: str=None, lastname: str=None, sortby: int=11, cultureid: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List agents with options and filters"
    recordsperpage: Number items returned per request, max 50
        specialties: One of the following : 2-Residential Property Management|4-Residential Brokerage|8-Residential Development|10-Residential Valuation|12-Residential Financing|14-Residential Leasing|16-Residential Legal|18-Residential Relocation|17-Relocation|28-2nd Home|33-Age Restricted/Active Adult Community Properties|36-Agriculture Land|9-Appraisal|3-Business Brokerage|35-Condos|5-Consulting|7-Development Land|24-Farm/Ranch|32-Golf Community Properties|25-Hospitality|21-Industrial|11-Investment|29-Luxury Homes|23-Multi-Family|22-Office|1-Property Management|26-Recreational|31-Resort Properties|20-Retail|30-Seasonal/Vacation Rentals|27-Waterfront Properties
        provinceids: One of the following : 1-Alberta|3-British Columbia|8-Manitoba|6-New Brunswick|10-Newfoundland &amp; Labrador|11-Northwest Territories|5-Nova Scotia|9-Nunavut|2-Ontario|12-Prince Edward Island|4-Quebec|7-Saskatchewan|13-Yukon
        designations: One of the following : 1-Accredited Buyer Representative|2-Accredited Buyer Representative Manager|3-At Home With Diversity Certification|4-Accredited Land Consultant|5-Accredited Residential Manager® |6-Associate Reserve Planner|7-Certified Commercial Investment Member|8-Certified International Property Specialist|9-Certified Leasing Officer|10-Certified Manager of Condominiums|11-Certified Property Manager® |12-Certified Real Estate Specialist|13-Certified Real Estate Brokerage Manager|14-Counselor of Real Estate|15-Certified in Real Estate Finance|16-Certified Reserve Planner|17-Certified Residential Specialist® |18-Certified Residential  Underwriter|19-REALTOR e-PRO®|20-Fellow of the Real Estate Institute|21-Fellow of the Real Estate Institute (Appraisal Specialist)|22-Fellow of the Real Estate Institute (Executive)|23-General Accredited Appraiser|24-Graduate REALTOR® Institute|26-Performance Management Network|27-Residential Accredited Appraiser|28-REALTOR® Association Certified Executive|29-Real Estate Professional Assistant|30-Accredited Appraiser Canadian Institute|31-Market Value Appraiser-Residential|33-Accredited Leasing Officer |34-Accredited Mortgage Professional |35-Accredited Senior Appraiser|36-Certified Valuation Analyst|37-Certified Mold Remediation|38-Certified Professional Residential Property Manager|39-Canadian Residential Appraiser|40-Fellows of the Royal Institution of Chartered Surveyors |41-Member, Appraisal Institute|42-Professional Land Economist|43-Real Property Administrator®|44-Society of Industrial and Office REALTORS®|45-Senior Residential Appraiser|46-Canadian Employee Relocation Professional|47-Canadian Employee Relocation Council Relocation Specialist™|48-Relocation Resort Specialist|49-Senior Real Estate Specialist|50-Certified Lease Professional Designation|51-Designated Agency Representatives |52-Distinguished Real Estate Instructor™|53-Unknown|54-Certified Condo Specialist|55-Canadian REALTOR® Association Executive|56-NAR's Green Designation|57-ACCREDITED GREENAGENT® -Residential|58-ACCREDITED GREENBROKER® - Commercial|59-EcoBroker®|60-Transnational Referral Certified|61-Certified Negotiation Expert|62-Member of the Royal Institute of Charter Surveyors|63-Certified Reserve Fund Planner|64-Accredited Staging Professional|65-Resort &amp; Second-Home Property Specialist|66-Short Sales and Foreclose Resource|67-Business Corporation Owned by a Real Estate Broker|68-Canadian Personal Property Appraiser|69-Seller Representative Specialist |70-Accredited Senior Agent|71-Certified Luxury Home Marketing Specialist|72-Royal Institute of Chartered Surveyors|73-Accredited Commercial Professional|74-Master Certified Negotiation Expert|75-Associate of the Canadian Condominium Institute|76-Real Estate Negotiation Expert|77-Military Relocation Professional|78-Pricing Strategy Advisor|79-Accredited Luxury Home Specialist|80-Performance Management Network Designation|81-Certified in Marketing of Real Estate|82-Residential Construction Certified|83-Certified New Home Specialist |84-Chartered Accountant|85-Chartered Professional Accountant|86-Certified General Accountant|87-Certified Management Accountant|88-BoundaryWise|89-Chartered Financial Analyst
        sortorder: A - ascending | D - descending
        companyname: Search by company name
        iscccmember: false|true : REALTORS® that are affiliates of the Canadian Commercial Network
        currentpage: For paging purpose
        firstname: Search by agent's first name
        languages: One of the following : 1-English|2-French|3-Chinese (Mandarin)|36-Chinese (Cantonese)|9-Punjabi |23-Hindi|13-Tagalog (Filipino)|11-Arabic |19-Russian|5-German |55-Aboriginal languages|50-Afrikaans|54-Albanian|22-American Sign Language (ASL)|56-Amharic|42-Armenian|106-Assyrian|57-Azeri|58-Bahasa Malaysia|39-Bangla|59-Belorussian|35-Bulgarian|40-Burmese|60-Catalan|105-Chaldean|16-Cree |61-Creole|25-Croatian|26-Czech|27-Danish|43-Dari|12-Dutch |62-Estonian|45-Farsi|51-Finnish|63-Flemish|64-Friesian|65-Galla|66-Gan|67-Georgian|14-Greek |47-Gujarati|68-Hakka|69-Hausa|28-Hebrew|29-Hungarian|70-Icelandic|71-Indonesian|17-Inuktitut (Inuit) |4-Italian |18-Japanese|72-Javanese|73-Jin-yu|74-Kannada|75-Kazakh|76-Kejia|77-Khmer|78-Kiswahili|79-Konkani|20-Korean|80-Kurdish|81-Lao|30-Latvian|53-Lithuanian|82-Luxembourgish|48-Macedonian|83-Malagasy|84-Malay|85-Malayalam|49-Malaysian|31-Maltese|86-Marathi|87-Miny|88-Mongolian|32-Norwegian |89-Pakhto|44-Pashto|24-Persian|6-Polish |8-Portuguese |41-Romanian|33-Serbian|90-Serbo-Croatian|91-Sindhi|92-Sinhala|21-Slovak|52-Slovenian|94-Somali|7-Spanish |95-Sunda|37-Swedish|104-Tajik|46-Tamil|96-Tchi|97-Telugu|98-Thai|99-Tigrinya|34-Turkish|10-Ukrainian |38-Urdu|100-Uzbek|15-Vietnamese |101-Xiang|102-Yiddish|103-Yue
        city: Search by city name
        lastname: Search by agent's last name
        sortby: 11-No Preference|3-Last Name|2-First Name|8-City|9-Province
        cultureid: 1 - English|2 - French
        
    """
    url = f"https://realty-in-ca1.p.rapidapi.com/agents/list"
    querystring = {}
    if recordsperpage:
        querystring['RecordsPerPage'] = recordsperpage
    if specialties:
        querystring['Specialties'] = specialties
    if provinceids:
        querystring['ProvinceIds'] = provinceids
    if designations:
        querystring['Designations'] = designations
    if sortorder:
        querystring['SortOrder'] = sortorder
    if companyname:
        querystring['CompanyName'] = companyname
    if iscccmember:
        querystring['isCccMember'] = iscccmember
    if currentpage:
        querystring['CurrentPage'] = currentpage
    if firstname:
        querystring['FirstName'] = firstname
    if languages:
        querystring['Languages'] = languages
    if city:
        querystring['City'] = city
    if lastname:
        querystring['LastName'] = lastname
    if sortby:
        querystring['SortBy'] = sortby
    if cultureid:
        querystring['CultureId'] = cultureid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-ca1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_list_commercial(latitudemin: int, longitudemax: int, latitudemax: int, longitudemin: int, sortorder: str='A', pricemin: int=None, unitrange: str=None, buildingtypeid: int=None, pricemax: int=None, farmtypeid: int=None, openhousestartdate: str=None, numberofdays: int=0, recordsperpage: int=10, constructionstyleid: int=None, openhouse: bool=None, bedrange: str='0-0', landsizerange: str=None, bathrange: str='0-0', currentpage: int=1, openhouseenddate: str=None, zoningtypegroupid: int=None, keywords: str=None, sortby: int=1, cultureid: int=1, buildingsizerange: str=None, propertysearchtypeid: int=None, transactiontypeid: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List commercial properties both for lease and for sale"
    latitudemin: South West latitude
        longitudemax: North East longitude
        longitudemin: South West longitude
        sortorder: A - ascending | D - descending
        pricemin: Filter by min price, applied when TransactionTypeId = 2
        unitrange: 0-0:Any|1-1:1|1-0:1+|2-2:2|2-0:2+|3-3:3|3-0:3+|4-4:4|4-0:4+|5-5:5|5-0:5+|….|9-0:9+
        buildingtypeid: 0-No Preference|1-House|2-Duplex|3-Triplex|5-Residential Commercial Mix|6-Mobile Home|12-Special Purpose|14-Other|16-Row / Townhouse|17-Apartment|19-Fourplex|20-Garden Home|26-Modular|27-Manufactured Home/Mobile|28-Commercial Apartment|29-Manufactured Home
        pricemax: Filter by max price, applied when TransactionTypeId = 2
        farmtypeid: 0-No Preference|1-Animal|2-Boarding|3-Cash Crop|4-Feed Lot|5-Nursery|6-Market Gardening|7-Hobby Farm|8-Vineyard|9-Orchard|10-Greenhouse|12-Mixed
        openhousestartdate: Format date as MM/dd/yyyy, ex : 03/20/2020
        numberofdays: Listed since
        recordsperpage: Number items returned per request, max 50
        constructionstyleid: 0-No Preference|1-Attached|3-Detached|5-Semi-detached|7-Stacked|9-Link
        openhouse: false/true (not 0/1). Need to use together with OpenHouseStartDate and OpenHouseEndDate parameters
        bedrange: 0-0:Any|1-1:1|1-0:1+|2-2:2|2-0:2+|3-3:3|3-0:3+|4-4:4|4-0:4+|5-5:5|5-0:5+
        landsizerange: 0-0:Any|1-0:1+ acres|2-0:2+ acres|5-0:5+ acres|10-0:10+ acres|50-0:50+ acres|100-0:100+ acres|200-0:200+ acres|300-0:300+ acres|400-0:400+ acres|500-0:500+ acres|1000-0:1000+ acres. Ex : 0-0
        bathrange: 0-0:Any|1-1:1|1-0:1+|2-2:2|2-0:2+|3-3:3|3-0:3+|4-4:4|4-0:4+|5-5:5|5-0:5+
        currentpage: For paging purpose
        openhouseenddate: Format date as MM/dd/yyyy, ex : 03/31/2020
        zoningtypegroupid: 1-Agricultural|2-Commercial Mixed|3-Commercial Office|4-Commercial Retail|5-Industrial|6-Industrial-Heavy|7-Industrial-Light|8-Industrial-Medium|9-Institutional|10-Other|11-Recreational|12-Residential-High Density|13-Residential-Low Density|14-Residential - Medium Density
        keywords: Get suitable values from …/keywords/list endpoint, separated by comma for multiple keywords, Ex : Inlaw suite,Income suite
        sortby: 1-Price($)|6-Date|11-Virtual Tour|12-Open Houses|13-More Photos
        cultureid: 1 - English|2 - French
        buildingsizerange: 0-5000:0-5,000 sqft|5001-10000:5,001-10,000 sqft|10001-15000:10,001-15,000 sqft|250001-0:Over 250,000 sqft. Ex : 0-5000
        propertysearchtypeid: 0-No Preference|1-Residential|2-Recreational|3-Condo/Strata|4-Agriculture|5-Parking|6-Vacant Land|8-Multi Family
        transactiontypeid: 2-For sale|3-For lease
        
    """
    url = f"https://realty-in-ca1.p.rapidapi.com/properties/list-commercial"
    querystring = {'LatitudeMin': latitudemin, 'LongitudeMax': longitudemax, 'LatitudeMax': latitudemax, 'LongitudeMin': longitudemin, }
    if sortorder:
        querystring['SortOrder'] = sortorder
    if pricemin:
        querystring['PriceMin'] = pricemin
    if unitrange:
        querystring['UnitRange'] = unitrange
    if buildingtypeid:
        querystring['BuildingTypeId'] = buildingtypeid
    if pricemax:
        querystring['PriceMax'] = pricemax
    if farmtypeid:
        querystring['FarmTypeId'] = farmtypeid
    if openhousestartdate:
        querystring['OpenHouseStartDate'] = openhousestartdate
    if numberofdays:
        querystring['NumberOfDays'] = numberofdays
    if recordsperpage:
        querystring['RecordsPerPage'] = recordsperpage
    if constructionstyleid:
        querystring['ConstructionStyleId'] = constructionstyleid
    if openhouse:
        querystring['OpenHouse'] = openhouse
    if bedrange:
        querystring['BedRange'] = bedrange
    if landsizerange:
        querystring['LandSizeRange'] = landsizerange
    if bathrange:
        querystring['BathRange'] = bathrange
    if currentpage:
        querystring['CurrentPage'] = currentpage
    if openhouseenddate:
        querystring['OpenHouseEndDate'] = openhouseenddate
    if zoningtypegroupid:
        querystring['ZoningTypeGroupId'] = zoningtypegroupid
    if keywords:
        querystring['Keywords'] = keywords
    if sortby:
        querystring['SortBy'] = sortby
    if cultureid:
        querystring['CultureId'] = cultureid
    if buildingsizerange:
        querystring['BuildingSizeRange'] = buildingsizerange
    if propertysearchtypeid:
        querystring['PropertySearchTypeId'] = propertysearchtypeid
    if transactiontypeid:
        querystring['TransactionTypeId'] = transactiontypeid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-ca1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_list_by_mls(referencenumber: str, cultureid: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List properties by listing ID or MLS number"
    referencenumber: List ID or MLS number
        cultureid: 1 - English|2 - French
        
    """
    url = f"https://realty-in-ca1.p.rapidapi.com/properties/list-by-mls"
    querystring = {'ReferenceNumber': referencenumber, }
    if cultureid:
        querystring['CultureId'] = cultureid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-ca1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_detail(propertyid: str, referencenumber: str, cultureid: int=1, preferedmeasurementunit: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details information of specific property"
    propertyid: The value of Id field from .../list-commercial or .../list-residential endpoints
        referencenumber: The value of MlsNumber field from .../list-commercial or .../list-residential endpoints
        cultureid: 1 - English|2 - French
        preferedmeasurementunit: 1-Metric|2-Imperial
        
    """
    url = f"https://realty-in-ca1.p.rapidapi.com/properties/detail"
    querystring = {'PropertyID': propertyid, 'ReferenceNumber': referencenumber, }
    if cultureid:
        querystring['CultureId'] = cultureid
    if preferedmeasurementunit:
        querystring['PreferedMeasurementUnit'] = preferedmeasurementunit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-ca1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_list_residential(longitudemin: int, latitudemax: int, latitudemin: int, longitudemax: int, sortorder: str='A', zoningtypegroupid: int=None, recordsperpage: int=10, cultureid: int=1, parkingtypeid: int=None, currentpage: int=1, openhouseenddate: str=None, bedrange: str='0-0', unitrange: str=None, keywords: str=None, pricemax: int=None, pricemin: int=None, constructionstyleid: int=None, openhouse: bool=None, rentmax: int=None, numberofdays: int=0, transactiontypeid: int=None, buildingtypeid: int=None, openhousestartdate: str=None, farmtypeid: int=None, sortby: int=1, bathrange: str='0-0', propertysearchtypeid: int=None, landsizerange: str=None, buildingsizerange: str=None, rentmin: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List residential properties both for rent and for sale"
    longitudemin: South West longitude
        latitudemin: South West latitude
        longitudemax: North East longitude
        sortorder: A - ascending | D - descending
        zoningtypegroupid: 1-Agricultural|2-Commercial Mixed|3-Commercial Office|4-Commercial Retail|5-Industrial|6-Industrial-Heavy|7-Industrial-Light|8-Industrial-Medium|9-Institutional|10-Other|11-Recreational|12-Residential-High Density|13-Residential-Low Density|14-Residential - Medium Density
        recordsperpage: Number items returned per request, max 50
        cultureid: 1 - English|2 - French
        parkingtypeid: 1-Attached garage|2-Integrated garage|3-Detached garage|4-Garage|5-Carport|6-Underground|7-Indoor|8-Open|9-Covered|10-Parking pad|11-Paved Yard|35-Boat House|36-Concrete|37-Heated Garage
        currentpage: For paging purpose
        openhouseenddate: Format date as MM/dd/yyyy, ex : 03/31/2020
        bedrange: 0-0:Any|1-1:1|1-0:1+|2-2:2|2-0:2+|3-3:3|3-0:3+|4-4:4|4-0:4+|5-5:5|5-0:5+
        unitrange: 0-0:Any|1-1:1|1-0:1+|2-2:2|2-0:2+|3-3:3|3-0:3+|4-4:4|4-0:4+|5-5:5|5-0:5+|….|9-0:9+
        keywords: Get suitable values from …/keywords/list endpoint, separated by comma for multiple keywords, Ex : Inlaw suite,Income suite
        pricemax: Filter by max price, applied when TransactionTypeId = 2
        pricemin: Filter by min price, applied when TransactionTypeId = 2
        constructionstyleid: 0-No Preference|1-Attached|3-Detached|5-Semi-detached|7-Stacked|9-Link
        openhouse: false/true (not 0/1). Need to use together with OpenHouseStartDate and OpenHouseEndDate parameters
        rentmax: Filter by max price, applied when TransactionTypeId = 3
        numberofdays: Listed since
        transactiontypeid: 2-For sale|3-For rent
        buildingtypeid: 0-No Preference|1-House|2-Duplex|3-Triplex|5-Residential Commercial Mix|6-Mobile Home|12-Special Purpose|14-Other|16-Row / Townhouse|17-Apartment|19-Fourplex|20-Garden Home|26-Modular|27-Manufactured Home/Mobile|28-Commercial Apartment|29-Manufactured Home
        openhousestartdate: Format date as MM/dd/yyyy, ex : 03/20/2020
        farmtypeid: 0-No Preference|1-Animal|2-Boarding|3-Cash Crop|4-Feed Lot|5-Nursery|6-Market Gardening|7-Hobby Farm|8-Vineyard|9-Orchard|10-Greenhouse|12-Mixed
        sortby: 1-Price($)|6-Date|11-Virtual Tour|12-Open Houses|13-More Photos
        bathrange: 0-0:Any|1-1:1|1-0:1+|2-2:2|2-0:2+|3-3:3|3-0:3+|4-4:4|4-0:4+|5-5:5|5-0:5+
        propertysearchtypeid: 0-No Preference|1-Residential|2-Recreational|3-Condo/Strata|4-Agriculture|5-Parking|6-Vacant Land|8-Multi Family
        landsizerange: 0-0:Any|1-0:1+ acres|2-0:2+ acres|5-0:5+ acres|10-0:10+ acres|50-0:50+ acres|100-0:100+ acres|200-0:200+ acres|300-0:300+ acres|400-0:400+ acres|500-0:500+ acres|1000-0:1000+ acres. Ex : 0-0
        buildingsizerange: 0-5000:0-5,000 sqft|5001-10000:5,001-10,000 sqft|10001-15000:10,001-15,000 sqft|250001-0:Over 250,000 sqft Ex : 0-5000
        rentmin: Filter by min price, applied when TransactionTypeId = 3
        
    """
    url = f"https://realty-in-ca1.p.rapidapi.com/properties/list-residential"
    querystring = {'LongitudeMin': longitudemin, 'LatitudeMax': latitudemax, 'LatitudeMin': latitudemin, 'LongitudeMax': longitudemax, }
    if sortorder:
        querystring['SortOrder'] = sortorder
    if zoningtypegroupid:
        querystring['ZoningTypeGroupId'] = zoningtypegroupid
    if recordsperpage:
        querystring['RecordsPerPage'] = recordsperpage
    if cultureid:
        querystring['CultureId'] = cultureid
    if parkingtypeid:
        querystring['ParkingTypeId'] = parkingtypeid
    if currentpage:
        querystring['CurrentPage'] = currentpage
    if openhouseenddate:
        querystring['OpenHouseEndDate'] = openhouseenddate
    if bedrange:
        querystring['BedRange'] = bedrange
    if unitrange:
        querystring['UnitRange'] = unitrange
    if keywords:
        querystring['Keywords'] = keywords
    if pricemax:
        querystring['PriceMax'] = pricemax
    if pricemin:
        querystring['PriceMin'] = pricemin
    if constructionstyleid:
        querystring['ConstructionStyleId'] = constructionstyleid
    if openhouse:
        querystring['OpenHouse'] = openhouse
    if rentmax:
        querystring['RentMax'] = rentmax
    if numberofdays:
        querystring['NumberOfDays'] = numberofdays
    if transactiontypeid:
        querystring['TransactionTypeId'] = transactiontypeid
    if buildingtypeid:
        querystring['BuildingTypeId'] = buildingtypeid
    if openhousestartdate:
        querystring['OpenHouseStartDate'] = openhousestartdate
    if farmtypeid:
        querystring['FarmTypeId'] = farmtypeid
    if sortby:
        querystring['SortBy'] = sortby
    if bathrange:
        querystring['BathRange'] = bathrange
    if propertysearchtypeid:
        querystring['PropertySearchTypeId'] = propertysearchtypeid
    if landsizerange:
        querystring['LandSizeRange'] = landsizerange
    if buildingsizerange:
        querystring['BuildingSizeRange'] = buildingsizerange
    if rentmin:
        querystring['RentMin'] = rentmin
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-ca1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_get_statistics(latitude: int, longitude: int, cultureid: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistic information of surround area by GEO location"
    latitude: Latitude of specific location
        longitude: Longitude of specific location
        cultureid: 1 - English|2 - French
        
    """
    url = f"https://realty-in-ca1.p.rapidapi.com/properties/get-statistics"
    querystring = {'Latitude': latitude, 'Longitude': longitude, }
    if cultureid:
        querystring['CultureId'] = cultureid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-ca1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

