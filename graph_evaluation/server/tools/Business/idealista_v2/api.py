import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def properties_list(operation: str, locationname: str, locationid: str, country: str='it', childrenallowed: bool=None, floorheights: str=None, hashousekeeper: bool=None, couplesallowed: bool=None, garden: bool=None, privatetoilet: bool=None, terrace: bool=None, swimmingpool: bool=None, privateowner: bool=None, elevator: bool=None, airconditioning: bool=None, accessible: bool=None, gaypartners: bool=None, distance: int=None, ispoi: bool=None, maxitems: int=40, zoiid: str=None, locale: str='en', sort: str='asc', numpage: int=1, maxprice: int=None, minprice: int=None, shape: str=None, propertytype: str=None, auction: str=None, bedtype: str=None, ownernotliving: bool=None, newgender: str=None, gallery: bool=None, storeroom: bool=None, builtinwardrobes: bool=None, maxsize: int=None, minsize: int=None, garage: bool=None, luxury: bool=None, housemates: str=None, sincedate: str=None, petspolicy: str=None, showruledouts: bool=None, smokingpolicy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List properties for sale or rent in Italy, Spain, Portugal with options and filters"
    operation: One of the following rent|sale
        locationname: The value of name field returned in .../auto-complete endpoint.
        locationid: The value of locationId field returned in .../auto-complete endpoint. Do NOT use together with zoiId parameter 
        country: One of the following it|es|pt
        childrenallowed: true|false
        floorheights: One of the following topFloor|intermediateFloor|groundFloor (separated by comma for multiple values. Ex :  topFloor,groundFloor)
        hashousekeeper: true|false
        couplesallowed: true|false
        garden: true|false
        privatetoilet: true|false
        terrace: true|false
        swimmingpool: true|false
        privateowner: true|false
        elevator: true|false
        airconditioning: true|false
        accessible: true|false - Whether or not is the property accessible
        gaypartners: true|false
        distance: The radius to look for properties inside
        ispoi: true|false - Whether or not is it a point of interest, this parameter only works with zoiId parameter
        maxitems: The number of items per response for paging purpose
        zoiid: The value of zoiId field returned in .../auto-complete endpoint. Do NOT use together with locationId parameter 
        locale: One of the following en|es|it|pt|de|fr|ro|ru|pl|sv|fi|nb|nl
        sort: One of the following desc|asc
        numpage: The page index for paging purpose
        maxprice: Max price
        minprice: Min price
        shape: The value of shape field returned in .../zois/detail endpoint. Simply pass the json object as string. Do NOT use together with locationId or zoiId. Ex : {\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"coordinates\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\":[[[[9.20289,45.469786,0],...,[9.20289,45.469786,0]]]],\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"type\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\":\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"MultiPolygon\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"}
        propertytype: One of the following : homes|bedrooms|newDevelopments|offices|premises|garages|lands|storageRooms|buildings
        auction: Leave empty or one of the following : excludeAuctions|onlyAuctions
        bedtype: One of the following none|single|double|twoBeds
        ownernotliving: true|false
        newgender: male|female
        gallery: true|false
        storeroom: true|false
        builtinwardrobes: true|false
        maxsize: Max area size
        minsize: Min area size
        garage: true|false
        luxury: true|false
        housemates: One of the following 2|3|4 (separated by comma for multiple values, Ex : 3,4)
        sincedate: One of the following Y|W|M  (Last 48 hours|Last week|Last month)
        petspolicy: allowed|disallowed
        showruledouts: true|false
        smokingpolicy: allowed|disallowed
        
    """
    url = f"https://idealista2.p.rapidapi.com/properties/list"
    querystring = {'operation': operation, 'locationName': locationname, 'locationId': locationid, }
    if country:
        querystring['country'] = country
    if childrenallowed:
        querystring['childrenAllowed'] = childrenallowed
    if floorheights:
        querystring['floorHeights'] = floorheights
    if hashousekeeper:
        querystring['hasHouseKeeper'] = hashousekeeper
    if couplesallowed:
        querystring['couplesAllowed'] = couplesallowed
    if garden:
        querystring['garden'] = garden
    if privatetoilet:
        querystring['privateToilet'] = privatetoilet
    if terrace:
        querystring['terrace'] = terrace
    if swimmingpool:
        querystring['swimmingPool'] = swimmingpool
    if privateowner:
        querystring['privateOwner'] = privateowner
    if elevator:
        querystring['elevator'] = elevator
    if airconditioning:
        querystring['airConditioning'] = airconditioning
    if accessible:
        querystring['accessible'] = accessible
    if gaypartners:
        querystring['gayPartners'] = gaypartners
    if distance:
        querystring['distance'] = distance
    if ispoi:
        querystring['isPoi'] = ispoi
    if maxitems:
        querystring['maxItems'] = maxitems
    if zoiid:
        querystring['zoiId'] = zoiid
    if locale:
        querystring['locale'] = locale
    if sort:
        querystring['sort'] = sort
    if numpage:
        querystring['numPage'] = numpage
    if maxprice:
        querystring['maxPrice'] = maxprice
    if minprice:
        querystring['minPrice'] = minprice
    if shape:
        querystring['shape'] = shape
    if propertytype:
        querystring['propertyType'] = propertytype
    if auction:
        querystring['auction'] = auction
    if bedtype:
        querystring['bedType'] = bedtype
    if ownernotliving:
        querystring['ownerNotLiving'] = ownernotliving
    if newgender:
        querystring['newGender'] = newgender
    if gallery:
        querystring['gallery'] = gallery
    if storeroom:
        querystring['storeRoom'] = storeroom
    if builtinwardrobes:
        querystring['builtinWardrobes'] = builtinwardrobes
    if maxsize:
        querystring['maxSize'] = maxsize
    if minsize:
        querystring['minSize'] = minsize
    if garage:
        querystring['garage'] = garage
    if luxury:
        querystring['luxury'] = luxury
    if housemates:
        querystring['housemates'] = housemates
    if sincedate:
        querystring['sinceDate'] = sincedate
    if petspolicy:
        querystring['petsPolicy'] = petspolicy
    if showruledouts:
        querystring['showRuledOuts'] = showruledouts
    if smokingpolicy:
        querystring['smokingPolicy'] = smokingpolicy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "idealista2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def zois_detail(zoiid: str, language: str='en', country: str='it', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about specific zone of interest"
    zoiid: The value of zoiId field returned in .../auto-complete endpoint
        language: One of the following en|es|it|pt|de|fr|ro|ru|pl|sv|fi|nb|nl
        country: One of the following it|es|pt
        
    """
    url = f"https://idealista2.p.rapidapi.com/zois/detail"
    querystring = {'zoiId': zoiid, }
    if language:
        querystring['language'] = language
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "idealista2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_detail(propertycode: int, country: str='it', language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get property detail"
    propertycode: The value of propertyCode field returned in .../properties/list endpoint
        country: One of the following it|es|pt
        language: One of the following en|es|it|pt|de|fr|ro|ru|pl|sv|fi|nb|nl
        
    """
    url = f"https://idealista2.p.rapidapi.com/properties/detail"
    querystring = {'propertyCode': propertycode, }
    if country:
        querystring['country'] = country
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "idealista2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_complete(prefix: str, country: str='it', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get auto complete suggestion by term or phrase"
    prefix: Any term or phrase that you are familiar with
        country: One of the following it|es|pt
        
    """
    url = f"https://idealista2.p.rapidapi.com/auto-complete"
    querystring = {'prefix': prefix, }
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "idealista2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

