import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_rooms(numpage: str, order: str, locationname: str, locale: str, locationid: str, location: str, maxitems: str, intermediatefloor: bool=None, sincedate: str=None, groundfloor: bool=None, occupation: str=None, ownernotliving: bool=None, gaypartners: bool=None, topfloor: bool=None, swimmingpool: bool=None, couplesallowed: bool=None, childrenallowed: bool=None, airconditioning: bool=None, housemates1: bool=None, elevator: bool=None, terrace: bool=None, garden: bool=None, accessible: bool=None, hashousekeeper: bool=None, exterior: bool=None, privatetoilet: bool=None, availablefrom: str=None, housemates3: bool=None, housemates2: bool=None, streetviewwindow: bool=None, maxprice: str=None, newgender: str=None, smokingpolicy: str=None, petspolicy: str=None, minprice: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List renting rooms with the requested parameters. Buy Rent > Rooms or Share > Homes."
    order: Order by one of the followings: relevance|lowestprice|highestprice|mostrecent|leastrecent|highestpricereduction|lowestpricem2|highestpricem2|biggest|smallest|highestfloors|lowestfloors

Default is relevance.
        sincedate: Publication Date Parameter. One of the following values:

Indifferent: leave blank (by default)
Last 48h: Y
Last week: W
Last month: M
        occupation: Flat sharing with... Pick one of these:

Indifferent: leave blank (default)
With workers: workers
With students: students
        ownernotliving: Owner doesn't live in the property.
        gaypartners: LGTB Friendly
        housemates1: Flat share with 1 person.
        elevator: Lift parameter.
        accessible: Accessible property.
        hashousekeeper: Cleaning services included.
        availablefrom: TODO. Don't use. Date the room is available. Pick between:

Any date: leave blank (default)
Available now: now
+1 month since now: 1m
+2 months since now: 2m
+3 months since now: 3m
+4 months since now: 4m
+5 months since now: 5m
        housemates3: Flat share with 3 or more people.
        housemates2: Flat share with 2 people.
        streetviewwindow: Street facing window.
        newgender: 'You are' parameter (your gender). Pick between: male|female
        smokingpolicy: Smokers. Pick between:

Indiferent: leave blank (default)
Smoking allowed: allowed
Smoking is not allowed: disallowed
        petspolicy: Pets. Pick between:

Indifferent: leave blank (default)
Pets allowed: allowed
Pets not allowed: disallowed
        
    """
    url = f"https://idealista7.p.rapidapi.com/listrooms"
    querystring = {'numPage': numpage, 'order': order, 'locationName': locationname, 'locale': locale, 'locationId': locationid, 'location': location, 'maxItems': maxitems, }
    if intermediatefloor:
        querystring['intermediateFloor'] = intermediatefloor
    if sincedate:
        querystring['sinceDate'] = sincedate
    if groundfloor:
        querystring['groundFloor'] = groundfloor
    if occupation:
        querystring['occupation'] = occupation
    if ownernotliving:
        querystring['ownerNotLiving'] = ownernotliving
    if gaypartners:
        querystring['gayPartners'] = gaypartners
    if topfloor:
        querystring['topFloor'] = topfloor
    if swimmingpool:
        querystring['swimmingPool'] = swimmingpool
    if couplesallowed:
        querystring['couplesAllowed'] = couplesallowed
    if childrenallowed:
        querystring['childrenAllowed'] = childrenallowed
    if airconditioning:
        querystring['airConditioning'] = airconditioning
    if housemates1:
        querystring['housemates1'] = housemates1
    if elevator:
        querystring['elevator'] = elevator
    if terrace:
        querystring['terrace'] = terrace
    if garden:
        querystring['garden'] = garden
    if accessible:
        querystring['accessible'] = accessible
    if hashousekeeper:
        querystring['hasHouseKeeper'] = hashousekeeper
    if exterior:
        querystring['exterior'] = exterior
    if privatetoilet:
        querystring['privateToilet'] = privatetoilet
    if availablefrom:
        querystring['availableFrom'] = availablefrom
    if housemates3:
        querystring['housemates3'] = housemates3
    if housemates2:
        querystring['housemates2'] = housemates2
    if streetviewwindow:
        querystring['streetViewWindow'] = streetviewwindow
    if maxprice:
        querystring['maxPrice'] = maxprice
    if newgender:
        querystring['newGender'] = newgender
    if smokingpolicy:
        querystring['smokingPolicy'] = smokingpolicy
    if petspolicy:
        querystring['petsPolicy'] = petspolicy
    if minprice:
        querystring['minPrice'] = minprice
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "idealista7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_offices(location: str, maxitems: str, locationid: str, locale: str, locationname: str, numpage: str, operation: str, order: str, bankoffer: str=None, sincedate: str=None, heating: bool=None, airconditioning: bool=None, hasplan: bool=None, garage: bool=None, security: bool=None, exterior: bool=None, elevator: bool=None, accessible: bool=None, buildingtype: str=None, hotwater: bool=None, minsize: str=None, layout: str=None, maxprice: str=None, minprice: str=None, maxsize: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List office properties with the requested parameters. Buy/Rent > Offices. Filters are optional parameters, please make sure the request follows the rules in each parameter."
    operation: Search sales or rentals. Pick between: sale|rent
        bankoffer: Bank-owned
        sincedate: Publication Date Parameter. One of the following values:

Indifferent: leave blank (by default)
Last 48h: Y
Last week: W
Last month: M
        hasplan: With floor plan
        security: Surveillance and security.
        elevator: Lift parameter.
        accessible: Accessible building
        buildingtype: Type of building parameter. Pick between:

Indifferent: leave blank (default)
Exclusively for offices: exclusive
Mixted use: mixed
        layout: Distribution parameter. Pick between:

Indifferent: leave blank (default)
Open plan: openPlan
Divided by walls: withWalls
        
    """
    url = f"https://idealista7.p.rapidapi.com/listoffices"
    querystring = {'location': location, 'maxItems': maxitems, 'locationId': locationid, 'locale': locale, 'locationName': locationname, 'numPage': numpage, 'operation': operation, 'order': order, }
    if bankoffer:
        querystring['bankOffer'] = bankoffer
    if sincedate:
        querystring['sinceDate'] = sincedate
    if heating:
        querystring['heating'] = heating
    if airconditioning:
        querystring['airConditioning'] = airconditioning
    if hasplan:
        querystring['hasPlan'] = hasplan
    if garage:
        querystring['garage'] = garage
    if security:
        querystring['security'] = security
    if exterior:
        querystring['exterior'] = exterior
    if elevator:
        querystring['elevator'] = elevator
    if accessible:
        querystring['accessible'] = accessible
    if buildingtype:
        querystring['buildingType'] = buildingtype
    if hotwater:
        querystring['hotWater'] = hotwater
    if minsize:
        querystring['minSize'] = minsize
    if layout:
        querystring['layout'] = layout
    if maxprice:
        querystring['maxPrice'] = maxprice
    if minprice:
        querystring['minPrice'] = minprice
    if maxsize:
        querystring['maxSize'] = maxsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "idealista7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_new_homes(numpage: str, location: str, operation: str, locationid: str, maxitems: str, locationname: str, locale: str, order: str, bedrooms2: bool=None, bedrooms0: bool=None, bankoffer: bool=None, statesubsidized: bool=None, bedrooms3: bool=None, flats: bool=None, housesorchalets: bool=None, renttoown: bool=None, finished: bool=None, bedrooms1: bool=None, maxprice: str=None, bedrooms4: bool=None, minprice: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List new home properties with the requested parameters. Buy/Rent > New home. Filters are optional parameters, please make sure the request follows the rules in each parameter."
    operation: Search sales or rentals. Pick between: sale|rent
        order: Order by one of the followings: relevance|lowestprice|highestprice|mostrecent|leastrecent|highestpricereduction|lowestpricem2|highestpricem2|biggest|smallest|highestfloors|lowestfloors

Default is relevance.
        statesubsidized: Set true if you want to filter by: Subsidized housing
        renttoown: Set true if you want to filter by: For rent with buying option
        finished: Set true if you want to filter by: Finished new home
        bedrooms4: Set true if you want to filter by 4 or more bedrooms.
        
    """
    url = f"https://idealista7.p.rapidapi.com/listnewhomes"
    querystring = {'numPage': numpage, 'location': location, 'operation': operation, 'locationId': locationid, 'maxItems': maxitems, 'locationName': locationname, 'locale': locale, 'order': order, }
    if bedrooms2:
        querystring['bedrooms2'] = bedrooms2
    if bedrooms0:
        querystring['bedrooms0'] = bedrooms0
    if bankoffer:
        querystring['bankOffer'] = bankoffer
    if statesubsidized:
        querystring['stateSubsidized'] = statesubsidized
    if bedrooms3:
        querystring['bedrooms3'] = bedrooms3
    if flats:
        querystring['flats'] = flats
    if housesorchalets:
        querystring['housesOrChalets'] = housesorchalets
    if renttoown:
        querystring['rentToOwn'] = renttoown
    if finished:
        querystring['finished'] = finished
    if bedrooms1:
        querystring['bedrooms1'] = bedrooms1
    if maxprice:
        querystring['maxPrice'] = maxprice
    if bedrooms4:
        querystring['bedrooms4'] = bedrooms4
    if minprice:
        querystring['minPrice'] = minprice
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "idealista7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_details(location: str, propertyid: str, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get more details about a property."
    location: One of the following values: es|pt|it
        propertyid: Property Id.
        language: The language of the ad. Pick between: es|it|pt|en|ca|de|fr|nl|nb
        
    """
    url = f"https://idealista7.p.rapidapi.com/propertydetails"
    querystring = {'location': location, 'propertyId': propertyid, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "idealista7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_home_properties(locale: str, order: str, locationname: str, maxitems: str, operation: str, numpage: str, locationid: str, location: str, onlyflats: bool=None, petsallowed: bool=None, sincedate: str=None, furnished: str=None, newdevelopment: bool=None, groundfloor: bool=None, bathroom3: bool=None, terrance: bool=None, bathrooms2: bool=None, hasplan: bool=None, builtinwardrobes: bool=None, swimmingpool: bool=None, topfloor: bool=None, storeroom: bool=None, garage: bool=None, luxury: bool=None, exterior: bool=None, intermediatefloor: bool=None, garden: bool=None, virtualtour: bool=None, bathrooms1: bool=None, renew: bool=None, bankoffer: bool=None, airconditioning: bool=None, elevator: bool=None, good: bool=None, accessible: bool=None, bedrooms0: bool=None, bedrooms4: bool=None, bedrooms3: bool=None, bedrooms1: bool=None, bedrooms2: bool=None, apartamentotype: bool=None, independanthouse: bool=None, penthouse: bool=None, lofttype: bool=None, villatype: bool=None, countryhouse: bool=None, casabajatype: bool=None, atticstudiotype: bool=None, cortijotype: bool=None, semidetachedhouse: bool=None, terracedhouse: bool=None, chalet: bool=None, flat: bool=None, duplex: bool=None, maxsize: int=None, minsize: int=None, micrositeshortname: str=None, minprice: int=None, maxprice: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List home properties with the requested parameters. Buy/Rent > Homes. Filters are optional parameters, please make sure the request follows the rules in each parameter.
		
		Boolean parameters should be true always if you want to use them. To make them false, don't include them in the requests (leave blank)."
    locale: The language of the ads. Pick between: es|it|pt|en|ca|de|fr|nl|nb
        order: Order by one of the followings: relevance|lowestprice|highestprice|mostrecent|leastrecent|highestpricereduction|lowestpricem2|highestpricem2|biggest|smallest|highestfloors|lowestfloors

Default is relevance.
        maxitems: By default the value is 40. NOT recommended to change this.
        operation: Search sales or rentals. Pick between: sale|rent
        location: One of the following values: es|pt|it
        onlyflats: Set true if you want to find flats.
        petsallowed: ONLY works with operation=rent.
        sincedate: Publication Date Parameter. One of the following values:

Indifferent: leave blank (by default)
Last 48h: Y
Last week: W
Last month: M

        furnished: Furnishings parameter. ONLY valid when operation=rent. Pick between:

Indifferent: leave blank (default)
Furnished: furnished
Only furnished kitchens: furnishedKitchen

        chalet: Set true if you want to get independantHouses, semidetachedHouses, terracedHouses and countryHouse (all of them).
        flat: Set true if you want to get flats, penthouses and duplex. (All of them)
        duplex: Set true if you want to find duplex.
        maxsize: Maximum size to search in m2.
        minsize: Minimum size to search in m2.
        micrositeshortname: Get ads from a specific real estate. For example if the URL from the real estate is: https://www.idealista.com/pro/cajal-gestion-inmobiliaria/

The parameter micrositeShortName should be cajal-gestion-inmobiliaria
        minprice: Minimum price to search.
        maxprice: Maximum price to search.
        
    """
    url = f"https://idealista7.p.rapidapi.com/listhomes"
    querystring = {'locale': locale, 'order': order, 'locationName': locationname, 'maxItems': maxitems, 'operation': operation, 'numPage': numpage, 'locationId': locationid, 'location': location, }
    if onlyflats:
        querystring['onlyFlats'] = onlyflats
    if petsallowed:
        querystring['petsAllowed'] = petsallowed
    if sincedate:
        querystring['sinceDate'] = sincedate
    if furnished:
        querystring['furnished'] = furnished
    if newdevelopment:
        querystring['newDevelopment'] = newdevelopment
    if groundfloor:
        querystring['groundFloor'] = groundfloor
    if bathroom3:
        querystring['bathroom3'] = bathroom3
    if terrance:
        querystring['terrance'] = terrance
    if bathrooms2:
        querystring['bathrooms2'] = bathrooms2
    if hasplan:
        querystring['hasPlan'] = hasplan
    if builtinwardrobes:
        querystring['builtinWardrobes'] = builtinwardrobes
    if swimmingpool:
        querystring['swimmingPool'] = swimmingpool
    if topfloor:
        querystring['topFloor'] = topfloor
    if storeroom:
        querystring['storeRoom'] = storeroom
    if garage:
        querystring['garage'] = garage
    if luxury:
        querystring['luxury'] = luxury
    if exterior:
        querystring['exterior'] = exterior
    if intermediatefloor:
        querystring['intermediateFloor'] = intermediatefloor
    if garden:
        querystring['garden'] = garden
    if virtualtour:
        querystring['virtualTour'] = virtualtour
    if bathrooms1:
        querystring['bathrooms1'] = bathrooms1
    if renew:
        querystring['renew'] = renew
    if bankoffer:
        querystring['bankOffer'] = bankoffer
    if airconditioning:
        querystring['airConditioning'] = airconditioning
    if elevator:
        querystring['elevator'] = elevator
    if good:
        querystring['good'] = good
    if accessible:
        querystring['accessible'] = accessible
    if bedrooms0:
        querystring['bedrooms0'] = bedrooms0
    if bedrooms4:
        querystring['bedrooms4'] = bedrooms4
    if bedrooms3:
        querystring['bedrooms3'] = bedrooms3
    if bedrooms1:
        querystring['bedrooms1'] = bedrooms1
    if bedrooms2:
        querystring['bedrooms2'] = bedrooms2
    if apartamentotype:
        querystring['apartamentoType'] = apartamentotype
    if independanthouse:
        querystring['independantHouse'] = independanthouse
    if penthouse:
        querystring['penthouse'] = penthouse
    if lofttype:
        querystring['loftType'] = lofttype
    if villatype:
        querystring['villaType'] = villatype
    if countryhouse:
        querystring['countryHouse'] = countryhouse
    if casabajatype:
        querystring['casaBajaType'] = casabajatype
    if atticstudiotype:
        querystring['atticStudioType'] = atticstudiotype
    if cortijotype:
        querystring['cortijoType'] = cortijotype
    if semidetachedhouse:
        querystring['semidetachedHouse'] = semidetachedhouse
    if terracedhouse:
        querystring['terracedHouse'] = terracedhouse
    if chalet:
        querystring['chalet'] = chalet
    if flat:
        querystring['flat'] = flat
    if duplex:
        querystring['duplex'] = duplex
    if maxsize:
        querystring['maxSize'] = maxsize
    if minsize:
        querystring['minSize'] = minsize
    if micrositeshortname:
        querystring['micrositeShortName'] = micrositeshortname
    if minprice:
        querystring['minPrice'] = minprice
    if maxprice:
        querystring['maxPrice'] = maxprice
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "idealista7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_suggestions(location: str, operation: str, propertytype: str, prefix: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get location suggestions"
    location: One of the following values: es|pt|it
        
    """
    url = f"https://idealista7.p.rapidapi.com/getsuggestions"
    querystring = {'location': location, 'operation': operation, 'propertyType': propertytype, 'prefix': prefix, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "idealista7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

