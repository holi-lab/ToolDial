import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def bulk_geocoding(addresses: str, f: str, token: str, matchoutofrange: str='false', preferredlabelvalues: str='localCity', outfields: str='AddNum,StAddr,City', category: str='Address', locationtype: str='street', searchextent: str='-104,35.6,-94.32,41', sourcecountry: str='USA', outsr: int=102100, langcode: str='fr', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Geocode an entire list of addresses in one request using the geocodeAddresses operation. Geocoding many addresses at once is also known as batch or bulk geocoding.
		
		![bulk geocoding](https://developers.arcgis.com/rest/geocode/api-reference/GUID-FD609701-B9B5-49DB-BFD9-A936280A09C6-web.png)
		
		[API reference `geocodeAddresses` operation](https://developers.arcgis.com/rest/geocode/api-reference/geocoding-geocode-addresses.htm)"
    addresses: A record set representing the addresses to be geocoded. Each record must include an OBJECTID attribute with a unique value as well as values for the address fields supported by the service.
        f: (Required) Format of the response: json, pjson, html, ...
        token: (Required) How to create an API key: https://youtu.be/Q1x4NZPK8Ws
        matchoutofrange: A Boolean which specifies if StreetAddress matches should be returned even when the input house number is outside of the house number range defined for the input street. Out-of-range matches have Addr_type=StreetAddressExt. The geometry of such matches is a point corresponding to the end of the street segment where the range value is closest to the input house number. If matchOutOfRange is not specified in a request, its value is set to true by default.
        preferredlabelvalues: Allows simple configuration of output fields returned in a response from the ArcGIS World Geocoding Service by specifying which address component values should be included in output fields. Supports a single value as input. If the parameter is blank or excluded from a request then default address label formats will be used.
        outfields: The list of fields to be returned within the attributes object of the geocodeAddresses response. Descriptions for each of these fields are available in the Service output topic.
        category: A place or address type that can be used to filter geocodeAddresses results. The parameter supports input of single category values or multiple comma-separated values. See Category filtering for complete details about the category parameter.
        locationtype: Specifies if the output geometry of PointAddress and Subaddress matches should be the rooftop point or street entrance location. Valid values are rooftop and street. The default value is rooftop.
        searchextent: A set of bounding box coordinates that limit the search area to a specific region. This is especially useful for applications in which a user will search for places and addresses within the current map extent. Helper to get search extent with https://arcgis-js-api-extent-helper.gavinr.com/
        sourcecountry: A value representing the country. When a value is passed for this parameter, all of the addresses in the input table are sent to the specified country to be geocoded. For example, if sourceCountry=USA is passed with a batch request, it is assumed that all of the addresses in the table are in the United States, so only matching addresses in USA are returned. Using this parameter can increase batch geocoding performance when all addresses are within a single country.
        outsr: The spatial reference of the x/y coordinates returned by a geocode request. This is useful for applications using a map with a spatial reference different than that of the geocode service.
        langcode: Sets the language in which reverse-geocoded addresses are returned. Addresses in many countries are available in more than one language; in these cases the langCode parameter can be used to specify which language should be used for addresses returned by the reverseGeocode operation. This is useful for ensuring that addresses are returned in the expected language. For example, a web application could be designed to get the browser language and pass it as the langCode parameter value in a reverseGeocode request.
        
    """
    url = f"https://arcgis-platform-geocoding.p.rapidapi.com/arcgis/rest/services/World/GeocodeServer/geocodeAddresses"
    querystring = {'addresses': addresses, 'f': f, 'token': token, }
    if matchoutofrange:
        querystring['matchOutOfRange'] = matchoutofrange
    if preferredlabelvalues:
        querystring['preferredLabelValues'] = preferredlabelvalues
    if outfields:
        querystring['outFields'] = outfields
    if category:
        querystring['category'] = category
    if locationtype:
        querystring['locationType'] = locationtype
    if searchextent:
        querystring['searchExtent'] = searchextent
    if sourcecountry:
        querystring['sourceCountry'] = sourcecountry
    if outsr:
        querystring['outSR'] = outsr
    if langcode:
        querystring['langCode'] = langcode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "arcgis-platform-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def autocomplete_suggestions(text: str, f: str, searchextent: str='-104,35.6,-94.32,41', category: str='Address,Postal', countrycode: str='USA', preferredlabelvalues: str='postalCity', maxsuggestions: int=10, location: str='-117.196,34.056', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The ArcGIS World Geocoding Service includes a method that allows character-by-character autocomplete suggestions to be generated for user input in a client application. This capability facilitates the interactive search user experience by reducing the number of characters that need to be typed before a suggested match is obtained. The idea is that a client application can provide a list of suggestions that is updated with each character entered by a user until the place they are looking for is returned in the list.
		
		![Autocomplete diagram](https://developers.arcgis.com/rest/geocode/api-reference/GUID-9A754AFE-8154-46C3-8A31-3566963F971E-web.png)
		
		[API reference `suggest` operation](https://developers.arcgis.com/rest/geocode/api-reference/geocoding-suggest.htm)"
    text: (Required) The input text entered by a user, which is used by the suggest operation to generate a list of possible matches.
        f: (Required) Format of the response: json or pjson
        searchextent: A set of bounding box coordinates that limit the search area for suggestions to a specific region. This is especially useful for applications in which a user will search for places and addresses within the current map extent. Helper to get search extent with https://arcgis-js-api-extent-helper.gavinr.com/
        category: A place or address type that can be used to filter suggest results. The parameter supports input of single category values or multiple comma-separated values. The category parameter must be passed in a request with the text parameter. See Category filtering for complete details about the category parameter.
        countrycode: Limits the returned suggestions to values in a particular country. Valid 2- and 3-character country code values for each country are available in Geocode coverage.
        preferredlabelvalues: Allows simple configuration of suggestion labels returned in a response from the ArcGIS World Geocoding Service by specifying which address component values should be included in the label. If the parameter is blank or excluded from a request then default address formats are used.
        maxsuggestions: The maximum number of suggestions returned by a suggest response, up to the maximum number allowed by the service. Currently, the ArcGIS World Geocoding Service allows up to 15 suggestions to be returned. If maxSuggestions is not included in the suggest request, the default value is 5.


        location: Defines an origin point that is used to prefer or boost geocoding candidates based on their proximity to the location. Candidates near the location are prioritized relative to those further away. This is useful in mobile applications where a user wants to search for places in the vicinity of their current GPS location, or in mapping applications where users want to search for places near the center of the map.
        
    """
    url = f"https://arcgis-platform-geocoding.p.rapidapi.com/arcgis/rest/services/World/GeocodeServer/suggest"
    querystring = {'text': text, 'f': f, }
    if searchextent:
        querystring['searchExtent'] = searchextent
    if category:
        querystring['category'] = category
    if countrycode:
        querystring['countryCode'] = countrycode
    if preferredlabelvalues:
        querystring['preferredLabelValues'] = preferredlabelvalues
    if maxsuggestions:
        querystring['maxSuggestions'] = maxsuggestions
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "arcgis-platform-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_address(f: str, token: str, searchextent: str, singleline: str, magickey: str='JS91CYhQDS5vDPhvSMyGZby0YFbaUDoaM5bHMoFF', location: str='-117.196,34.056', address2: str='Beetham Tower', region: str='Florida', outfields: str='*', category: str='Address', postal: int=92373, preferredlabelvalues: str='postalCity', sourcecountry: str='USA', countrycode: str='USA', neighborhood: str='Herrera', outsr: int=102100, maxlocations: int=10, subregion: str='Vienne', forstorage: str='true', address3: str='Suite 4208', langcode: str='fr', locationtype: str='street', matchoutofrange: str='false', city: str='Los Angeles', address: str='Avenida Revolucion 8208', postalext: int=1112, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The findAddressCandidates operation geocodes one location per request; the input address can be combined into a single input field or divided among multiple parameters.
		
		![geoceode screenshot](https://developers.arcgis.com/rest/geocode/api-reference/GUID-F2F78690-5FB5-4BF1-8633-26BF184C45A9-web.png)
		
		[API reference for `findAddressCandidates` operation](https://developers.arcgis.com/rest/geocode/api-reference/geocoding-find-address-candidates.htm)"
    f: Search for an address, POI category, intersection, etc.
        token: (Required) Format of the response: json, pjson, html, ...
        searchextent: The list of fields to be returned within the attributes object of the response. Descriptions for each of these fields are available in the Output fields section of this document. Helper to get search extent with https://arcgis-js-api-extent-helper.gavinr.com/
        singleline: Specifies whether the results of the operation will be persisted
        magickey: Defines an origin point that is used to prefer or boost geocoding candidates based on their proximity to the location. Candidates near the location are prioritized relative to those further away. This is useful in mobile applications where a user wants to search for places in the vicinity of their current GPS location, or in mapping applications where users want to search for places near the center of the map.
        location: A place or address type that can be used to filter findAddressCandidates results. The parameter supports input of single category values or multiple comma-separated values. The category parameter can be passed in a request with the SingleLine or address parameters. It can also be passed in a request on its own without the singleline or address parameters. See Category filtering for complete details about the category parameter.
        address2: A string that represents the third line of a street address. This can include street name/house number, building name, place name, or subunit.
        region: The standard postal code for an address, typically, a 3–6-digit alphanumeric code.
        outfields: A string that represents the first line of a street address. In most cases this field will be used for street name and house number input, but it can also be used to input building name or place name.
        category: The spatial reference of the x/y coordinates returned by a geocode request. This is useful for applications using a map with a spatial reference different than that of the geocode service.
        postal: A postal code extension, such as the United States Postal Service ZIP+4 code, provides finer resolution or higher accuracy when also passing postal.
        sourcecountry: Allows simple configuration of output fields returned in a response from the ArcGIS World Geocoding Service by specifying which address component values should be included in output fields. Supports a single value or a comma-delimited collection of values as input. If the parameter is blank or excluded from a request then default address label formats will be used.


        countrycode: The findAddressCandidates operation retrieves results quicker when you pass in valid SingleLine and magicKey values than when you don't pass in magicKey. However, to get these advantages, you need to make a prior request to suggest, which provides a magicKey. This may or may not be relevant to your workflow.
        neighborhood: The next largest administrative division associated with an address, typically, a city or municipality. A city is a subdivision of a subregion or a region.
        outsr: The maximum number of locations to be returned by a search, up to the maximum number allowed by the service. If not specified, then all matching candidates up to the service maximum are returned.
        maxlocations: A Boolean which specifies if StreetAddress matches should be returned even when the input house number is outside of the house number range defined for the input street. Out-of-range matches have Addr_type=StreetAddressExt. The geometry of such matches is a point corresponding to the end of the street segment where the range value is closest to the input house number. If matchOutOfRange is not specified in a request, its value is set to true by default.
        subregion: The largest administrative division associated with an address, typically, a state or province.
        forstorage: Get search extent with https://arcgis-js-api-extent-helper.gavinr.com/
        address3: The smallest administrative division associated with an address, typically, a neighborhood or a section of a larger populated place. A neighborhood is a subdivision of a city.
        langcode: Limits the candidates returned by the findAddressCandidates operation to the specified country or countries. For example, if sourceCountry=USA is included in the request, it is assumed that the address is in the United States, so only matching addresses in USA are returned. Using this parameter prevents potential unexpected results in other countries for ambiguous searches.
        locationtype: Sets the language in which geocode results are returned. Addresses and places in many countries are available in more than one language; in these cases the langCode parameter can be used to specify which language should be used for results returned by the findAddressCandidates operation. This is useful for ensuring that results are returned in the expected language. For example, a web application could be designed to get the browser language and pass it as the langCode parameter value in a findAddressCandidates request.
        matchoutofrange: Specifies if the output geometry of PointAddress and Subaddress matches should be the rooftop point or street entrance location. Valid values are rooftop and street. The default value is rooftop.
        city: The next largest administrative division associated with an address. Depending on the country, a subregion can represent a county, state, or province.
        address: A string that represents the second line of a street address. This can include street name/house number, building name, place name, or subunit.
        postalext: A value representing the country. Providing this value increases geocoding speed. Acceptable values include the full country name in English or the official language of the country, the 2-character country code, or the 3-character country code. A list of supported countries and codes is available in the Geocode coverage topic.
        
    """
    url = f"https://arcgis-platform-geocoding.p.rapidapi.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates"
    querystring = {'f': f, 'token': token, 'searchExtent': searchextent, 'singleLine': singleline, }
    if magickey:
        querystring['magicKey'] = magickey
    if location:
        querystring['location'] = location
    if address2:
        querystring['address2'] = address2
    if region:
        querystring['region'] = region
    if outfields:
        querystring['outFields'] = outfields
    if category:
        querystring['category'] = category
    if postal:
        querystring['postal'] = postal
    if preferredlabelvalues:
        querystring['preferredLabelValues'] = preferredlabelvalues
    if sourcecountry:
        querystring['sourceCountry'] = sourcecountry
    if countrycode:
        querystring['countryCode'] = countrycode
    if neighborhood:
        querystring['neighborhood'] = neighborhood
    if outsr:
        querystring['outSR'] = outsr
    if maxlocations:
        querystring['maxLocations'] = maxlocations
    if subregion:
        querystring['subregion'] = subregion
    if forstorage:
        querystring['forStorage'] = forstorage
    if address3:
        querystring['address3'] = address3
    if langcode:
        querystring['langCode'] = langcode
    if locationtype:
        querystring['locationType'] = locationtype
    if matchoutofrange:
        querystring['matchOutOfRange'] = matchoutofrange
    if city:
        querystring['city'] = city
    if address:
        querystring['address'] = address
    if postalext:
        querystring['postalExt'] = postalext
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "arcgis-platform-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

