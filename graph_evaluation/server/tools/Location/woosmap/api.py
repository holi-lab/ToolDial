import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getzone(zone_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to retrieve a zone from his `zone_id`
		"
    zone_id: ID of the zone to get
        
    """
    url = f"https://woosmap1.p.rapidapi.com/zones/{zone_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "woosmap1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstore(storeid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to retrieve an asset from his `storeId`
		"
    storeid: ID of the asset to get
        
    """
    url = f"https://woosmap1.p.rapidapi.com/stores/{storeid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "woosmap1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def localitiesautocomplete(input: str, language: str='en', extended: str='postal_code', data: str='advanced', types: str='postal_code', components: str='country:fr|country:es', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Autocomplete on worldwide suggestions for a for text-based geographic searches. It can match on full words as well as substrings. You can therefore send queries as the user types, to provide on-the-fly city names, postal codes or suburb name suggestions.
		"
    input: The text string on which to search, for example: "london" or "123 Cross Road". The Woosmap Localities API will return predictions matches based on this string and order the results based on their perceived relevance.

        language: The language code, using ISO 3166-1 Alpha-2 country codes, indicating in which language the results should be returned, if possible. If language is not supplied, the Localities service will use english as default language. No language necessary for `postal_code` request.

        extended: If set, this parameter allows a refined search over locality names that bears the same postal code. By triggering this parameter, integrators will benefit from a search spectrum on the `locality` type that ***includes postal codes***. To avoid confusion, it is recommended not to activate this parameter along with the `postal_code` type which could lead to duplicate locations. Also, the default description returned by the API changes to `name (postal code), admin_1, admin_0`. It is only available for France and Italy.

        data: Two values for this parameter: `standard` or `advanced`. By default, if the parameter is not defined, value is set as `standard`. The `advanced` value opens suggestions to worldwide postal codes in addition to postal codes for Western Europe. ***A dedicated option subject to specific billing on your license is needed to use this parameter. Please contact us if you are interested in using this parameter and you do not have subscribed the proper option yet.***

        types: The types of suggestion to return.

  * `locality`: includes locality names (from city to village) and suburbs
  * `postal_code`: publicly-used postal codes around the world
  * `address`: addresses (only available for UK and France)
  * `admin_level`: most commonly used administrative areas
  * `country`: countries as whole point of interest
  * `airport`: includes all medium sized to international sized airports
  * `train_station`: includes all train stations
  * `metro_station`: includes all metro stations
  * `shopping`: includes shopping malls (or "shopping centers") - *may include private retail brands*
  * `museum`: includes museums
  * `tourist_attraction`: includes tourist attractions like the Eiffel tower
  * `amusement_park`: includes amusement parks like Disneyland Paris
  * `art_gallery`: includes art galleries
  * `zoo`: includes zoos

> The information returned on an `address` suggestion contain only a "single-line" description. A request to the [Details endpoint](/products/localities/details/) of the API must be performed to retrieve the location (geographic coordinates) and the address components (street address, zipcode, city..).
Not specifying any type will only query `locality` and `postal_code`. Multiple types can be passed using the pipe character (`|`) as a separator. For example: `types=locality|airport|admin_level`.

        components: A grouping of places to which you would like to restrict your results. Currently, you can use `components` to filter over countries. Countries must be passed as a two character, ISO 3166-1 Alpha-2 compatible country code. For example: `components=country:fr` would restrict your results to places within France and `components=country:fr-fr` returns locations only in Metropolitan France. Multiple countries must be passed as multiple country:XX filters, with the pipe character (`|`) as a separator. For example: `components=country:gb|country:fr|country:be|country:sp|country:it` would restrict your results to city names or postal codes within the United Kingdom, France, Belgium, Spain and Italy.

        
    """
    url = f"https://woosmap1.p.rapidapi.com/localities/autocomplete"
    querystring = {'input': input, }
    if language:
        querystring['language'] = language
    if extended:
        querystring['extended'] = extended
    if data:
        querystring['data'] = data
    if types:
        querystring['types'] = types
    if components:
        querystring['components'] = components
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "woosmap1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def localitiesdetails(public_id: str, fields: str='geometry', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides details of an autocomplete suggestion (using the suggestion’s `public_id`).
		"
    public_id: A textual identifier that uniquely identifies a locality, returned from a [Localities Autocomplete](https://developers.woosmap.com/products/localities/autocomplete/).

        fields: Used to limit the returning fields (by default, all fields are return). Available fields are (geometry) (fields should be separated by a ,). By using this parameter you will limit content of responses to the geometry part. No address component will be returned.

        
    """
    url = f"https://woosmap1.p.rapidapi.com/localities/details"
    querystring = {'public_id': public_id, }
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "woosmap1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def addressautocomplete(input: str, language: str='en', location: str='5.2,-2.3', components: str='country:CAN|country:BEL', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Autocomplete on worldwide suggestions for a for text-based geographic searches. It can match on full words as well as substrings.
		"
    input: The text string on which to search, for example: "london" or "123 Cross Road". The Woosmap Address API will return predictions matches based on this string and order the results based on their perceived relevance.

        language: The language code, using ISO 639-1 country codes, indicating in which language the results should be returned, if possible. If language is not supplied, the Localities service will use the default language of each country.

        location: Bias for the results. Should be pass in `lat`,`lng` format.

        components: To restrict your results to specific areas. Currently, you can use components to filter over countries. Countries must be passed as three characters string (ISO 3166-1 Alpha 3). Components should be passed as an array of different options which are separated by a `|`.

        
    """
    url = f"https://woosmap1.p.rapidapi.com/address/autocomplete/json"
    querystring = {'input': input, }
    if language:
        querystring['language'] = language
    if location:
        querystring['location'] = location
    if components:
        querystring['components'] = components
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "woosmap1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def addressdetails(public_id: str, fields: str='geometry', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides details of an address autocomplete suggestion (using the suggestion’s `public_id`).
		"
    public_id: A textual identifier that uniquely identifies a locality, returned from a [Localities Autocomplete](https://developers.woosmap.com/products/localities/autocomplete/).

        fields: Used to limit the returning fields (by default, all fields are return). Available fields are (geometry) (fields should be separated by a `,`). By using this parameter you will limit content of responses to the geometry part. No address component will be returned.

        
    """
    url = f"https://woosmap1.p.rapidapi.com/address/details/json"
    querystring = {'public_id': public_id, }
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "woosmap1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdistancematrix(destinations: str, origins: str, language: str='en', method: str='distance', mode: str='driving', elements: str='duration_distance', units: str='metric', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get distances and durations for a matrix of origins and destinations, based on the recommended route between start and end points for a specified travel mode.
		"
    destinations: One or more locations to use as the finishing point for calculating travel distance. The options for the destinations parameter are the same as for the origins parameter, described above. In order to reduce URL size, [encoded polylines](https://developers.google.com/maps/documentation/utilities/polylinealgorithm) are also supported using `enc:encoded-polyline:`

        origins: The starting point for calculating travel distance. You can supply one or more locations separated by the pipe character (|), in the form of latitude/longitude coordinates. They are used unchanged to calculate distance. Ensure that no space exists between the latitude and longitude values. In order to reduce URL size, [encoded polylines](https://developers.google.com/maps/documentation/utilities/polylinealgorithm) are also supported using `enc:encoded-polyline:`

        language: The language code, indicating in which language the results should be returned, if possible. If language is not supplied, the Distance API service will use the navigator language or “en”.

        method: Specifies the method to compute the route between the start point and the end point:
- `time`: fastest route (default) - `distance`: shortest route

        mode: Specifies the mode of transport to use when calculating distance

        elements: Specifies element values that will be part of the API response (distance and/or duration). if not specified default is `distance`

        units: Specifies the unit system to use when expressing distance as text. Two different units supported:
  * `metric` (default) returns distances in kilometers and meters
  * `imperial` returns distances in miles and feet

        
    """
    url = f"https://woosmap1.p.rapidapi.com/distance/distancematrix/json"
    querystring = {'destinations': destinations, 'origins': origins, }
    if language:
        querystring['language'] = language
    if method:
        querystring['method'] = method
    if mode:
        querystring['mode'] = mode
    if elements:
        querystring['elements'] = elements
    if units:
        querystring['units'] = units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "woosmap1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getroute(origin: str, destination: str, alternatives: bool=None, waypoints: str='48.850077,3.311124|48.850077,3.411124', language: str='en', mode: str='driving', units: str='metric', method: str='distance', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get distance, duration and path (as a polyline) for a pair of origin and destination, based on the recommended route between those two points for a specified travel mode.
		"
    origin: The starting point for the route. It should be supplied in the form of latitude/longitude coordinates. Ensure that no space exists between the latitude and longitude values.

        destination: The ending point for the route. It should be supplied in the form of latitude/longitude coordinates. Ensure that no space exists between the latitude and longitude values.

        alternatives: Specifies if alternative routes should be returned. default is `false`. Depending on the calculated route, alternatives may not be provided.
`alternatives` and `waypoints` can not be used at the same time.

        waypoints: A list of points by which the route should pass. route response is divided into legs, one leg corresponding to a route between two waypoints. Waypoints should be separated by `|` character.
`alternatives` and `waypoints` can not be used at the same time.

        language: The language code, indicating in which language the results should be returned, if possible. If language is not supplied, the Distance API service will use the navigator language or “en”.

        mode: Specifies the mode of transport to use when calculating distance

        units: Specifies the unit system to use when expressing distance as text. Two different units supported:
  * `metric` (default) returns distances in kilometers and meters
  * `imperial` returns distances in miles and feet

        method: Specifies the method to compute the route between the start point and the end point:
- `time`: fastest route (default) - `distance`: shortest route

        
    """
    url = f"https://woosmap1.p.rapidapi.com/distance/route/json"
    querystring = {'origin': origin, 'destination': destination, }
    if alternatives:
        querystring['alternatives'] = alternatives
    if waypoints:
        querystring['waypoints'] = waypoints
    if language:
        querystring['language'] = language
    if mode:
        querystring['mode'] = mode
    if units:
        querystring['units'] = units
    if method:
        querystring['method'] = method
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "woosmap1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstoresfromgeolocationposition(ip_address: str='75.134.29.90', query: str="name:'My cool store'|type:'click_and_collect'", limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the stores nearby an ip location. Stores are returned only if a relevant ip location is found - for an accuracy of 20km or less.
		"
    ip_address: The ip_address you want to geolocate. For **server call with private_key** only. Without this parameter, the API will geolocate the IP Address attached to the raw TCP request.

        query: Search query combining one or more search clauses. Each search clause is made up of three parts structured as `field` `:` `operator` `value`. , e.g. `name:="My cool store"`
### Vocabulary

  - **Field**: attribute of the Store that is searched, e.g. the attribute `name` of the store.

  - **Operator**: test that is performed on the data to provide a match, e.g. `=`.
  Each field has a default operator. If none operator follow the `:`, the default one is used.

  - **Value**: the content of the attribute that is tested, e.g. the name of the store `"My cool store"`.

Combine clauses with the conjunctions `AND` or `OR`, and negate the query with `NOT`.
### Fields

  - `type`: An element is contained within `type` collection. e.g. `type:"myType"`

  - `tag`: An element is contained within `tag` collection. e.g. `tag:"myTag"`

  - `city`: text matching: the value match the `city` field. e.g. `city:="Paris"`

  - `country`: text matching: the value match the `countryCode` field. e.g. `country:="FR"`

  - `name`: text matching: the value match the `name` field. e.g. `name:="myName"`

  - `idstore`: text matching: the value match the `idstore` field.  e.g. `idstore:="myIdStore"`

  - `user`: concerns all fields inside `user_properties`. text matching or numerical comparison.  e.g. `user.myAttribute:="myValue"`

  - `localized`: used for [localizedNames](https://developers.woosmap.com/products/data-api/data-structure/#localizednames) to search in native language. text matching in collection: the value match one of the the `localizedNames`. e.g. `localized:="centro"`

> **userProperties** field has no restriction regarding the data you can put in it (Arrays, Object, Boolean, String, Numeric...) but you can only query for **text matching or numerical comparison**.
### Operators

  - `:` : Default and mandatory operator. For `type` and `tag` fields, define that an element is contained within a collection.

  - `=` : The content of a string or a number is equal to the other.

  - `>` : A number is greater than another.

  - `<` : A number is smaller than another.

  - `>=` : A number is greater than or equal to another.

  - `<=` : A number is smaller than or equal to another.

  - `AND` : Return assets that match both clauses.

  - `OR` : Return assets that match either clauses.

  - `NOT` : Negates a search clause.

For compound clauses, you can use parentheses to group clauses together. For example: ```(type:"type1" OR type:"type2") AND tag:"hockey"```
You can use `NOT` operator to negates a search clause. For example: ```not type:"type1"```

        limit: To limit number of assets retrieved from an IP location.

        
    """
    url = f"https://woosmap1.p.rapidapi.com/geolocation/stores"
    querystring = {}
    if ip_address:
        querystring['ip_address'] = ip_address
    if query:
        querystring['query'] = query
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "woosmap1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getgeolocationposition(ip_address: str='75.134.29.90', query: str="name:'My cool store'|type:'click_and_collect'", toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The `/position` returns JSON location of your users thanks to IP address of their devices.
		"
    ip_address: The ip_address you want to geolocate. For **server call with private_key** only. Without this parameter, the API will geolocate the IP Address attached to the raw TCP request.

        query: Search query combining one or more search clauses. Each search clause is made up of three parts structured as `field` `:` `operator` `value`. , e.g. `name:="My cool store"`
### Vocabulary

  - **Field**: attribute of the Store that is searched, e.g. the attribute `name` of the store.

  - **Operator**: test that is performed on the data to provide a match, e.g. `=`.
  Each field has a default operator. If none operator follow the `:`, the default one is used.

  - **Value**: the content of the attribute that is tested, e.g. the name of the store `"My cool store"`.

Combine clauses with the conjunctions `AND` or `OR`, and negate the query with `NOT`.
### Fields

  - `type`: An element is contained within `type` collection. e.g. `type:"myType"`

  - `tag`: An element is contained within `tag` collection. e.g. `tag:"myTag"`

  - `city`: text matching: the value match the `city` field. e.g. `city:="Paris"`

  - `country`: text matching: the value match the `countryCode` field. e.g. `country:="FR"`

  - `name`: text matching: the value match the `name` field. e.g. `name:="myName"`

  - `idstore`: text matching: the value match the `idstore` field.  e.g. `idstore:="myIdStore"`

  - `user`: concerns all fields inside `user_properties`. text matching or numerical comparison.  e.g. `user.myAttribute:="myValue"`

  - `localized`: used for [localizedNames](https://developers.woosmap.com/products/data-api/data-structure/#localizednames) to search in native language. text matching in collection: the value match one of the the `localizedNames`. e.g. `localized:="centro"`

> **userProperties** field has no restriction regarding the data you can put in it (Arrays, Object, Boolean, String, Numeric...) but you can only query for **text matching or numerical comparison**.
### Operators

  - `:` : Default and mandatory operator. For `type` and `tag` fields, define that an element is contained within a collection.

  - `=` : The content of a string or a number is equal to the other.

  - `>` : A number is greater than another.

  - `<` : A number is smaller than another.

  - `>=` : A number is greater than or equal to another.

  - `<=` : A number is smaller than or equal to another.

  - `AND` : Return assets that match both clauses.

  - `OR` : Return assets that match either clauses.

  - `NOT` : Negates a search clause.

For compound clauses, you can use parentheses to group clauses together. For example: ```(type:"type1" OR type:"type2") AND tag:"hockey"```
You can use `NOT` operator to negates a search clause. For example: ```not type:"type1"```

        
    """
    url = f"https://woosmap1.p.rapidapi.com/geolocation/position"
    querystring = {}
    if ip_address:
        querystring['ip_address'] = ip_address
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "woosmap1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def storeautocomplete(query: str="name:'My cool store'|type:'click_and_collect'", lng: int=None, lat: int=None, encoded_polyline: str='_p~iF~ps|U_ulLnnqC_mqNvxq`@', page: int=2, stores_by_page: int=150, radius: int=3000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Autocomplete on `localizedNames` with highlighted results on asset name. Use the field `localized` in your query parameter to search for localized names.
		"
    query: Search query combining one or more search clauses. Each search clause is made up of three parts structured as `field` `:` `operator` `value`. , e.g. `name:="My cool store"`
### Vocabulary

  - **Field**: attribute of the Store that is searched, e.g. the attribute `name` of the store.

  - **Operator**: test that is performed on the data to provide a match, e.g. `=`.
  Each field has a default operator. If none operator follow the `:`, the default one is used.

  - **Value**: the content of the attribute that is tested, e.g. the name of the store `"My cool store"`.

Combine clauses with the conjunctions `AND` or `OR`, and negate the query with `NOT`.
### Fields

  - `type`: An element is contained within `type` collection. e.g. `type:"myType"`

  - `tag`: An element is contained within `tag` collection. e.g. `tag:"myTag"`

  - `city`: text matching: the value match the `city` field. e.g. `city:="Paris"`

  - `country`: text matching: the value match the `countryCode` field. e.g. `country:="FR"`

  - `name`: text matching: the value match the `name` field. e.g. `name:="myName"`

  - `idstore`: text matching: the value match the `idstore` field.  e.g. `idstore:="myIdStore"`

  - `user`: concerns all fields inside `user_properties`. text matching or numerical comparison.  e.g. `user.myAttribute:="myValue"`

  - `localized`: used for [localizedNames](https://developers.woosmap.com/products/data-api/data-structure/#localizednames) to search in native language. text matching in collection: the value match one of the the `localizedNames`. e.g. `localized:="centro"`

> **userProperties** field has no restriction regarding the data you can put in it (Arrays, Object, Boolean, String, Numeric...) but you can only query for **text matching or numerical comparison**.
### Operators

  - `:` : Default and mandatory operator. For `type` and `tag` fields, define that an element is contained within a collection.

  - `=` : The content of a string or a number is equal to the other.

  - `>` : A number is greater than another.

  - `<` : A number is smaller than another.

  - `>=` : A number is greater than or equal to another.

  - `<=` : A number is smaller than or equal to another.

  - `AND` : Return assets that match both clauses.

  - `OR` : Return assets that match either clauses.

  - `NOT` : Negates a search clause.

For compound clauses, you can use parentheses to group clauses together. For example: ```(type:"type1" OR type:"type2") AND tag:"hockey"```
You can use `NOT` operator to negates a search clause. For example: ```not type:"type1"```

        lng: Longitude bias for the results. Should be pass with `lat`.

        lat: Latitude bias for the results. Should be pass with `lng`.

        encoded_polyline: Find assets nearby an [encoded polyline](https://developers.google.com/maps/documentation/utilities/polylinealgorithm) and inside a defined radius.

        page: Page number when accessing paginated assets feature collection

        stores_by_page: If your request returns a high number of assets, the result will be paginated. If so, you can request assets by page using `page` and `stores_by_page` parameters (Default is 100, max is 300).

        radius: Unit in meters. Used to combine with lat/lng or encoded polyline. To bias the results within a given circular area. 3000 means to search for Assets that are at the most far from 3kms to search area (latlng or polyline).

        
    """
    url = f"https://woosmap1.p.rapidapi.com/stores/autocomplete"
    querystring = {}
    if query:
        querystring['query'] = query
    if lng:
        querystring['lng'] = lng
    if lat:
        querystring['lat'] = lat
    if encoded_polyline:
        querystring['encoded_polyline'] = encoded_polyline
    if page:
        querystring['page'] = page
    if stores_by_page:
        querystring['stores_by_page'] = stores_by_page
    if radius:
        querystring['radius'] = radius
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "woosmap1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def storesearch(lng: int=None, radius: int=3000, encoded_polyline: str='_p~iF~ps|U_ulLnnqC_mqNvxq`@', page: int=2, stores_by_page: int=150, lat: int=None, query: str="name:'My cool store'|type:'click_and_collect'", toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to retrieve assets from query
		"
    lng: Longitude bias for the results. Should be pass with `lat`.

        radius: Unit in meters. Used to combine with lat/lng or encoded polyline. To bias the results within a given circular area. 3000 means to search for Assets that are at the most far from 3kms to search area (latlng or polyline).

        encoded_polyline: Find assets nearby an [encoded polyline](https://developers.google.com/maps/documentation/utilities/polylinealgorithm) and inside a defined radius.

        page: Page number when accessing paginated assets feature collection

        stores_by_page: If your request returns a high number of assets, the result will be paginated. If so, you can request assets by page using `page` and `stores_by_page` parameters (Default is 100, max is 300).

        lat: Latitude bias for the results. Should be pass with `lng`.

        query: Search query combining one or more search clauses. Each search clause is made up of three parts structured as `field` `:` `operator` `value`. , e.g. `name:="My cool store"`
### Vocabulary

  - **Field**: attribute of the Store that is searched, e.g. the attribute `name` of the store.

  - **Operator**: test that is performed on the data to provide a match, e.g. `=`.
  Each field has a default operator. If none operator follow the `:`, the default one is used.

  - **Value**: the content of the attribute that is tested, e.g. the name of the store `"My cool store"`.

Combine clauses with the conjunctions `AND` or `OR`, and negate the query with `NOT`.
### Fields

  - `type`: An element is contained within `type` collection. e.g. `type:"myType"`

  - `tag`: An element is contained within `tag` collection. e.g. `tag:"myTag"`

  - `city`: text matching: the value match the `city` field. e.g. `city:="Paris"`

  - `country`: text matching: the value match the `countryCode` field. e.g. `country:="FR"`

  - `name`: text matching: the value match the `name` field. e.g. `name:="myName"`

  - `idstore`: text matching: the value match the `idstore` field.  e.g. `idstore:="myIdStore"`

  - `user`: concerns all fields inside `user_properties`. text matching or numerical comparison.  e.g. `user.myAttribute:="myValue"`

  - `localized`: used for [localizedNames](https://developers.woosmap.com/products/data-api/data-structure/#localizednames) to search in native language. text matching in collection: the value match one of the the `localizedNames`. e.g. `localized:="centro"`

> **userProperties** field has no restriction regarding the data you can put in it (Arrays, Object, Boolean, String, Numeric...) but you can only query for **text matching or numerical comparison**.
### Operators

  - `:` : Default and mandatory operator. For `type` and `tag` fields, define that an element is contained within a collection.

  - `=` : The content of a string or a number is equal to the other.

  - `>` : A number is greater than another.

  - `<` : A number is smaller than another.

  - `>=` : A number is greater than or equal to another.

  - `<=` : A number is smaller than or equal to another.

  - `AND` : Return assets that match both clauses.

  - `OR` : Return assets that match either clauses.

  - `NOT` : Negates a search clause.

For compound clauses, you can use parentheses to group clauses together. For example: ```(type:"type1" OR type:"type2") AND tag:"hockey"```
You can use `NOT` operator to negates a search clause. For example: ```not type:"type1"```

        
    """
    url = f"https://woosmap1.p.rapidapi.com/stores/search"
    querystring = {}
    if lng:
        querystring['lng'] = lng
    if radius:
        querystring['radius'] = radius
    if encoded_polyline:
        querystring['encoded_polyline'] = encoded_polyline
    if page:
        querystring['page'] = page
    if stores_by_page:
        querystring['stores_by_page'] = stores_by_page
    if lat:
        querystring['lat'] = lat
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "woosmap1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

