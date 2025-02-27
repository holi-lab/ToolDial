import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def correlations(sort: str='correlationCoefficient', cause: str=None, limit: int=None, offset: int=None, effect: str='Overall Mood', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get correlations"
    sort: Sort by given field. If the field is prefixed with `-, it will sort in descending order.
        cause: ORIGINAL variable name of the cause variable for which the user desires correlations
        limit: The LIMIT is used to limit the number of results returned. So if you have 1000 results, but only want to the first 10, you would set this to 10 and offset to 0.
        offset: Now suppose you wanted to show results 11-20. You'd set the offset to 10 and the limit to 10.
        effect: ORIGINAL variable name of the effect variable for which the user desires correlations
        
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/correlations"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if cause:
        querystring['cause'] = cause
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if effect:
        querystring['effect'] = effect
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_embeddable_connect_javascript(t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get embeddable connect javascript"
    t: User token
        
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/api/v1/connect.js"
    querystring = {'t': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def disconnect(connector: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The disconnect method deletes any stored tokens or connection information from the connectors database."
    connector: Lowercase system name of the source application or device. Get a list of available connectors from the /connectors/list endpoint.
        
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/connectors/{connector}/disconnect"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def measurement_time_range(sources: str=None, user: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Unix time-stamp (epoch time) of the user's first and last measurements taken."
    sources: Enter source name to limit to specific source (varchar)
        user: If not specified, uses currently logged in user (bigint)
        
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/measurementsRange"
    querystring = {}
    if sources:
        querystring['sources'] = sources
    if user:
        querystring['user'] = user
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def connect(connector: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Attempt to obtain a token from the data provider, store it in the database. With this, the connector to continue to obtain new user data until the token is revoked."
    connector: Lowercase system name of the source application or device. Get a list of available connectors from the /connectors/list endpoint.
        
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/connectors/{connector}/connect"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def measurement_sources(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all the apps from which measurement data is obtained."
    
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/measurementSources"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_info_about_a_variable(variablename: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all of the settings and information about a variable by its name.  If the logged in user has modified the settings for the variable, these will be provided instead of the default settings for that variable."
    variablename: Variable name
        
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/variables/{variablename}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_variables(search: str, source: str=None, categoryname: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get variables containing the search characters for which the currently logged in user has measurements. Used to provide auto-complete function in variable search boxes."
    search: Search query which may be an entire variable name or a fragment of one.
        source: Specify a data source name to only return variables from a specific data source.
        categoryname: Filter variables by category name. The variable categories include Activity, Causes of Illness, Cognitive Performance, Conditions, Environment, Foods, Location, Miscellaneous, Mood, Nutrition, Physical Activity, Physique, Sleep, Social Interactions, Symptoms, Treatments, Vital Signs, and Work.
        
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/variables/search/{search}"
    querystring = {}
    if source:
        querystring['source'] = source
    if categoryname:
        querystring['categoryname'] = categoryname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def variables(category: str='Treatments', sort: str='-id', limit: int=10, offset: int=10, userid: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all variables for which the user has measurements by the category name"
    category: Filter data by category
        sort: Sort by given field. If the field is prefixed with `-, it will sort in descending order.
        limit: The LIMIT is used to limit the number of results returned. So if you have 1000 results, but only want to the first 10, you would set this to 10 and offset to 0.
        offset: Now suppose you wanted to show results 11-20. You'd set the offset to 10 and the limit to 10.
        userid: User Id
        
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/variables"
    querystring = {}
    if category:
        querystring['category'] = category
    if sort:
        querystring['sort'] = sort
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if userid:
        querystring['userId'] = userid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_correlations(search: str, effectorcause: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the average correlations from all users for all public variables that contain the characters in the search query."
    search: Name of the variable that you want to know the causes or effects of.
        effectorcause: Specifies whehter to return the effects or causes of the searched variable.
        
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/public/correlations/search/{search}"
    querystring = {'effectorcause': effectorcause, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def connection_instructions(url: str, parameters: str, connector: str, usepopup: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns instructions that describe what parameters and endpoint to use to connect to the given data provider."
    url: URL which should be used to enable the connector.
        parameters: JSON Array of Parameters for the request to enable connector.
        connector: Lowercase system name of the source application or device. Get a list of available connectors from the /connectors/list endpoint.
        usepopup: Should use popup when enabling connector
        
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/connectors/{connector}/connectInstructions"
    querystring = {'url': url, 'parameters': parameters, }
    if usepopup:
        querystring['usePopup'] = usepopup
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mobile_connect_page(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mobile connect page"
    
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/api/v1/connect/mobile"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def update(connector: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The update method tells the QM Connector Framework to check with the data provider (such as Fitbit or MyFitnessPal) and retrieve any new measurements available."
    connector: Lowercase system name of the source application or device
        
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/connectors/{connector}/update"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_authorization_token(response_type: str, redirect_uri: str, client_id: str, client_secret: str, scope: str=None, realm: str=None, state: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ask the user if they want to allow a client applications to submit or obtain data from their QM account."
    response_type: If the value is code, launches a Basic flow, requiring a POST to the token endpoint to obtain the tokens. If the value is token id_token or id_token token, launches an Implicit flow, requiring the use of Javascript at the redirect URI to retrieve tokens from the URI #fragment.
        redirect_uri: The redirect URI is the URL within your client application that will receive the OAuth2 credentials.
        client_id: This is the unique ID that QuantiModo uses to identify your application. Obtain a client id by emailing info@quantimo.do.
        client_secret: This is the secret for your obtained clietn_id. QuantiModo uses this to validate that only your application uses the client_id.
        scope: Scopes include basic, readmeasurements, and writemeasurements.  The "basic" scope allows you to read user info (displayname, email, etc). The "readmeasurements" scope allows one to read a user's data. The "writemeasurements" scope allows you to write user data. Separate multiple scopes by a space.
        realm: Name of the realm representing the users of your distributed applications and services.  A "realm" attribute MAY be included to indicate the scope of protection.
        state: An opaque string that is round-tripped in the protocol; that is to say, it is returned as a URI parameter in the Basic flow, and in the URI #fragment in the Implicit flow.  The state can be useful for correlating requests and responses. Because your redirect_uri can be guessed, using a state value can increase your assurance that an incoming connection is the result of an authentication request. If you generate a random string or encode the hash of some client state (e.g., a cookie) in this state variable, you can validate the response to additionally ensure that the request and response originated in the same browser. This provides protection against attacks such as cross-site request forgery.
        
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/oauth2/authorize"
    querystring = {'response_type': response_type, 'redirect_uri': redirect_uri, 'client_id': client_id, 'client_secret': client_secret, }
    if scope:
        querystring['scope'] = scope
    if realm:
        querystring['realm'] = realm
    if state:
        querystring['state'] = state
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_user_correlations_for_a_given_effect(variablename: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns average of all correlations and votes for all user cause variables for a given effect."
    
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/v1/variables/{variablename}/causes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_user_correlations_for_a_given_cause(variablename: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns average of all correlations and votes for all user effect variables for a given cause."
    
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/v1/variables/{variablename}/effects"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_public_correlations_for_a_given_cause(variablename: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns average of all correlations and votes for all public cause variables for a given cause."
    
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/v1/variables/{variablename}/public/effects"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def me(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns user info for the currently authenticated user."
    
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/user/me"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def public_variables(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves an array of all public variables. Public variables are things like foods, medications, symptoms, conditions, and anything not unique to a particular user. For instance, a telephone number or name would not be a public variable."
    
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/public/variables"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def unit_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of the categories of measurement units such as 'Distance', 'Duration', 'Energy', 'Frequency', 'Miscellany', 'Pressure', 'Proportion', 'Rating', 'Temperature', 'Volume', and 'Weight'."
    
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/unitCategories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_tokens_for_existing_users_create_new_users(body: str, organizationid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user tokens for existing users or Create new users"
    body: Provides organization token and user ID
        
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/v1/organizations/{organizationid}/users"
    querystring = {'body': body, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_public_correlations_for_a_given_effect(variablename: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns average of all correlations and votes for all public cause variable for a given effect."
    variablename: Effect variable name
        
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/v1/variables/{variablename}/public/causes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def measurements(variablename: str, unit: str=None, starttime: str=None, endtime: str=None, groupingwidth: int=None, groupingtimezone: str=None, limit: int=None, offset: int=None, sort: str='startTime', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get measurements for this user. Measurements are any value that can be recorded like daily steps, a mood rating, or apples eaten."
    variablename: Name of the variable you want measurements for
        unit: The unit your want the measurements in
        starttime: The lower limit of measurements returned (Epoch)
        endtime: The upper limit of measurements returned (Epoch)
        groupingwidth: The time (in seconds) over which measurements are grouped together
        groupingtimezone: The time (in seconds) over which measurements are grouped together
        limit: The LIMIT is used to limit the number of results returned. So if you have 1000 results, but only want to the first 10, you would set this to 10 and offset to 0.
        offset: Now suppose you wanted to show results 11-20. You'd set the offset to 10 and the limit to 10.
        sort: Sort by given field. If the field is prefixed with `-, it will sort in descending order.
        
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/api/measurements"
    querystring = {'variablename': variablename, }
    if unit:
        querystring['unit'] = unit
    if starttime:
        querystring['startTime'] = starttime
    if endtime:
        querystring['endTime'] = endtime
    if groupingwidth:
        querystring['groupingWidth'] = groupingwidth
    if groupingtimezone:
        querystring['groupingTimezone'] = groupingtimezone
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_available_units(unitname: str=None, abbreviatedunitname: str=None, categoryname: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available units"
    unitname: Unit Name
        abbreviatedunitname: Restric the results to a specific unit by providing the unit abbreviation
        categoryname: Restrict the results to a specific unit category by  providing the unit category name.
        
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/units"
    querystring = {}
    if unitname:
        querystring['unitName'] = unitname
    if abbreviatedunitname:
        querystring['abbreviatedUnitName'] = abbreviatedunitname
    if categoryname:
        querystring['categoryName'] = categoryname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_connectors(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A connector pulls data from other data providers using their API or a screenscraper. Returns a list of all available connectors and information about them such as their id, name, whether the user has provided access, logo url, connection instructions, and the update history."
    
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/connectors/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def connector_information(connector: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns information about the connector such as the connector id, whether or not is connected for this user (i.e. we have a token or credentials), and its update history for the user."
    connector: Lowercase system name of the source application or device. Get a list of available connectors from the /connectors/list endpoint.
        
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/connectors/{connector}/info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pairs(cause: str, effect: str, duration: str=None, delay: str=None, starttime: str=None, endtime: str=None, causesource: str=None, effectsource: str=None, causeunit: str=None, effectunit: str=None, limit: int=None, offset: int=None, sort: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pairs cause measurements with effect measurements grouped over the duration of action after the onset delay."
    cause: Original variable name for the explanatory or independent variable
        effect: Original variable name for the outcome or dependent variable
        duration: Duration of action (in seconds) from the cause variable settings.
        delay: Delay before onset of action (in seconds) from the cause variable settings.
        starttime: The earliest date (in epoch time) for which we should return measurements
        endtime: The most recent date (in epoch time) for which we should return measurements
        causesource: Name of data source that the cause measurements should come from
        effectsource: Name of data source that the effectmeasurements should come from
        causeunit: Abbreviated name for the unit cause measurements to be returned in
        effectunit: Abbreviated name for the unit effect measurements to be returned in
        limit: The LIMIT is used to limit the number of results returned. So if you have 1000 results, but only want to the first 10, you would set this to 10 and offset to 0.
        offset: Now suppose you wanted to show results 11-20. You'd set the offset to 10 and the limit to 10.
        sort: Sort by given field. If the field is prefixed with `-, it will sort in descending order.
        
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/pairs"
    querystring = {'cause': cause, 'effect': effect, }
    if duration:
        querystring['duration'] = duration
    if delay:
        querystring['delay'] = delay
    if starttime:
        querystring['startTime'] = starttime
    if endtime:
        querystring['endTime'] = endtime
    if causesource:
        querystring['causeSource'] = causesource
    if effectsource:
        querystring['effectSource'] = effectsource
    if causeunit:
        querystring['causeUnit'] = causeunit
    if effectunit:
        querystring['effectUnit'] = effectunit
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def variable_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The variable categories include Activity, Causes of Illness, Cognitive Performance, Conditions, Environment, Foods, Location, Miscellaneous, Mood, Nutrition, Physical Activity, Physique, Sleep, Social Interactions, Symptoms, Treatments, Vital Signs, and Work."
    
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/variableCategories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def units_for_variable(variable: str=None, abbreviatedunitname: str=None, categoryname: str=None, unitname: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of all possible units to use for a given variable"
    variable: Name of the variable you want units for
        abbreviatedunitname: Abbreviated Unit Name of the unit you want
        categoryname: Name of the category you want units for
        unitname: Name of Unit you want to retrieve
        
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/unitsVariable"
    querystring = {}
    if variable:
        querystring['variable'] = variable
    if abbreviatedunitname:
        querystring['abbreviatedunitname'] = abbreviatedunitname
    if categoryname:
        querystring['categoryname'] = categoryname
    if unitname:
        querystring['unitname'] = unitname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_user_correlations_for_a_given_cause_filtered_by_organization(organizationid: str, userid: str, variablename: str, organization_token: str, include_public: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns average of all correlations and votes for all user cause variables for a given effect. If parameter "include_public" is used, it also returns public correlations. User correlation overwrites or supersedes public correlation."
    organization_token: Organization access token
        include_public: Include bublic correlations, Can be "1" or empty.
        
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/v1/organizations/{organizationid}/users/{userid}/variables/{variablename}/effects"
    querystring = {'organization_token': organization_token, }
    if include_public:
        querystring['include_public'] = include_public
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_user_correlation_for_a_given_effect_filtering_by_organization(organizationid: str, userid: str, variablename: str, organization_token: str, include_public: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns average of all correlations and votes for all user cause variables for a given effect. If parameter "include_public" is used, it also returns public correlations. User correlation overwrites or supersedes public correlation."
    organization_token: Organization access token
        include_public: Include bublic correlations, Can be "1" or empty.
        
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/v1/organizations/{organizationid}/users/{userid}/variables/{variablename}/causes"
    querystring = {'organization_token': organization_token, }
    if include_public:
        querystring['include_public'] = include_public
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_public_variables(search: str, limit: int=None, offset: int=None, sort: str=None, effectorcause: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top 5 PUBLIC variables with the most correlations containing the entered search characters. For example, search for 'mood' as an effect. Since 'Overall Mood' has a lot of correlations with other variables, it should be in the autocomplete list."
    search: Search query can be some fraction of a variable name.
        limit: The LIMIT is used to limit the number of results returned. So if you have 1000 results, but only want to the first 10, you would set this to 10 and offset to 0.
        offset: Now suppose you wanted to show results 11-20. You'd set the offset to 10 and the limit to 10.
        sort: Sort by given field. If the field is prefixed with `-, it will sort in descending order.
        effectorcause: Allows us to specify which column in the correlations table will be searched. Choices are effect or cause.
        
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/public/variables/search/{search}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if sort:
        querystring['sort'] = sort
    if effectorcause:
        querystring['effectorcause'] = effectorcause
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_access_token(client_id: str, grant_type: str, client_secret: str, response_type: str=None, scope: str=None, redirect_uri: str=None, state: str=None, realm: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Client provides authorization token obtained from /api/oauth2/authorize to this endpoint and receives an access token.  Access token can then"
    client_id: This is the unique ID that QuantiModo uses to identify your application. Obtain a client id by emailing info@quantimo.do.
        grant_type: The grant type can be `authorization_code ` or `refresh_token`
        client_secret: This is the secret for your obtained client_id. QuantiModo uses this to validate that only your application uses the client_id.
        response_type: If the value is `code`, launches a Basic flow, requiring a POST to the token endpoint to obtain the tokens. If the value is `token id_token` or `id_token token`, launches an Implicit flow, requiring the use of Javascript at the redirect URI to retrieve tokens from the URI fragment.
        scope: Space-delimited set of permissions that the application requests. Scopes include "basic", "readmeasurements", and "writemeasurements". The "basic" scope allows you to read user info (displayname, email, etc). The "readmeasurements" scope allows one to read a user's data. The "writemeasurements" scope allows you to write user data. Separate multiple scopes by a space.
        redirect_uri: The redirect URI is the URL within your client application that will receive the OAuth2 credentials.
        state: An opaque string that is round-tripped in the protocol; that is to say, it is returned as a URI parameter in the Basic flow, and in the URI
        realm: Name of the realm representing the users of your distributed applications and services. A "realm" attribute MAY be included to indicate the scope of protection.
        
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/api/oauth2/token"
    querystring = {'client_id': client_id, 'grant_type': grant_type, 'client_secret': client_secret, }
    if response_type:
        querystring['response_type'] = response_type
    if scope:
        querystring['scope'] = scope
    if redirect_uri:
        querystring['redirect_uri'] = redirect_uri
    if state:
        querystring['state'] = state
    if realm:
        querystring['realm'] = realm
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def connect_parameter(connector: str, displayname: str, key: str, type: str, placeholder: str, defaultvalue: str=None, usepopup: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns instructions that describe what parameters and endpoint to use to connect to the given data provider."
    connector: Lowercase system name of the source application or device. Get a list of available connectors from the /connectors/list endpoint.
        displayname: Name of the parameter that is user visible in the form
        key: Name of the property that the user has to enter such as username or password Connector (used in HTTP request)
        type: Type of input field such as those found here http://www.w3schools.com/tags/tag_input.asp
        placeholder: Placeholder hint value for the parameter input tag.
        defaultvalue: Default parameter value
        usepopup: Should use popup when enabling connector
        
    """
    url = f"https://quantimodo-quantimodo-v1.p.rapidapi.com/connectors/{connector}/connectParameter"
    querystring = {'displayName': displayname, 'key': key, 'type': type, 'placeholder': placeholder, }
    if defaultvalue:
        querystring['defaultValue'] = defaultvalue
    if usepopup:
        querystring['usePopup'] = usepopup
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantimodo-quantimodo-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

