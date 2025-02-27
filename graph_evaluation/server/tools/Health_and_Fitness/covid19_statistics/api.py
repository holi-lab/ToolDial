import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def allghicsse(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://covid19-statistics.p.rapidapi.com/v2/jhucsse"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid19-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def allcontinents(sort: str, yesterday: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a JSON array with an element for each continent that has stats available."
    sort: For this parameter you can provide a key from the country model (e.g. cases, todayCases, deaths, recovered, active, etc) to sort the countries from greatest to least, depending on the key

Available values : cases, todayCases, deaths, todayDeaths, recovered, active, critical, casesPerOneMillion, deathsPerOneMillion.
        yesterday: For this parameter you can show yesterday data

Available values : true, false, 1, 0.
        
    """
    url = f"https://covid19-statistics.p.rapidapi.com/v2/continents"
    querystring = {'sort': sort, 'yesterday': yesterday, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid19-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historicaldata(lastdays: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get time series info from the JHU CSSE Data Repository. Every date since 1/22/20 has an entry tracking deaths, cases, and recoveries for each country. Updated each day at 23:59 UTC. Data is updated every 10 minutes."
    lastdays: number of days you want the data to go back to. Default is 30. Use all for full data set. Ex: 15, all, 24
        
    """
    url = f"https://covid19-statistics.p.rapidapi.com/v2/historical"
    querystring = {'lastdays': lastdays, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid19-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def statedata(state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all NYT state data or individual state data if specified. Each entry returned represents data for a given day."
    state: The state that you'd like to search for, separated by comma if you want to search for multiple (i.e. 'California, Washington'. Default is full data set.
        
    """
    url = f"https://covid19-statistics.p.rapidapi.com/v2/nyt/states"
    querystring = {'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid19-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def specificcontry(strict: str, yesterday: str, query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the same data from the /countries endpoint, but filter down to a specific country."
    strict: Defaults to true. Setting to false gives you the ability to fuzzy search countries. Example Oman vs. rOMANia
        yesterday: For this parameter you can show yesterday data

Available values : true, false, 1, 0.
        query: The continent name.
        query: For this parameter you can use Country Name && Country Id && ISOs (ISO 2 | ISO 3) 3166 Country Standards
        
    """
    url = f"https://covid19-statistics.p.rapidapi.com/v2/countries/{query}"
    querystring = {'strict': strict, 'yesterday': yesterday, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid19-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countydata(county: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all NYT county data or individual county data if specified. Each entry returned represents data for a given day."
    county: The county that you'd like to search for, separated by comma if you want to search for multiple (i.e. 'Alameda, Humboldt'). Default is full data set.
        
    """
    url = f"https://covid19-statistics.p.rapidapi.com/v2/nyt/counties"
    querystring = {'county': county, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid19-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def state(yesterday: str, states: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    yesterday: For this parameter you can show yesterday data
Available values : true, false, 1, 0.
        states: state name or comma separated names spelled correctly.
        
    """
    url = f"https://covid19-statistics.p.rapidapi.com/v2/states/{states}"
    querystring = {'yesterday': yesterday, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid19-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def states(yesterday: str, sort: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stats on United States of America States with COVID-19, including cases, new cases, deaths, new deaths, and active cases. Data is updated every 10 minutes."
    yesterday: For this parameter you can show yesterday data

Available values : true, false, 1, 0.
        sort: For this parameter you can provide a key from the country model (e.g. cases, todayCases, deaths, active, etc) to sort the states from greatest to least, depending on the key.

Available values : cases, todayCases, deaths, todayDeaths, active.
        
    """
    url = f"https://covid19-statistics.p.rapidapi.com/v2/states"
    querystring = {'yesterday': yesterday, 'sort': sort, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid19-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historicaldatabycountryname(query: str, lastdays: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a country's time series using a country iso code, country ID, or of course, country name. Data is updated every 10 minutes."
    query: Required. For this parameter you can use Country Name && Country Id && ISOs (ISO 2 | ISO 3) 3166 Country Standards
        lastdays: Number of days you want the data to go back to. Default is 30. Use all for full data set. Ex: 15, all, 24


        
    """
    url = f"https://covid19-statistics.p.rapidapi.com/v2/historical/{query}"
    querystring = {'lastdays': lastdays, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid19-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def allcountries(yesterday: str, sort: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a JSON array with an element for each country that has stats available. This includes iso codes, lat/long, a link to the country flag, cases, new cases, deaths, new deaths, recovered, active cases, critical cases, and cases/deaths per one million people. Data is updated every 10 minutes."
    yesterday: For this parameter you can show yesterday data

Available values : true, false, 1, 0.
        sort: For this parameter you can provide a key from the country model (e.g. cases, todayCases, deaths, recovered, active, etc) to sort the countries from greatest to least, depending on the key

Available values : cases, todayCases, deaths, todayDeaths, recovered, active, critical, casesPerOneMillion, deathsPerOneMillion.
        
    """
    url = f"https://covid19-statistics.p.rapidapi.com/v2/countries"
    querystring = {'yesterday': yesterday, 'sort': sort, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid19-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def usnationwidedata(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all NYT US nationwide data. Each entry returned represents data for a given day."
    
    """
    url = f"https://covid19-statistics.p.rapidapi.com/v2/nyt/usa"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid19-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def usastatestoquery_historical_usacounties_state_endpoint(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of states that are available for querying the /historical/usacounties/{state} endpoint with."
    
    """
    url = f"https://covid19-statistics.p.rapidapi.com/v2/historical/usacounties"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid19-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def specificcontinent(yesterday: str, strict: str, query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the same data from the /v2/continents endpoint, but filter down to a specific continent."
    yesterday: For this parameter you can show yesterdays data

Available values : true, false, 1, 0.
        strict: Defaults to true. Setting to false gives you the ability to fuzzy search countries. Example Oman vs. rOMANia

Available values : true, false.
        query: The continent name.
        
    """
    url = f"https://covid19-statistics.p.rapidapi.com/v2/continents/{query}"
    querystring = {'yesterday': yesterday, 'strict': strict, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid19-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historicaltimeseriesdata(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return time series data globally. Data is updated every 10 minutes."
    
    """
    url = f"https://covid19-statistics.p.rapidapi.com/v2/historical/all"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid19-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all(yesterday: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get global stats: cases, deaths, recovered, time last updated, and active cases. Data is updated every 10 minutes."
    yesterday: For this parameter you can show yesterday data. Available values : true, false, 1, 0.
        
    """
    url = f"https://covid19-statistics.p.rapidapi.com/v2/all"
    querystring = {'yesterday': yesterday, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid19-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historicaldataforallcountiesinaspecifiedstate(lastdays: int, state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get time series info from the JHU CSSE Data Repository. Every date since 1/22/20 has an entry tracking deaths and cases. Updated each day at 23:59 UTC. Data is updated every 10 minutes."
    lastdays: number of days you want the data to go back to. Default is 30. Use all for full data set. Ex: 15, all, 24. 
Default value: 30
        state: Required. A valid US state name, validated in the array returned from /v2/historical/usacounties.
        
    """
    url = f"https://covid19-statistics.p.rapidapi.com/v2/historical/usacounties/{state}"
    querystring = {'lastdays': lastdays, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid19-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def singlecountydata(countyname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get JHU CSSE county specific data. This includes confirmed cases, deaths, recovered, and coordinates. Returns array because there are duplicate names. Data is updated every 10 minutes."
    
    """
    url = f"https://covid19-statistics.p.rapidapi.com/v2/jhucsse/counties/{countyname}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid19-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alluscountydata(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get JHU CSSE county specific data. This includes confirmed cases, deaths, recovered, and coordinates. Data is updated every 10 minutes"
    
    """
    url = f"https://covid19-statistics.p.rapidapi.com/v2/jhucsse/counties"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid19-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historicaldatabyprovincename(lastdays: str, province: str, query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a province within a country's time series. Example is /v2/historical/chn/tibet. CHN is China's iso3 code. Data is updated every 10 minutes."
    lastdays: Number of days you want the data to go back to. Default is 30. Use all for full data set. Ex: 15, all, 24.
        province: Required. Province name.
        query: Required. For this parameter you can use Country Name && Country Id && ISOs (ISO 2 | ISO 3) 3166 Country Standards.
        
    """
    url = f"https://covid19-statistics.p.rapidapi.com/v2/historical/{query}/{province}"
    querystring = {'lastdays': lastdays, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid19-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historicaldatabymultipleprovincenames(lastdays: int, query: str, provinces: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get multiple provinces' time series using multiple Country Names && Country Ids && ISOs (ISO 2 | ISO 3) 3166 Country Standards. Data is updated every 10 minutes."
    lastdays: Number of days you want the data to go back to. Default is 30. Use all for full data set. Ex: 15, all, 24


        query: Required. For this parameter you can use multiple Country Names && Country Ids && ISOs (ISO 2 | ISO 3) 3166 Country Standards separated by commas.
        provinces: Required. Provinces spelled correctly separated by ',' or '|' delimiters, never both in the same query.
        
    """
    url = f"https://covid19-statistics.p.rapidapi.com/v2/historical/{query}/{provinces}"
    querystring = {'lastdays': lastdays, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid19-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

