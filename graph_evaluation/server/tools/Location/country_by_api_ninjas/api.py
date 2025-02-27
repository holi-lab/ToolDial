import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_country(min_area: int=None, name: str='United States', min_fertility: int=None, min_gdp_growth: int=None, limit: int=None, min_population: int=None, max_unemployment: int=None, max_urban_pop_rate: int=None, max_population: int=None, max_gdp: int=None, min_urban_pop_rate: int=None, min_unemployment: int=None, max_area: int=None, min_gdp: int=None, min_infant_mortality: int=None, max_gdp_growth: int=None, max_infant_mortality: int=None, currency: str=None, max_fertility: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Country API endpoint."
    min_area: Minimum surface area of country in km2.
        name: Plain English name, 2-letter ISO-3166 alpha-2, or 3-letter ISO-3166 alpha-3 code of country.
        min_fertility: Minimum fertility rate in %.
        min_gdp_growth: Minimum GDP growth rate in %.
        limit: How many results to return. Must be between **1** and **30**. Default is **5**.
        min_population: Minimum population of country.
        max_unemployment: Maximum unemployment rate in %.
        max_urban_pop_rate: Maximum urban population rate in %.
        max_population: Maximum population of country.
        max_gdp: Maximum gross domestic product (GDP) of country, in US Dollars.
        min_urban_pop_rate: Minimum urban population rate in %.
        min_unemployment: Minimum unemployment rate in %.
        max_area: Maximum surface area of country in km2.
        min_gdp: Minimum gross domestic product (GDP) of country, in US Dollars.
        min_infant_mortality: Minimum infant mortality rate in %.
        max_gdp_growth: Maximum GDP growth rate in %.
        max_infant_mortality: Maximum infant mortality rate in %.
        currency: 3-letter currency code of country (e.g. **USD**).
        max_fertility: Maximum fertility rate in %.
        
    """
    url = f"https://country-by-api-ninjas.p.rapidapi.com/v1/country"
    querystring = {}
    if min_area:
        querystring['min_area'] = min_area
    if name:
        querystring['name'] = name
    if min_fertility:
        querystring['min_fertility'] = min_fertility
    if min_gdp_growth:
        querystring['min_gdp_growth'] = min_gdp_growth
    if limit:
        querystring['limit'] = limit
    if min_population:
        querystring['min_population'] = min_population
    if max_unemployment:
        querystring['max_unemployment'] = max_unemployment
    if max_urban_pop_rate:
        querystring['max_urban_pop_rate'] = max_urban_pop_rate
    if max_population:
        querystring['max_population'] = max_population
    if max_gdp:
        querystring['max_gdp'] = max_gdp
    if min_urban_pop_rate:
        querystring['min_urban_pop_rate'] = min_urban_pop_rate
    if min_unemployment:
        querystring['min_unemployment'] = min_unemployment
    if max_area:
        querystring['max_area'] = max_area
    if min_gdp:
        querystring['min_gdp'] = min_gdp
    if min_infant_mortality:
        querystring['min_infant_mortality'] = min_infant_mortality
    if max_gdp_growth:
        querystring['max_gdp_growth'] = max_gdp_growth
    if max_infant_mortality:
        querystring['max_infant_mortality'] = max_infant_mortality
    if currency:
        querystring['currency'] = currency
    if max_fertility:
        querystring['max_fertility'] = max_fertility
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "country-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

