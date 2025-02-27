import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def location(location_name: str, country_code: str='GB', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Location API lets you search for supported Google search locations. You can pass the `location_name` returned by the Location API as the `location_name` parameter and set the `location_parameters_auto` value to true in a Search API query to retrieve search results localized to that location."
    
    """
    url = f"https://g-search.p.rapidapi.com/location"
    querystring = {'location_name': location_name, }
    if country_code:
        querystring['country_code'] = country_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "g-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(q: str, google_domain: str=None, cr: str=None, tbs: str=None, nfpr: str=None, gl: str=None, safe: str=None, filter: str=None, location_parameters_auto: str='true', location_name: str='London,Ontario,Canada', num: str=None, hl: str=None, lr: str=None, start: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Real-time Search Result Page"
    q: The parameter defines the query you want to search. You can use anything that you would use in a regular Google search. (e.g., `inurl:`, `site:`,` intitle:`, etc.) . You can also pass the absolute google search URL (**URL encoded**) here. For example, if you want to scrape the 2nd page of the SERP, you can pass the URL from the navigation field in the result response. If the absolute URL is passed here then others parameters will be ignored.
        google_domain: The parameter defines the Google domain to use. It defaults to google.com
        cr: The `cr` parameter instructs Google to limit the results to websites in the specified country.
        tbs: (to be searched) the parameter defines advanced search parameters that aren't possible in the regular query field. (e.g., advanced search for patents, dates, news, videos, images, apps, or text contents).
        nfpr: The parameter defines the exclusion of results from an auto-corrected query that is spelled wrong. It can be set to 1 to exclude these results, or 0 to include them (default).
        gl: The `gl` parameter determines the Google country to use for the query.
        safe: The parameter defines the level of filtering for adult content. It can be set to active, or off (default).
        filter: The parameter defines if the filters for 'Similar Results' and 'Omitted Results' are on or off. It can be set to 1 (default) to enable these filters, or 0 to disable these filters.
        location_parameters_auto: If the `location_name` parameter is set to a location from the Location Endpoint, and `location_parameters_auto` is set to true (default) then the `google_domain`, `gl` and hl parameters are automatically updated to the domain, country, and language that match the location. Valid values are true (default) to enable this behavior or false to disable it.
        location_name: Specifies the geographic location where the query is executed. You can enter `location_name` with free text or use the result from the `/location` endpoint for a more valid value. When you select a `location_name` and assign `true` to the `location_parameters_auto` parameter, then the values in google_domain, `gl` and `hl` are automatically updated to the domain, country, and language matching the default location (note that this behavior can be disabled via location_parameters_auto).
        num: The parameter defines the maximum number of results to return. (e.g., 10 (default) returns 10 results, 40 returns 40 results, and 100 returns 100 results).
        hl: The `hl` parameter determines the Google UI (display) language to return results.
        lr: The `lr` parameter limits the results to websites containing the specified language.
        start: `start` parameter defines the resulting offset. It skips the given number of results. It's used for pagination. (e.g., 0 (default) is the first page of results, 10 is the 2nd page of results, 20 is the 3rd page of results, etc.).
        
    """
    url = f"https://g-search.p.rapidapi.com/search"
    querystring = {'q': q, }
    if google_domain:
        querystring['google_domain'] = google_domain
    if cr:
        querystring['cr'] = cr
    if tbs:
        querystring['tbs'] = tbs
    if nfpr:
        querystring['nfpr'] = nfpr
    if gl:
        querystring['gl'] = gl
    if safe:
        querystring['safe'] = safe
    if filter:
        querystring['filter'] = filter
    if location_parameters_auto:
        querystring['location_parameters_auto'] = location_parameters_auto
    if location_name:
        querystring['location_name'] = location_name
    if num:
        querystring['num'] = num
    if hl:
        querystring['hl'] = hl
    if lr:
        querystring['lr'] = lr
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "g-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

