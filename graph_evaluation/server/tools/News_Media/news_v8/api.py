import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def news_v1_search(organizations: str='[
  [
    "Glaad",
    "Friends of Hawaiian Islands National Wildlife Refuge",
    "Running Rebels Community Organization Inc"
  ]
]', sort_order: str=None, filter_operator: str=None, advisory_content: bool=None, start_date: str='2021-09-01', sources_include: str='[
  [
    "www.philanthropynewsdigest.org/",
    "www.nytimes.com/"
  ]
]', subjects: str='[
  [
    "ST000000",
    "SR040000",
    "SA010200"
  ]
]', staff_change_mentioned: bool=None, search_terms: str='Tree planting', grant_mentioned: bool=None, populations: str='[
  [
    "PC000000",
    "PG010000",
    "PE050000"
  ]
]', page: int=1, end_date: str='2021-10-01', rfp_mentioned: bool=None, sources_exclude: str='[
  [
    "philanthropy.com/",
    "www.niemanlab.org/"
  ]
]', added_end_date: str='2021-10-02T17:32:18.1169712Z', eins: str='[
  [
    "13-1837418",
    "58-1721923",
    742150288
  ]
]', locations: str='[
  [
    5791132,
    7007157,
    7007919
  ]
]', added_start_date: str='2021-09-01T17:32:18.1169712Z', people: str='[
  [
    "Bradford K. Smith",
    "Mizmun Kusairi"
  ]
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access the only API on the market delivering real-time, actionable news on nonprofits, philanthropy, and the social sector that is curated by Candid from over 65,000 sources."
    organizations: Organization names. Accepts up to 25, multiple values are always joined using OR.
        sort_order: Sort the results by publication date. desc = descending; asc = ascending. Default is desc.
        filter_operator: Toggle the and/or union type for searches for the following parameters Subjects, Populations, Locations, People, and Organizations. Default is AND. Please note that inter-array parameter values are not affected by the filter operator.
        advisory_content: Advisory content is flagged for organizations or people in the social sector that may or may not require due diligence. True = Only return articles flagged as advisory content. False = Do not return articles flagged as advisory content. Null = Return articles regardless of advisory content.
        start_date: The oldest article publication date to search from in the format yyyy-MM-dd. Default is one month before today's date. The maximum date range size is one year. The earliest date to search from is 2018-09-24.
        sources_include: Names of news source to include. Accepts up to 25, multiple values are always joined using OR.
        subjects: Candid Philanthropy Classification System (PCS) Subject Facet codes (eg. ST000000, SR040000, SA010200, etc). Top level PCS codes will include searches for all related child codes. Accepts up to 25, multiple values are always joined using OR.
        staff_change_mentioned: True = Only return articles containing mentions of staff changes. False = Do not return articles containing mentions of staff changes. Null = Return articles regardless of staff changes mentioned.
        search_terms: Free text news search.
        grant_mentioned: True = Only return articles containing mentions of grants. False = Do not return articles containing mentions of grants. Null = Return articles regardless of grants mentioned.
        populations: Candid Philanthropy Classification System (PCS) Population Facet codes (eg. PC000000, PG010000, PE050000, etc). Top level PCS codes will include searches for all related child codes. Accepts up to 25, multiple values are always joined using OR.
        page: Select the page number for search results. If page is not specified, it defaults to 1. The maximum page value is the total number of results divided by 25.
        end_date: The most recent article publication date to search to in the format yyyy-MM-dd. Default is today's date. The maximum date range size is one year.
        rfp_mentioned: True = Only return articles containing mentions of RFPs. False = Do not return articles containing mentions of RFPs. Null = Return articles regardless of RFPs mentioned.
        sources_exclude: Names of news source to exclude. Accepts up to 25, multiple values are always joined using OR.
        added_end_date: The most recent date and time to search from when articles were added to the Candid News database in the format yyyy-MM-ddTHH:mm:ss.fffffffZ. Default is today's date. The maximum date window size is one year.
        eins: Organization EINs that can be formatted with or without a dash. Accepts up to 25, multiple values are always joined using OR.
        locations: GeoName IDs for locations. Accepts up to 25, multiple values are always joined using OR. Candid can provide you a GeoNames ID seed file in JSON format for all Candid-supported GeoNames IDs.
        added_start_date: The oldest date and time to search from when articles were added to the Candid News database in the format yyyy-MM-ddTHH:mm:ss.fffffffZ. Default is one month before today's date. The maximum date window size is one year. The earliest date to search from is 2018-09-24
        people: People by which to filter news articles. Accepts up to 25, multiple values are always joined using OR.
        
    """
    url = f"https://news180.p.rapidapi.com/v1/search"
    querystring = {}
    if organizations:
        querystring['organizations'] = organizations
    if sort_order:
        querystring['sort_order'] = sort_order
    if filter_operator:
        querystring['filter_operator'] = filter_operator
    if advisory_content:
        querystring['advisory_content'] = advisory_content
    if start_date:
        querystring['start_date'] = start_date
    if sources_include:
        querystring['sources_include'] = sources_include
    if subjects:
        querystring['subjects'] = subjects
    if staff_change_mentioned:
        querystring['staff_change_mentioned'] = staff_change_mentioned
    if search_terms:
        querystring['search_terms'] = search_terms
    if grant_mentioned:
        querystring['grant_mentioned'] = grant_mentioned
    if populations:
        querystring['populations'] = populations
    if page:
        querystring['page'] = page
    if end_date:
        querystring['end_date'] = end_date
    if rfp_mentioned:
        querystring['rfp_mentioned'] = rfp_mentioned
    if sources_exclude:
        querystring['sources_exclude'] = sources_exclude
    if added_end_date:
        querystring['added_end_date'] = added_end_date
    if eins:
        querystring['eins'] = eins
    if locations:
        querystring['locations'] = locations
    if added_start_date:
        querystring['added_start_date'] = added_start_date
    if people:
        querystring['people'] = people
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "news180.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

