import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def name_concept_type_specific_concept_json(query: str, concept_type: str, specific_concept: str, fields: str='all', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    query: Precedes the search term string. Used in a Search Query. Except for &lt;specific_concept_name&gt;, Search Query will take the required parameters listed above (&lt;concept_type&gt;, &lt;concept_uri&gt;, &lt;article_uri&gt;) as an optional_parameter in addition to the query=&lt;query_term&gt;.
        concept_type: The type of the concept, used for Constructing a Semantic API Request by Concept Type and Specific Concept Name. The parameter is defined as a name-value pair, as in "concept_type=[nytd_geo|nytd_per|nytd_org|nytd_des]".

        specific_concept: The name of the concept, used for Constructing a Semantic API Request by Concept Type and Specific Concept Name. The parameter is defined in the URI path, as the element immediately preceding ".json" like with "Baseball.json".

        fields: "all" or comma-separated list of specific optional fields: pages, ticker_symbol, links, taxonomy, combinations, geocodes, article_list, scope_notes, search_api_query

Optional fields are returned in result_set. They are briefly explained here:

pages: A list of topic pages associated with a specific concept.
ticker_symbol: If this concept is a publicly traded company, this field contains the ticker symbol.
links: A list of links from this concept to external data resources.
taxonomy: For descriptor concepts, this field returns a list of taxonomic relations to other concepts.
combinations: For descriptor concepts, this field returns a list of the specific meanings tis concept takes on when combined with other concepts.
geocodes: For geographic concepts, the full GIS record from geonames.
article_list: A list of up to 10 articles associated with this concept.
scope_notes: Scope notes contains clarifications and meaning definitions that explicate the relationship between the concept and an article.
search_api_query: Returns the request one would need to submit to the Article Search API to obtain a list of articles annotated with this concept.

        
    """
    url = f"https://ny-time-semantic.p.rapidapi.com/name/{concept_type}/{specific_concept}.json"
    querystring = {'query': query, }
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ny-time-semantic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_json(query: str, fields: str='all', offset: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    query: Precedes the search term string. Used in a Search Query. Except for &lt;specific_concept_name&gt;, Search Query will take the required parameters listed above (&lt;concept_type&gt;, &lt;concept_uri&gt;, &lt;article_uri&gt;) as an optional_parameter in addition to the query=&lt;query_term&gt;.
        fields: "all" or comma-separated list of specific optional fields: pages, ticker_symbol, links, taxonomy, combinations, geocodes, article_list, scope_notes, search_api_query

Optional fields are returned in result_set. They are briefly explained here:

pages: A list of topic pages associated with a specific concept.
ticker_symbol: If this concept is a publicly traded company, this field contains the ticker symbol.
links: A list of links from this concept to external data resources.
taxonomy: For descriptor concepts, this field returns a list of taxonomic relations to other concepts.
combinations: For descriptor concepts, this field returns a list of the specific meanings tis concept takes on when combined with other concepts.
geocodes: For geographic concepts, the full GIS record from geonames.
article_list: A list of up to 10 articles associated with this concept.
scope_notes: Scope notes contains clarifications and meaning definitions that explicate the relationship between the concept and an article.
search_api_query: Returns the request one would need to submit to the Article Search API to obtain a list of articles annotated with this concept.

        offset: Integer value for the index count from the first concept to the last concept, sorted alphabetically. Used in a Search Query. A Search Query will return up to 10 concepts in its results.
        
    """
    url = f"https://ny-time-semantic.p.rapidapi.com/search.json"
    querystring = {'query': query, }
    if fields:
        querystring['fields'] = fields
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ny-time-semantic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

