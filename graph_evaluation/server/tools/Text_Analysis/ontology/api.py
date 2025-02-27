import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def expand_concept_by_id(is_id: str, output_lang: str='en', with_cids: str=None, levels_up: str='1', exclude_tags: str=None, levels_down: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API endpoint used for creating an expanded query using concept child-parent relationships, synonyms, and “similar” relationships"
    id: format: numeric id of the concept. The concept must be in one of the allowable branches: occupation, specialization, function, skill
        output_lang: format: 2 letter ISO code for language to return the affinity table in

default value: de

effect: Return preferred label for results in this language if available.
        with_cids: format: true/false

default value: false

effect: include the concept id with every result
        levels_up: format: integer

default value: 1

effect: represents the number of parent levels to go up from given concept
        exclude_tags: format: character

default value: empty

effect: exclude labels which have this tag, such as ‘e’.
        levels_down: format: integer

default value: 1

effect represents the number of children levels to go up from given concept
        
    """
    url = f"https://ontology1.p.rapidapi.com/expand_concept_id"
    querystring = {'id': is_id, }
    if output_lang:
        querystring['output_lang'] = output_lang
    if with_cids:
        querystring['with_cids'] = with_cids
    if levels_up:
        querystring['levels_up'] = levels_up
    if exclude_tags:
        querystring['exclude_tags'] = exclude_tags
    if levels_down:
        querystring['levels_down'] = levels_down
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ontology1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def occupation_suggest(relation: str, occupation: str, limit: str=None, exclude_tags: str=None, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API endpoint used for getting recommended functions, specializations, skills, soft skills for a given occupation"
    relation: allowable relations: ‘specialization’, ‘function’, ‘skill’, ‘softskill’

default value: none, and nothing is returned. At least one relation must be specified.
        occupation: The string which must match the concept label in the search_lang
        limit: integer representing maximum number of results, from 1 to 50, default: 25
        exclude_tags: exclude labels which have this tag, such as ‘e’. default: e
        lang: 2 letters ISO 369-1 language code for which to search and return results in, default: en

        
    """
    url = f"https://ontology1.p.rapidapi.com/occupation_suggest"
    querystring = {'relation': relation, 'occupation': occupation, }
    if limit:
        querystring['limit'] = limit
    if exclude_tags:
        querystring['exclude_tags'] = exclude_tags
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ontology1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def concept_detail(concept_id: str, country: str=None, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives the details of a specific Ontology's concept given its concept_id."
    concept_id: The concept_id must be an integer corresponding to the internal ID of the concept, which is returned in the listing call
        country: format: ISO 3166-1 numeric code. Example: 756 for Switzerland, 040 for Austria, etc.

default value: 0 (no preferred country)

effect: Labels with the given country code set on them will be chosen over regular or preffered labels
        lang: format: ISO 369-1, 2 character language code, example: de, en, fr, …

default value: browser locale

effect: Return preferred label for results in this language if available.
        
    """
    url = f"https://ontology1.p.rapidapi.com/concepts/{concept_id}"
    querystring = {}
    if country:
        querystring['country'] = country
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ontology1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def concept_relations_by_branch(lang: str, branch: str, label: str, output_classifications: str=None, include_descendants: str=None, limit_output_lang: str='en', include_umbrella: str=None, num_results: str=None, relation: str='children', descendant_levels: str=None, country: str=None, start_at: str=None, child_level: str='2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API endpoint used to explore the relationships in the concept graph when the concept id is not known."
    lang: format: ISO 369-1, 2 character language code, example: de, en, fr,

default value: browser locale

effect: search for concepts in this language and display preferred label in this language
        branch: format: string of a single branch to find label in.

allowable branches: ‘occupation’, ‘specialization’, ‘function’, ‘skill’, ‘softskill’, ‘education’
        label: format: any string which is an exact match for a concept.

effect: if ambiguous labels exist matching this concept, it will return the first matched.

hint: use = before string to avoid normalizing during search to improve performance
        output_classifications: format: | separated list of classifications

effect: return all classification values related to the concept_id of each returned value

list of available classifications: [Classifications and Taxonomies](https://www.janzz.jobs/static/doc/apiv1/classifications.html#classifications-and-taxonomies)
        include_descendants: format: true|false

default: false

effect: when set to true, it will return all descendants of the related concepts. Does not apply to child/parent relations.
        limit_output_lang: format: true|false

default value: false

effect: Only display concepts having at least one label in the language provided in lang parameter

        include_umbrella: format: true|false

default: true

effect: when set to false, it will not return any umbrella concepts
        num_results: format: integer between 0 and 1000

default value: 100

effect: maximum number of results to return
        relation: **child/parent**:

children: child concepts
parents: parent concepts

**by relationship name**:

specializations: has specialization
functions: has function
skills: has skill
softskills: has softskill
skillsonet: has skill O*Net 1,2,3,4,5
softskillsonet: has softskill O*Net 1,2,3,4,5
skillsonet1: has skill O*Net 1
skillsonet2: has skill O*Net 2
skillsonet3: has skill O*Net 3
skillsonet4: has skill O*Net 4
skillsonet5: has skill O*Net 5
softskillsonet1: has softskill O*Net 1
softskillsonet2: has softskill O*Net 2
softskillsonet3: has softskill O*Net 3
softskillsonet4: has softskill O*Net 4
softskillsonet5: has softskill O*Net 5
specialized: is specialized in
education: has education
similar1: same but different
similar2: same but different 2
similar3: same but different 3
similar4: same but different 4
similar5: same but different 5
similar: same but different 1, 2, 3, 4, 5
nss: not suitable for
related_occupations: special case when on a non-occupation concept, find related occupations

effect:
display all relationships to the specified concept which use this relationship type

        descendant_levels: format: 1,2,3

default: -1

effect: when set, include only n levels of descendants, not all. Does not apply to child/parent relations.
        country: format: ISO 3166-1 numeric code. Example: 756 for Switzerland, 040 for Austria, etc.

default value: 0 (no preferred country)

effect: Labels with the given country code set on them will be chosen over regular or preffered labels
        start_at: format: integer

default value: 0

effect: start at for pagination
        child_level: format: 1,2,3

default: 1

effect: Applies only when relation_type is children, setting to 2 for example will return node.children.children, and no direct children of the current node.
        
    """
    url = f"https://ontology1.p.rapidapi.com/concept_relation_by_branch/"
    querystring = {'lang': lang, 'branch': branch, 'label': label, }
    if output_classifications:
        querystring['output_classifications'] = output_classifications
    if include_descendants:
        querystring['include_descendants'] = include_descendants
    if limit_output_lang:
        querystring['limit_output_lang'] = limit_output_lang
    if include_umbrella:
        querystring['include_umbrella'] = include_umbrella
    if num_results:
        querystring['num_results'] = num_results
    if relation:
        querystring['relation'] = relation
    if descendant_levels:
        querystring['descendant_levels'] = descendant_levels
    if country:
        querystring['country'] = country
    if start_at:
        querystring['start_at'] = start_at
    if child_level:
        querystring['child_level'] = child_level
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ontology1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def expand_concept_by_input(q: str, search_lang: str, branch: str, output_lang: str='en', exclude_tags: str=None, levels_up: str='1', with_cids: str=None, levels_down: str='2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API endpoint used for creating an expanded query using concept child-parent relationships, synonyms, and “similar” relationships"
    q: format: string

effect: The string which must match the concept label in the search_lang
        search_lang: format: 2 letter ISO code

effect: language for which to search for q in, or “all” to include all languages

default value: all
        branch: format: comma separated list of branches to limit search to.

allowed values: occupation, specialization, function, skill

effect: search for concepts matching q in these branches
        output_lang: format: 2 letter ISO code for language to return the affinity table in

default value: de

effect: Return preferred label for results in this language if available.
        exclude_tags: format: character

default value: empty

effect: exclude labels which have this tag, such as ‘e’.
        levels_up: format: integer

default value: 1

effect: represents the number of parent levels to go up from given concept
        with_cids: format: true/false

default value: false

effect: include the concept id with every result
        levels_down: format: integer

default value: 1

effect represents the number of children levels to go up from given concept
        
    """
    url = f"https://ontology1.p.rapidapi.com/expand_concept"
    querystring = {'q': q, 'search_lang': search_lang, 'branch': branch, }
    if output_lang:
        querystring['output_lang'] = output_lang
    if exclude_tags:
        querystring['exclude_tags'] = exclude_tags
    if levels_up:
        querystring['levels_up'] = levels_up
    if with_cids:
        querystring['with_cids'] = with_cids
    if levels_down:
        querystring['levels_down'] = levels_down
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ontology1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def more_specific(q: str, branch: str, output_lang: str='en', exclude_umbrella: str=None, limit: str=None, exclude_tags: str=None, levels_down: str='2', search_lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API endpoint used for finding concepts which are more specific than the one provided. This uses only strings to search and it can return duplicate concepts if more than one label exists in child concepts."
    q: The string which must match the concept label in the search_lang
        branch: comma-separated list of branches in which to search for the given term. It is recommended to use a single branch to isolate the concept you are really looking for.

allowable branches: ‘occupation’, ‘specialization’, ‘function’, ‘skill’, ‘softskill’

default value: none, and no search is performed. At least one branch must be specified
        output_lang: 2 letter ISO code for language to return the affinity table in, default: en
        exclude_umbrella: exclude labels which belong to umbrella terms. default: False
        limit: integer representing the maximum number of results, from 1 to 50, default: 25
        exclude_tags: exclude labels which have this tag, such as ‘e’. default: e
        levels_down: integer representing number of children levels to go down from found concept, from 1 to 3, default: 1
        search_lang: 2 letter ISO code for language for which to search in, or “all” to include all languages, default: en
        
    """
    url = f"https://ontology1.p.rapidapi.com/more_specific"
    querystring = {'q': q, 'branch': branch, }
    if output_lang:
        querystring['output_lang'] = output_lang
    if exclude_umbrella:
        querystring['exclude_umbrella'] = exclude_umbrella
    if limit:
        querystring['limit'] = limit
    if exclude_tags:
        querystring['exclude_tags'] = exclude_tags
    if levels_down:
        querystring['levels_down'] = levels_down
    if search_lang:
        querystring['search_lang'] = search_lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ontology1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def classification_details(val: str, cls: str, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API endpoint used for getting classification details"
    val: the classification value
        cls: the classification name, for example, ISCO-08. Any / characters in cls need to be escaped using a double-underscore, __
        lang: 2 letter ISO code for language for which to search and return results in, default: en
        
    """
    url = f"https://ontology1.p.rapidapi.com/classification/{cls}/{val}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ontology1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cross_classification(cls: str, clsx: str, val: str, cls1_lang: str=None, cid_filter_cls: str=None, cls2_lang: str=None, cid_lang: str=None, cid_filter_val: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API endpoint used for returning equivalent classifications"
    cls: The source classification name, for example, ISCO-08. Any / characters in cls need to be escaped using a double-underscore, __
        clsx: The target classification name, for example, ISCO-08. Any / characters in cls need to be escaped using a double-underscore, __
        val: The source classification value
        cls1_lang: 2 letter ISO code for language for to provide a classification value for cls, default: en
        cid_filter_cls: only consider concepts also having this classification default: empty
        cls2_lang: 2 letter ISO code for language for to provide a classification value for cls2, default: en
        cid_lang: 2 letter ISO code for language for to provide a best label for returned cids, default: en
        cid_filter_val: when cid_filter_cls is used, filter concepts having this value for cid_filter_cls. Leave empty to include any value, as long as the cid_filter_cls classification is set. default: empty
        
    """
    url = f"https://ontology1.p.rapidapi.com/cross_classification/{cls}/{val}/{clsx}"
    querystring = {}
    if cls1_lang:
        querystring['cls1_lang'] = cls1_lang
    if cid_filter_cls:
        querystring['cid_filter_cls'] = cid_filter_cls
    if cls2_lang:
        querystring['cls2_lang'] = cls2_lang
    if cid_lang:
        querystring['cid_lang'] = cid_lang
    if cid_filter_val:
        querystring['cid_filter_val'] = cid_filter_val
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ontology1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def concept_relations_by_id(lang: str, relation: str, concept_id: str, include_descendants: str=None, child_level: str='2', descendant_levels: str=None, start_at: str=None, include_umbrella: str=None, output_classifications: str=None, num_results: str=None, limit_output_lang: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API endpoint used to explore the relationships in the concept graph when the concept id is known."
    lang: format: ISO 369-1, 2 character language code, example: de, en, fr,

default value: browser locale

effect: search for concepts in this language and display preferred label in this language
        relation: **child/parent**:

children: child concepts
parents: parent concepts

**by relationship name**:

specializations: has specialization
functions: has function
skills: has skill
softskills: has softskill
skillsonet: has skill O*Net 1,2,3,4,5
softskillsonet: has softskill O*Net 1,2,3,4,5
skillsonet1: has skill O*Net 1
skillsonet2: has skill O*Net 2
skillsonet3: has skill O*Net 3
skillsonet4: has skill O*Net 4
skillsonet5: has skill O*Net 5
softskillsonet1: has softskill O*Net 1
softskillsonet2: has softskill O*Net 2
softskillsonet3: has softskill O*Net 3
softskillsonet4: has softskill O*Net 4
softskillsonet5: has softskill O*Net 5
specialized: is specialized in
education: has education
similar1: same but different
similar2: same but different 2
similar3: same but different 3
similar4: same but different 4
similar5: same but different 5
similar: same but different 1, 2, 3, 4, 5
nss: not suitable for
related_occupations: special case when on a non-occupation concept, find related occupations

effect:
display all relationships to the specified concept which use this relationship type

        include_descendants: format: true|false

default: false

effect: when set to true, it will return all descendants of the related concepts. Does not apply to child/parent relations.
        child_level: format: 1,2,3

default: 1

effect: Applies only when relation_type is children, setting to 2 for example will return node.children.children, and no direct children of the current node.
        descendant_levels: format: 1,2,3

default: -1

effect: when set, include only n levels of descendants, not all. Does not apply to child/parent relations.
        start_at: format: integer

default value: 0

effect: start at for pagination
        include_umbrella: format: true|false

default: true

effect: when set to false, it will not return any umbrella concepts
        output_classifications: format: | separated list of classifications

effect: return all classification values related to the concept_id of each returned value

list of available classifications: [Classifications and Taxonomies](https://www.janzz.jobs/static/doc/apiv1/classifications.html#classifications-and-taxonomies)
        num_results: format: integer between 0 and 1000

default value: 100

effect: maximum number of results to return
        limit_output_lang: format: true|false

default value: false

effect: Only display concepts having at least one label in the language provided in lang parameter

        country: format: ISO 3166-1 numeric code. Example: 756 for Switzerland, 040 for Austria, etc.

default value: 0 (no preferred country)

effect: Labels with the given country code set on them will be chosen over regular or preffered labels
        
    """
    url = f"https://ontology1.p.rapidapi.com/concept_relation_by_id/"
    querystring = {'lang': lang, 'relation': relation, 'concept_id': concept_id, }
    if include_descendants:
        querystring['include_descendants'] = include_descendants
    if child_level:
        querystring['child_level'] = child_level
    if descendant_levels:
        querystring['descendant_levels'] = descendant_levels
    if start_at:
        querystring['start_at'] = start_at
    if include_umbrella:
        querystring['include_umbrella'] = include_umbrella
    if output_classifications:
        querystring['output_classifications'] = output_classifications
    if num_results:
        querystring['num_results'] = num_results
    if limit_output_lang:
        querystring['limit_output_lang'] = limit_output_lang
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ontology1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def education_level_by_classification(class_esco: str, output_lang: str='en', country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return an education level for a given standard classificationon and a clascode or value, where the result will be the result of mapping these to the following educational scale:
		
		0: No qualifications/No A-Level/No International Baccalaureate (IB)/No high school diploma
		1: In training (e.g. school, college, apprenticeship, etc.), On-the-job training, Course, …
		2: A Level/High School Diploma/International Baccalaureate (IB), University studies (not completed), …
		3: Vocational qualification (e.g. Swiss Federal VET Diploma)
		4: Certificate/Credentials/Diploma/Title, Higher vocational degree/Higher technical degree, …
		5: Bachelor’s degree/University primary qualification/Undergraduate degree
		6: Master’s degree/Postgraduate degree
		7: Doctorate/PhD"
    class_esco: CLASS_{CLASSIFICATION_NAME}

search for concepts having a given classification. Use * for the value to indicate all concepts which have that classification name, regardless of the value

example: ?q=*&CLASS_BIS/AMS=647520 (normally you would set q=* when searching by specific classification ID)

example: ?q=painter&CLASS_BIS/AMS=*
list of available classifications: [Classifications and Taxonomies](https://www.janzz.jobs/static/doc/apiv1/classifications.html#classifications-and-taxonomies)
        output_lang: format: ISO 369-1, 2 character language code, example: de, en, fr, …

default value: browser locale

effect: Return preferred label for results in this language if available.
        country: format: ISO 3166-1 numeric code. Example: 756 for Switzerland, 040 for Austria, etc.

default value: 0 (no preferred country)

effect: Labels with the given country code set on them will be chosen over regular or preffered labels
        
    """
    url = f"https://ontology1.p.rapidapi.com/education_level/"
    querystring = {'CLASS_ESCO': class_esco, }
    if output_lang:
        querystring['output_lang'] = output_lang
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ontology1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def concept_list(q: str, branch: str, exclude_nss: str=None, output_lang: str='en', best_label: str=None, exact_match: str=None, search_lang: str='en', normalized: str=None, output_classifications: str=None, is_class: str=None, country: str=None, override_limit_level: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of ontology concepts that contains the input text.
		
		If you would like to convert a given string to specific ontology concepts, you can use the concepts api endpoint which works like the labels search but does not return duplicate concepts. 
		
		There is no guarantee that the entered text will appear in the result (the preferred label of a concept is returned, not the search term entered), therefore this might not be suitable for certain typeahead applications. 
		
		With this tool you can also search in one language and output the results in another language."
    q: format: any string

default value: ‘*’ (search for everything)

effect: performs a search for concepts that have labels which partially match this query string

        branch: format: comma separated list of branches to limit search to.

allowable branches: ‘occupation’, ‘specialization’, ‘function’, ‘skill’, ‘softskill’, ‘language’, ‘industry’, ‘contract_type’, ‘education’, ‘education2’, ‘authorization’, ‘filter__xx’ (where xx is the name of the filter, example filter__location_no)

default value: none, and no serach is performed. At least one branch must be specified.
        exclude_nss: format: | separated list of “person class” concept labels

effect: does not include concepts in results having a relation “not suitable for” where arg1 matches one of provided concepts.
        output_lang: format: ISO 369-1, 2 character language code, example: de, en, fr, …

default value: browser locale

effect: Return preferred label for results in this language if available.
        best_label: format: true|false

default value: false

effect: if true, return the preferred label of the concept, otherwise return label most closely matching search query
        exact_match: format: true|false

default value: false

effect: search for this exact string. When false, search for q
        search_lang: format: ISO 369-1, 2 character language code, example: de, en, fr, … Choose ‘all’ to search in all languages

default value: ‘all’

effect: search for concept labels only in this language.
        normalized: format: true|false

default value: false

effect: normalize the input string, ü->ue, A->a, collapse multiple spaces, etc.
        output_classifications: format: | separated list of classifications

effect: return all classification values related to the concept_id of each returned value

list of available classifications:  [Classifications and Taxonomies](https://www.janzz.jobs/static/doc/apiv1/classifications.html#classifications-and-taxonomies)
        is_class: search for concepts having a given classification. 

Use * for the value to indicate all concepts which have that classification name, regardless of the value

example: ?q=*&CLASS_BIS/AMS=647520 (normally you would set q=* when searching by specific classification ID)

example: ?q=painter&CLASS_BIS/AMS=*
list of available classifications:  [Classifications and Taxonomies](https://www.janzz.jobs/static/doc/apiv1/classifications.html#classifications-and-taxonomies)
        country: format: ISO 3166-1 numeric code. Example: 756 for Switzerland, 040 for Austria, etc.

default value: 0 (no preferred country)

effect: Labels with the given country code set on them will be chosen over regular or preffered labels
        override_limit_level: format: true|false

default value: false

effect: when searching in industry or contract type, default searches only 1 level deep. This allows you to remove this limit - for internal use only.
        
    """
    url = f"https://ontology1.p.rapidapi.com/concepts"
    querystring = {'q': q, 'branch': branch, }
    if exclude_nss:
        querystring['exclude_nss'] = exclude_nss
    if output_lang:
        querystring['output_lang'] = output_lang
    if best_label:
        querystring['best_label'] = best_label
    if exact_match:
        querystring['exact_match'] = exact_match
    if search_lang:
        querystring['search_lang'] = search_lang
    if normalized:
        querystring['normalized'] = normalized
    if output_classifications:
        querystring['output_classifications'] = output_classifications
    if is_class:
        querystring['CLASS_'] = is_class
    if country:
        querystring['country'] = country
    if override_limit_level:
        querystring['override_limit_level'] = override_limit_level
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ontology1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def education_level_by_id(is_id: str, country: str=None, is_class: str=None, output_lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This end-point will return the required education level for a given concept id, where the values indicate the following educational scale:
		
		0: No qualifications/No A-Level/No International Baccalaureate (IB)/No high school diploma
		1: In training (e.g. school, college, apprenticeship, etc.), On-the-job training, Course, …
		2: A Level/High School Diploma/International Baccalaureate (IB), University studies (not completed), …
		3: Vocational qualification (e.g. Swiss Federal VET Diploma)
		4: Certificate/Credentials/Diploma/Title, Higher vocational degree/Higher technical degree, …
		5: Bachelor’s degree/University primary qualification/Undergraduate degree
		6: Master’s degree/Postgraduate degree
		7: Doctorate/PhD"
    id: The search concept cid
        country: format: ISO 3166-1 numeric code. Example: 756 for Switzerland, 040 for Austria, etc.

default value: 0 (no preferred country)

effect: Labels with the given country code set on them will be chosen over regular or preffered labels
        is_class: CLASS_{CLASSIFICATION_NAME}

search for concepts having a given classification. Use * for the value to indicate all concepts which have that classification name, regardless of the value

example: ?q=*&CLASS_BIS/AMS=647520 (normally you would set q=* when searching by specific classification ID)

example: ?q=painter&CLASS_BIS/AMS=*
list of available classifications: [Classifications and Taxonomies](https://www.janzz.jobs/static/doc/apiv1/classifications.html#classifications-and-taxonomies)
        output_lang: format: ISO 369-1, 2 character language code, example: de, en, fr, …

default value: browser locale

effect: Return preferred label for results in this language if available.
        
    """
    url = f"https://ontology1.p.rapidapi.com/education_level/{is_id}"
    querystring = {}
    if country:
        querystring['country'] = country
    if is_class:
        querystring['CLASS_'] = is_class
    if output_lang:
        querystring['output_lang'] = output_lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ontology1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def labels_list(q: str, branch: str, children_of: str=None, children_of_relation: str=None, exclude_nss: str=None, exclude_tags: str=None, limit: str=None, word_breaks: str=None, is_class: str=None, lang: str='en', output_classifications: str=None, include_umbrella: str=None, exact_match: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of ontology labels that contains the input text.
		
		The input text will always appear in the results, making it suitable for most typeaheads applications.
		
		Normally this would be used in a typeahead to encourage people to enter something which already exists in the ontology.
		
		It may return duplicate concepts."
    q: format: any string

default value: ‘*’ (search for everything)

effect: performs a search for concepts that have labels which partially match this query string
        branch: format: comma separated list of branches to limit search to.

allowable branches: ‘occupation’, ‘specialization’, ‘function’, ‘skill’, ‘softskill’, ‘language’, ‘industry’, ‘contract_type’, ‘education’, ‘education2’, ‘authorization’, ‘filter__xx’ (where xx is the name of the filter, example filter__location_no

default value: none, and no search is performed. At least one branch must be specified.
        children_of: format: any string
default value: none

effect: search under a specific concept. The specific concept must exist with a label matching this value, in the provided branches, in language “lang”.

notes: This parameter is not compatible with the following branches and will be ignored: education, industry
        children_of_relation: format: one of: [‘and_descendants’, ‘descendants’, ‘and_children’, ‘children’]

default value: descendants

effect: used in conjunction with children_of. After matching concepts in children_of, search for q in the related concepts. and means include also the concept in children_of.
        exclude_nss: format: | separated list of “person class” concept labels

effect: does not include labels of concepts having a relation “not suitable for” where arg1 matches one of provided concepts.
        exclude_tags: format: character

default value: empty

effect: exclude labels which have this tag, such as ‘e’.
        limit: format: number of results to return, maximum 500

default value: 20
        word_breaks: format: string of characters, example %20,-*

default: ‘*’ - for compatibility, later will be changed to %20 (‘ ‘)

effect: only include labels where q is preceded by a word-break character in this string. ‘*’ allows any preceding characters, so searching for “er” will return “engineer”.
        is_class: search for concepts having a given classification. 

Use * for the value to indicate all concepts which have that classification name, regardless of the value

example: ?q=*&CLASS_BIS/AMS=647520 (normally you would set q=* when searching by specific classification ID)

example: ?q=painter&CLASS_BIS/AMS=*
list of available classifications:  [Classifications and Taxonomies](https://www.janzz.jobs/static/doc/apiv1/classifications.html#classifications-and-taxonomies)
        lang: format: ISO 369-1, 2 character language code, example: de, en, fr, … Choose ‘all’ to search in all languages

default value: ‘all’

effect: search for concept labels only in this language.
        output_classifications: format: | separated list of classifications

effect: return all classification values related to the concept_id of each returned value

list of available classifications:  [Classifications and Taxonomies](https://www.janzz.jobs/static/doc/apiv1/classifications.html#classifications-and-taxonomies)
        include_umbrella: format: true|false

default: false

effect: when set to false, it will not return labels of any umbrella concepts
        exact_match: format: true|false

default value: false

effect: search for this exact string. When false, search for q
        
    """
    url = f"https://ontology1.p.rapidapi.com/labels/"
    querystring = {'q': q, 'branch': branch, }
    if children_of:
        querystring['children_of'] = children_of
    if children_of_relation:
        querystring['children_of_relation'] = children_of_relation
    if exclude_nss:
        querystring['exclude_nss'] = exclude_nss
    if exclude_tags:
        querystring['exclude_tags'] = exclude_tags
    if limit:
        querystring['limit'] = limit
    if word_breaks:
        querystring['word_breaks'] = word_breaks
    if is_class:
        querystring['CLASS_'] = is_class
    if lang:
        querystring['lang'] = lang
    if output_classifications:
        querystring['output_classifications'] = output_classifications
    if include_umbrella:
        querystring['include_umbrella'] = include_umbrella
    if exact_match:
        querystring['exact_match'] = exact_match
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ontology1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def typeahead(branch: str, q: str, output_classifications: str=None, limit: str=None, is_class: str=None, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A more performant version of the labels list endpoint with some limitations. The main differences to /labels/ are:
		
		not all branches are available
		lang parameter is required
		exclude_tags is not needed because ‘e’ tags are always excluded and not configurable
		exact_match is not possible
		children_of and relations is not possible
		word_breaks are not configurable
		exclude_nss is not possible
		the results do not reflect the current state of the concept graph, they are up to 20 minutes delayed."
    branch: format: comma separated list of branches to limit search to.

allowable branches: occupation, function, skill, softskill, education, education level, specialization, language, industry, authorization

default value: none, and no serach is performed. At least one branch must be specified.
        q: format: any string

default value: ‘*’ (search for everything)

effect: performs a search for concepts that have labels which partially match this query string
        output_classifications: format: | separated list of classifications

effect: return all classification values related to the concept_id of each returned value

list of available classifications:  [Classifications and Taxonomies](https://www.janzz.jobs/static/doc/apiv1/classifications.html#classifications-and-taxonomies)
        limit: format: number of results to return, maximum 500

default value: 20
        is_class: search for concepts having a given classification. 

Use * for the value to indicate all concepts which have that classification name, regardless of the value

example: ?q=*&CLASS_BIS/AMS=647520 (normally you would set q=* when searching by specific classification ID)

example: ?q=painter&CLASS_BIS/AMS=*
list of available classifications:  [Classifications and Taxonomies](https://www.janzz.jobs/static/doc/apiv1/classifications.html#classifications-and-taxonomies)
        lang: format: ISO 369-1, 2 character language code, example: de, en, fr, …

default value: None, a language must be specified

effect: search for concept labels only in this language.
        
    """
    url = f"https://ontology1.p.rapidapi.com/typeahead/"
    querystring = {'branch': branch, 'q': q, }
    if output_classifications:
        querystring['output_classifications'] = output_classifications
    if limit:
        querystring['limit'] = limit
    if is_class:
        querystring['CLASS_'] = is_class
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ontology1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

