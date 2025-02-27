import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def topics_2_0(lang: str, tt: str, txt: str, accept: str='application/json;', uw: str='n', rt: str='n', dm: str='s', sdg: str='l', timeref: str=None, of: str='json', txtf: str='plain', st: str='n', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Topics Extraction 2.0 is MeaningCloud's solution for extracting the different elements present in sources of information. This detection process is carried out by combining a number of complex natural language processing techniques that allow to obtain morphological, syntactic and semantic analyses of a text and use them to identify different types of significant elements."
    lang: It specifies the language in which the text must be analyzed.
        tt: The list of topic types to extract will be specified through a string with the letters assigned to each one of the topic types that are to be extracted.
        txt: Input text that's going to be analyzed.
        uw: Deal with unknown words. This feature adds a stage to the topic extraction in which the engine, much like a spellchecker, tries to find a suitable analysis to the unknown words resulted from the initial analysis assignment. It is specially useful to decrease the impact typos have in text analyses.
        rt: Deal with relaxed typography. This parameter indicates how reliable the text (as far as spelling, typography, etc. are concerned) to analyze is, and influences how strict the engine will be when it comes to take these factors into account in the topic extraction.
        dm: Type of disambiguation applied. It is accumulative, that is, the semantic disambiguation mode will also include morphosyntactic disambiguation
        sdg: Semantic disambiguation grouping. This parameter will only apply when semantic disambiguation is activated (dm=s).
        timeref: This value allows to set a specific time reference to detect the actual value of all the relative time expressions detected in the text. Format: YYYY-MM-DD hh:mm:ss GMT±HH:MM
        of: Output format.
        txtf: The text format parameter specifies if the text included in the txt parameter uses markup language that needs to be interpreted (known HTML tags and HTML code will be interpreted, and unknown tags will be ignored).
        st: Show subtopics.
        
    """
    url = f"https://topics-extraction.p.rapidapi.com/topics-2.0"
    querystring = {'lang': lang, 'tt': tt, 'txt': txt, }
    if accept:
        querystring['accept'] = accept
    if uw:
        querystring['uw'] = uw
    if rt:
        querystring['rt'] = rt
    if dm:
        querystring['dm'] = dm
    if sdg:
        querystring['sdg'] = sdg
    if timeref:
        querystring['timeref'] = timeref
    if of:
        querystring['of'] = of
    if txtf:
        querystring['txtf'] = txtf
    if st:
        querystring['st'] = st
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "topics-extraction.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

