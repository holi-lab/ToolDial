import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_by_name_or_companyid(ruesiegefr: str=None, nomcommercialfr: str='Tunisair', notinstatuslist: str='EN_COURS_CREATION', limit: int=10, detailactivitefr: str=None, nomresponsablear: str=None, nomresponsablefr: str=None, ruesiegear: str=None, nomcommercialar: str=None, detailactivitear: str=None, nomsocietefr: str=None, idunique: str=None, anneedecreation: str=None, bureauregional: str=None, nomsocietear: str=None, etatregistre: str=None, identitepersonne: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search the Tunisian company register by company name or by company ID"
    
    """
    url = f"https://tunisia-company-data.p.rapidapi.com/rne-api/public/registres/pm"
    querystring = {}
    if ruesiegefr:
        querystring['rueSiegeFr'] = ruesiegefr
    if nomcommercialfr:
        querystring['nomCommercialFr'] = nomcommercialfr
    if notinstatuslist:
        querystring['notInStatusList'] = notinstatuslist
    if limit:
        querystring['limit'] = limit
    if detailactivitefr:
        querystring['detailActiviteFr'] = detailactivitefr
    if nomresponsablear:
        querystring['nomResponsableAr'] = nomresponsablear
    if nomresponsablefr:
        querystring['nomResponsableFr'] = nomresponsablefr
    if ruesiegear:
        querystring['rueSiegeAr'] = ruesiegear
    if nomcommercialar:
        querystring['nomCommercialAr'] = nomcommercialar
    if detailactivitear:
        querystring['detailActiviteAr'] = detailactivitear
    if nomsocietefr:
        querystring['nomSocieteFr'] = nomsocietefr
    if idunique:
        querystring['idUnique'] = idunique
    if anneedecreation:
        querystring['anneeDeCreation'] = anneedecreation
    if bureauregional:
        querystring['bureauRegional'] = bureauregional
    if nomsocietear:
        querystring['nomSocieteAr'] = nomsocietear
    if etatregistre:
        querystring['etatRegistre'] = etatregistre
    if identitepersonne:
        querystring['identitePersonne'] = identitepersonne
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tunisia-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_company_details_from_company_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get company details from the Tunisian Company identifier."
    
    """
    url = f"https://tunisia-company-data.p.rapidapi.com/rne-api/public/registres/pm/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tunisia-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

