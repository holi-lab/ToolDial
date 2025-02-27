import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def balanco_ticker_ano_tri(ano_tri: str, ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "balanco-ticker-ano_tri"
    
    """
    url = f"https://info_acoes_bolsa_brasil_b3.p.rapidapi.com/balanco-ticker-ano_tri/"
    querystring = {'ano_tri': ano_tri, 'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "info_acoes_bolsa_brasil_b3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def relatorio_financeiro_ticker_ano_tri(ano_tri: str, ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "relatorio_financeiro-ticker-ano_tri"
    
    """
    url = f"https://info_acoes_bolsa_brasil_b3.p.rapidapi.com/relatorio_financeiro-ticker-ano_tri/"
    querystring = {'ano_tri': ano_tri, 'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "info_acoes_bolsa_brasil_b3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def empresa_passado_ticker_ano_tri(ano_tri: str, ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "empresa_passado-ticker-ano_tri"
    
    """
    url = f"https://info_acoes_bolsa_brasil_b3.p.rapidapi.com/empresa_passado-ticker-ano_tri/"
    querystring = {'ano_tri': ano_tri, 'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "info_acoes_bolsa_brasil_b3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def empresa_atual_ticker_ano_tri(ticker: str, ano_tri: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "empresa_atual-ticker-ano_tri"
    
    """
    url = f"https://info_acoes_bolsa_brasil_b3.p.rapidapi.com/empresa_atual-ticker-ano_tri/"
    querystring = {'ticker': ticker, 'ano_tri': ano_tri, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "info_acoes_bolsa_brasil_b3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def indicadores_ticker_data_base(ticker: str, data_val: str='2021-07-30', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "indicadores-ticker-data_base"
    
    """
    url = f"https://info_acoes_bolsa_brasil_b3.p.rapidapi.com/indicadores-ticker-data_base/"
    querystring = {'ticker': ticker, }
    if data_val:
        querystring['data_val'] = data_val
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "info_acoes_bolsa_brasil_b3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def preco_ticker_data_base(ticker: str, data_val: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint que pega o preço histórico da empresa através do ticker e uma data base."
    
    """
    url = f"https://info_acoes_bolsa_brasil_b3.p.rapidapi.com/preco-ticker-data_base/"
    querystring = {'ticker': ticker, 'data_val': data_val, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "info_acoes_bolsa_brasil_b3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def preco_corrigido_ticker_data_base(ticker: str, data_val: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "preco-corrigido-ticker-data_base"
    
    """
    url = f"https://info_acoes_bolsa_brasil_b3.p.rapidapi.com/preco-corrigido-ticker-data_base/"
    querystring = {'ticker': ticker, 'data_val': data_val, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "info_acoes_bolsa_brasil_b3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def crescimento_ticker(ticker: str, tipo: str='trimestral', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "crescimento-ticker"
    
    """
    url = f"https://info_acoes_bolsa_brasil_b3.p.rapidapi.com/crescimento-ticker/"
    querystring = {'ticker': ticker, }
    if tipo:
        querystring['tipo'] = tipo
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "info_acoes_bolsa_brasil_b3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

