import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def pesquisa_por_campo(campo: str, q: str, situacao: str='Ativa', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Busque uma lista de empresas utilizando outros campos, como:  **nome, email, telefone, socio, cep**.
		Campos retornados: CNPJ, localização, UF, Cidade, bairro, nome fantasia e razão social."
    campo: Informar um campo disponível para busca. Campos disponíveis: **nome, email, telefone, socio, cep.**
        situacao: Ativa, Inativa, Inapta ou Baixada
        
    """
    url = f"https://lista-de-empresas-por-segmento.p.rapidapi.com/buscar-por-segmento.php"
    querystring = {'campo': campo, 'q': q, }
    if situacao:
        querystring['situacao'] = situacao
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lista-de-empresas-por-segmento.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def detalhes_empresa(cnpj: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Detalhes da empresa utilizando o CNPJ. Dados retornados:
		CNPJ
		Razão social
		Nome fantasia
		Data criação
		Status empresa
		Natureza jurídica
		CNAE principal
		CNAEs secundários
		Porte empresa
		Endereço
		Telefone
		E-mail
		Quadro de sócios e administradores
		Participações em outras empresas"
    
    """
    url = f"https://lista-de-empresas-por-segmento.p.rapidapi.com/buscar-base.php"
    querystring = {'cnpj': cnpj, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lista-de-empresas-por-segmento.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cnae_ou_localiza_o(page: str, localizacao: str='PR', situacao: str='Ativa', cnae: str='0151201', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Busque uma lista de empresas filtrando pelo CNAE ou localização. Você pode filtrar usando apenas CNAE tendo abrangência nacional, filtrar usando apenas a localização, ou utilizar ambos os filtros combinados. Dados retornados da empresa:
		**CNPJ, endereço, UF, cidade, razão social e nome fantasia**."
    page: Paginação da lista, cada requisição retorna 10 registros por página.
        localizacao: Informar UF ou UF-Cidade
        situacao: Ativa, Suspensa, Inapta, Baixada
        cnae: Código CNAE do segmento da empresa. Você pode informar 1 código CNAE ou vários códigos CNAE. Para informar mais de 1 código CNAE use o separador pipeline (|). Ex:  6911701|4399101
        
    """
    url = f"https://lista-de-empresas-por-segmento.p.rapidapi.com/buscar-por-segmento.php"
    querystring = {'page': page, }
    if localizacao:
        querystring['localizacao'] = localizacao
    if situacao:
        querystring['situacao'] = situacao
    if cnae:
        querystring['cnae'] = cnae
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lista-de-empresas-por-segmento.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

