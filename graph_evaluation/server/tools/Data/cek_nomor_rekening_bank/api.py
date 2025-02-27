import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def daftar_bank(key_bank: str='bank_mandiri', nomor_rekening: str='1320022373493', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "key bank :
		- bank_mandiri
		- bank_bri
		- bank_bni
		- bank_bca
		- bank_danamon
		- bank_bsi
		- bank_permata
		- maybank_indonesia
		- bank_panin
		- bank_cimb_niaga
		- bank_kalbar
		- bank_kaltimtara
		- bpr_supra_artapersada
		- bpr_karyajatnika_sadaya
		- bangkok_bank
		- bank_j_trust_indonesia
		- bank_banten
		- bank_shinhan_indonesia
		- bank_mega_syariah
		- bank_artha_graha_internasional
		- bank_panin_dubai_syariah
		- bank_bumi_arta
		- bank_mestika_dharma
		- bank_mayapada
		- bpd_diy
		- jp_morgan_chase_bank_n.a.
		- bank_of_america
		- bank_aceh_syariah
		- bank_resona_perdania
		- bank_capital_indonesia
		- bank_bnp_paribas_indonesia
		- bank_dki
		- bank_jatim
		- bank_dbs_indonesia
		- bank_bjb
		- bank_jateng
		- bank_bca_syariah
		- bank_bjb_syariah
		- bank_of_china_(hong_kong)
		- bank_ntb_syariah
		- bank_bengkulu
		- bank_nusantara_parahyangan
		- bank_of_india_indonesia
		- bank_muamalat_indonesia
		- bank_btpn
		- bank_ibk_indonesia
		- bank_digital_bca
		- bank_raya_indonesia
		- bank_uob_indonesia
		- bank_hsbc_indonesia
		- bank_icbc_indonesia
		- bank_qnb_indonesia
		- bank_kb_bukopin
		- bank_bisnis_internasional
		- bank_jasa_jakarta
		- bank_mnc_internasional
		- bank_amar_indonesia
		- bank_ina_perdana
		- seabank_indonesia
		- bank_fama_internasional
		- bank_sahabat_sampoerna
		- bank_btn
		- citibank
		- bank_ganesha
		- bank_keb_hana
		- bank_woori_saudara
		- bank_nationalnobu
		- allo_bank_indonesia
		- bank_mandiri_taspen
		- bank_sumitomo_mitsui_indonesia
		- bank_mizuho_indonesia
		- standard_chartered_bank
		- bank_sumut
		- bank_lampung
		- bpd_kalteng
		- bank_sulselbar
		- bank_sulutgo
		- bank_maluku_malut
		- bank_sulteng
		- bank_sultra
		- bank_maspion_indonesia
		- bank_mega
		- bank_multiarta_sentosa
		- bank_ocbc_nisp
		- bank_ntt
		- bank_riau_kepri
		- bank_jago
		- bank_sinarmas
		- bank_victoria_syariah
		- bank_neo_commerce
		- bank_sbi_indonesia
		- prima_master_bank
		- bank_kb_bukopin_syariah
		- bank_oke_indonesia
		- bank_btpn_syariah
		- bank_mayora
		- bank_index_selindo
		- bank_victoria_international
		- bank_aladin_syariah
		- bank_ctbc_indonesia
		- bank_papua
		- bank_commonwealth
		- bpd_bali
		- bank_ccb_indonesia
		- mufg_bank
		- bank_anz_indonesia
		- deutsche_bank_ag.
		- bpd_jambi
		- bank_nagari
		- bank_sumsel_babel
		- bpd_kalsel"
    
    """
    url = f"https://cek-nomor-rekening-bank.p.rapidapi.com/check_bank_lq/{key_bank}/{nomor_rekening}"
    querystring = {}
    if key_bank:
        querystring['key_bank'] = key_bank
    if nomor_rekening:
        querystring['nomor_rekening'] = nomor_rekening
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cek-nomor-rekening-bank.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cek_nomor_rekening_digibank_dbs_indonesia(nomor_rekening: str='3320117753', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Untuk Cek Pemilik Nomor Rekening DIGIBANK (DBS) Indonesia
		
		
		More Info : [Beli Pulsa Murah](https://belipulsamurah.net/)"
    
    """
    url = f"https://cek-nomor-rekening-bank.p.rapidapi.com/check_bank_lq/bank_dbs_indonesia/{nomor_rekening}"
    querystring = {}
    if nomor_rekening:
        querystring['nomor_rekening'] = nomor_rekening
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cek-nomor-rekening-bank.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cek_nomor_rekening_bank_permata(nomor_rekening: str='1219619770', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Untuk Cek Pemilik Nomor Rekening Bank Permata
		
		
		More Info : [Beli Pulsa Murah](https://belipulsamurah.net/)"
    
    """
    url = f"https://cek-nomor-rekening-bank.p.rapidapi.com/check_bank_lq/bank_permata/{nomor_rekening}"
    querystring = {}
    if nomor_rekening:
        querystring['nomor_rekening'] = nomor_rekening
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cek-nomor-rekening-bank.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cek_nomor_rekening_bank_danamon(nomor_rekening: str='003522428212', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Untuk Cek Pemilik Nomor Rekening Bank Danamon
		
		
		More Info : [Beli Pulsa Murah](https://belipulsamurah.net/)"
    
    """
    url = f"https://cek-nomor-rekening-bank.p.rapidapi.com/check_bank_lq/bank_danamon/{nomor_rekening}"
    querystring = {}
    if nomor_rekening:
        querystring['nomor_rekening'] = nomor_rekening
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cek-nomor-rekening-bank.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cek_nomor_rekening_bri(nomor_rekening: str='412001011548536', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Untuk Cek Pemilik Nomor Rekening Bank BRI
		
		
		More Info : [Beli Pulsa Murah](https://belipulsamurah.net/)"
    
    """
    url = f"https://cek-nomor-rekening-bank.p.rapidapi.com/check_bank_lq/bank_bri/{nomor_rekening}"
    querystring = {}
    if nomor_rekening:
        querystring['nomor_rekening'] = nomor_rekening
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cek-nomor-rekening-bank.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cek_nomor_rekening_bni(nomor_rekening: str='1179192735', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Untuk Cek Pemilik Nomor Rekening Bank BNI
		
		
		More Info : [Beli Pulsa Murah](https://belipulsamurah.net/)"
    
    """
    url = f"https://cek-nomor-rekening-bank.p.rapidapi.com/check_bank_lq/bank_bni/{nomor_rekening}"
    querystring = {}
    if nomor_rekening:
        querystring['nomor_rekening'] = nomor_rekening
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cek-nomor-rekening-bank.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cek_nomor_rekening_btpn_jenius(nomor_rekening: str='90130058734', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Untuk Cek Pemilik Nomor Rekening Bank BTPN / Jenius
		
		
		More Info : [Beli Pulsa Murah](https://belipulsamurah.net/)"
    
    """
    url = f"https://cek-nomor-rekening-bank.p.rapidapi.com/check_bank_lq/bank_btpn/{nomor_rekening}"
    querystring = {}
    if nomor_rekening:
        querystring['nomor_rekening'] = nomor_rekening
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cek-nomor-rekening-bank.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cek_nomor_rekening_cimb_niaga(nomor_rekening: str='1770100731002', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Untuk Cek Pemilik Nomor Rekening Bank CIMB Niaga
		
		
		More Info : [Beli Pulsa Murah](https://belipulsamurah.net/)"
    
    """
    url = f"https://cek-nomor-rekening-bank.p.rapidapi.com/check_bank_lq/bank_cimb_niaga/{nomor_rekening}"
    querystring = {}
    if nomor_rekening:
        querystring['nomor_rekening'] = nomor_rekening
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cek-nomor-rekening-bank.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cek_nomor_rekening_bca(nomor_rekening: str='6042888890', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Untuk Cek Pemilik Nomor Rekening Bank BCA
		
		
		More Info : [Beli Pulsa Murah](https://belipulsamurah.net/)"
    
    """
    url = f"https://cek-nomor-rekening-bank.p.rapidapi.com/check_bank_lq/bank_bca/{nomor_rekening}"
    querystring = {}
    if nomor_rekening:
        querystring['nomor_rekening'] = nomor_rekening
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cek-nomor-rekening-bank.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cek_nomor_rekening_bank_mandiri(nomor_rekening: str='1840002783726', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Untuk Cek Pemilik Nomor Rekening Bank Mandiri
		
		
		More Info : [Beli Pulsa Murah](https://belipulsamurah.net/)"
    
    """
    url = f"https://cek-nomor-rekening-bank.p.rapidapi.com/check_bank_lq/bank_mandiri/{nomor_rekening}"
    querystring = {}
    if nomor_rekening:
        querystring['nomor_rekening'] = nomor_rekening
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cek-nomor-rekening-bank.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cek_nomor_rekening_btn(nomor_rekening: str='3401500929585', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Untuk Cek Pemilik Nomor Rekening Bank BTN
		
		
		More Info : [Beli Pulsa Murah](https://belipulsamurah.net/)"
    
    """
    url = f"https://cek-nomor-rekening-bank.p.rapidapi.com/check_bank_lq/bank_btn/{nomor_rekening}"
    querystring = {}
    if nomor_rekening:
        querystring['nomor_rekening'] = nomor_rekening
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cek-nomor-rekening-bank.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cek_nomor_rekening_bsi_indonesia(nomor_rekening: str='7179287735', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Untuk Cek Pemilik Nomor Rekening Bank BSI Indonesia
		
		
		More Info : [Beli Pulsa Murah](https://belipulsamurah.net/)"
    
    """
    url = f"https://cek-nomor-rekening-bank.p.rapidapi.com/check_bank_lq/bank_bsi/{nomor_rekening}"
    querystring = {}
    if nomor_rekening:
        querystring['nomor_rekening'] = nomor_rekening
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cek-nomor-rekening-bank.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

