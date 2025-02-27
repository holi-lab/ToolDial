import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def return_both_real_time_price_of_gold_and_silver_for_specific_currency(cur: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return both real-time price of gold and silver for specific currency."
    cur: USD : United States Dollar
AED : United Arab Emirates Dirham
AFN : Afghan Afghani
ALL : Albanian Lek
AMD : Armenian Dram
ANG : Netherlands Antillean Guilder
AOA : Angolan Kwanza
ARS : Argentine Peso
AUD : Australian Dollar
AWG : Aruban Florin
AZN : Azerbaijani Manat
BAM : Bosnia-Herzegovina Convertible Mark
BBD : Barbadian Dollar
BDT : Bangladeshi Taka
BGN : Bulgarian Lev
BHD : Bahraini Dinar
BIF : Burundian Franc
BMD : Bermudan Dollar
BND : Brunei Dollar
BOB : Bolivian Boliviano
BRL : Brazilian Real
BSD : Bahamian Dollar
BTN : Bhutanese Ngultrum
BWP : Botswanan Pula
BYN : New Belarusian Ruble
BZD : Belize Dollar
CAD : Canadian Dollar
CDF : Congolese Franc
CHF : Swiss Franc
CLP : Chilean Peso
CNY : Chinese Yuan
COP : Colombian Peso
CRC : Costa Rican Colón
CUC : Cuban Convertible Peso
CUP : Cuban Peso
CVE : Cape Verdean Escudo
CZK : Czech Republic Koruna
DJF : Djiboutian Franc
DKK : Danish Krone
DOP : Dominican Peso
DZD : Algerian Dinar
EGP : Egyptian Pound
ERN : Eritrean Nakfa
ETB : Ethiopian Birr
EUR : Euro
FJD : Fijian Dollar
FKP : Falkland Islands Pound
GBP : British Pound Sterling
GEL : Georgian Lari
GGP : Guernsey Pound
GHS : Ghanaian Cedi
GIP : Gibraltar Pound
GMD : Gambian Dalasi
GNF : Guinean Franc
GTQ : Guatemalan Quetzal
GYD : Guyanaese Dollar
HKD : Hong Kong Dollar
HNL : Honduran Lempira
HRK : Croatian Kuna
HTG : Haitian Gourde
HUF : Hungarian Forint
IDR : Indonesian Rupiah
ILS : Israeli New Sheqel
IMP : Manx pound
INR : Indian Rupee
IQD : Iraqi Dinar
IRR : Iranian Rial
ISK : Icelandic Króna
JEP : Jersey Pound
JMD : Jamaican Dollar
JOD : Jordanian Dinar
JPY : Japanese Yen
KES : Kenyan Shilling
KGS : Kyrgystani Som
KHR : Cambodian Riel
KMF : Comorian Franc
KPW : North Korean Won
KRW : South Korean Won
KWD : Kuwaiti Dinar
KYD : Cayman Islands Dollar
KZT : Kazakhstani Tenge
LAK : Laotian Kip
LBP : Lebanese Pound
LKR : Sri Lankan Rupee
LRD : Liberian Dollar
LSL : Lesotho Loti
LYD : Libyan Dinar
MAD : Moroccan Dirham
MDL : Moldovan Leu
MGA : Malagasy Ariary
MKD : Macedonian Denar
MMK : Myanma Kyat
MNT : Mongolian Tugrik
MOP : Macanese Pataca
MRO : Mauritanian Ouguiya
MUR : Mauritian Rupee
MVR : Maldivian Rufiyaa
MWK : Malawian Kwacha
MXN : Mexican Peso
MYR : Malaysian Ringgit
MZN : Mozambican Metical
NAD : Namibian Dollar
NGN : Nigerian Naira
NIO : Nicaraguan Córdoba
NOK : Norwegian Krone
NPR : Nepalese Rupee
NZD : New Zealand Dollar
OMR : Omani Rial
PAB : Panamanian Balboa
PEN : Peruvian Nuevo Sol
PGK : Papua New Guinean Kina
PHP : Philippine Peso
PKR : Pakistani Rupee
PLN : Polish Zloty
PYG : Paraguayan Guarani
QAR : Qatari Rial
RON : Romanian Leu
RSD : Serbian Dinar
RUB : Russian Ruble
RWF : Rwandan Franc
SAR : Saudi Riyal
SBD : Solomon Islands Dollar
SCR : Seychellois Rupee
SDG : Sudanese Pound
SEK : Swedish Krona
SGD : Singapore Dollar
SHP : Saint Helena Pound
SLL : Sierra Leonean Leone
SOS : Somali Shilling
SRD : Surinamese Dollar
STD : São Tomé and Príncipe Dobra
SVC : Salvadoran Colón
SYP : Syrian Pound
SZL : Swazi Lilangeni
THB : Thai Baht
TJS : Tajikistani Somoni
TMT : Turkmenistani Manat
TND : Tunisian Dinar
TOP : Tongan Paʻanga
TRY : Turkish Lira
TTD : Trinidad and Tobago Dollar
TWD : New Taiwan Dollar
TZS : Tanzanian Shilling
UAH : Ukrainian Hryvnia
UGX : Ugandan Shilling
USD : United States Dollar
UYU : Uruguayan Peso
UZS : Uzbekistan Som
VEF : Venezuelan Bolívar Fuerte
VND : Vietnamese Dong
VUV : Vanuatu Vatu
WST : Samoan Tala
XAF : CFA Franc BEAC
XCD : East Caribbean Dollar
XOF : CFA Franc BCEAO
XPF : CFP Franc
YER : Yemeni Rial
ZAR : South African Rand
ZMW : Zambian Kwacha
        
    """
    url = f"https://gold-price6.p.rapidapi.com/GetGold"
    querystring = {'CUR': cur, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gold-price6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

