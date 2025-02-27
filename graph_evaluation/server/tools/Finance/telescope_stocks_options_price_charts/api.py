import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def stocks(modules: str, symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides an access to stock information"
    modules: **Available modules:** assetProfile, summaryProfile, summaryDetail, esgScores, price, incomeStatementHistory, incomeStatementHistoryQuarterly, balanceSheetHistory, balanceSheetHistoryQuarterly, cashflowStatementHistory, cashflowStatementHistoryQuarterly, defaultKeyStatistics, financialData, calendarEvents, secFilings, recommendationTrend, upgradeDowngradeHistory, institutionOwnership, fundOwnership, majorDirectHolders, majorHoldersBreakdown, insiderTransactions, insiderHolders, netSharePurchaseActivity, earnings, earningsHistory, earningsTrend, industryTrend, indexTrend, sectorTrend.
Pass the list of required modules separated by commas as the parameter value. 
**Example**: *?modules=assetProfile,
summaryProfile,price*.
        symbol: The symbol is a **stock exchange ticker**. Some exchanges also require you to specify a suffix. For example, for Mercedes-Benz Group AG, you need to specify the ticker MBG with the suffix .DE, i.e. */stocks/MBG.DE*.
Below is a list of **suffixes for various exchanges**.
№ / Country / Market, or Index / Suffix
0 / United States of America / Chicago Board of Trade (CBOT) / .CBT
1 / United States of America / Chicago Mercantile Exchange (CME) / .CME
2 / United States of America / Dow Jones Indexes / nan
3 / United States of America / Nasdaq Stock Exchange / nan
4 / United States of America / ICE Futures US / .NYB
5 / United States of America / New York Commodities Exchange (COMEX) / .CMX
6 / United States of America / New York Mercantile Exchange (NYMEX) / .NYM
7 / United States of America / Options Price Reporting Authority (OPRA) / nan
8 / United States of America / OTC Bulletin Board Market / nan
9 / United States of America / OTC Markets Group / nan
10 / United States of America / S & P Indices / nan
11 / Argentina / Buenos Aires Stock Exchange (BYMA) / .BA
12 / Austria / Vienna Stock Exchange / .VI
13 / Australia / Australian Stock Exchange (ASX) / .AX
14 / Belgium / Euronext Brussels / .BR
15 / Brazil / Sao Paolo Stock Exchange (BOVESPA) / .SA
16 / Canada / Canadian Securities Exchange / .CN
17 / Canada / NEO Exchange / .NE
18 / Canada / Toronto Stock Exchange (TSX) / .TO
19 / Canada / TSX Venture Exchange (TSXV) / .V
20 / Chile / Santiago Stock Exchange / .SN
21 / China / Shanghai Stock Exchange / .SS
22 / China / Shenzhen Stock Exchange / .SZ
23 / Czech Republic / Prague Stock Exchange Index / .PR
24 / Denmark / Nasdaq OMX Copenhagen / .CO
25 / Egypt / Egyptian Exchange Index (EGID) / .CA
26 / Estonia / Nasdaq OMX Tallinn / .TL
27 / Europe / Euronext / .NX
28 / Finland / Nasdaq OMX Helsinki / .HE
29 / France / Euronext Paris / .PA
30 / Germany / Berlin Stock Exchange / .BE
31 / Germany / Bremen Stock Exchange / .BM
32 / Germany / Dusseldorf Stock Exchange / .DU
33 / Germany / Frankfurt Stock Exchange / .F
34 / Germany / Hamburg Stock Exchange / .HM
35 / Germany / Hanover Stock Exchange / .HA
36 / Germany / Munich Stock Exchange / .MU
37 / Germany / Stuttgart Stock Exchange / .SG
38 / Germany / Deutsche Boerse XETRA / .DE
39 / Greece / Athens Stock Exchange (ATHEX) / .AT
40 / Hong Kong / Hong Kong Stock Exchange (HKEX)*** / .HK
41 / Hungary / Budapest Stock Exchange / .BD
42 / Iceland / Nasdaq OMX Iceland / .IC
43 / India / Bombay Stock Exchange / .BO
44 / India / National Stock Exchange of India / .NS
45 / Indonesia / Indonesia Stock Exchange (IDX) / .JK
46 / Ireland / Euronext Dublin / .IR
47 / Israel / Tel Aviv Stock Exchange / .TA
48 / Italy / EuroTLX / .TI
49 / Italy / Italian Stock Exchange / .MI
50 / Japan / Nikkei Indices / nan
51 / Japan / Tokyo Stock Exchange / .T
52 / Latvia / Nasdaq OMX Riga / .RG
53 / Lithuania / Nasdaq OMX Vilnius / .VS
54 / Malaysia / Malaysian Stock Exchange / .KL
55 / Mexico / Mexico Stock Exchange (BMV) / .MX
56 / Netherlands / Euronext Amsterdam / .AS
57 / New Zealand / New Zealand Stock Exchange (NZX) / .NZ
58 / Norway / Oslo Stock Exchange / .OL
59 / Portugal / Euronext Lisbon / .LS
60 / Qatar / Qatar Stock Exchange / .QA
61 / Russia / Moscow Exchange (MOEX) / .ME
62 / Singapore / Singapore Stock Exchange (SGX) / .SI
63 / South Africa / Johannesburg Stock Exchange / .JO
64 / South Korea / Korea Stock Exchange / .KS
65 / South Korea / KOSDAQ / .KQ
66 / Spain / Madrid SE C.A.T.S. / .MC
67 / Saudi Arabia / Saudi Stock Exchange (Tadawul) / .SAU
68 / Sweden / Nasdaq OMX Stockholm / .ST
69 / Switzerland / Swiss Exchange (SIX) / .SW
70 / Taiwan / Taiwan OTC Exchange / .TWO
71 / Taiwan / Taiwan Stock Exchange (TWSE) / .TW
72 / Thailand / Stock Exchange of Thailand (SET) / .BK
73 / Turkey / Borsa Istanbul / .IS
74 / United Kingdom / FTSE Indices / nan
75 / United Kingdom / London Stock Exchange / .L
76 / United Kingdom / London Stock Exchange / .IL
77 / Venezuela / Caracas Stock Exchange / .CR
        
    """
    url = f"https://telescope-stocks-options-price-charts.p.rapidapi.com/stocks/{symbol}"
    querystring = {'modules': modules, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "telescope-stocks-options-price-charts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def price(symbol: str, period2: str='1666411200', interval: str='1d', period1: str='1663911200', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides an access to share price history."
    symbol: The symbol is a **stock exchange ticker**. Some exchanges also require you to specify a suffix. For example, for Mercedes-Benz Group AG, you need to specify the ticker MBG with the suffix .DE, i.e. */stocks/MBG.DE*.
Below is a list of **suffixes for various exchanges**.
№ / Country / Market, or Index / Suffix
0 / United States of America / Chicago Board of Trade (CBOT) / .CBT
1 / United States of America / Chicago Mercantile Exchange (CME) / .CME
2 / United States of America / Dow Jones Indexes / nan
3 / United States of America / Nasdaq Stock Exchange / nan
4 / United States of America / ICE Futures US / .NYB
5 / United States of America / New York Commodities Exchange (COMEX) / .CMX
6 / United States of America / New York Mercantile Exchange (NYMEX) / .NYM
7 / United States of America / Options Price Reporting Authority (OPRA) / nan
8 / United States of America / OTC Bulletin Board Market / nan
9 / United States of America / OTC Markets Group / nan
10 / United States of America / S & P Indices / nan
11 / Argentina / Buenos Aires Stock Exchange (BYMA) / .BA
12 / Austria / Vienna Stock Exchange / .VI
13 / Australia / Australian Stock Exchange (ASX) / .AX
14 / Belgium / Euronext Brussels / .BR
15 / Brazil / Sao Paolo Stock Exchange (BOVESPA) / .SA
16 / Canada / Canadian Securities Exchange / .CN
17 / Canada / NEO Exchange / .NE
18 / Canada / Toronto Stock Exchange (TSX) / .TO
19 / Canada / TSX Venture Exchange (TSXV) / .V
20 / Chile / Santiago Stock Exchange / .SN
21 / China / Shanghai Stock Exchange / .SS
22 / China / Shenzhen Stock Exchange / .SZ
23 / Czech Republic / Prague Stock Exchange Index / .PR
24 / Denmark / Nasdaq OMX Copenhagen / .CO
25 / Egypt / Egyptian Exchange Index (EGID) / .CA
26 / Estonia / Nasdaq OMX Tallinn / .TL
27 / Europe / Euronext / .NX
28 / Finland / Nasdaq OMX Helsinki / .HE
29 / France / Euronext Paris / .PA
30 / Germany / Berlin Stock Exchange / .BE
31 / Germany / Bremen Stock Exchange / .BM
32 / Germany / Dusseldorf Stock Exchange / .DU
33 / Germany / Frankfurt Stock Exchange / .F
34 / Germany / Hamburg Stock Exchange / .HM
35 / Germany / Hanover Stock Exchange / .HA
36 / Germany / Munich Stock Exchange / .MU
37 / Germany / Stuttgart Stock Exchange / .SG
38 / Germany / Deutsche Boerse XETRA / .DE
39 / Greece / Athens Stock Exchange (ATHEX) / .AT
40 / Hong Kong / Hong Kong Stock Exchange (HKEX)*** / .HK
41 / Hungary / Budapest Stock Exchange / .BD
42 / Iceland / Nasdaq OMX Iceland / .IC
43 / India / Bombay Stock Exchange / .BO
44 / India / National Stock Exchange of India / .NS
45 / Indonesia / Indonesia Stock Exchange (IDX) / .JK
46 / Ireland / Euronext Dublin / .IR
47 / Israel / Tel Aviv Stock Exchange / .TA
48 / Italy / EuroTLX / .TI
49 / Italy / Italian Stock Exchange / .MI
50 / Japan / Nikkei Indices / nan
51 / Japan / Tokyo Stock Exchange / .T
52 / Latvia / Nasdaq OMX Riga / .RG
53 / Lithuania / Nasdaq OMX Vilnius / .VS
54 / Malaysia / Malaysian Stock Exchange / .KL
55 / Mexico / Mexico Stock Exchange (BMV) / .MX
56 / Netherlands / Euronext Amsterdam / .AS
57 / New Zealand / New Zealand Stock Exchange (NZX) / .NZ
58 / Norway / Oslo Stock Exchange / .OL
59 / Portugal / Euronext Lisbon / .LS
60 / Qatar / Qatar Stock Exchange / .QA
61 / Russia / Moscow Exchange (MOEX) / .ME
62 / Singapore / Singapore Stock Exchange (SGX) / .SI
63 / South Africa / Johannesburg Stock Exchange / .JO
64 / South Korea / Korea Stock Exchange / .KS
65 / South Korea / KOSDAQ / .KQ
66 / Spain / Madrid SE C.A.T.S. / .MC
67 / Saudi Arabia / Saudi Stock Exchange (Tadawul) / .SAU
68 / Sweden / Nasdaq OMX Stockholm / .ST
69 / Switzerland / Swiss Exchange (SIX) / .SW
70 / Taiwan / Taiwan OTC Exchange / .TWO
71 / Taiwan / Taiwan Stock Exchange (TWSE) / .TW
72 / Thailand / Stock Exchange of Thailand (SET) / .BK
73 / Turkey / Borsa Istanbul / .IS
74 / United Kingdom / FTSE Indices / nan
75 / United Kingdom / London Stock Exchange / .L
76 / United Kingdom / London Stock Exchange / .IL
77 / Venezuela / Caracas Stock Exchange / .CR
        period2: The **period2** parameter is a UNIX timestam that specifies the end of the interval to be uploaded.
For example, *?period2=1666411200*  (2022-10-22 04:00:00).
        interval: The **interval** parameter defines the step for the requested time series. 
This parameter takes the following values: 1m, 5m, 15m, 30m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo. For example, *?interval=1d* means that the data will be uploaded at 1 day intervals.
        period1: The **period1** parameter is a UNIX timestam that specifies the start of the interval to be uploaded.
For example, *?period1=1679011200*  (2022-09-23 05:33:20).
        
    """
    url = f"https://telescope-stocks-options-price-charts.p.rapidapi.com/price/{symbol}"
    querystring = {}
    if period2:
        querystring['period2'] = period2
    if interval:
        querystring['interval'] = interval
    if period1:
        querystring['period1'] = period1
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "telescope-stocks-options-price-charts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def options(symbol: str, date: str='1679011200', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides an access to options information"
    symbol: The symbol is a **stock exchange ticker**. Some exchanges also require you to specify a suffix. For example, for Mercedes-Benz Group AG, you need to specify the ticker MBG with the suffix .DE, i.e. */stocks/MBG.DE*.
Below is a list of **suffixes for various exchanges**.
№ / Country / Market, or Index / Suffix
0 / United States of America / Chicago Board of Trade (CBOT) / .CBT
1 / United States of America / Chicago Mercantile Exchange (CME) / .CME
2 / United States of America / Dow Jones Indexes / nan
3 / United States of America / Nasdaq Stock Exchange / nan
4 / United States of America / ICE Futures US / .NYB
5 / United States of America / New York Commodities Exchange (COMEX) / .CMX
6 / United States of America / New York Mercantile Exchange (NYMEX) / .NYM
7 / United States of America / Options Price Reporting Authority (OPRA) / nan
8 / United States of America / OTC Bulletin Board Market / nan
9 / United States of America / OTC Markets Group / nan
10 / United States of America / S & P Indices / nan
11 / Argentina / Buenos Aires Stock Exchange (BYMA) / .BA
12 / Austria / Vienna Stock Exchange / .VI
13 / Australia / Australian Stock Exchange (ASX) / .AX
14 / Belgium / Euronext Brussels / .BR
15 / Brazil / Sao Paolo Stock Exchange (BOVESPA) / .SA
16 / Canada / Canadian Securities Exchange / .CN
17 / Canada / NEO Exchange / .NE
18 / Canada / Toronto Stock Exchange (TSX) / .TO
19 / Canada / TSX Venture Exchange (TSXV) / .V
20 / Chile / Santiago Stock Exchange / .SN
21 / China / Shanghai Stock Exchange / .SS
22 / China / Shenzhen Stock Exchange / .SZ
23 / Czech Republic / Prague Stock Exchange Index / .PR
24 / Denmark / Nasdaq OMX Copenhagen / .CO
25 / Egypt / Egyptian Exchange Index (EGID) / .CA
26 / Estonia / Nasdaq OMX Tallinn / .TL
27 / Europe / Euronext / .NX
28 / Finland / Nasdaq OMX Helsinki / .HE
29 / France / Euronext Paris / .PA
30 / Germany / Berlin Stock Exchange / .BE
31 / Germany / Bremen Stock Exchange / .BM
32 / Germany / Dusseldorf Stock Exchange / .DU
33 / Germany / Frankfurt Stock Exchange / .F
34 / Germany / Hamburg Stock Exchange / .HM
35 / Germany / Hanover Stock Exchange / .HA
36 / Germany / Munich Stock Exchange / .MU
37 / Germany / Stuttgart Stock Exchange / .SG
38 / Germany / Deutsche Boerse XETRA / .DE
39 / Greece / Athens Stock Exchange (ATHEX) / .AT
40 / Hong Kong / Hong Kong Stock Exchange (HKEX)*** / .HK
41 / Hungary / Budapest Stock Exchange / .BD
42 / Iceland / Nasdaq OMX Iceland / .IC
43 / India / Bombay Stock Exchange / .BO
44 / India / National Stock Exchange of India / .NS
45 / Indonesia / Indonesia Stock Exchange (IDX) / .JK
46 / Ireland / Euronext Dublin / .IR
47 / Israel / Tel Aviv Stock Exchange / .TA
48 / Italy / EuroTLX / .TI
49 / Italy / Italian Stock Exchange / .MI
50 / Japan / Nikkei Indices / nan
51 / Japan / Tokyo Stock Exchange / .T
52 / Latvia / Nasdaq OMX Riga / .RG
53 / Lithuania / Nasdaq OMX Vilnius / .VS
54 / Malaysia / Malaysian Stock Exchange / .KL
55 / Mexico / Mexico Stock Exchange (BMV) / .MX
56 / Netherlands / Euronext Amsterdam / .AS
57 / New Zealand / New Zealand Stock Exchange (NZX) / .NZ
58 / Norway / Oslo Stock Exchange / .OL
59 / Portugal / Euronext Lisbon / .LS
60 / Qatar / Qatar Stock Exchange / .QA
61 / Russia / Moscow Exchange (MOEX) / .ME
62 / Singapore / Singapore Stock Exchange (SGX) / .SI
63 / South Africa / Johannesburg Stock Exchange / .JO
64 / South Korea / Korea Stock Exchange / .KS
65 / South Korea / KOSDAQ / .KQ
66 / Spain / Madrid SE C.A.T.S. / .MC
67 / Saudi Arabia / Saudi Stock Exchange (Tadawul) / .SAU
68 / Sweden / Nasdaq OMX Stockholm / .ST
69 / Switzerland / Swiss Exchange (SIX) / .SW
70 / Taiwan / Taiwan OTC Exchange / .TWO
71 / Taiwan / Taiwan Stock Exchange (TWSE) / .TW
72 / Thailand / Stock Exchange of Thailand (SET) / .BK
73 / Turkey / Borsa Istanbul / .IS
74 / United Kingdom / FTSE Indices / nan
75 / United Kingdom / London Stock Exchange / .L
76 / United Kingdom / London Stock Exchange / .IL
77 / Venezuela / Caracas Stock Exchange / .CR
        date: The date parameter is a **UNIX timestamp** that defines the option's expiration date in the future. For example, *?date=1679011200* (March 17, 2023 expiration).
        
    """
    url = f"https://telescope-stocks-options-price-charts.p.rapidapi.com/options/{symbol}"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "telescope-stocks-options-price-charts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

