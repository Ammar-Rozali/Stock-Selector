from requests_html import HTMLSession
import pandas as pd


url = 'https://www.bursamalaysia.com/market_information/shariah_compliant_equities_prices?top_stock=top_active&per_page=50&page=1'

session = HTMLSession()
r = session.get(url)
r.html.render(timeout=200)

dfs = pd.read_html(r.text)
print(len(dfs))
stock_list = dfs[0]

print(list(stock_list))

stock_list = stock_list[~stock_list.Name.str.contains("-WA")]
stock_list = stock_list[~stock_list.Name.str.contains("-WB")]
stock_list = stock_list[~stock_list.Name.str.contains("-WD")]

selected_sl = stock_list[['Name', 'LOW', 'Last Done', 'HIGH', 'Code', 'STOCK CODE']]

# hammer = selected_sl['LOW'] > selected_sl['Last Done']

# Find green only
hammer = selected_sl['LOW'] < selected_sl['Last Done']

df = selected_sl[hammer]

print(df)
total_hammer = 0
total_hammer_shooting_star = 0
link_list_filter = []
stock_name_list_filter = []
link_list = []
stock_name_list = []

for code in df.Code:
    print('################################################')
    print(code)
    link = f'https://www.bursamalaysia.com/trade/trading_resources/listing_directory/company-profile?stock_code={code}'

    r = session.get(link)

    r.html.render(timeout=200)
    dfs = pd.read_html(r.text)

    price = dfs[2].T
    price.columns = price.iloc[0]
    price = price[1:]

    print(price)
    # print(price[['Open']])

    LD = df[df['Code'].str.match(code)]
    close = LD[['Last Done']]
    print('close 2', close)
    stock_name = LD[['Name']]
    print('close', close['Last Done'].values)
    print('Open', price['Open'].values)

    ac = price['High'].astype(float) - price['Open'].astype(float)
    bc = price['Open'].astype(float) - price['Low'].astype(float)
    print('ac', ac.values, 'bc', bc.values)


    # Bullish
    if close['Last Done'].values > price['Open'].values:
        # Bearish
        # if close['Last Done'].values <  price['Open'].values:
        print('This is green candle', code)
        print(link)

        total_hammer += 1
        link_list.append(link)
        stock_name_list.append(stock_name)

        # Bullish
        # close = high price is hammer candel

        #if close['Last Done'].values == price['High'].values:
        if ac.values <= bc.values:
            # print('YES BOSS')
            # Bearish
            # if price['Open'].values == price['High'].values:

            total_hammer_shooting_star += 1
            link_list_filter.append(link)
            stock_name_list_filter.append(stock_name)
            print('This is hammer', code)


for one, two in zip(stock_name_list, link_list):
    print(str(one))
    print(two)

print('total green candle:', total_hammer)
print('total hammer candle:', total_hammer_shooting_star)

for one, two in zip(stock_name_list_filter, link_list_filter):
    print(str(one))
    print(two)