import requests
import matplotlib.pyplot as plt

api_key = "EMx4kcStc3lOf2APX89nVsYGRevSBLn9"
company = "FB"
years = 2

url = f"https://financialmodelingprep.com/api/v3/financials/income-statement/{company}?period=annual&limit={years}&apikey={api_key}"

response = requests.get(url)
if response.status_code == 200:
    income_statement = response.json()['financials']

    revenues = list(reversed([int(income_statement[i]['Revenue']) for i in range(len(income_statement))]))
    profits = list(reversed([int(income_statement[i]['Gross Profit']) for i in range(len(income_statement))]))

    plt.plot(revenues, label='Revenues')
    plt.plot(profits, label='Profits')
    plt.xlabel('Year')
    plt.ylabel('Amount (USD)')
    plt.title(f'{company} Revenue and Profit Over Last {years} Years')
    plt.legend(loc='upper left')
    plt.show()
else:
    print("Failed to fetch data. Please check your API key and company symbol.")
