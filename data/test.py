import pandas as pd
import numpy as np

# ============================================================
# LOAD DATA
# ============================================================

dir_carbon = r'/Users/mohsenasghariilani/Downloads/anticipatory-energy-stress/data/raw/Carbon Emissions Futures Historical Data UK.csv'

dir_gas = r'/Users/mohsenasghariilani/Downloads/anticipatory-energy-stress/data/raw/gas.csv'

dir_elec = r'/Users/mohsenasghariilani/Downloads/anticipatory-energy-stress/data/raw/electricity.csv'
dir_gdp = r'/Users/mohsenasghariilani/Downloads/anticipatory-energy-stress/data/raw/mgdp.csv'

dir_inflation = r'/Users/mohsenasghariilani/Downloads/anticipatory-energy-stress/data/raw/cpih08_188.xlsx'

dir_weather = r'/Users/mohsenasghariilani/Downloads/anticipatory-energy-stress/data/raw/monthly-temperature-anomalies.csv'


carbon = pd.read_csv(dir_carbon)
gas = pd.read_csv(dir_gas)
elec = pd.read_csv(dir_elec)
gdp = pd.read_csv(dir_gdp)
inflation = pd.read_excel(dir_inflation, header=None)
weather = pd.read_csv(dir_weather)


# ============================================================
# WEATHER
# ============================================================

weather = weather[
    weather['Entity'] == 'United Kingdom'
].copy()

weather['date'] = pd.to_datetime(weather['Day'])

weather['date'] = (
    weather['date']
    .dt.to_period('M')
    .dt.to_timestamp()
)

weather = weather.rename(
    columns={
        'Temperature anomaly': 'temp_anomaly'
    }
)

weather = weather[
    ['date', 'temp_anomaly']
]

weather = weather.sort_values('date')

# rolling volatility
weather['weather_volatility'] = (
    weather['temp_anomaly']
    .rolling(window=12)
    .std()
)


# ============================================================
# CARBON
# raw price -> log returns
# ============================================================

carbon['date'] = pd.to_datetime(
    carbon['Date'],
    dayfirst=True
)

carbon = carbon.sort_values('date')

carbon['Price'] = (
    carbon['Price']
    .astype(str)
    .str.replace(',', '')
    .astype(float)
)

# LOG RETURN
carbon['carbon_return'] = np.log(
    carbon['Price'] /
    carbon['Price'].shift(1)
)

carbon = carbon[
    ['date', 'Price', 'carbon_return']
]

carbon = carbon.rename(
    columns={
        'Price': 'carbon_price'
    }
)


# ============================================================
# GAS
# YoY % -> convert to MoM %
# ============================================================

gas = gas[
    gas['Title'].str.match(
        r'^\d{4} [A-Z]{3}$',
        na=False
    )
].copy()

gas['date'] = pd.to_datetime(
    gas['Title'],
    format='%Y %b'
)

gas_col = gas.columns[1]

gas[gas_col] = pd.to_numeric(
    gas[gas_col],
    errors='coerce'
)

gas = gas.rename(
    columns={
        gas_col: 'gas_yoy'
    }
)

# OPTIONAL:
# convert YoY proxy to smoother monthly dynamics

gas['gas_mom_proxy'] = (
    gas['gas_yoy'] / 12
)

gas = gas[
    ['date', 'gas_yoy', 'gas_mom_proxy']
]


# ============================================================
# ELECTRICITY
# CPI INDEX -> monthly growth
# ============================================================

elec = elec[
    elec['Title'].str.match(
        r'^\d{4} [A-Z]{3}$',
        na=False
    )
].copy()

elec['date'] = pd.to_datetime(
    elec['Title'],
    format='%Y %b'
)

elec_col = elec.columns[1]

elec[elec_col] = pd.to_numeric(
    elec[elec_col],
    errors='coerce'
)

elec = elec.rename(
    columns={
        elec_col: 'electricity_index'
    }
)

# MONTHLY GROWTH
elec['electricity_growth'] = (
    elec['electricity_index']
    .pct_change() * 100
)

elec = elec[
    ['date', 'electricity_index', 'electricity_growth']
]


# ============================================================
# CPIH / INFLATION CLEANING
# ============================================================

inflation_raw = pd.read_excel(
    dir_inflation,
    header=None
)

# Row 2 contains dates
date_row = inflation_raw.iloc[2]

# Row 3 contains UK values
value_row = inflation_raw.iloc[3]

dates = []
values = []

for i in range(len(date_row)):

    dt = pd.to_datetime(
        str(date_row.iloc[i]),
        format='%b-%y',
        errors='coerce'
    )

    if pd.notna(dt):

        val = pd.to_numeric(
            value_row.iloc[i],
            errors='coerce'
        )

        dates.append(dt)
        values.append(val)

inflation_clean = pd.DataFrame({
    'date': dates,
    'cpih_index': values
})

inflation_clean['date'] = (
    inflation_clean['date']
    .dt.to_period('M')
    .dt.to_timestamp()
)

inflation_clean = inflation_clean.sort_values('date')

# CPIH -> monthly inflation growth
inflation_clean['inflation_growth'] = (
    inflation_clean['cpih_index']
    .pct_change(fill_method=None) * 100
)

print(inflation_clean.head())
print(inflation_clean.tail())

print("\nMissing Values:")
print(inflation_clean.isna().sum())

# ============================================================
# GDP
# already MoM growth
# ============================================================

gdp = gdp[
    gdp['Title'].str.match(
        r'^\d{4} [A-Z]{3}$',
        na=False
    )
].copy()

gdp['date'] = pd.to_datetime(
    gdp['Title'],
    format='%Y %b'
)

gdp_col = (
    'Gross Value Added - Monthly '
    '(period on period growth) :CVM SA'
)

gdp[gdp_col] = pd.to_numeric(
    gdp[gdp_col],
    errors='coerce'
)

gdp = gdp.rename(
    columns={
        gdp_col: 'gdp_growth'
    }
)

gdp = gdp[
    ['date', 'gdp_growth']
]


# ============================================================
# MERGE ALL DATASETS
# ============================================================

dfs = [
    gas,
    elec,
    carbon,
    inflation_clean,
    gdp,
    weather
]

from functools import reduce

merged = reduce(
    lambda left, right:
    pd.merge(
        left,
        right,
        on='date',
        how='outer'
    ),
    dfs
)

merged = merged.sort_values('date')


# ============================================================
# RESTRICT PERIOD
# IMPORTANT:
# carbon begins much later
# ============================================================

merged = merged[
    (merged['date'] >= '2005-01-01') &
    (merged['date'] <= '2017-12-01')
]


# ============================================================
# FINAL CLEANING
# ============================================================

merged = merged.reset_index(drop=True)

print(merged.head())

print("\n")
print(merged.describe())

print("\nMissing Values:")
print(merged.isna().sum())