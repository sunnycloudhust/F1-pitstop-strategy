import pandas as pd


drivers = pd.read_csv('../formula-1-race-data/versions/116/drivers.csv')
circuits = pd.read_csv('../formula-1-race-data/versions/116/circuits.csv')
races = pd.read_csv('../formula-1-race-data/versions/116/races.csv')
results = pd.read_csv('../formula-1-race-data/versions/116/results.csv')        


print(drivers.head())
print(circuits.head())
print(races.head())
print(results.head())

comparing_drivers = drivers[drivers['surname'].isin(['Hamilton', 'Verstappen', 'Alonso'])]
print(comparing_drivers)

results_with_drivers = results.merge(comparing_drivers, on='driverId')
print(results_with_drivers.head())