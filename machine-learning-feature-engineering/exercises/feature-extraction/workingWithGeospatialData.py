import pandas as pd
import matplotlib.pyplot as plt
import geopandas
import descartes

pd.options.display.max_rows = 10

df = pd.DataFrame(
    {
        'City': ['Buenos Aires', 'Brasilia', 'Santiago', 'Bogota', 'Caracas'],
        'Country': ['Argentina', 'Brazil', ' Chile', 'Colombia', ' Venezuela'],
        'Latitude': [-34.58, -15.78, -33.45, 4.60, 10.48],
        'Longitude': [-58.66, -47.91, -70.66, -74.08, -66.86]
    }
)

print(df.head())

gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude))
print(gdf.head())

# world = geopandas.read_file(geopandas.dataset.get_path('naturalearth_lowres'))
# ax = world[world.continent == 'South America'].plot(color='white', edgecolor='black')
# gdf.plot(ax=ax, color='red')
# plt.show()

countries = geopandas.read_file('../geographic/ne_110m_admin_0_countries.shx')
print(countries.head())
print(type(countries))

# visualization
# countries.plot(figsize=(15, 8))
# plt.show()

cities = geopandas.read_file('zip://../geographic/ne_10m_populated_places.zip')
print(type(cities))
print(cities.tail(5))

# get 2 cities in index 0 and index 241
two_cities = cities.iloc[[0, 241]]
print(two_cities)

# two_cities.plot(color='r')
# plt.show()

# plot the two cities on the map
ax = countries.plot(figsize=(12, 6))
two_cities.plot(ax=ax, color='r')
plt.show()
