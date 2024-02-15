# %%
import pandas as pd
import folium
import re

def plot_grand_prix():
    pd.set_option('display.precision', 2)

    file_path = '/Users/pal/Desktop/senior_design/code/app/data/purdue_grand_prix.csv'

    with open(file_path, 'r') as fid:
        data = [line for line in fid.readlines()]

    header = data[0]

    lat_lon = re.findall(r'-*\d+.\d+ -*\d+.\d+', data[1])
    lat_lon = [data.split(' ') for data in lat_lon]
    lon = [float(data[0]) for data in lat_lon]
    lat = [float(data[1]) for data in lat_lon]

    df_route = pd.DataFrame({
        'lat': lat,
        'lon': lon
    })

    avg_location = df_route[['lat', 'lon']].mean()
    map_grand_prix = folium.Map(location=(avg_location[0], avg_location[1]), zoom_start=13)

    df_route.index.name = 'visit_order'

    df_route_segments = df_route.join(
        df_route.shift(-1),  # map each stop to its next stop
        rsuffix='_next'
    ).dropna()  # last stop has no "next one", so drop it

    map_grand_prix = folium.Map(location=avg_location, zoom_start=18)

    for stop in df_route_segments.itertuples():
        # print(stop)
        # marker for current stop
        marker = folium.Marker(location=(stop.lat, stop.lon))
        # line for the route segment connecting current to next stop
        line = folium.PolyLine(
            locations=[(stop.lat, stop.lon), 
                    (stop.lat_next, stop.lon_next)],
            # tooltip=f"{stop.site} to {stop.site_next}",
        )
        # add elements to the map
        marker.add_to(map_grand_prix)
        line.add_to(map_grand_prix)

    # maker for last stop wasn't added in for loop, so adding it now 
    folium.Marker(location=(stop.lat_next, stop.lon_next)).add_to(map_grand_prix)

    map_grand_prix.save('./data/grand_prix.html')

    return map_grand_prix

# %%
if __name__ == '__main__':
    plot_grand_prix()