import streamlit as st
import pandas as pd
import requests
import time
import pdfplumber
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import folium
from streamlit_folium import folium_static
from shapely.geometry import Polygon, Point
import random


# Load API Key
if "API_KEY" in st.secrets:
    API_KEY = st.secrets["API_KEY"]
else:
    st.error("⚠️ API Key not found! Make sure it's in `~/.streamlit/secrets.toml`")

# --- Streamlit UI ---
st.title("🚪 Door Knock Route Optimizer")

# Option 1: Upload a file
uploaded_file = st.file_uploader("📤 Upload Addresses TXT File", type="txt")

# Option 2: Manual Address Entry
manual_addresses = st.text_area("📌 Or, enter addresses manually (one per line)", "")

addresses = []

if uploaded_file is not None:
    addresses = uploaded_file.read().decode("utf-8").splitlines()
elif manual_addresses.strip():
    addresses = manual_addresses.strip().split("\n")

addresses = [addr.strip() for addr in addresses if addr.strip()]  # Clean empty lines

if not addresses:
    st.warning("⚠️ Please upload a file or enter addresses manually.")
else:
    st.success(f"✅ {len(addresses)} addresses loaded!")

    # --- Step 2: Convert Address to Coordinates ---
    def get_coordinates(address):
        full_address = f"{address}, Metro Vancouver, BC, Canada"  # Add more location context
        url = f'https://maps.googleapis.com/maps/api/geocode/json?address={full_address}&key={API_KEY}'
        
        try:
            response = requests.get(url).json()
            if response['status'] == 'OK':
                location = response['results'][0]['geometry']['location']
                return location['lat'], location['lng']
            else:
                st.warning(f"Geocoding failed for {address}: {response['status']}")
                return None, None  
        except Exception as e:
            st.error(f"Error during geocoding {address}: {e}")
            return None, None  



    coordinates = []
    valid_addresses = []

    for address in addresses:
        coords = get_coordinates(address)
        if coords:
            coordinates.append(coords)
            valid_addresses.append(address)
            st.write(f"📍 {address} → {coords}")
        time.sleep(0.5)  # Prevent API rate limiting

    # --- Step 3: Extract TOD Stations from PDF ---
    tod_pdf_path = "tod_areas_june_30_2024.pdf"

    def extract_tod_stations(pdf_path):
        tod_stations = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    lines = text.split("\n")
                    for line in lines:
                        if "List of transit stations" in line:
                            start_extracting = True
                            continue
                        if start_extracting and line.strip():
                            tod_stations.append(line.strip())
        return tod_stations

    tod_stations = extract_tod_stations(tod_pdf_path)
    
    # Fetch TOD station coordinates
    tod_coordinates = []
    for station in tod_stations:
        lat, lng = get_coordinates(station)
        if lat is not None and lng is not None:
            tod_coordinates.append((lat, lng))  # Only add valid coordinates
        else:
            st.write(f"🚫 Skipping {station}, no coordinates found.")  # Avoid crash
        time.sleep(0.5)



    if len(coordinates) < 2:
        st.error("❌ Not enough valid addresses.")
    else:
        # --- Step 4: Avoidance Zones (Railways & Cemeteries) ---

        def get_places_nearby(location, place_type, radius=5000, max_retries=3):
            url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            params = {
                "location": f"{location[0]},{location[1]}",
                "radius": radius,
                "type": place_type,
                "key": API_KEY
            }

            for attempt in range(max_retries):
                try:
                    response = requests.get(url, params=params, timeout=10).json()
                    if response.get("status") == "OK":
                        return [(p["geometry"]["location"]["lat"], p["geometry"]["location"]["lng"]) for p in response["results"]]
                    elif response.get("status") == "OVER_QUERY_LIMIT":
                        st.warning("⚠️ Google API Rate Limit reached. Waiting before retrying...")
                        time.sleep(2 + random.uniform(0, 2))  # Wait before retrying
                    else:
                        break  # Exit loop if it's another error
                except requests.exceptions.RequestException as e:
                    st.error(f"API request failed: {e}")
                    time.sleep(2 + random.uniform(0, 2))  # Wait before retrying

            return []  # Return empty list if all retries fail


        def is_in_avoid_zone(lat, lng, polygon):
            if polygon:
                return polygon.contains(Point(lat, lng))
            return False

        filtered_coordinates = []
        filtered_addresses = []
        railway_locations = []
        cemetery_locations = []

        for i, coords in enumerate(coordinates):
            railways = get_places_nearby(coords, "train_station")
            cemeteries = get_places_nearby(coords, "cemetery")

            railway_polygon = Polygon(railways) if railways else None
            cemetery_polygon = Polygon(cemeteries) if cemeteries else None

            if is_in_avoid_zone(coords[0], coords[1], railway_polygon) or is_in_avoid_zone(coords[0], coords[1], cemetery_polygon):
                st.write(f"🚫 Skipped {valid_addresses[i]} (Near Railway or Cemetery)")
            else:
                filtered_coordinates.append(coords)
                filtered_addresses.append(valid_addresses[i])

            railway_locations.extend(railways)
            cemetery_locations.extend(cemeteries)

       # --- Step 5: Display Map with TOD Station Circles ---
        st.write("🗺️ **Plotting Addresses and TOD Stations on Map...**")
        m = folium.Map(location=filtered_coordinates[0] if filtered_coordinates else coordinates[0], zoom_start=14)

        # Plot Valid Addresses
        for i, (lat, lng) in enumerate(filtered_coordinates):
            folium.Marker([lat, lng], popup=filtered_addresses[i]).add_to(m)

        # Plot Railway Locations (Blue)
        for lat, lng in railway_locations:
            folium.Marker([lat, lng], icon=folium.Icon(color="blue", icon="train"), popup="Railway Station").add_to(m)

        # Plot Cemetery Locations (Black)
        for lat, lng in cemetery_locations:
            folium.Marker([lat, lng], icon=folium.Icon(color="black", icon="cross"), popup="Cemetery").add_to(m)

        # Plot TOD Stations and Draw Circles
        # Plot TOD Stations and Draw Concentric Circles
        for lat, lng in tod_coordinates:
            # TOD Station Marker
            folium.Marker([lat, lng], icon=folium.Icon(color="red", icon="cloud"), popup="TOD Station").add_to(m)

            # 200m Radius (High Impact Zone - Dark Red)
            folium.Circle(
                location=[lat, lng],
                radius=200,
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.3,
                popup="200m TOD Zone"
            ).add_to(m)

            # 400m Radius (Moderate Impact Zone - Orange)
            folium.Circle(
                location=[lat, lng],
                radius=400,
                color="orange",
                fill=True,
                fill_color="orange",
                fill_opacity=0.2,
                popup="400m TOD Zone"
            ).add_to(m)

            # 800m Radius (Low Impact Zone - Light Red)
            folium.Circle(
                location=[lat, lng],
                radius=800,
                color="darkred",
                fill=True,
                fill_color="darkred",
                fill_opacity=0.15,
                popup="800m TOD Zone"
            ).add_to(m)

                # Render the map in Streamlit
        folium_static(m)

        # --- Step 6: Solve Traveling Salesman Problem (TSP) ---
        st.write("🧠 **Solving Optimal Route...**")
        def get_distance(origin, destination, mode='walking'):
            """
            Retrieves the walking distance between two locations using the Google Directions API.
            """
            url = f"https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={destination}&mode={mode}&key={API_KEY}"
            
            for _ in range(3):  # Retry up to 3 times
                try:
                    response = requests.get(url, timeout=10).json()
                    if response['status'] == 'OK':
                        return response['routes'][0]['legs'][0]['distance']['value']  # Distance in meters
                    elif response['status'] == 'OVER_QUERY_LIMIT':
                        st.warning("⚠️ Google API Rate Limit reached. Retrying...")
                        time.sleep(2 + random.uniform(0, 2))  # Wait before retrying
                    else:
                        break  # Exit loop if another error
                except requests.exceptions.RequestException as e:
                    st.error(f"API error for {origin} → {destination}: {e}")
                    time.sleep(2)

            return float('inf')  # Return a very large number if all retries fail


        def build_distance_matrix(coordinates):
            """
            Builds a distance matrix for all locations using Google Maps distances.
            """
            n = len(coordinates)
            distance_matrix = [[0] * n for _ in range(n)]

            for i in range(n):
                for j in range(n):
                    if i != j:
                        origin = f"{coordinates[i][0]},{coordinates[i][1]}"
                        destination = f"{coordinates[j][0]},{coordinates[j][1]}"
                        distance_matrix[i][j] = get_distance(origin, destination)
                        time.sleep(1 + random.uniform(0, 1))  # Increase delay to prevent rate limits

            return distance_matrix



        distance_matrix = build_distance_matrix(filtered_coordinates)

        def solve_tsp(distance_matrix):
            n = len(distance_matrix)
            manager = pywrapcp.RoutingIndexManager(n, 1, 0)
            routing = pywrapcp.RoutingModel(manager)

            # Distance callback function
            def distance_callback(from_index, to_index):
                return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

            # Improve the search strategy to ensure the best route
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            search_parameters.time_limit.seconds = 10  # Ensure we find the best path within 10 seconds

            # Solve the TSP
            solution = routing.SolveWithParameters(search_parameters)

            if solution:
                index = routing.Start(0)
                route = []
                while not routing.IsEnd(index):
                    route.append(manager.IndexToNode(index))
                    index = solution.Value(routing.NextVar(index))
                route.append(manager.IndexToNode(index))
                return route  # Returns optimal order of addresses
            return None


        optimal_order = solve_tsp(distance_matrix)

        if optimal_order is not None and len(optimal_order) > 1:
            st.write("🚶‍♀️ **Optimal Route Order with Distances:**")
            total_distance = 0  # Track total route distance

            for i in range(len(optimal_order) - 1):
                from_idx = optimal_order[i]
                to_idx = optimal_order[i + 1]
                dist = distance_matrix[from_idx][to_idx] / 1000  # Convert to km
                total_distance += dist
                st.write(f"📍 {filtered_addresses[from_idx]} → {filtered_addresses[to_idx]}: **{dist:.2f} km**")

            st.write(f"🚗 **Total Route Distance: {total_distance:.2f} km**")

            # --- Step 7: Generate Google Maps URL ---
            def build_google_maps_url(coords, optimal_order):
                base_url = "https://www.google.com/maps/dir/"
                ordered_coords = [coords[i] for i in optimal_order]
                waypoints = "/".join(f"{lat},{lng}" for lat, lng in ordered_coords)
                return base_url + waypoints

            st.write("🌍 **Open Route in Google Maps:**")
            url = build_google_maps_url(filtered_coordinates, optimal_order)
            st.markdown(f"[📍 View Route in Google Maps]({url})", unsafe_allow_html=True)
        else:
            st.error("❌ Could not solve TSP problem. Check distance matrix or input addresses.")