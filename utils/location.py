import math
from geopy.geocoders import Nominatim
import random  # For demo purposes only

# Initialize geocoder with a custom user agent
geolocator = Nominatim(user_agent="skinhealth_app")


def get_user_location_from_ip():
    """
    Get approximate user location from IP address.
    In a real app, you would use a service like ipinfo.io or similar.
    This is a simplified version for demo purposes.
    """
    return {
        "latitude": 55.7558,
        "longitude": 37.6173,
        "city": "Moscow",
        "state": "Msc",
        "country": "Russia"
    }


def geocode_address(address):
    """Convert address to latitude and longitude."""
    try:
        location = geolocator.geocode(address)
        if location:
            return {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "address": location.address
            }
        return None
    except Exception as e:
        print(f"Geocoding error: {e}")
        return None


def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two coordinates using Haversine formula.
    Returns distance in kilometers.
    """
    # Convert coordinates from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Earth radius in kilometers

    return c * r


def get_doctors_in_radius(user_lat, user_lon, radius_km, all_doctors):
    """Filter doctors within specified radius of user location."""
    doctors_in_radius = []

    for doctor in all_doctors:
        # In a real app, you would have actual coordinates for each doctor
        # For demo, we're generating random nearby coordinates
        if "latitude" not in doctor or "longitude" not in doctor:
            # Generate random nearby coordinates for demo
            doctor["latitude"] = user_lat + (random.random() - 0.5) * 0.1
            doctor["longitude"] = user_lon + (random.random() - 0.5) * 0.1

        distance = calculate_distance(user_lat, user_lon, doctor["latitude"], doctor["longitude"])
        doctor["distance"] = round(distance, 1)  # Round to 1 decimal place

        if distance <= radius_km:
            doctors_in_radius.append(doctor)

    # Sort by distance
    return sorted(doctors_in_radius, key=lambda x: x["distance"])
