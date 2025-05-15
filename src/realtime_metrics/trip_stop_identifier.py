
class TripStopIdentifier:
    """
    A class to uniquely identify the event of a vehicle of a trip halting at a stop. This is unique across an arbitrary time span
    It contains the trip_start_date, route_id, trip_id and stop_id.
    """
    def __init__(self, trip_start_date: str, route_id: str, trip_id: str, stop_id: str):
        self.trip_start_date = trip_start_date
        self.route_id = route_id
        self.trip_id = trip_id
        self.stop_id = stop_id

    def __hash__(self):
        return hash((self.trip_start_date, self.route_id, self.trip_id, self.stop_id))
    
    def __eq__(self, other):
        return (self.trip_start_date, self.route_id, self.trip_id, self.stop_id) == (other.trip_start_date, other.route_id, other.trip_id, other.stop_id)
