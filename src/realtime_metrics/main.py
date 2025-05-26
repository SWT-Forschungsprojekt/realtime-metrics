import logging
import sys
from optparse import OptionParser
from typing import Dict
from datetime import datetime, timezone, timedelta
from collections import defaultdict

from gtfsrdb.model import Base, TripUpdate, StopTimeUpdate, VehiclePosition
from sqlalchemy import create_engine, inspect, func
from realtime_metrics.gtfsdb_models import StopTime
from realtime_metrics.trip_stop_identifier import TripStopIdentifier
from sqlalchemy.orm import sessionmaker

import numpy

def run_stop_time_analysis():
    """
    Fetches all stop times and corresponding trip updates, computes the following metrics and prints their results to the console:
    - accuracy using mean squared error
    - ETA accuracy (defined here: https://github.com/TransitApp/ETA-Accuracy-Benchmark)
    - experienced wait time delay (defined here: https://docs.google.com/document/d/1-AOtPaEViMcY6B5uTAYj7oVkwry3LfAQJg3ihSRTVoU/edit#heading=h.yxonst7620cc)
    - availability of acceptable stop time updates (defined here: https://docs.google.com/document/d/1-AOtPaEViMcY6B5uTAYj7oVkwry3LfAQJg3ihSRTVoU/edit#heading=h.opcobjkpjpd5)
    - prediction reliability (defined here: https://docs.google.com/document/d/1-AOtPaEViMcY6B5uTAYj7oVkwry3LfAQJg3ihSRTVoU/edit#heading=h.jxkc7wix2vyb)
    - prediction inconsistency (defined here: https://docs.google.com/document/d/1-AOtPaEViMcY6B5uTAYj7oVkwry3LfAQJg3ihSRTVoU/edit#heading=h.uzxz2nt2mgh0)
    """
    stoptimes = session.query(
        StopTime.trip_id,
        func.min(StopTime.arrival_time).label('min_arrival_time')
    ).group_by(StopTime.trip_id).all()

    trip_updates = session.query(TripUpdate).all()

    trip_stop_time_updates = session.query(StopTimeUpdate, TripUpdate).join(
        TripUpdate, StopTimeUpdate.trip_update_id == TripUpdate.oid
    ).filter(
        StopTimeUpdate.arrival_time > 0
    ).order_by(
        TripUpdate.route_id.desc(),
        TripUpdate.trip_id.desc(),
        StopTimeUpdate.stop_id.desc()
    ).all()

    # build the stop_time_updates that is used for the metrics computation
    stop_time_updates: list[tuple[TripUpdate, StopTimeUpdate]] = []
    trips: Dict[TripStopIdentifier, list[tuple[TripUpdate, StopTimeUpdate]]] = defaultdict(list)

    for stop_time_update, trip_update in trip_stop_time_updates:
        stop_time_updates.append((trip_update, stop_time_update))

        trip_stop_identifier = TripStopIdentifier(
            trip_update.trip_start_date,
            trip_update.route_id,
            trip_update.trip_id,
            stop_time_update.stop_id
        )

        trips[trip_stop_identifier].append((trip_update, stop_time_update))

        if stop_time_update.arrival_uncertainty > 0:
            continue

        timestamp = trip_update.timestamp.replace(tzinfo=timezone.utc).timestamp()
        current = actual_arrival_times.get(trip_stop_identifier)

        if not current or timestamp > current[0]:
            actual_arrival_times[trip_stop_identifier] = (timestamp, stop_time_update)

    # MSE accuracy ------------------------------------------------------------------------------------------------------------------
    print("---------------------------------------------------------------------------")
    print("Computing MSE accuracy ...")
    mse_accuracy_result = mse_accuracy(stop_time_updates=stop_time_updates)
    if mse_accuracy_result is None:
        print("MSE accuracy could not be computed, no data provided!")
    else:
        print("MSE accuracy: ", round(mse_accuracy_result, 2))

    # ETA accuracy ------------------------------------------------------------------------------------------------------------------
    print("---------------------------------------------------------------------------")
    print("Computing ETA accuracy ...")
    eta_accuracy_result = eta_accuracy(stop_time_updates=stop_time_updates)
    if eta_accuracy_result is None:
        print("ETA accuracy could not be computed, no data provided!")
    else:
        print(f"ETA accuracy: {round(eta_accuracy_result, 2)}%")

    # experienced wait time delay ---------------------------------------------------------------------------------------------------
    print("---------------------------------------------------------------------------")
    print("Computing experienced wait time delay ...")
    experienced_wait_time_delay_result = experienced_wait_time_delay(stop_time_updates)
    if experienced_wait_time_delay_result is None:
        print("Experienced Wait Time Delay could not be computed, no data provided!")
    else:
        print(f"Experienced Wait Time Delay: {round(experienced_wait_time_delay_result, 2)} seconds")

    # availability of acceptable stop time updates ----------------------------------------------------------------------------------
    print("---------------------------------------------------------------------------")
    print("Computing availability of acceptable stop time updates ...")
    availabilities = []
    
    for trip_stop_identifier in trips.keys():
        trip_updates: list[tuple[TripUpdate, StopTimeUpdate]] = trips[trip_stop_identifier]
        if len(trip_updates) == 0:
            continue
        trip_updates.sort(key=lambda u: u[0].timestamp.replace(tzinfo=timezone.utc).timestamp())
        time_frame_start = trip_updates[0][0].timestamp.replace(tzinfo=timezone.utc).replace(second=0).timestamp()
        time_frame_end = trip_updates[-1][0].timestamp.replace(tzinfo=timezone.utc).replace(second=0).timestamp()
        trip_availability = availability_acceptable_stop_time_updates(trip_updates, time_frame_start, time_frame_end)
        availabilities.append(trip_availability)

    if len(availabilities) == 0:
        availability_acceptable_stop_time_updates_result = 0
        print("No stop time updates provided!")
    else:
        availability_acceptable_stop_time_updates_result = sum(availabilities) / len(availabilities)

    print(f"Availability of acceptable stop time updates: {round(availability_acceptable_stop_time_updates_result, 2)}%")

    # prediction reliability --------------------------------------------------------------------------------------------------------
    print("---------------------------------------------------------------------------")
    print("Computing prediction reliability ...")
    prediction_reliability_result = prediction_reliability(stop_time_updates)
    print(f"Prediction reliability: {round(prediction_reliability_result * 100, 2)}%")

    # prediction inconsistency ------------------------------------------------------------------------------------------------------
    print("---------------------------------------------------------------------------")
    print("Computing prediction inconsistency ...")
    inconsistencies = []

    for trip_stop_identifier in actual_arrival_times.keys():
        actual_arrival_time = actual_arrival_times[trip_stop_identifier][1].arrival_time
        updates = trips[trip_stop_identifier]
        inconsistency = prediction_inconsistency(actual_arrival_time, updates)
        inconsistencies.append(inconsistency)
        logger.debug(f"Prediction inconsistency for route {trip_stop_identifier.route_id}, trip {trip_stop_identifier.trip_id}, stop {trip_stop_identifier.stop_id}: {round(inconsistency, 2)} seconds")

    if len(inconsistencies) == 0:
        print(f"Prediction inconsistency: could not be computed, no data provided!")
    else:
        prediction_inconsistency_result = numpy.mean(inconsistencies)
        print(f"Prediction inconsistency: {round(prediction_inconsistency_result, 2)} seconds")


def run_vehicle_position_analysis():
    """
    Fetches all vehicle positions, computes the following metrics and prints their results to the console:
    - availability of acceptable vehicle updates (defined here: https://docs.google.com/document/d/1-AOtPaEViMcY6B5uTAYj7oVkwry3LfAQJg3ihSRTVoU/edit#heading=h.14woewhhbqwk)
    """
    vehicle_positions: dict[str, list[VehiclePosition]] = dict()
    for vehicle_position in session.query(VehiclePosition).all():
        if vehicle_position.trip_id not in vehicle_positions.keys():
            vehicle_positions[vehicle_position.trip_id] = []
        vehicle_positions[vehicle_position.trip_id].append(vehicle_position)
    
    min_max_timestamps: dict[str, tuple[datetime, datetime]] = dict()
    # get for each trip the first and last arrival time
    # for each trip in vehicle position get first and last arrival time
    for vehicle_position_min_max in session.query(VehiclePosition.trip_id, func.min(VehiclePosition.timestamp).label('min_timestamp'), func.max(VehiclePosition.timestamp).label('max_timestamp')).group_by(VehiclePosition.trip_id).all():
        min_timestamp = vehicle_position_min_max.min_timestamp.replace(tzinfo=timezone.utc)
        max_timestamp = vehicle_position_min_max.max_timestamp.replace(tzinfo=timezone.utc)
        min_max_timestamps[vehicle_position_min_max.trip_id] = (min_timestamp, max_timestamp)

    # availability of acceptable vehicle position updates ----------------------------------------------------------------------------------
    print("---------------------------------------------------------------------------")
    print("Computing availability of acceptable vehicle position updates ...")
    availabilities: list[float] = [] 

    for trip_id, vehicle_positions in vehicle_positions.items():
        # get min and max timestamp for the trip
        min_timestamp, max_timestamp = min_max_timestamps[trip_id]

        # availability of acceptable vehicle positions ----------------------------------------------------------------------------------
        availability_acceptable_vehicle_positions_result = availability_acceptable_vehicle_positions(vehicle_positions, min_timestamp, max_timestamp)
        availabilities.append(availability_acceptable_vehicle_positions_result)

    if len(availabilities) == 0:
        availability_acceptable_vehicle_positions_result = 0
        print("No vehicle positions provided!")
    else:
        availability_acceptable_vehicle_positions_result = numpy.mean(availabilities)
    print(f"Availability of acceptable vehicle positions: {round(availability_acceptable_vehicle_positions_result, 2)}%")


def mse_accuracy(stop_time_updates: list[tuple[TripUpdate, StopTimeUpdate]]) -> float | None:
    """
    Computes the accuracy of the given stop time updates using mean squared error. 
    Stop time updates with unknown actual arrival time are omitted.

    Parameters
    ----------
    stop_time_updates : list[tuple[TripUpdate, StopTimeUpdate]
        pairs of StopTimeUpdates and the corresponding TripUpdate
    
    Returns
    -------
    float | None
        the mean squared deviation of the true arrival time
        None, if an empty list is provided or no true arrival times are known.
    """

    if len(stop_time_updates) <= 0:
        logger.info("No samples provided!")
        return

    # compute prediction error for each stop time update
    samples = []
    for update in stop_time_updates:
        trip_update = update[0]
        stop_time_update = update[1]
        actual_arrival_time = get_actual_arrival_time(trip_update=trip_update, stop_time_update=stop_time_update)
        if actual_arrival_time is None:
            continue # skip updates with unknown actual arrival time
        predicted_arrival_time = stop_time_update.arrival_time
        prediction_error = actual_arrival_time - predicted_arrival_time
        samples.append(prediction_error)

    # compute MSE
    if len(samples) == 0:
        logger.warning("No data provided!")
        return None

    sum = 0.0
    for entry in samples:
        sum += entry * entry
    mean_squared_error = sum / len(samples)

    return mean_squared_error


def eta_accuracy(stop_time_updates: list[tuple[TripUpdate, StopTimeUpdate]]) -> float | None:
    """
    Computes the accuracy of the given stop time updates using the ETA bucketing approach.
    The metric is defined here: https://github.com/TransitApp/ETA-Accuracy-Benchmark.
    It sorts the stop time updates into buckets based on how much time is left until the arrival.
    In each bucket, the predictions are allowed to be within a different range around the true arrival time.
    Samples within this range are labeled 'accurate', the others 'inaccurate'.
    For each bucket, the percentage of accurate updates is computed.
    The ETA accuracy is then defined as the mean accuracy of all buckets.

    Parameters
    ----------
    stop_time_updates : list[tuple[TripUpdate, StopTimeUpdate]
        pairs of StopTimeUpdates with the corresponding TripUpdate
    
    Returns
    -------
    float | None
        the mean squared deviation of the true arrival time. 
        None, if an empty list is provided or no stop time updates are within the valid range of the buckets.
    """

    # initialize buckets with zero correct and incorrect samples each
    buckets: list[list] = []
    buckets.append([0,0])
    buckets.append([0,0])
    buckets.append([0,0])
    buckets.append([0,0])

    # sort stop time updates into the correct buckets
    for update in stop_time_updates:
        trip_update = update[0]
        stop_time_update = update[1]
        actual_arrival_time = get_actual_arrival_time(trip_update=trip_update, stop_time_update=stop_time_update)
        if actual_arrival_time is None:
            continue # skip updates with unknown actual arrival time
        time_variance = actual_arrival_time - trip_update.timestamp.replace(tzinfo=timezone.utc).timestamp()

        bucket_index = 0
        upper_limit = 0
        lower_limit = 0
        if time_variance <= 0:
            # remove stop time updates published after arrival
            continue
        elif time_variance <= 180:
            # 0-3 min bucket
            bucket_index = 0
            upper_limit = 90
            lower_limit = -30
        elif time_variance <= 360:
            # 3-6 min bucket
            bucket_index = 1
            upper_limit = 150
            lower_limit = -60
        elif time_variance <= 600:
            # 6-10 min bucket
            bucket_index = 2
            upper_limit = 210
            lower_limit = -60
        elif time_variance <= 900:
            # 10-15 min bucket
            bucket_index = 3
            upper_limit = 270
            lower_limit = -90
        else: 
            # too far in the past
            continue

        # check if inside allowed interval
        actual_arrival_time = get_actual_arrival_time(trip_update=trip_update, stop_time_update=stop_time_update)
        if actual_arrival_time is None:
            continue # skip updates with unknown actual arrival time
        predicted_arrival_time = stop_time_update.arrival_time
        difference =  predicted_arrival_time - actual_arrival_time

        bucket = buckets[bucket_index]

        if difference <= upper_limit and difference >= lower_limit:
            bucket[0] += 1
        bucket[1] += 1
        
        buckets[bucket_index] = bucket

    logger.debug("ETA buckets: %s", buckets)

    # compute accuracy in each bucket
    accuracies = []
    for i in range(0,len(buckets)):
        bucket = buckets[i]
        correct = bucket[0]
        total = bucket[1]
        if total == 0:
            continue
        accuracies.append(correct / total)

    # compute total accuracy
    if len(accuracies) == 0:
        logger.warning("No samples in any bucket!")
        return None
    
    mean_accuracy = numpy.mean(accuracies) * 100
    return mean_accuracy


def get_actual_arrival_time(trip_update: TripUpdate, stop_time_update: StopTimeUpdate) -> int | None:
    """
    Returns the actual arrival time for the route and stop of the given TripUpdate.
    If no actual arrival time is known, None is returned.

    Parameters
    ----------
    trip_update : TripUpdate
        the TripUpdate that published the StopTimeUpdate
    stop_time_update : StopTimeUpdate
        the StopTimeUpdate for the concrete stop

    Returns
    -------
    int | None
        unix timestamp (in seconds) of the actual arrival time.
        None, if no actual arrival time is known.
    """
    trip_stop_identifier = TripStopIdentifier(trip_update.trip_start_date, trip_update.route_id, trip_update.trip_id, stop_time_update.stop_id)
    if trip_stop_identifier in actual_arrival_times.keys():
        newest_stop_time_update: StopTimeUpdate = actual_arrival_times[trip_stop_identifier][1]
        return newest_stop_time_update.arrival_time
    else:
        return None


def experienced_wait_time_delay(trip_stop_time_updates: list[tuple[TripUpdate, StopTimeUpdate]]) -> float | None:
    """
    Computes the average Experienced Wait Time Delay metrics for the given stop time updates.
    The metric is defined here: https://docs.google.com/document/d/1-AOtPaEViMcY6B5uTAYj7oVkwry3LfAQJg3ihSRTVoU/edit#heading=h.yxonst7620cc. 
    It computes the average amount of time in minutes a passenger has to wait at a stop, 
    if he arrives at the arrival time of the most up-to-date stop time update.

    Parameters
    ----------
    stop_time_updates : list[tuple[TripUpdate, StopTimeUpdate]]
        pairs of corresponding trip updates and stop time updates

    Returns
    -------
    float | None
       the average wait time
        None, if no experienced wait time could be computed.
    """
    logger.info("Calculating Experienced Wait Time Delay...")

    delays = []

    logger.debug(trip_stop_time_updates)

    indexed_updates: Dict[tuple[str, str], list[tuple[TripUpdate, StopTimeUpdate]]] = {}

    for trip_stop_identifier in actual_arrival_times.keys():
        logger.debug("Key: %s", trip_stop_identifier)
        
        new_key = (trip_stop_identifier.route_id, trip_stop_identifier.stop_id)

        indexed_updates[new_key] = []

    for trip_stop_time_update in trip_stop_time_updates:
        trip_update: TripUpdate = trip_stop_time_update[0]
        stop_time_update: StopTimeUpdate = trip_stop_time_update[1]

        route_id = trip_update.route_id
        stop_id = stop_time_update.stop_id
        new_key = (route_id, stop_id)

        if new_key not in indexed_updates.keys():
            continue

        indexed_updates[new_key].append((trip_update, stop_time_update))

    logger.debug("Indexed updates: %s", indexed_updates)

    for index_key, trip_stop_updates in indexed_updates.items():
        if len(trip_stop_updates) < 2:
            logger.debug("Not enough updates for %s", index_key)
            continue
        route_id, stop_id = index_key
        # sort updates by published time
        trip_stop_updates.sort(key=lambda u: u[0].timestamp.replace(tzinfo=timezone.utc).timestamp())
        logger.debug("Sorted updates for %s: %s", index_key, [datetime.fromtimestamp(update[0].timestamp.replace(tzinfo=timezone.utc).timestamp()).strftime("%Y-%m-%d %H:%M:%S") for update in trip_stop_updates])

        # Determine the range of days based on arrival times
        min_arrival_time = trip_stop_updates[0][0].timestamp.replace(tzinfo=timezone.utc).timestamp()
        max_arrival_time = trip_stop_updates[-1][0].timestamp.replace(tzinfo=timezone.utc).timestamp()

        current_time = min_arrival_time - (min_arrival_time % 60)
        end_time = max_arrival_time

        for current_time in range(int(current_time), int(end_time) + 1, 60):
            # find the last stop time update
            next_predicted_arrival = get_last_predicted_update(current_time, trip_stop_updates)
            if not next_predicted_arrival:
                continue  # No prediction available
            next_predicted_arrival_time = next_predicted_arrival[1].arrival_time
            if next_predicted_arrival_time > current_time + 3600:
                continue  # prediction is more than 60 minutes in the future
            # find next experienced arrival at or after predicted 
            next_actual_arrival = get_next_actual_arrival(next_predicted_arrival_time, route_id, stop_id)
            if not next_actual_arrival:
                continue  # no actual arrival after the given timestamp found
            next_actual_arrival_time = next_actual_arrival.arrival_time
            delay = next_actual_arrival_time - next_predicted_arrival_time
            delays.append(delay)

    if len(delays) <= 0:
        logger.warning("No delay samples found.")
        return None

    logger.debug("Delays: %s", delays)    
    average_delay = numpy.mean(delays)
    return average_delay


def availability_acceptable_stop_time_updates(stop_time_updates: list[tuple[TripUpdate, StopTimeUpdate]], 
                                              time_frame_start: int, 
                                              time_frame_end: int) -> float:
    """
    Computes the availability of acceptable stop time updates metrics for the given stop time updates in the given time frame.
    The metric is defined here: https://docs.google.com/document/d/1-AOtPaEViMcY6B5uTAYj7oVkwry3LfAQJg3ihSRTVoU/edit#heading=h.opcobjkpjpd5. 
    It computes the percentage of one-minute slots with two or more updates.

    Parameters
    ----------
    stop_time_updates : list[tuple[TripUpdate, StopTimeUpdate]
        list of corresponding trip updates and stop time updates
    time_frame_start : int
        start of the time frame to observe, in seconds since 1970
    time_frame_end : int 
        end of the time frame to observe, in seconds since 1970

    Returns
    -------
    float
        The percentage of one minute slots with two or more updates.
    """
    logger.debug("Time frame start: %s", time_frame_start)
    logger.debug("Time frame end: %s", time_frame_end)

    if len(stop_time_updates) == 0:
        logger.debug("No stop time updates provided!")
        return 0.0

    # dict to store how many stop time updates are available for each minute slot
    time_slots: Dict[int, int] = dict()

    for update in stop_time_updates:
        trip_update = update[0]

        # get time in minutes by removing part of the timestamp responsible for seconds
        time_rounded_in_minutes = trip_update.timestamp.replace(tzinfo=timezone.utc).replace(second=0).timestamp()

        # skip, if outside of time frame
        if time_rounded_in_minutes < time_frame_start or time_rounded_in_minutes > time_frame_end:
            continue

        # increase respective stop time counter
        if time_rounded_in_minutes not in time_slots.keys():
            time_slots[time_rounded_in_minutes] = 1
        else:
            time_slots[time_rounded_in_minutes] = time_slots[time_rounded_in_minutes] + 1

    # calculate percentage of minutes with two or more stop time updates
    number_of_time_slots = int(time_frame_end / 60) - int(time_frame_start / 60) + 1
    logger.debug("Time slots: %s", number_of_time_slots)

    time_slots_with_enough_updates = 0
    for time_slot in time_slots.keys():
        if time_slots[time_slot] >= 2:
            time_slots_with_enough_updates += 1
    logger.debug("Time slots with enough updates: %s", time_slots_with_enough_updates)

    return (time_slots_with_enough_updates / number_of_time_slots) * 100


def availability_acceptable_vehicle_positions(vehicle_positions: list[VehiclePosition], 
                                              time_frame_start: datetime, 
                                              time_frame_end: datetime) -> float:
    """
    Computes the availability of acceptable vehicle position messages metrics for the given vehicle positions in the given time frame.
    The metric is defined here: https://docs.google.com/document/d/1-AOtPaEViMcY6B5uTAYj7oVkwry3LfAQJg3ihSRTVoU/edit#heading=h.14woewhhbqwk. 
    It computes the percentage of one-minute slots with two or more vehicle positions.

    Parameters
    ----------
    vehicle_positions: list[VehiclePosition]
        list of vehicle positions
    time_frame_start: datetime
        start of the time frame to observe, datetime in utc
    time_frame_end: datetime
        end of the time frame to observe, datetime in utc

    Returns
    -------
    float
        A float containing the percentage of one minute slots with two or more updates.
    """
    logger.debug("Time frame start: %s", time_frame_start)
    logger.debug("Time frame end: %s", time_frame_end)

    if len(vehicle_positions) == 0:
        logger.debug("No vehicle positions provided!")
        return 0.0

    # all vehicle positions for a given minute
    amount_vehicle_positions_per_minute: dict[int, int] = dict()

    for vehicle_position in vehicle_positions:
        timestamp: datetime = vehicle_position.timestamp.replace(tzinfo=timezone.utc)

        if timestamp < time_frame_start or timestamp > time_frame_end:
            continue

        minutes_since_time_frame_start = int((timestamp - time_frame_start).total_seconds() / 60)
        if minutes_since_time_frame_start not in amount_vehicle_positions_per_minute.keys():
            amount_vehicle_positions_per_minute[minutes_since_time_frame_start] = 0
        amount_vehicle_positions_per_minute[minutes_since_time_frame_start] += 1

    time_slots_with_enough_updates = 0

    for minute, amount_of_updates in amount_vehicle_positions_per_minute.items():
        if amount_of_updates >= 2:
            time_slots_with_enough_updates += 1

    if len(amount_vehicle_positions_per_minute) == 0:
        logger.info("No vehicle positions fall within the specified time frame!")
        return 0.0

    return time_slots_with_enough_updates / len(amount_vehicle_positions_per_minute) * 100


def get_last_predicted_update(timestamp: int, updates: list[tuple[TripUpdate, StopTimeUpdate]]) -> tuple[TripUpdate, StopTimeUpdate] | None:
    """
    Returns the last stop time update in the given list, that what published before or at the given timestamp.

    Parameters
    ----------
    timestamp : int
        An unix timestamp (in seconds) restricting which stop time updates are in the past
    updates : list[tuple[TripUpdate, StopTimeUpdate]]
        pairs of StopTimeUpdates and corresponding TripUpdates to find the last before the timestamp
    
    Returns
    -------
    tuple[TripUpdate, StopTimeUpdate] | None
        The last StopTimeUpdate published before or at the timestamp, with its corresponding TripUpdate.
        None, if no such StopTimeUpdate is found.
    """
    updates_before_timestamp = [update for update in updates if update[0].timestamp.replace(tzinfo=timezone.utc).timestamp() <= timestamp and update[1].arrival_time > 0]
    if len(updates_before_timestamp) <= 0:
        return None
    latest_timestamp = updates_before_timestamp[-1][0].timestamp.replace(tzinfo=timezone.utc).timestamp()
    updates_published_at_last_timestamp = [update for update in updates_before_timestamp if update[0].timestamp.replace(tzinfo=timezone.utc).timestamp() == latest_timestamp]
    updates_published_at_last_timestamp.sort(key=lambda update: update[1].arrival_time)
    return updates_published_at_last_timestamp[0]


def get_next_actual_arrival(timestamp: int, route_id: str, stop_id: str) -> StopTimeUpdate | None:
    """
    Returns the stop time update containing the next actual arrival time for the given route and stop after the given timestamp.
    
    Parameters
    ----------
    timestamp : int
        An unix timestamp (in seconds) defining the lower boundary to consider stop time updates.
    route_id : str
        The route to get the next actual arrival time of.
    stop_id : str
        The stop (of the route) to get the next actual arrival time of.
    """
    actual_arrivals: list[StopTimeUpdate] = []
    # collect all actual arrivals after the given timestamp
    for trip_stop_identifier, update in actual_arrival_times.items():
        if trip_stop_identifier.route_id == route_id and trip_stop_identifier.stop_id == stop_id:
            stop_time_update: StopTimeUpdate = update[1]
            if stop_time_update.arrival_time >= timestamp:
                actual_arrivals.append(stop_time_update)

    if len(actual_arrivals) == 0:
        return None # no actual arrival after the given timestamp found
        
    actual_arrivals.sort(key=lambda update: update.arrival_time)
    return actual_arrivals[0]


def prediction_reliability(stop_time_updates: list[tuple[TripUpdate, StopTimeUpdate]]) -> float:
    """
    Computes the prediction reliability metrics for a route, trip and stop combination.
    The metric is defined here: https://docs.google.com/document/d/1-AOtPaEViMcY6B5uTAYj7oVkwry3LfAQJg3ihSRTVoU/edit#heading=h.jxkc7wix2vyb. 
    It computes the percentage of stop time updates classified as reliable.
    Whether or not a stop time update counts as reliable depends on the deviation of the predicted arrival time from the actual arrival time, 
    as well as how far in the future the prediction is.

    Parameters
    ----------
    stop_time_updates : list[tuple[TripUpdate, StopTimeUpdate]]
        pairs of corresponding trip updates and stop time updates

    Returns
    -------
    float
        the prediction reliability.
    """
    logger.debug(f"amount of updates: {len(stop_time_updates)}")
    if len(stop_time_updates) == 0:
        logger.warning("No stop time updates provided!")
        return 0.0 # no stop time updates are not reliable
    
    reliable_updates = 0
    unreliable_updates = 0
    
    for update in stop_time_updates:
        trip_id = update[0].trip_id

        logger.debug("-----------------------------------------------------------------------------")
        logger.debug(f"update from: {update[0].timestamp}")
        actual_arrival_time = get_actual_arrival_time(update[0], update[1])

        if actual_arrival_time is None:
            continue # skip updates without known actual arrival time
        prediction_error = actual_arrival_time - update[1].arrival_time
        logger.debug(f"actual arrival time: {datetime.fromtimestamp(actual_arrival_time, timezone.utc)}")
        logger.debug(f"predicted arrival time: {datetime.fromtimestamp(update[1].arrival_time, timezone.utc)}")
        logger.debug(f"prediction error: {prediction_error}")

        predicted_arrival_time = update[1].arrival_time
        prediction_published_datetime: datetime = update[0].timestamp
        prediction_published = prediction_published_datetime.replace(tzinfo=timezone.utc).timestamp()
    
        if trip_id not in first_trip_arrival_times.keys():
            logger.debug("Min arrival_time not found for trip_id: %s", trip_id)
            continue

        first_trip_arrival_time = datetime.strptime(first_trip_arrival_times[trip_id], "%H:%M:%S").replace(
            year=prediction_published_datetime.year, 
            month=prediction_published_datetime.month, 
            day=prediction_published_datetime.day
        )

        # check prediction_published is greater than scheduled trip stop arrival time - 60 minutes
        if prediction_published_datetime < first_trip_arrival_time - timedelta(minutes=60):
            continue

        time_to_prediction = (predicted_arrival_time - prediction_published) / 60 # time is required in minutes
        logger.debug(f"time to prediction: {time_to_prediction}")

        if time_to_prediction < 0:
            continue # skip updates with negative time to predictions (updates published for the past or without an arrival time)

        lower_bound = -60 * numpy.log(time_to_prediction + 1.3)
        upper_bound = 60 * numpy.log(time_to_prediction + 1.5)

        logger.debug(f"lower bound: {lower_bound}")
        logger.debug(f"upper bound: {upper_bound}")

        if lower_bound < prediction_error < upper_bound:
            reliable_updates += 1
            logger.debug("reliable")
        else:
            unreliable_updates += 1
            logger.debug("unreliable")

    total = reliable_updates + unreliable_updates

    if total == 0:
        logger.warning("No valid updates provided!")
        return 0.0 # no stop time updates are not reliable
    
    return reliable_updates / total


def prediction_inconsistency(actual_arrival_time: int, updates: list[tuple[TripUpdate, StopTimeUpdate]]) -> float:
    """
    Computes the prediction inconsistency metrics for a route, trip and stop combination.
    The metric is defined here: https://docs.google.com/document/d/1-AOtPaEViMcY6B5uTAYj7oVkwry3LfAQJg3ihSRTVoU/edit#heading=h.uzxz2nt2mgh0. 
    It computes the predicted arrival time spread in 30 two minute time frames, starting every minute.
    It computes the spread as the difference of the minimal and maximal predicted arrival time in each time frame. 
    The prediction inconsistency is then the average spread of all windows.

    Parameters
    ----------
    actual_arrival_time : int
        An unix timestamp (in seconds), containing the actual arrival time of the vehicle (for that route, trip and stop)
    updates : list[tuple[TripUpdate, StopTimeUpdate]]
        pairs of corresponding trip updates and stop time updates (for that route, trip and stop)

    Returns
    -------
    float
        the prediction inconsistency.
    """
    thirty_one_minutes_earlier = actual_arrival_time - 1860 # 31 minutes to fit 30 time windows of 2 minutes, starting every minute
    valid_updates = [update for update in updates if update[0].timestamp.replace(tzinfo=timezone.utc).timestamp() >= thirty_one_minutes_earlier]

    if len(valid_updates) == 0:
        logger.debug("No stop time updates provided!")
        return 0.0 # no inconsistency, if no updates in the last 31 minutes
    
    two_minutes_earlier = actual_arrival_time - 120
    spreads = []

    logger.debug("----------------------------------------------------------")
    logger.debug(f"actual arrival time: {actual_arrival_time} ({datetime.fromtimestamp(actual_arrival_time, timezone.utc)})")

    for current_time_frame_start in range(thirty_one_minutes_earlier, two_minutes_earlier + 1, 60):
        updates_in_time_frame = [update for update in valid_updates if 
                                 update[0].timestamp.replace(tzinfo=timezone.utc).timestamp() >= current_time_frame_start and 
                                 update[0].timestamp.replace(tzinfo=timezone.utc).timestamp() < current_time_frame_start + 120]
        
        if len(updates_in_time_frame) == 0:
            logger.debug(f"time frame {current_time_frame_start} ({datetime.fromtimestamp(current_time_frame_start, timezone.utc)}): no entries")
            continue # skip time frames with no updates

        updates_in_time_frame.sort(key=lambda update: update[1].arrival_time)
        min_prediction = updates_in_time_frame[0][1].arrival_time
        max_prediction = updates_in_time_frame[-1][1].arrival_time
        spread = max_prediction - min_prediction
        logger.debug(f"time frame {current_time_frame_start} ({datetime.fromtimestamp(current_time_frame_start, timezone.utc)}): {spread}")
        spreads.append(spread)

    logger.debug("----------------------------------------------------------")

    logger.debug(spreads)

    if len(spreads) == 0:
        logger.debug("No valid updates provided!")
        return 0.0
    
    average_spread = numpy.mean(spreads)
    return average_spread


if __name__ == "__main__":
    option_parser = OptionParser()
    option_parser.add_option('-d', '--database', default=None, dest='dsn',
                help='Database connection string', metavar='DSN')
    option_parser.add_option('-v', '--verbose', default=False, dest='verbose',
                action='store_true', help='Print generated SQL')
    option_parser.add_option('-q', '--quiet', default=False, dest='quiet',
                action='store_true', help="Don't print warnings and status messages")
    option_parser.add_option('--debug', default=False, dest='debug',
                action='store_true', help="Print debug logs")
    option_parser.add_option('-a', '--analysis', default='stop_time', dest='analysis',
                help="The analysis to run. Can be 'stop_time' or 'vehicle_position'")
    options, arguments = option_parser.parse_args()

    if options.quiet:
        level = logging.ERROR
    elif options.verbose:
        level = logging.INFO
    elif options.debug:
        level = logging.DEBUG
    else:
        level = logging.WARNING
    # Set up a logger
    logger = logging.getLogger()
    logger.setLevel(level)
    loghandler = logging.StreamHandler(sys.stdout)
    logformatter = logging.Formatter(fmt='%(message)s')
    loghandler.setFormatter(logformatter)
    logger.addHandler(loghandler)

    if options.dsn is None:
        logging.error('No database specified!')
        exit(1)

    # Connect to the database
    engine = create_engine(options.dsn, echo=options.quiet)
    # Create a database inspector
    inspector = inspect(engine)
    session = sessionmaker(bind=engine)()

    # dict for newest stop time update
    actual_arrival_times: Dict[TripStopIdentifier, tuple[int, StopTimeUpdate]] = dict()

    # dict for trip stop times (gtfsdb)
    first_trip_arrival_times: Dict[str, str] = dict()

    # Check if it has the tables
    # Base from model.py
    for table in Base.metadata.tables.keys():
        if not inspector.has_table(table):
            logging.error('Missing table %s! Use gtfsrdb.py -c to create it.', table)
            exit(1)
        logger.debug("Table %s exists" % table)

    logger.info("Successfully connected to database")
    
    if options.analysis == 'stop_time':
        print("Running stop time analysis...")
        run_stop_time_analysis()
    elif options.analysis == 'vehicle_position':
        print("Running vehicle position analysis...")
        run_vehicle_position_analysis()
    else:
        logger.error("Unknown analysis type %s", options.analysis)
        exit(1)
