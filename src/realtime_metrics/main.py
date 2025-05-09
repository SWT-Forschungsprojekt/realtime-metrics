import logging
import sys
from optparse import OptionParser
from typing import Dict
from datetime import datetime, timezone, timedelta

from gtfsrdb.model import Base, TripUpdate, StopTimeUpdate
from sqlalchemy import create_engine, inspect, func
from realtime_metrics.gtfsdb_models import StopTime
from sqlalchemy.orm import sessionmaker

import numpy

def run_analysis():
    """
    get all stop time updates and compute the different metrics
    """
    stoptimes = session.query(StopTime.trip_id, func.min(StopTime.arrival_time).label('min_arrival_time')).group_by(StopTime.trip_id)

    for stop_time in stoptimes:
        first_trip_arrival_times[stop_time.trip_id] = stop_time.min_arrival_time

    tripUpdates = session.query(TripUpdate).group_by(TripUpdate.trip_id).all()

    stop_time_updates: list[tuple[TripUpdate, StopTimeUpdate]] = []

    trips: Dict[tuple[str, str, str], list[tuple[TripUpdate, StopTimeUpdate]]]  = dict()

    for tripUpdate in tripUpdates:
        trip_stop_time_updates = session.query(StopTimeUpdate, TripUpdate).join(TripUpdate).filter(TripUpdate.trip_id == tripUpdate.trip_id).filter(StopTimeUpdate.arrival_time > 0).order_by(TripUpdate.route_id.desc(), TripUpdate.trip_id.desc(), StopTimeUpdate.stop_id.desc()).all()
        for trip_stop_time_update in trip_stop_time_updates:
            stop_time_updates.append((trip_stop_time_update.TripUpdate, trip_stop_time_update.StopTimeUpdate))

            # add stop time update to trips
            stop_time_update: StopTimeUpdate = trip_stop_time_update.StopTimeUpdate
            key = (tripUpdate.route_id, tripUpdate.trip_id, stop_time_update.stop_id)
            if key in trips.keys():
                trip_updates: list[TripUpdate] = trips[key]
            else:
                trip_updates: list[TripUpdate] = []
            trip_updates.append((trip_stop_time_update.TripUpdate, trip_stop_time_update.StopTimeUpdate))
            trips[key] = trip_updates

            # add stop_time_update to dictionary if not present or update if timestamp of tripUpdate is newer than the one in the dictionary
            if stop_time_update.arrival_uncertainty > 0:
                continue # only consider updates with arrival uncertainty 0 as actual arrivals
            if key in actual_arrival_times.keys():
                if trip_stop_time_update.TripUpdate.timestamp.replace(tzinfo=timezone.utc).timestamp() > actual_arrival_times[key][0]:
                    actual_arrival_times[key] = (trip_stop_time_update.TripUpdate.timestamp.replace(tzinfo=timezone.utc).timestamp(), stop_time_update)
            else:
                actual_arrival_times[key] = (trip_stop_time_update.TripUpdate.timestamp.replace(tzinfo=timezone.utc).timestamp(), stop_time_update)

    # MSE accuracy ------------------------------------------------------------------------------------------------------------------
    mse_accuracy_result = mse_accuracy(stop_time_updates=stop_time_updates)
    if mse_accuracy_result is None:
        print("MSE accuracy could not be computed, no data provided!")
    else:
        print("MSE accuracy: ", round(mse_accuracy_result, 2))

    # ETA accuracy ------------------------------------------------------------------------------------------------------------------
    eta_accuracy_result = eta_accuracy(stop_time_updates=stop_time_updates)
    if eta_accuracy_result is None:
        print("ETA accuracy could not be computed, no data provided!")
    else:
        print(f"ETA accuracy: {round(eta_accuracy_result, 2)}%")

    # experienced wait time delay ---------------------------------------------------------------------------------------------------
    experienced_wait_time_delay_result = experienced_wait_time_delay(stop_time_updates)
    if experienced_wait_time_delay_result is None:
        print("Experienced Wait Time Delay could not be computed, no data provided!")
    else:
        print(f"Experienced Wait Time Delay: {round(experienced_wait_time_delay_result, 2)} seconds")

    # availability of acceptable stop time updates ----------------------------------------------------------------------------------
    availabilities = []
    
    for trip_id in trips.keys():
        trip_updates: list[tuple[TripUpdate, StopTimeUpdate]] = trips[trip_id]
        if len(trip_updates) == 0:
            continue
        trip_updates.sort(key=lambda u: u[0].timestamp.replace(tzinfo=timezone.utc).timestamp())
        time_frame_start = int(trip_updates[0][0].timestamp.replace(tzinfo=timezone.utc).timestamp() / 60) * 60
        time_frame_end = int(trip_updates[-1][0].timestamp.replace(tzinfo=timezone.utc).timestamp() / 60) * 60
        trip_availability = availability_acceptable_stop_time_updates(trip_updates, time_frame_start, time_frame_end)
        availabilities.append(trip_availability)

    if len(availabilities) == 0:
        availability_acceptable_stop_time_updates_result = 0
    else:
        availability_acceptable_stop_time_updates_result = sum(availabilities) / len(availabilities)

    print(f"Availability of acceptable stop time updates: {round(availability_acceptable_stop_time_updates_result, 2)}%")

    # prediction reliability --------------------------------------------------------------------------------------------------------
    prediction_reliability_result = prediction_reliability(stop_time_updates)
    print(f"Prediction reliability: {round(prediction_reliability_result * 100, 2)}%")
    
    # prediction inconsistency ------------------------------------------------------------------------------------------------------
    inconsistencies = []

    for trip_stop in actual_arrival_times.keys():
        actual_arrival_time = actual_arrival_times[trip_stop][1].arrival_time
        updates = trips[trip_stop]
        inconsistency = prediction_inconsistency(actual_arrival_time, updates)
        inconsistencies.append(inconsistency)
        logger.info(f"Prediction inconsistency for route {trip_stop[0]}, trip {trip_stop[1]}, stop {trip_stop[2]}: {inconsistency} seconds")

    prediction_inconsistency_result = numpy.mean(inconsistencies)
    print(f"Prediction inconsistency: {prediction_inconsistency_result} seconds")


def mse_accuracy(stop_time_updates: list[tuple[TripUpdate, StopTimeUpdate]]) -> float | None:
    """
    computes the accuracy of the given stop time updates using mean squared error
    """

    if len(stop_time_updates) <= 0:
        logger.info("No samples provided!")
        return

    # compute prediction error for each stop time update
    samples = []
    for update in stop_time_updates:
        trip_update = update[0]
        stop_time_update = update[1]
        actual_delay = get_actual_delay(trip_update=trip_update, stop_time_update=stop_time_update)
        if actual_delay is None:
            continue # skip updates with unknown actual delay
        predicted_delay = stop_time_update.arrival_delay
        prediction_error = actual_delay - predicted_delay
        samples.append(prediction_error)

    # compute MSE
    if len(samples) == 0:
        logger.info("No data provided!")
        return None

    sum = 0.0
    for entry in samples:
        sum += entry * entry
    mean_squared_error = sum / len(samples)

    return mean_squared_error


def eta_accuracy(stop_time_updates: list[tuple[TripUpdate, StopTimeUpdate]]) -> float | None:
    """
    computes the accuracy of the given stop time updates using the eta bucketing approach
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
        actual_delay = get_actual_delay(trip_update=trip_update, stop_time_update=stop_time_update)
        if actual_delay is None:
            continue # skip updates with unknown actual delay
        predicted_delay = stop_time_update.arrival_delay
        difference =  predicted_delay - actual_delay

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
        logger.info("No data provided!")
        return None
    mean_accuracy = numpy.mean(accuracies) * 100
    return mean_accuracy


def get_actual_delay(trip_update: TripUpdate, stop_time_update: StopTimeUpdate) -> int | None:
    """
    Returns the actual delay for the route and stop of the given TripUpdate.
    If no actual delay is known, None is returned.
    """
    key = (trip_update.route_id, trip_update.trip_id, stop_time_update.stop_id)
    if key in actual_arrival_times.keys():
        newest_stop_time_update: StopTimeUpdate = actual_arrival_times[key][1]
        return newest_stop_time_update.arrival_delay
    else:
        return None


def get_actual_arrival_time(trip_update: TripUpdate, stop_time_update: StopTimeUpdate) -> int | None:
    """
    Returns the actual arrival time for the route and stop of the given TripUpdate.
    If no actual arrival time is known, None is returned.
    """
    key = (trip_update.route_id, trip_update.trip_id, stop_time_update.stop_id)
    if key in actual_arrival_times.keys():
        newest_stop_time_update: StopTimeUpdate = actual_arrival_times[key][1]
        return newest_stop_time_update.arrival_time
    else:
        return None


def experienced_wait_time_delay(trip_stop_time_updates: list[tuple[TripUpdate, StopTimeUpdate]]) -> float | None:
    """
    Computes the average Experienced Wait Time Delay metrics for the given stop time updates.
    The metric is defined here: https://docs.google.com/document/d/1-AOtPaEViMcY6B5uTAYj7oVkwry3LfAQJg3ihSRTVoU. 
    It computes the average amount of time in minutes a passenger has to wait at a stop, 
    if he arrives at the arrival time of the most up-to-date stop time update.

    Parameters:
    stop_time_updates: list of corresponding trip updates and stop time updates

    Returns:
    A float containing the average wait time.
    """
    logger.info("Calculating Experienced Wait Time Delay...")

    delays = []

    logger.debug(trip_stop_time_updates)

    indexed_updates: Dict[tuple[str, str], list[tuple[TripUpdate, StopTimeUpdate]]] = {}

    for key in actual_arrival_times.keys():
        logger.debug("Key: %s", key)
        route_id, _, stop_id = key
        
        new_key = (route_id, stop_id)

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
        logger.info("No delay samples found.")
        return None

    logger.debug("Delays: %s", delays)    
    average_delay = numpy.mean(delays)
    return average_delay


def availability_acceptable_stop_time_updates(stop_time_updates: list[tuple[TripUpdate, StopTimeUpdate]], 
                                              time_frame_start: int, 
                                              time_frame_end: int) -> float:
    """
    Computes the availability of acceptable stop time updates metrics for the given stop time updates in the given time frame.
    The metric is defined here: https://docs.google.com/document/d/1-AOtPaEViMcY6B5uTAYj7oVkwry3LfAQJg3ihSRTVoU. 
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

    # dict to store how many stop time updates are available for each minute slot
    time_slots: Dict[int, int] = dict()

    for update in stop_time_updates:
        trip_update = update[0]

        # get time in minutes by removing part of the timestamp responsible for seconds
        time_rounded_in_minutes = int(trip_update.timestamp.replace(tzinfo=timezone.utc).timestamp() / 60) * 60

        # skip, if outide of time frame
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
    for key in time_slots.keys():
        if time_slots[key] >= 2:
            time_slots_with_enough_updates += 1
    logger.debug("Time slots with enough updates: %s", time_slots_with_enough_updates)

    return (time_slots_with_enough_updates / number_of_time_slots) * 100


def get_last_predicted_update(timestamp: int, updates: list[tuple[TripUpdate, StopTimeUpdate]]) -> tuple[TripUpdate, StopTimeUpdate] | None:
    """
    Returns the last stop time update in the given list, that what published before or at the given timestamp.
    If no such stop time update is in the list, None is returned.
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
    """
    actual_arrivals: list[StopTimeUpdate] = []
    # collect all actual arrivals after the given timestamp
    for key, update in actual_arrival_times.items():
        if key[0] == route_id and key[2] == stop_id:
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
    The metric is defined here: https://docs.google.com/document/d/1-AOtPaEViMcY6B5uTAYj7oVkwry3LfAQJg3ihSRTVoU. 
    It computes the percentage of stop time updates classified as reliable.
    Whether or not a stop time update counts as reliable depends on the deviation of the predicted arrival time from the actual arrival time, 
    as well as how far in the future the prediction is.

    Parameters:
    stop_time_updates: list of corresponding trip updates and stop time updates

    Returns:
    A float containing the prediction reliability.
    """
    logger.debug(f"amount of updates: {len(stop_time_updates)}")
    if len(stop_time_updates) == 0:
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

        first_trip_arrival_time = datetime.strptime(first_trip_arrival_times[trip_id], "%H:%M:%S").replace(year=prediction_published_datetime.year, month=prediction_published_datetime.month, day=prediction_published_datetime.day)

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
        logger.info("No valid updates pprovided!")
        return 0.0 # no stop time updates are not reliable
    
    return reliable_updates / total


def prediction_inconsistency(actual_arrival_time: int, updates: list[tuple[TripUpdate, StopTimeUpdate]]) -> float:
    """
    Computes the prediction inconsistency metrics for a route, trip and stop combination.
    The metric is defined here: https://docs.google.com/document/d/1-AOtPaEViMcY6B5uTAYj7oVkwry3LfAQJg3ihSRTVoU. 
    It computes the predicted arrival time spread in 30 two minute time frames, starting every minute.
    It computes the spread as the difference of the minimal and maximal predicted arrival time in each time frame. 
    The prediction inconsistency is then the average spread of all windows.

    Parameters:
    actual_arrival_time: the actual arrival time of the vehicle (for that route, trip and stop)
    updates: list of corresponding trip updates and stop time updates (for that route, trip and stop)

    Returns:
    A float containing the prediction inconsistency.
    """
    thirty_one_minutes_earlier = actual_arrival_time - 1860 # 31 minutes to fit 30 time windows of 2 minutes, starting every minute
    valid_updates = [update for update in updates if update[0].timestamp.replace(tzinfo=timezone.utc).timestamp() >= thirty_one_minutes_earlier]

    if len(valid_updates) == 0:
        return 0.0 # no inconsistency, if no updates in the last 31 minutes
    
    two_minutes_earlier = actual_arrival_time - 120
    spreads = []

    logger.info("----------------------------------------------------------")
    logger.info(f"actual arrival time: {actual_arrival_time} ({datetime.fromtimestamp(actual_arrival_time, timezone.utc)})")

    for current_time_frame_start in range(thirty_one_minutes_earlier, two_minutes_earlier + 1, 60):
        updates_in_time_frame = [update for update in valid_updates if 
                                 update[0].timestamp.replace(tzinfo=timezone.utc).timestamp() >= current_time_frame_start and 
                                 update[0].timestamp.replace(tzinfo=timezone.utc).timestamp() < current_time_frame_start + 120]
        
        if len(updates_in_time_frame) == 0:
            logger.info(f"time frame {current_time_frame_start} ({datetime.fromtimestamp(current_time_frame_start, timezone.utc)}): no entries")
            continue # skip time frames with no updates

        updates_in_time_frame.sort(key=lambda update: update[1].arrival_time)
        min_prediction = updates_in_time_frame[0][1].arrival_time
        max_prediction = updates_in_time_frame[-1][1].arrival_time
        spread = max_prediction - min_prediction
        logger.info(f"time frame {current_time_frame_start} ({datetime.fromtimestamp(current_time_frame_start, timezone.utc)}): {spread}")
        spreads.append(spread)

    logger.info("----------------------------------------------------------")

    logger.info(spreads)

    if len(spreads) == 0:
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
    actual_arrival_times: Dict[tuple[str, str, str], tuple[int, StopTimeUpdate]] = dict()

    # dict for trip stop times (gtfsdb)
    first_trip_arrival_times: Dict[tuple[str, str], str] = dict()

    # Check if it has the tables
    # Base from model.py
    for table in Base.metadata.tables.keys():
        if not inspector.has_table(table):
            logging.error('Missing table %s! Use gtfsrdb.py -c to create it.', table)
            exit(1)
        logger.debug("Table %s exists" % table)

    logger.info("Successfully connected to database")
    
    run_analysis()
