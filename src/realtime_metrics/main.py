import logging
import sys
from optparse import OptionParser
from typing import Dict
from datetime import datetime, timedelta

from gtfsrdb.model import Base, TripUpdate, StopTimeUpdate
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

import numpy

def run_analysis():
    """
    get all stop time updates and compute the different metrics
    """
    tripUpdates = session.query(TripUpdate).group_by(TripUpdate.trip_id).all()

    stop_time_updates: list[tuple[TripUpdate, StopTimeUpdate]] = []

    for tripUpdate in tripUpdates:
        trip_stop_time_updates = session.query(StopTimeUpdate, TripUpdate).join(TripUpdate).filter(TripUpdate.trip_id == tripUpdate.trip_id).filter(StopTimeUpdate.arrival_time > 0).order_by(TripUpdate.route_id.desc(), TripUpdate.trip_id.desc(), StopTimeUpdate.stop_id.desc()).all()
        for trip_stop_time_update in trip_stop_time_updates:
            stop_time_updates.append((trip_stop_time_update.TripUpdate, trip_stop_time_update.StopTimeUpdate))

            # add stop_time_update to dictionary if not present or update if timestamp of tripUpdate is newer than the one in the dictionary
            stop_time_update: StopTimeUpdate = trip_stop_time_update.StopTimeUpdate
            key = (tripUpdate.route_id, tripUpdate.trip_id, stop_time_update.stop_id)
            if key in actual_arrival_times.keys():
                if tripUpdate.timestamp.timestamp() > actual_arrival_times[key][0]:
                    actual_arrival_times[key] = (tripUpdate.timestamp.timestamp(), stop_time_update)
            else:
                actual_arrival_times[key] = (tripUpdate.timestamp.timestamp(), stop_time_update)

    mse_accuracy_result = mse_accuracy(stop_time_updates=stop_time_updates)
    print("MSE accuracy: ", mse_accuracy_result)
    eta_accuracy_result = eta_accuracy(stop_time_updates=stop_time_updates)
    print("ETA accuracy: ", eta_accuracy_result)

    experienced_wait_time_delay_result = experienced_wait_time_delay(stop_time_updates)
    print("Experienced Wait Time Delay: ", experienced_wait_time_delay_result)


def mse_accuracy(stop_time_updates: list[tuple[TripUpdate, StopTimeUpdate]]):
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
        predicted_delay = stop_time_update.arrival_delay
        prediction_error = actual_delay - predicted_delay
        samples.append(prediction_error)

    # compute MSE
    sum = 0
    for entry in samples:
        sum += entry * entry
    mean_squared_error = sum / len(samples)

    # print MSE
    return mean_squared_error


def eta_accuracy(stop_time_updates: list[tuple[TripUpdate, StopTimeUpdate]]):
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
        time_variance = actual_arrival_time - trip_update.timestamp.timestamp()

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
        return
    mean_accuracy = numpy.mean(accuracies) * 100
    return mean_accuracy


def get_actual_delay(trip_update: TripUpdate, stop_time_update: StopTimeUpdate) -> int:
    """
    returns the actual delay for the route and stop of the given TripUpdate
    """
    key = (trip_update.route_id, trip_update.trip_id, stop_time_update.stop_id)
    newest_stop_time_update: StopTimeUpdate = actual_arrival_times[key][1]
    return newest_stop_time_update.arrival_delay


def get_actual_arrival_time(trip_update: TripUpdate, stop_time_update: StopTimeUpdate) -> int:
    """
    return the actual arrival time for the route and stop of the given TripUpdate
    """
    key = (trip_update.route_id, trip_update.trip_id, stop_time_update.stop_id)
    newest_stop_time_update: StopTimeUpdate = actual_arrival_times[key][1]
    return newest_stop_time_update.arrival_time


def experienced_wait_time_delay(trip_stop_time_updates: list[tuple[TripUpdate, StopTimeUpdate]]) -> float:
    """
    Computes the average Experienced Wait Time Delay.
    """
    logger.info("Calculating Experienced Wait Time Delay...")

    delays = []

    logger.info(trip_stop_time_updates)

    indexed_updates: Dict[tuple[str, str], list[StopTimeUpdate]] = {}

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

        indexed_updates[new_key].append(stop_time_update)

    logger.debug("Indexed updates: %s", indexed_updates)

    for index_key, stop_updates in indexed_updates.items():
        if len(stop_updates) < 2:
            logger.debug("Not enough updates for %s", index_key)
            continue
        route_id, stop_id = index_key
        # sort updates by arrival time
        stop_updates.sort(key=lambda u: u.arrival_time)
        logger.debug("Sorted updates for %s: %s", index_key, [datetime.fromtimestamp(u.arrival_time).strftime("%Y-%m-%d %H:%M:%S") for u in stop_updates])

        # simulate a full day based on the days of arrival times in stop_updates
        if not stop_updates:
            logger.info("No stop updates available for simulation.")
            return None

        # Determine the range of days based on arrival times
        min_arrival_time = stop_updates[0].arrival_time
        max_arrival_time = stop_updates[-1].arrival_time
        start_of_day = datetime.fromtimestamp(min_arrival_time).replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = datetime.fromtimestamp(max_arrival_time).replace(hour=23, minute=59, second=59, microsecond=0)

        current_time = int(start_of_day.timestamp())
        end_time = int(end_of_day.timestamp())

        # TODO: check from here if this is correct
        while current_time <= end_time:
            # find the next predicted arrival time within 60 min
            next_predicted = next((u.arrival_time for u in stop_updates if current_time <= u.arrival_time <= current_time + 3600), None)
            if next_predicted is None:
                current_time += 60  # Increment by 1 minute
                continue  # No prediction available in the 60 min window

            # find next experienced arrival at or after predicted
            next_experienced = next((u.arrival_time for u in stop_updates if u.arrival_time >= next_predicted), None)
            if next_experienced is None:
                current_time += 60  # Increment by 1 minute
                continue

            delay = next_experienced - next_predicted
            delays.append(delay)

            # Move to the next minute after the current prediction
            current_time = next_predicted + 60

    if not delays:
        logger.info("No delay samples found.")
        return None

    logger.info("Delays: %s", delays)    
    average_delay = numpy.mean(delays)
    return average_delay


if __name__ == "__main__":
    option_parser = OptionParser()
    option_parser.add_option('-d', '--database', default=None, dest='dsn',
                help='Database connection string', metavar='DSN')
    option_parser.add_option('-v', '--verbose', default=False, dest='verbose',
                action='store_true', help='Print generated SQL')

    option_parser.add_option('-q', '--quiet', default=False, dest='quiet',
                action='store_true', help="Don't print warnings and status messages")
    options, arguments = option_parser.parse_args()

    if options.quiet:
        level = logging.ERROR
    elif options.verbose:
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

    # Check if it has the tables
    # Base from model.py
    for table in Base.metadata.tables.keys():
        if not inspector.has_table(table):
            logging.error('Missing table %s! Use gtfsrdb.py -c to create it.', table)
            exit(1)
        logger.debug("Table %s exists" % table)

    logger.info("Successfully connected to database")
    
    run_analysis()
