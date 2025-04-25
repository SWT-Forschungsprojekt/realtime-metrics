import logging
import sys
from optparse import OptionParser
from typing import Dict

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

    trips: Dict[tuple[str, str, str], list[TripUpdate]]  = dict()

    time_frame_start = sys.maxsize
    time_frame_end = 0

    for tripUpdate in tripUpdates:
        trip_stop_time_updates = session.query(StopTimeUpdate, TripUpdate).join(TripUpdate).filter(TripUpdate.trip_id == tripUpdate.trip_id).filter(StopTimeUpdate.arrival_time > 0).all()

        # update time range
        time_in_minutes = int(tripUpdate.timestamp.timestamp() / 60)
        if time_in_minutes < time_frame_start:
            time_frame_start = time_in_minutes
        if time_in_minutes > time_frame_end:
            time_frame_end = time_in_minutes

        for trip_stop_time_update in trip_stop_time_updates:
            stop_time_updates.append((trip_stop_time_update.TripUpdate, trip_stop_time_update.StopTimeUpdate))

            # add stop_time_update to dictionary if not present or update if timestamp of tripUpdate is newer than the one in the dictionary
            stop_time_update: StopTimeUpdate = trip_stop_time_update.StopTimeUpdate
            key = (tripUpdate.route_id, tripUpdate.trip_id, stop_time_update.stop_id)
            if key in dictionary.keys():
                if tripUpdate.timestamp.timestamp() > dictionary[key][0]:
                    dictionary[key] = (tripUpdate.timestamp.timestamp(), stop_time_update)
            else:
                dictionary[key] = (tripUpdate.timestamp.timestamp(), stop_time_update)

            # add stop time update to trips
            if key in trips.keys():
                trip_updates: list[TripUpdate] = trips[key]
            else:
                trip_updates: list[TripUpdate] = []
            trip_updates.append((trip_stop_time_update.TripUpdate, trip_stop_time_update.StopTimeUpdate))
            trips[key] = trip_updates

    mse_accuracy_result = mse_accuracy(stop_time_updates=stop_time_updates)
    print("MSE accuracy: ", mse_accuracy_result)
    eta_accuracy_result = eta_accuracy(stop_time_updates=stop_time_updates)
    print("ETA accuracy: ", eta_accuracy_result)

    availabilities = []
    
    for trip_id in trips.keys():
        trip_updates = trips[trip_id]
        trip_availability = availability_acceptable_stop_time_updates(trip_updates, time_frame_start, time_frame_end)
        availabilities.append(trip_availability)

    if len(availabilities) == 0:
        availability_acceptable_stop_time_updates_result = 0
    else:
        availability_acceptable_stop_time_updates_result = sum(availabilities) / len(availabilities)

    print("Availability of acceptable stop time updates: ", availability_acceptable_stop_time_updates_result)


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


def availability_acceptable_stop_time_updates(stop_time_updates: list[tuple[TripUpdate, StopTimeUpdate]], 
                                              time_frame_start: int, 
                                              time_frame_end: int):
    """
    Computes the availability of acceptable stop time updates metrics for the given stop time updates in the given time frame.
    The metric is defined here: https://docs.google.com/document/d/1-AOtPaEViMcY6B5uTAYj7oVkwry3LfAQJg3ihSRTVoU. 
    It computes the percentage of one minute slots with two or more updates.

    Parameters:
    stop_time_updates: list of corresponding trip updates and stop time updates
    time_frame_start: start of the time frame to observe, in minutes since 1970
    time_frame_end: end of the time frame to observe, in minutes since 1970

    Returns:
    A float containing the percentage of one minute slots with two or more updates.
    """
    logger.info("Time frame start: %s", time_frame_start)
    logger.info("Time frame end: %s", time_frame_end)

    # dict to store how many stop time updates are available for each minute slot
    time_slots: Dict[int, int] = dict()

    for update in stop_time_updates:
        trip_update = update[0]

        # get time in minutes
        time_in_minutes = int(trip_update.timestamp.timestamp() / 60)

        # skip, if outide of time frame
        if time_in_minutes < time_frame_start or time_in_minutes > time_frame_end:
            continue

        # increase respective stop time counter
        if time_in_minutes not in time_slots.keys():
            time_slots[time_in_minutes] = 1
        else:
            time_slots[time_in_minutes] = time_slots[time_in_minutes] + 1

    # calculate percentage of minutes with two or more stop time updates
    number_of_time_slots = time_frame_end - time_frame_start + 1
    logger.info("Time slots: %s", number_of_time_slots)

    time_slots_with_enough_updates = 0
    for key in time_slots.keys():
        if time_slots[key] >= 2:
            time_slots_with_enough_updates += 1
    logger.info("Time slots with enough updates: %s", time_slots_with_enough_updates)

    return (time_slots_with_enough_updates / number_of_time_slots) * 100


def get_actual_delay(trip_update: TripUpdate, stop_time_update: StopTimeUpdate) -> int:
    """
    returns the actual delay for the route and stop of the given TripUpdate
    """
    key = (trip_update.route_id, trip_update.trip_id, stop_time_update.stop_id)
    newest_stop_time_update: StopTimeUpdate = dictionary[key][1]
    return newest_stop_time_update.arrival_delay


def get_actual_arrival_time(trip_update: TripUpdate, stop_time_update: StopTimeUpdate) -> int:
    """
    return the actual arrival time for the route and stop of the given TripUpdate
    """
    key = (trip_update.route_id, trip_update.trip_id, stop_time_update.stop_id)
    newest_stop_time_update: StopTimeUpdate = dictionary[key][1]
    return newest_stop_time_update.arrival_time


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
    dictionary: Dict[tuple[str, str, str], tuple[int, StopTimeUpdate]] = dict()

    # Check if it has the tables
    # Base from model.py
    for table in Base.metadata.tables.keys():
        if not inspector.has_table(table):
            logging.error('Missing table %s! Use gtfsrdb.py -c to create it.', table)
            exit(1)
        logger.debug("Table %s exists" % table)

    logger.info("Successfully connected to database")
    
    run_analysis()
