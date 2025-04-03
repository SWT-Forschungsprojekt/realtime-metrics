import logging
import sys
from optparse import OptionParser

from gtfsrdb.model import Base, TripUpdate, StopTimeUpdate
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

import numpy

def run_analysis():
    # get all stop time updates and run compute the different metrics
    tripUpdates = session.query(TripUpdate).group_by(TripUpdate.trip_id).all()

    stop_time_updates: list[tuple[TripUpdate, StopTimeUpdate]] = []

    for tripUpdate in tripUpdates:
        trip_stop_time_updates = session.query(StopTimeUpdate, TripUpdate).join(TripUpdate).filter(TripUpdate.trip_id == tripUpdate.trip_id).filter(StopTimeUpdate.arrival_time > 0).all()
        for trip_stop_time_update in trip_stop_time_updates:
            stop_time_updates.append((trip_stop_time_update.TripUpdate, trip_stop_time_update.StopTimeUpdate))

            # check dict
            stop_time_update: StopTimeUpdate = trip_stop_time_update.StopTimeUpdate
            key = (tripUpdate.route_id, tripUpdate.trip_id, stop_time_update.stop_id)
            if key in dictionary.keys():
                if tripUpdate.timestamp.timestamp() > dictionary[key][0]:
                    dictionary[key] = (tripUpdate.timestamp.timestamp(), stop_time_update)
            else:
                dictionary[key] = (tripUpdate.timestamp.timestamp(), stop_time_update)

    # compute accuracy metric
    logger.info("MSE accuracy: ")
    mse_accuracy(stop_time_updates=stop_time_updates)
    logger.info("ETA accuracy:")
    eta_accuracy(stop_time_updates=stop_time_updates)


def mse_accuracy(stop_time_updates: list[tuple[TripUpdate, StopTimeUpdate]]):
    # computes the accuracy of the given stop time updates using mean squared error

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
    logger.info("MSE: %s", mean_squared_error)


def eta_accuracy(stop_time_updates: list[tuple[TripUpdate, StopTimeUpdate]]):
    # computes the accuracy of the given stop time updates using the eta bucketing approach

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
            #print("Stop Time Update is outside the allowed interval.")
            continue

        # check if inside allowed interval
        actual_delay = get_actual_delay(trip_update=trip_update, stop_time_update=stop_time_update)
        predicted_delay = stop_time_update.arrival_delay
        # TODO: fix calculation
        difference =  predicted_delay - actual_delay

        bucket = buckets[bucket_index]

        if difference <= upper_limit and difference >= lower_limit:
            bucket[0] += 1
        bucket[1] += 1
        
        buckets[bucket_index] = bucket

    logger.debug("buckets: %s", buckets)

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
    logger.info(f"ETA accuracy: {mean_accuracy}%")


def get_actual_delay(trip_update: TripUpdate, stop_time_update: StopTimeUpdate) -> int:
    # return the actual delay for the route and stop of the given TripUpdate
    #return -numpy.random.randint(-90, 270)
    key = (trip_update.route_id, trip_update.trip_id, stop_time_update.stop_id)
    newest_stop_time_update: StopTimeUpdate = dictionary[key][1]
    return newest_stop_time_update.arrival_delay


def get_actual_arrival_time(trip_update: TripUpdate, stop_time_update: StopTimeUpdate) -> int:
    # return the actual arrival time for the route and stop of the given TripUpdate
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
    #dictionary: Dict[(str, str, str), (int, StopTimeUpdate)] = dict()
    dictionary = dict()

    # Check if it has the tables
    # Base from model.py
    for table in Base.metadata.tables.keys():
        if not inspector.has_table(table):
            logging.error('Missing table %s! Use gtfsrdb.py -c to create it.', table)
            exit(1)
        logger.debug("Table %s exists" % table)

    logger.info("Successfully connected to database")
    
    run_analysis()
