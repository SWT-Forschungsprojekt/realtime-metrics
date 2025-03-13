import logging
import sys
from optparse import OptionParser

from gtfsrdb.model import Base, TripUpdate, StopTimeUpdate
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

import pandas as pd

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
    engine = create_engine(options.dsn, echo=options.verbose)
    # Create a database inspector
    inspector = inspect(engine)
    session = sessionmaker(bind=engine)()

    # Check if it has the tables
    # Base from model.py
    for table in Base.metadata.tables.keys():
        if not inspector.has_table(table):
            logging.error('Missing table %s! Use gtfsrdb.py -c to create it.', table)
            exit(1)
        logger.debug("Table %s exists" % table)

    logger.info("Successfully connected to database")

    trips = session.query(TripUpdate).group_by(TripUpdate.trip_id).all()

    data = []

    for trip in trips:
        logger.warning("Trip ID: %s", trip.trip_id)
        logger.warning("Vehicle ID: %s", trip.vehicle_id)

        stop_time_updates = session.query(StopTimeUpdate, TripUpdate).join(TripUpdate).filter(TripUpdate.trip_id == trip.trip_id).all()

        if len(stop_time_updates) <= 1:
            logger.info(f"Trip {trip.trip_id} has no stop time updates")
            logger.info("")
            logger.info("=====================================")
            continue

        total_arrival_delay = 0

        for i in range(1, len(stop_time_updates) - 1):

            stop_time_update: StopTimeUpdate = stop_time_updates[i].StopTimeUpdate
            trip_update: TripUpdate = stop_time_updates[i].TripUpdate

            logger.debug("Stop ID: %s", stop_time_update.stop_id)
            logger.debug("Stop sequence: %s", stop_time_update.stop_sequence)
            logger.debug("Arrival delay: %s", stop_time_update.arrival_delay)
            logger.debug("Arrival uncertainty: %s", stop_time_update.arrival_uncertainty)
            logger.debug("TimeStamp: %s", trip_update.timestamp)
            logger.debug("")
            logger.debug("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

            total_arrival_delay += stop_time_update.arrival_delay

        last: StopTimeUpdate = stop_time_updates[-1].StopTimeUpdate
        #last.arrival_delay = 0

        # TODO: Remove the trip if we know it is ongoing
        if last.arrival_delay != 0:
            pass

        average_arrival_delay = total_arrival_delay / len(stop_time_updates) - 1
        real_difference = last.arrival_time - average_arrival_delay

        # add to dataframe
        data.append({
            'trip_id': trip.trip_id,
            'avg_delay_difference': average_arrival_delay,
            'real_difference': real_difference
        })

        logger.warning("Total number of stop time updates: %d", len(stop_time_updates))
        logger.warning("Average arrival delay: %f", average_arrival_delay)
        logger.warning("Real difference: %f", real_difference)
        logger.warning("")
        logger.warning("=====================================")

    df = pd.DataFrame(data)
    logger.warning(df)
    df.to_csv('metrics.csv')

