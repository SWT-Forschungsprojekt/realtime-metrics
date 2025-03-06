import logging
import sys
from optparse import OptionParser

from gtfsrdb.model import Base, TripUpdate, StopTimeUpdate
from sqlalchemy import create_engine, inspect, Row
from sqlalchemy.orm import sessionmaker

from typing import Tuple

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
        print("Table %s exists" % table)

    print("Successfully connected to database")

    trips = session.query(TripUpdate).group_by(TripUpdate.trip_id).all()

    for trip in trips:
        print("Trip ID: ", trip.trip_id)
        print("Vehicle ID: ", trip.vehicle_id)

        trip_updates = session.query(TripUpdate).filter(TripUpdate.trip_id == trip.trip_id).all()
        stop_time_updates = session.query(StopTimeUpdate, TripUpdate).join(TripUpdate).filter(TripUpdate.trip_id == trip.trip_id).all()

        if len(stop_time_updates) <= 1:
            print(f"Trip {trip.trip_id} has no stop time updates")
            print("")
            print("=====================================")
            continue

        total_arrival_delay = 0

        for i in range(1, len(stop_time_updates) - 1):

            stop_time_update = stop_time_updates[i]

            # print("Stop ID: ", stop_time_update.StopTimeUpdate.stop_id)
            # print("Stop sequence: ", stop_time_update.StopTimeUpdate.stop_sequence)
            # print("Arrival delay: ", stop_time_update.StopTimeUpdate.arrival_delay)
            # print("Arrival uncertainty: ", stop_time_update.StopTimeUpdate.arrival_uncertainty)
            # print("TimeStamp: ", stop_time_update.TripUpdate.timestamp)
            # print("")
            # print("================================")

            total_arrival_delay += stop_time_update.StopTimeUpdate.arrival_delay

        last = stop_time_updates[-1].StopTimeUpdate
        #last.arrival_delay = 0

        # TODO: Remove the trip if we know it is ongoing
        if last.arrival_delay != 0:
            pass

        average_arrival_delay = total_arrival_delay / len(stop_time_updates) - 1
        real_difference = last.arrival_time - average_arrival_delay

        print("Total number of stop time updates: ", len(stop_time_updates))
        print("Average arrival delay: ", average_arrival_delay)
        print("Real difference: ", real_difference)
        print("")
        print("=====================================")

        exit(0)

