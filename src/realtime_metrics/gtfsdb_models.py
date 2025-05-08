from gtfsrdb.model import Base
from sqlalchemy import Column
from sqlalchemy.types import SmallInteger, Integer, Numeric, String

class StopTime(Base):
    __tablename__ = 'stop_times'

    trip_id = Column(String(255), primary_key=True, index=True, nullable=False)
    stop_id = Column(String(255), index=True, nullable=False)
    stop_sequence = Column(Integer, primary_key=True, nullable=False)
    arrival_time = Column(String(9))
    departure_time = Column(String(9), index=True)
    stop_headsign = Column(String(255))
    pickup_type = Column(Integer, default=0)
    drop_off_type = Column(Integer, default=0)
    shape_dist_traveled = Column(Numeric(20, 10))
    timepoint = Column(SmallInteger, index=True, default=0)