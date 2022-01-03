from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer
from geoalchemy2 import Geometry
from sqlalchemy.sql.sqltypes import TIMESTAMP, Boolean, Float

Base = declarative_base()


class Noise(Base):
    __table_args__ = {'quote': False}
    __tablename__ = 'project.real_noise'

    id = Column(Integer, primary_key=True)

    timestamp = Column(TIMESTAMP)
    location = Column(Geometry('POINT'))
    noise = Column(Float)
