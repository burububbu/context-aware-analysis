
from database_connection.schemas import Noise
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, func

import os

engine = None
Session = None


def engine_create():

    db_string = 'postgresql://{0}:{1}@{2}/{3}'.format(
        os.environ.get("TYPEORM_USERNAME"),
        os.environ.get("TYPEORM_PASSWORD"),
        os.environ.get("TYPEORM_HOST"),
        os.environ.get("TYPEORM_DATABASE"))

    global engine
    engine = create_engine(db_string, echo=False)

    return db_string


def getAll():
    if not engine:
        engine_create()

    global Session
    Session = sessionmaker(bind=engine)

    session = Session()

    noises = session.query(
        Noise.id.label('id'),
        Noise.timestamp.label('timestamp'),
        func.st_x(Noise.location).label('longitude'),
        func.st_y(Noise.location).label('latitude'),
        Noise.noise.label('noise'))

    return noises
