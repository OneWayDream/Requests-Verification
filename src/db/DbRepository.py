from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from sqlalchemy.sql import text as sa_text
from db.Requests import *


def init_repo(db_url):
    global engine
    engine = create_engine(db_url)
    DeclarativeBase.metadata.create_all(engine)


@contextmanager
def connect():
    connection = engine.connect()
    db_session = sessionmaker(autocommit=False, autoflush=True, bind=engine)
    yield db_session()
    db_session.close_all()
    connection.close()


def add_requests(requests):
    with connect() as session:
        session.add_all(requests)
        session.commit()


def get_all(clazz):
    with connect() as session:
        return session.query(clazz).all()


def truncate_table(clazz):
    sql = "TRUNCATE TABLE %s" % clazz.__tablename__
    engine.execute(sa_text(sql).execution_options(autocommit=True))


def get_count_of_rows(clazz):
    with connect() as session:
        return session.query(clazz).count()
