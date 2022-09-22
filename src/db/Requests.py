from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

DeclarativeBase = declarative_base()


class Request(DeclarativeBase):
    __tablename__ = 'all_requests'
    id = Column(Integer, primary_key=True)
    vector = Column('vector', String, nullable=False)

    def __repr__(self):
        return 'Request [%s, %s]' % (self.id, self.vector)


class NewRequest(DeclarativeBase):
    __tablename__ = 'new_requests'
    id = Column(Integer, primary_key=True)
    vector = Column('vector', String)

    def __repr__(self):
        return 'New Request [%s, %s]' % (self.id, self.vector)
