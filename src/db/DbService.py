from db.DbRepository import get_all, get_count_of_rows, add_requests, truncate_table, init_repo
from db.Requests import Request, NewRequest
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def init(db_url, max_request_in_table, max_new_request_in_table):
    global max_requests
    global max_new_requests

    max_requests = max_request_in_table
    max_new_requests = max_new_request_in_table
    init_repo(db_url)


def get_all_new_requests():
    return get_all(NewRequest)


def get_all_requests():
    return get_all(Request)


def add_new_requests(requests):
    logger.info("Adding new requests...")
    if get_count_of_new_requests() > max_new_requests:
        raise OverflowError('Limit of new requests has been reached')
    if len(requests) < 1:
        logger.info("No requests to add")
        return
    add_requests(requests)
    logger.info("New requests added")


def merge_new_requests():
    if get_count_of_requests() > max_requests:
        logger.warning("Limit of request has been reached")
    new_requests = get_all_new_requests()
    requests = list()
    for new_request in new_requests:
        requests.append(Request(vector=new_request.vector))
    add_requests(requests)
    truncate_table(NewRequest)


def get_count_of_new_requests():
    return get_count_of_rows(NewRequest)


def get_count_of_requests():
    return get_count_of_rows(Request)
