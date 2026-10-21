import logging

logger = logging.getLogger(__name__)


def fetch_remote(source):
    logger.debug("fetching from %s", source.name)
    items = source.list()
    if not items:
        logger.warning("%s returned no items", source.name)
    logger.info("fetched %d items from %s", len(items), source.name)
    return items
