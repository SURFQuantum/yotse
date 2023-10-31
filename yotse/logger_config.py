import logging


def configure_logger() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("yotse.log"), logging.StreamHandler()],
    )
    logger = logging.getLogger("qcg.pilotjob")
    logger.setLevel(
        logging.ERROR
    )  # This should set the logging level to ERROR for 'qcg.pilotjob'
    # TODO: this doesn't work. All our logging messages are drowned out by huge amount of qcg logging.
