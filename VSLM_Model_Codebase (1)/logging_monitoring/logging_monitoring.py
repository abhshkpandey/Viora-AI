import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VSLM Logging")

def log_message(message):
    logger.info(message)

if __name__ == "__main__":
    log_message("Logging system initialized.")
