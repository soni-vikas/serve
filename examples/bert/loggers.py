import logging

loggers = {}

def get_logger(name: str,
               level = logging.INFO,
               format =  "%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s"
    ) -> logging.Logger:
    """
    returns a logger with predefined formatting
    """
    if name in loggers:
        return loggers[name]

    # disable roor logger
    logging.getLogger().handlers = [logging.NullHandler()]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(format))
    logger.addHandler(handler)
    loggers[name] = logger

    return logger