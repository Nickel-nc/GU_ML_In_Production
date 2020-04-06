import logging
import configparser

dictLogConfig = {
    "version": 1,
    "handlers": {
        "mainHandler": {
            "class": "logging.FileHandler",
            "formatter": "mainFormatter",
            "filename": "logs/logger.log"
        },
        "ProcessingHandler": {
            "class": "logging.FileHandler",
            "formatter": "processFormatter",
            "filename": "logs/logger.log"
        },
    },
    "loggers": {
        "runSession": {
            "handlers": ["mainHandler"],
            "level": "INFO",
        },
        "progress": {
            "handlers": ["ProcessingHandler"],
            "level": "INFO",
        }

    },
    "formatters": {
        "mainFormatter": {
            "format": "\n%(asctime)s %(levelname)s %(message)s\n"
        },
        "processFormatter": {
            "format": "%(filename)s[LINE:%(lineno)d] %(levelname)s %(message)s "
        }
    }
}