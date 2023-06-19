import os, os.path as osp
import logging
from collections import namedtuple
import torch
import datetime

Session = namedtuple('Session', ['sess_dir'])


def generate_new_session(root, obj=None):
    from datetime import datetime
    dt = datetime.now()
    dt_str = dt.strftime("%Y-%m-%d %H:%M:%S") # Convert datetime to a string
    if obj is None:
        obj = ""
    sess_dir = osp.join(root, f"exp-{obj}-{dt_str}")
    if not osp.exists(sess_dir):
        os.makedirs(sess_dir, exist_ok=True)
    session = Session(sess_dir)
    return session


def save_model(sess, model, epoch):
    modelpath = osp.join(sess.sess_dir, f"epoch-{epoch}.pth")
    torch.save(model.cpu(), modelpath)


def load_model_session(session, epoch):
    modelpath = osp.join(session.sess_dir, f"epoch-{epoch}.pth")
    assert osp.exists(modelpath)
    return modelpath


def create_logger(name, file_dir):
    filename = osp.join(file_dir, "exp-log.log")
    # Configure the logger with timestamp
    logging.basicConfig(filename=filename, level=logging.INFO, format='%(asctime)s - %(message)s')

    # Create a logger object
    logger = logging.getLogger(name)

    # Add a custom handler to write to the log file
    file_handler = logging.FileHandler('log.txt')
    logger.addHandler(file_handler)

    # Add a stream handler to print log messages to the console
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    # Set the format for the log messages
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Set the formatter for the handlers
    file_handler.setFormatter(log_formatter)
    stream_handler.setFormatter(log_formatter)

    # Add colors to the stream handler
    class ColorFormatter(logging.Formatter):
        BLACK = '\033[30m'
        RED = '\033[31m'
        GREEN = '\033[32m'
        YELLOW = '\033[33m'
        BLUE = '\033[34m'
        MAGENTA = '\033[35m'
        CYAN = '\033[36m'
        WHITE = '\033[37m'
        RESET = '\033[0m'

        def format(self, record):
            color = self._get_color(record.levelname)
            message = super().format(record)
            return f'{color}{message}{self.RESET}'

        def _get_color(self, levelname):
            if levelname == 'DEBUG':
                return self.CYAN
            elif levelname == 'INFO':
                return self.GREEN
            elif levelname == 'WARNING':
                return self.YELLOW
            elif levelname == 'ERROR':
                return self.RED
            elif levelname == 'CRITICAL':
                return self.MAGENTA
            else:
                return self.RESET

    color_formatter = ColorFormatter('%(asctime)s::%(levelname)s:: %(message)s')

    stream_handler.setFormatter(color_formatter)
    return logger


