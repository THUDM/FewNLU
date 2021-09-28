# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains basic logging logic.
"""
import logging
import os

# logging
LOG_NAME = "root"
LOG_PATH = "../save/logs"

names = set()


def __setup_custom_logger(name: str) -> logging.Logger:
    # root_logger = logging.getLogger()
    # root_logger.handlers.clear()

    names.add(name)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(filename)s - %(message)s')

    # Add file handler
    file_handler = logging.FileHandler(os.path.join(LOG_PATH, name + ".txt"))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Add stream handler
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(file_handler)

    return logger


def get_logger(name:str=None) -> logging.Logger:
    if name is None:
        name = LOG_NAME
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    if name in names:
        return logging.getLogger(name)
    else:
        return __setup_custom_logger(name)