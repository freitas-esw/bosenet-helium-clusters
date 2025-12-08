
"""Writer utility classes."""

import contextlib
import os
from typing import Mapping, Optional, Sequence

from absl import logging


class Writer(contextlib.AbstractContextManager):
  """Write data to CSV, as well as logging data to stdout if desired."""

  def __init__(self,
               name: str,
               schema: Sequence[str],
               directory: str = 'logs/',
               iteration_key: Optional[str] = 't',
               log: bool = True):
    """Initialise Writer.

    Args:
      name: file name for CSV.
      schema: sequence of keys, corresponding to each data item.
      directory: directory path to write file to.
      iteration_key: if not None or a null string, also include the iteration
        index as the first column in the CSV output with the given key.
      log: Also log each entry to stdout.
    """
    self._schema = schema
    if not os.path.isdir(directory):
      os.mkdir(directory)
    i = 1
    while os.path.isfile(os.path.join(directory, name + f'_{i:02d}.csv')) and i < 99: i+=1
    if i > 99: raise RuntimeError("Directory has too much files.")
    self._filename = os.path.join(directory, name + f'_{i:02d}.csv')
    self._iteration_key = iteration_key
    self._log = log

  def __enter__(self):
    self._file = open(self._filename, 'w', encoding='UTF-8')
    # write top row of csv
    if self._iteration_key:
      self._file.write(f'{self._iteration_key},')
    self._file.write(','.join(self._schema) + '\n')
    return self

  def write(self, t: int, **data):
    """Writes to file and stdout.

    Args:
      t: iteration index.
      **data: data items with keys as given in schema.
    """
    row = [str(data.get(key, '')) for key in self._schema]
    if self._iteration_key:
      row.insert(0, str(t))
    for key in data:
      if key not in self._schema:
        raise ValueError('Not a recognized key for writer: %s' % key)

    # write the data to csv
    self._file.write(','.join(row) + '\n')

    # write the data to abseil logs
    if self._log:
      logging.info('Iteration %s: %s', t, data)

  def __exit__(self, exc_type, exc_val, exc_tb):
    self._file.close()

