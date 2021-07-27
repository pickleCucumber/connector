import cx_Oracle
import pandas as pd
import os
import numpy as np
import sys

bor=cx_Oracle.connect('login', 'password', 'DB_name')
#запрос к dwh

def read_query(connection, query):
    c = connection.cursor()
    try:
      c.execute(query)
      names=[x[0] for x in c.description]
      rows=c.fetchall()
      return pd.DataFrame(rows, columns=names)
    finally:
      if c is not None:
        c.close()
df=ready_query(bor, 'select * from DB_name')
