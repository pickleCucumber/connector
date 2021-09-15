import os
import gc
import pickle
import getpass

import pandas as pd
import cx_Oracle as cx
from pandas import read_sql_query

PASSWORD = getpass.getpass()

ROLE = 'YOUR_NAME'
DSN_TNS = '(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST=host)(PORT=1521))(CONNECT_DATA=(SID=DOME)))'


def upload_data(query, chunksize=10 ** 5):
    with cx.connect(ROLE, PASSWORD, DSN_TNS, encoding="UTF-8", nencoding="UTF-8") as  conn:
        try:
            iterator = read_sql_query(query, conn, chunksize=chunksize)
            df_data = pd.concat([df for df in tqdm(iterator)], ignore_index=True, copy=False)
        except:
            df_data = read_sql_query(query, conn)

    return df_data
query_get_sample=\
f"""
select *
from your_table

"""
df = upload_data(query_get_sample)
df.head()
