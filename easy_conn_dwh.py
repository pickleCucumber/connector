import os
import gc
import pickle
import getpass

import pandas as pd
import cx_Oracle as cx
from pandas import read_sql_query

PASSWORD = getpass.getpass()

ROLE = 'YOUR_NAME'
DSN_TNS = '(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST=dome-db.isb)(PORT=1521))(CONNECT_DATA=(SID=DOME)))'


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
select 
 cl.AGREEMENT_ID
,(nvl(r.ON_BALANCE_PRINCIPAL_EXPOSURE,0) + nvl(r.NET_ON_BALANCE_INTEREST_EXP,0)*nvl(cl.IFRS_CCF_RATE,0)) as EAD
,cl.PD_FINAL
,cl.LGD_FINAL
,cl.IFRS_DEFAULT_FLG
,dm.loan_type_id
FROM dwh.recode_agreement_ss s
right join DACU.GDS_ON_BALANCE_EXPOSURE r
On s.ss_01='VBX' and s.ss_02='CFT'
And r.contract_id=s.ss_01_agreement_rk
And substr(s.ss_02_agreement_id,1,1)!='-'
inner join scoring_app.scr_dm_portfolio dm
on dm.agreement_rk=nvl(s.ss_02_agreement_rk,r.contract_id)
and dm.report_dt=last_day(to_date('202107', 'yyyymm'))
inner join RISK_OLAP.HIST_IFRS_CLOSING_AGR cl
on dm.AGREEMENT_ID = cl.AGREEMENT_ID 
and dm.REPORT_DT = cl.REPORT_DT
where r.ON_BALANCE_PRINCIPAL_EXPOSURE + r.NET_ON_BALANCE_INTEREST_EXP > 0
and ENTRY_DT = 202107
and cl.SCENARIO_ID = 1

"""
df = upload_data(query_get_sample)
df.head()
