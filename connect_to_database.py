from DATA.login import login_info
import psycopg2

dbname_, user_, password_, host_ = login_info()

try:
    with psycopg2.connect(
        dbname= dbname_,
        user= user_,
        password= password_,
        host= host_
    ) as conn:
        cur = conn.cursor()
    cur.execute('rollback')
    print("Database connected successfully")
except:
    print("Database not connected successfully")