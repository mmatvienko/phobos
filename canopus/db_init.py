import os, logging
import psycopg2 as psql

from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from canopus.secrets import set_test_env

def setup_db():
    if not os.environ["ENV_TYPE"]:
        raise ValueError("ENV_TYPE environment variable is not set, but is needed to run")
    
    db_name = os.environ["ENV_TYPE"] + "_db"
    con = psql.connect(
        host="localhost", 
        user="marcmatvienko", 
        password=None, 
    )
    con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = con.cursor()
    cur.execute("SELECT * from pg_database")
    res = [x[1] for x in cur.fetchall()]
    print(res)
    if db_name not in res:
        # make the db
        cur.execute("CREATE DATABASE " + db_name)
        logging.info(f"Set up succeeded. Created {db_name}")
    else:
        logging.info(f"{db_name} already exists")
    cur.close()


if __name__ == "__main__":
    set_test_env()
    setup_db()