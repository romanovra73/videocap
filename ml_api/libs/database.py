import psycopg2
from psycopg2.extensions import register_adapter, AsIs
import numpy as np

def adapt_numpy_float32(value):
    return AsIs(value)

def get_postgres_connection():
    register_adapter(np.float32, adapt_numpy_float32)
    try:
        db = psycopg2.connect(
            host="localhost",
            port=5432,
            user = "videocap",
            password = "some-password",
            database = "ml",
            options = "-c client_encoding=utf-8")
    except:
        print('DB connection failed')
        return False
    return db

