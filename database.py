import sqlite3

def create_data_table():
    conn = sqlite3.connect("data.db", detect_types=sqlite3.PARSE_DECLTYPES)
    
    c = conn.cursor()
    c.execute("""DROP TABLE IF EXISTS data""")
    c.execute("""CREATE TABLE data (score FLOAT)""")
    
    conn.commit()
    conn.close()
    
def create_tables():
    create_data_table()
        
def select(statement):
    conn = sqlite3.connect("data.db", detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()
    c.execute(statement)
    results = c.fetchall()
    conn.commit()
    conn.close()
    return results

def insert(statement, insert):
    conn = sqlite3.connect("data.db", detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()
    c.execute(statement, insert)
    rowid = c.lastrowid
    conn.commit()
    conn.close()
    return rowid

def update(statement):
    conn = sqlite3.connect("data.db", detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()
    c.execute(statement)
    conn.commit()
    conn.close()