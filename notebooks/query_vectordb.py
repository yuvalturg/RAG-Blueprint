#!/usr/bin/env python3

import psycopg2
import argparse
import pandas as pd
from tabulate import tabulate

def connect_to_db(host, port, dbname, user, password):
    """Connect to PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def list_tables(conn):
    """List all tables in the database"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
    """)
    tables = cursor.fetchall()
    cursor.close()
    return [table[0] for table in tables]

def query_table(conn, table_name, limit=10, exclude_vector_cols=True):
    """Query a table and return results as a DataFrame"""
    cursor = conn.cursor()
    
    # Get column names first
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 0")
    column_names = [desc[0] for desc in cursor.description]
    
    # If excluding vector columns, identify them
    if exclude_vector_cols:
        vector_cols = []
        for col in column_names:
            try:
                cursor.execute(f"SELECT pg_typeof({col}) FROM {table_name} LIMIT 1")
                col_type = cursor.fetchone()[0]
                if 'vector' in col_type.lower():
                    vector_cols.append(col)
            except:
                pass
        
        # Create a query that selects all columns except vector columns
        if vector_cols:
            select_cols = [col for col in column_names if col not in vector_cols]
            query = f"SELECT {', '.join(select_cols)} FROM {table_name} LIMIT {limit}"
        else:
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
    else:
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    # Get the actual column names from the cursor description
    columns = [desc[0] for desc in cursor.description]
    
    cursor.close()
    return pd.DataFrame(rows, columns=columns)

def describe_table(conn, table_name):
    """Get the structure of a table"""
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT column_name, data_type, character_maximum_length 
        FROM information_schema.columns 
        WHERE table_name = '{table_name}'
    """)
    columns = cursor.fetchall()
    cursor.close()
    return pd.DataFrame(columns, columns=['Column', 'Type', 'Max Length'])

def main():
    parser = argparse.ArgumentParser(description='Query PostgreSQL Vector Database')
    parser.add_argument('--host', default='localhost', help='Database host')
    parser.add_argument('--port', default='5432', help='Database port')
    parser.add_argument('--dbname', required=True, help='Database name')
    parser.add_argument('--user', required=True, help='Database user')
    parser.add_argument('--password', required=True, help='Database password')
    parser.add_argument('--table', help='Specific table to query')
    parser.add_argument('--limit', type=int, default=10, help='Limit number of rows')
    parser.add_argument('--show-vectors', action='store_true', help='Include vector columns in output')
    
    args = parser.parse_args()
    
    conn = connect_to_db(args.host, args.port, args.dbname, args.user, args.password)
    if not conn:
        return
    
    try:
        tables = list_tables(conn)
        print(f"Found {len(tables)} tables: {', '.join(tables)}\n")
        
        if args.table:
            if args.table not in tables:
                print(f"Table '{args.table}' not found in database.")
                return
            
            print(f"Table structure for '{args.table}':")
            structure = describe_table(conn, args.table)
            print(tabulate(structure, headers='keys', tablefmt='psql'))
            print()
            
            print(f"Sample data from '{args.table}' (limit: {args.limit}):")
            df = query_table(conn, args.table, args.limit, not args.show_vectors)
            print(tabulate(df, headers='keys', tablefmt='psql'))
        else:
            for table in tables:
                print(f"Table structure for '{table}':")
                structure = describe_table(conn, table)
                print(tabulate(structure, headers='keys', tablefmt='psql'))
                print()
                
                print(f"Sample data from '{table}' (limit: {args.limit}):")
                df = query_table(conn, table, args.limit, not args.show_vectors)
                print(tabulate(df, headers='keys', tablefmt='psql'))
                print("\n" + "-" * 80 + "\n")
    
    finally:
        conn.close()

if __name__ == "__main__":
    main()
