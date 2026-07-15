from database import get_connection

conn = get_connection()
cur = conn.cursor()

cur.execute("""
SELECT
    id,
    name,
    industry
FROM companies
ORDER BY name
""")

rows = cur.fetchall()

for row in rows:
    print(row)

cur.close()
conn.close()