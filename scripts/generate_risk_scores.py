from database import get_connection
from datetime import date


def calculate_score(layoff_events, total_laid_off, latest_layoff, max_percentage):
    score = 0
    reasons = []

    # 1. Layoff Events
    if layoff_events >= 1:
        score += 10
        reasons.append("At least one layoff event")

    if 2 <= layoff_events <= 3:
        score += 15
        reasons.append("Multiple layoff events")

    elif layoff_events >= 4:
        score += 25
        reasons.append("Frequent layoff events")

    # 2. Employees Laid Off
    if total_laid_off >= 100:
        score += 10
        reasons.append("100+ employees laid off")

    if total_laid_off >= 500:
        score += 15
        reasons.append("500+ employees laid off")

    if total_laid_off >= 5000:
        score += 25
        reasons.append("5000+ employees laid off")

    # 3. Percentage Laid Off
    if max_percentage >= 10:
        score += 10
        reasons.append("10%+ workforce reduction")

    if max_percentage >= 25:
        score += 15
        reasons.append("25%+ workforce reduction")

    if max_percentage >= 50:
        score += 20
        reasons.append("50%+ workforce reduction")

    # 4. Recent Layoff
    if latest_layoff and latest_layoff.year >= 2024:
        score += 15
        reasons.append("Recent layoff activity")

    # Cap score
    score = min(score, 100)

    # Risk Level
    if score <= 20:
        level = "Very Low"
    elif score <= 40:
        level = "Low"
    elif score <= 60:
        level = "Medium"
    elif score <= 80:
        level = "High"
    else:
        level = "Critical"

    return score, level, "; ".join(reasons)


# ----------------------------
# Database Connection
# ----------------------------

conn = get_connection()
cur = conn.cursor()

# Remove previous scores
cur.execute("DELETE FROM workforce_risk;")

# Aggregate company statistics
cur.execute("""
SELECT
    c.id,
    c.name,

    COUNT(l.id) AS layoff_events,

    SUM(COALESCE(l.employees_laid_off,0)) AS total_laid_off,

    MAX(l.layoff_date) AS latest_layoff,

    MAX(COALESCE(l.percentage_laid_off,0)) AS max_percentage

FROM companies c

LEFT JOIN layoffs l
ON c.id = l.company_id

GROUP BY c.id, c.name;
""")

rows = cur.fetchall()

values = []

for row in rows:
    company_id = row[0]
    company_name = row[1]
    layoff_events = row[2]
    total_laid_off = row[3]
    latest_layoff = row[4]
    max_percentage = row[5]

    score, level, reasons = calculate_score(
        layoff_events,
        total_laid_off,
        latest_layoff,
        max_percentage
    )
    
    values.append((
        company_id,
        score,
        level,
        layoff_events,
        total_laid_off,
        latest_layoff,
        max_percentage,
        reasons,
        "rule_v1"
    ))

print(f"Batch inserting {len(values)} risk scores...")
from psycopg2.extras import execute_values
execute_values(cur, """
    INSERT INTO workforce_risk(
        company_id,
        workforce_risk_score,
        risk_level,
        layoff_events,
        total_employees_laid_off,
        latest_layoff,
        max_percentage_laid_off,
        risk_reasons,
        model_version
    )
    VALUES %s
""", values)

conn.commit()
cur.close()
conn.close()

print("=" * 50)
print(f"Generated risk scores for {len(values)} companies.")
print("Model Version : rule_v1")
print("=" * 50)


