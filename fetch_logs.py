#!/usr/bin/env python3
# usage: python3 fetch_logs.py

from tapi_yandex_metrika import YandexMetrikaLogsapi
from datetime import date, timedelta
import ast, csv, sys, time, requests

TOKEN   = ""
COUNTER = ""

DATE2 = date.today() - timedelta(days=1)
DATE1 = DATE2 - timedelta(days=30)

PARAMS = {
    "fields": ",".join([
        "ym:s:visitID",
        "ym:s:date",
        "ym:s:clientID",
        "ym:s:isNewUser",           # new vs returning visitor
        "ym:s:visitDuration",       # total session time (seconds)
        "ym:s:pageViews",           # screens/pages seen in session
        "ym:s:goalsID",             # which goals fired per session
        "ym:s:deviceCategory",      # desktop / mobile / tablet
        "ym:s:browser",             # Chrome, Safari, etc.
        "ym:s:operatingSystem",     # iOS, Android, Windows, etc.
        "ym:s:UTMSource",           # UTM source (e.g. google, vk, direct)
        "ym:s:UTMMedium",           # UTM medium (e.g. cpc, organic, referral)
        "ym:s:regionCity",          # city of the visitor
        "ym:s:parsedParamsKey1",    # game event param keys
        "ym:s:parsedParamsKey2",    # game event param values
    ]),
    "source": "visits",
    "date1": DATE1,
    "date2": DATE2,
}

# Fields that belong to a level attempt (reset after each level_status event)
LEVEL_FIELDS = {"level", "theme_letter", "words_found", "words_total",
                "completion_pct", "duration_sec", "hints_used",
                "drop_off_pct", "level_status",
                "level_seq",              # ordinal position of level in session
                "time_to_first_word_sec"} # seconds from level load to first found word

# Fields that are session-level and persist across level attempts
SESSION_FIELDS = {"ab_group", "is_returning",
                  "visit_count",   # cumulative visits by this user
                  "hour_of_day"}   # local hour when session started (0–23)

UNROLLED_COLUMNS = [
    "session_id", "date", "client_id",
    "is_new_user", "visit_duration_sec", "page_views",
    "device_category", "browser", "os", "utm_source", "utm_medium", "region",
    "ab_group", "is_returning", "visit_count", "hour_of_day",
    "level", "theme_letter", "level_status", "level_seq",
    "words_found", "words_total", "completion_pct",
    "duration_sec", "hints_used", "drop_off_pct",
    "time_to_first_word_sec",
]


def unroll_session(row):
    """
    Convert one session row from the Logs API into a list of dicts,
    one dict per level attempt (each level_status event = one attempt).
    Returns an empty list for sessions with no level events.
    """
    try:
        keys = ast.literal_eval(row["ym:s:parsedParamsKey1"])
        vals = ast.literal_eval(row["ym:s:parsedParamsKey2"])
    except (ValueError, SyntaxError):
        return []

    base = {
        "session_id":       row["ym:s:visitID"],
        "date":             row["ym:s:date"],
        "client_id":        row["ym:s:clientID"],
        "is_new_user":      row["ym:s:isNewUser"],
        "visit_duration_sec": row["ym:s:visitDuration"],
        "page_views":       row.get("ym:s:pageViews", ""),
        "device_category":  row.get("ym:s:deviceCategory", ""),
        "browser":          row.get("ym:s:browser", ""),
        "os":               row.get("ym:s:operatingSystem", ""),
        "utm_source":       row.get("ym:s:UTMSource", ""),
        "utm_medium":       row.get("ym:s:UTMMedium", ""),
        "region":           row.get("ym:s:regionCity", ""),
    }

    attempts = []
    ctx = {}  # accumulated key→value context within the current level attempt

    for k, v in zip(keys, vals):
        ctx[k] = v

        if k == "level_status":
            attempt = dict(base)
            for field in (SESSION_FIELDS | LEVEL_FIELDS):
                attempt[field] = ctx.get(field, "")
            ir = str(attempt.get("is_returning", "0")).lower()
            attempt["is_returning"] = "1" if ir in ("1", "true") else "0"
            # Ensure level_status is set from current event (not stale ctx)
            attempt["level_status"] = v
            attempts.append(attempt)

            # Reset level-specific context; keep session-level fields
            for field in LEVEL_FIELDS:
                ctx.pop(field, None)

    return attempts

client = YandexMetrikaLogsapi(
    access_token=TOKEN,
    default_url_params={"counterId": COUNTER},
    wait_report=True,
)

print(f"Evaluating · counter {COUNTER} · {DATE1} → {DATE2}")
evaluation = client.evaluate().get(params=PARAMS)
if not evaluation["log_request_evaluation"]["possible"]:
    print("Not enough data yet — try again tomorrow.")
    sys.exit(0)

print("Creating report…")
report = client.create().post(params=PARAMS)
request_id = report["log_request"]["request_id"]
print(f"Request ID: {request_id}")

while True:
    info = client.info(requestId=request_id).get()
    status = info["log_request"]["status"]
    print(f"Status: {status}")
    if status == "processed":
        break
    if status in ("cleaned_by_user", "processing_failed"):
        print("Request failed.")
        sys.exit(1)
    time.sleep(10)

parts = info["log_request"].get("parts", [])
print(f"Downloading {len(parts)} part(s)…")

lines_all = []
for part in parts:
    r = requests.get(
        f"https://api-metrika.yandex.net/management/v1/counter/{COUNTER}"
        f"/logrequest/{request_id}/part/{part['part_number']}/download",
        headers={"Authorization": f"OAuth {TOKEN}"},
    )
    r.raise_for_status()
    lines = [l for l in r.text.splitlines() if l]
    if not lines_all:
        lines_all.append(lines[0])  # header once
    lines_all.extend(lines[1:])

if len(lines_all) <= 1:
    print("No rows returned.")
    sys.exit(0)

header = lines_all[0].split("\t")
sessions = [dict(zip(header, l.split("\t"))) for l in lines_all[1:]]
print(f"Fetched {len(sessions)} sessions.")

all_attempts = []
for session in sessions:
    all_attempts.extend(unroll_session(session))

print(f"Unrolled to {len(all_attempts)} level attempts.")

out = f"reports/metrica-sessions-{DATE1}-{DATE2}.tsv"
with open(out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=UNROLLED_COLUMNS,
                            delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(all_attempts)

print(f"Saved → {out}")

client.clean(requestId=request_id).post()
print("Log request cleaned.")
