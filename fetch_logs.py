#!/usr/bin/env python3
# usage: python3 fetch_logs.py
# output: reports/metrica-sessions-<date1>-<date2>.tsv
# install: pip install tapi-yandex-metrika

from tapi_yandex_metrika import YandexMetrikaLogsapi
from datetime import date, timedelta
import csv, sys, time, requests

TOKEN   = ""
COUNTER = ""

DATE2 = date.today() - timedelta(days=1)
DATE1 = DATE2 - timedelta(days=30)

PARAMS = {
    "fields": ",".join([
        "ym:s:visitID",
        "ym:s:date",
        "ym:s:clientID",
        "ym:s:isNewUser",         # revisit rate
        "ym:s:visitDuration",     # time on site
        "ym:s:goalsID",           # which goals fired per session
        "ym:s:parsedParamsKey1",  # game param keys   e.g. ab_group, level_status
        "ym:s:parsedParamsKey2",  # game param values e.g. A, completed
    ]),
    "source": "visits",
    "date1": DATE1,
    "date2": DATE2,
}

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
rows = [dict(zip(header, l.split("\t"))) for l in lines_all[1:]]

out = f"reports/metrica-sessions-{DATE1}-{DATE2}.tsv"
with open(out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=header, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} sessions → {out}")

client.clean(requestId=request_id).post()
print("Log request cleaned.")
