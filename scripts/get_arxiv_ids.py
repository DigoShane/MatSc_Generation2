import requests
import xml.etree.ElementTree as ET
import time
import os


categories = [
    "cond-mat.supr-con",
    "cond-mat.str-el",
    "cond-mat.mtrl-sci",
    "cond-mat.dis-nn"
]

BASE_URL = "http://export.arxiv.org/api/query"

OUTPUT_DIR = "data_v4/arxiv_ids"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_ids(category, max_results=20000):

    ids = []
    start = 0
    batch_size = 100

    while start < max_results:

        print(f"{category} start={start}")

        params = {
            "search_query": f"cat:{category}",
            "start": start,
            "max_results": batch_size
        }

        try:

            r = requests.get(BASE_URL, params=params, timeout=30)

            if r.status_code != 200:
                print("HTTP error", r.status_code)
                time.sleep(5)
                continue

            if not r.text.strip():
                print("Empty response")
                time.sleep(5)
                continue

            root = ET.fromstring(r.text)

        except Exception as e:

            print("Request failed:", e)
            time.sleep(5)
            continue

        entries = root.findall("{http://www.w3.org/2005/Atom}entry")

        if not entries:
            break

        for entry in entries:

            id_url = entry.find("{http://www.w3.org/2005/Atom}id").text
            arxiv_id = id_url.split("/")[-1]
            ids.append(arxiv_id)

        start += batch_size

        time.sleep(3)

    return ids


def main():

    for cat in categories:

        ids = fetch_ids(cat)

        outfile = os.path.join(OUTPUT_DIR, f"{cat}.txt")

        with open(outfile, "w") as f:

            for i in ids:
                f.write(i + "\n")

        print(cat, "Total IDs:", len(ids))


if __name__ == "__main__":
    main()
