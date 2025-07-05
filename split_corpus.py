"""Split the combined publications export into one CSV per researcher.

Input:  all_university_publications.csv (one row per publication, with a
        "Researcher Name" column). This file was produced by a publication
        scraper that is not part of this repository, so the committed
        researcher_corpora/ folder is a frozen snapshot. See README.md.
Output: researcher_corpora/<Researcher_Name>_publications.csv
"""
import csv
import os
import re
import sys
from collections import defaultdict

INPUT_CSV = "all_university_publications.csv"
OUTPUT_DIR = "researcher_corpora"


def sanitize_filename(name):
    return re.sub(r'\W+', '_', name.strip())


def main():
    if not os.path.exists(INPUT_CSV):
        sys.exit(
            f"{INPUT_CSV} not found. The committed {OUTPUT_DIR}/ folder is a frozen "
            "snapshot; this script is only needed if you have a new raw export."
        )

    publications_by_researcher = defaultdict(list)
    with open(INPUT_CSV, mode="r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            publications_by_researcher[row.get("Researcher Name", "Unknown")].append(row)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for researcher, pubs in publications_by_researcher.items():
        filepath = os.path.join(OUTPUT_DIR, sanitize_filename(researcher) + "_publications.csv")
        with open(filepath, mode="w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=pubs[0].keys())
            writer.writeheader()
            writer.writerows(pubs)
        print(f"Saved {len(pubs)} publications for: {researcher}")

    print(f"Wrote {len(publications_by_researcher)} researcher files to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
