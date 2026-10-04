import os
import json
import time

from dotenv import load_dotenv
from firecrawl import Firecrawl


# CONFIG

BASE_URL = "https://thuocbietduoc.com.vn/thuoc/drgsearch.aspx"

TARGET_SAMPLES = 500
MAX_PAGES = 50

URL_FILE = "../data/drug_urls.json"
STATE_FILE = "../data/crawl_state.json"

RETRY_DELAYS = [15, 30, 60]
PAGE_DELAY = 15


# SETUP

load_dotenv()

app = Firecrawl(
    api_key=os.getenv("FIRECRAWL_API_KEY")
)

os.makedirs("../data", exist_ok=True)


# LOAD PREVIOUS PROGRESS

drug_urls = set()

if os.path.exists(URL_FILE):
    with open(URL_FILE, "r", encoding="utf-8") as f:
        drug_urls = set(json.load(f))

    print(f"Loaded existing URLs: {len(drug_urls)}")


last_completed_page = 0

if os.path.exists(STATE_FILE):
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)

    last_completed_page = state.get("last_completed_page", 0)

    print(
        f"Last completed page: "
        f"{last_completed_page}"
    )


start_page = last_completed_page + 1


# CRAWL PAGINATION

for page in range(start_page, MAX_PAGES + 1):

    # Stop if enough samples
    if len(drug_urls) >= TARGET_SAMPLES:
        break

    if page == 1:
        url = BASE_URL
    else:
        url = f"{BASE_URL}?page={page}"

    print("\n" + "=" * 60)
    print(f"Crawling listing page {page}...")
    print(f"URL: {url}")
    print(f"Current unique URLs: {len(drug_urls)}")

    result = None

    # RETRY

    for attempt in range(len(RETRY_DELAYS) + 1):

        try:
            result = app.scrape(
                url,
                formats=["links"],
                only_main_content=True
            )

            break

        except Exception as e:

            print(
                f"\nRequest failed "
                f"(attempt {attempt + 1})"
            )

            print(f"Error: {e}")

            if attempt < len(RETRY_DELAYS):

                delay = RETRY_DELAYS[attempt]

                print(
                    f"Retrying after "
                    f"{delay} seconds..."
                )

                time.sleep(delay)

            else:

                print(
                    "\nMaximum retries reached."
                )

                print(
                    "Progress has been saved."
                )

                print(
                    "Run the script again "
                    "to continue."
                )

                exit(1)

    # EXTRACT DRUG URLS

    page_drug_urls = [
        link
        for link in result.links
        if "/thuoc-" in link
        and link.endswith(".aspx")
    ]

    print(
        f"Drug URLs found on page: "
        f"{len(page_drug_urls)}"
    )

    # Remove duplicates
    before_count = len(drug_urls)

    drug_urls.update(page_drug_urls)

    new_count = len(drug_urls) - before_count

    print(
        f"New unique URLs: {new_count}"
    )

    print(
        f"Total unique URLs: "
        f"{len(drug_urls)}"
    )

    # SAVE URLS

    with open(
        URL_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            sorted(drug_urls),
            f,
            ensure_ascii=False,
            indent=2
        )

    # SAVE CHECKPOINT

    state = {
        "last_completed_page": page,
        "total_urls": len(drug_urls)
    }

    with open(
        STATE_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            state,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"Progress saved "
        f"(page {page})"
    )

    # STOP CONDITION

    if len(drug_urls) >= TARGET_SAMPLES:

        print(
            "\nTarget reached!"
        )

        break

    # RATE LIMIT PROTECTION

    print(
        f"Waiting {PAGE_DELAY} seconds "
        f"before next page..."
    )

    time.sleep(PAGE_DELAY)


# FINAL RESULT

print("\n" + "=" * 60)
print("CRAWLING FINISHED")
print("=" * 60)

print(
    f"Total unique drug URLs: "
    f"{len(drug_urls)}"
)

print(
    f"Saved to: {URL_FILE}"
)