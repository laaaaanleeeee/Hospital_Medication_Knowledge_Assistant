import os
import json
import time

from dotenv import load_dotenv
from firecrawl import Firecrawl


# CONFIG

URL_FILE = "../data/drug_urls.json"
RAW_DIR = "../data/raw"
STATE_FILE = "../data/scrape_state.json"

RETRY_DELAYS = [15, 30, 60]
PAGE_DELAY = 15


# SETUP

load_dotenv()

app = Firecrawl(
    api_key=os.getenv("FIRECRAWL_API_KEY")
)

os.makedirs(RAW_DIR, exist_ok=True)


# LOAD URLS

with open(
    URL_FILE,
    "r",
    encoding="utf-8"
) as f:

    drug_urls = json.load(f)

print(
    f"Total drug URLs: {len(drug_urls)}"
)


# LOAD CHECKPOINT

last_completed_index = -1

if os.path.exists(STATE_FILE):

    with open(
        STATE_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        state = json.load(f)

    last_completed_index = state.get(
        "last_completed_index",
        -1
    )

    print(
        f"Last completed index: "
        f"{last_completed_index}"
    )


start_index = last_completed_index + 1


# SCRAPE DRUG PAGES

for index in range(
    start_index,
    len(drug_urls)
):

# for index in range(
#     start_index,
#     min(start_index + 3, len(drug_urls))
# ):

    url = drug_urls[index]

    print("\n" + "=" * 60)
    print(
        f"Processing "
        f"{index + 1}/{len(drug_urls)}"
    )

    print(f"URL: {url}")

    result = None

    # RETRY

    for attempt in range(
        len(RETRY_DELAYS) + 1
    ):

        try:

            result = app.scrape(
                url,
                formats=["markdown"],
                only_main_content=True
            )

            break

        except Exception as e:

            print(
                f"Request failed "
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

                exit(1)

    # GET MARKDOWN

    markdown = result.markdown

    if not markdown:

        print(
            "Warning: Empty markdown."
        )

        continue

    # CREATE FILE NAME

    filename = (
        f"drug_{index + 1:03d}.md"
    )

    filepath = os.path.join(
        RAW_DIR,
        filename
    )

    # SAVE RAW MARKDOWN

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(markdown)

    print(
        f"Saved: {filepath}"
    )

    # SAVE CHECKPOINT

    state = {
        "last_completed_index": index,
        "total_scraped": index + 1
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

    # RATE LIMIT PROTECTION

    if index < len(drug_urls) - 1:

        print(
            f"Waiting {PAGE_DELAY} seconds..."
        )

        time.sleep(PAGE_DELAY)


# FINISHED

print("\n" + "=" * 60)
print("SCRAPING FINISHED")
print("=" * 60)

print(
    f"Total scraped: "
    f"{len(drug_urls)}"
)

# print(
#     f"Total scraped: "
#     f"{min(start_index + 3, len(drug_urls))}"
# )

print(
    f"Raw data directory: "
    f"{RAW_DIR}"
)