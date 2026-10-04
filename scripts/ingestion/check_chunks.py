import os
import json
from collections import Counter


CHUNKS_FILE = "D:\\HMKA\\data\\chunks\\chunks.jsonl"

MIN_CHUNK_LENGTH = 50
MAX_CHUNK_LENGTH = 1000


def load_chunks():

    chunks = []

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                chunks.append(json.loads(line))

    return chunks


def main():

    if not os.path.exists(CHUNKS_FILE):

        print(f"ERROR: File not found: {CHUNKS_FILE}")
        return

    chunks = load_chunks()

    if not chunks:

        print("ERROR: No chunks found.")
        return

    # Basic statistics

    documents = set()

    lengths = []

    sections = Counter()

    short_chunks = 0
    long_chunks = 0

    for chunk in chunks:

        metadata = chunk["metadata"]

        documents.add(
            metadata["drug_id"]
        )

        sections[
            metadata["section"]
        ] += 1

        length = len(chunk["text"])

        lengths.append(length)

        if length < MIN_CHUNK_LENGTH:
            short_chunks += 1

        if length > MAX_CHUNK_LENGTH:
            long_chunks += 1


    # Metadata check

    required_fields = {
        "drug_id",
        "drug_name",
        "section",
        "source_url",
        "chunk_index"
    }

    metadata_errors = 0

    for chunk in chunks:

        metadata = chunk.get(
            "metadata",
            {}
        )

        if not required_fields.issubset(
            metadata.keys()
        ):

            metadata_errors += 1


    # Print summary

    print("\n" + "=" * 50)
    print("CHUNK QUALITY CHECK")
    print("=" * 50)

    print(
        f"Documents             : {len(documents)}"
    )

    print(
        f"Total chunks          : {len(chunks)}"
    )

    print(
        f"Avg chunks/document   : "
        f"{len(chunks) / len(documents):.2f}"
    )

    print(
        f"Average chunk length  : "
        f"{sum(lengths) / len(lengths):.2f} chars"
    )

    print(
        f"Minimum chunk length  : "
        f"{min(lengths)} chars"
    )

    print(
        f"Maximum chunk length  : "
        f"{max(lengths)} chars"
    )

    print(
        f"Chunks < 50 chars    : "
        f"{short_chunks}"
    )

    print(
        f"Chunks > 1000 chars  : "
        f"{long_chunks}"
    )

    print(
        f"Metadata errors       : "
        f"{metadata_errors}"
    )


    # Section statistics
    print("\n" + "=" * 50)
    print("TOP SECTIONS")
    print("=" * 50)

    for section, count in sections.most_common(15):

        print(
            f"{count:5d}  {section}"
        )

    # Final status

    print("\n" + "=" * 50)
    print("STATUS")
    print("=" * 50)

    if (
        len(documents) == 500
        and short_chunks == 0
        and long_chunks == 0
        and metadata_errors == 0
    ):

        print("PASS - Chunking quality looks good.")

    else:

        print(
            "CHECK - Some values need inspection."
        )

    print("=" * 50)


if __name__ == "__main__":
    main()