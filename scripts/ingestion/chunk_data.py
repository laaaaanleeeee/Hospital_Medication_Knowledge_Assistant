import os
import re
import json


PROCESSED_DIR = "../data/processed"
OUTPUT_DIR = "../data/chunks"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "chunks.jsonl")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


SKIP_SECTIONS = {
    "Thông tin công ty",
    "Mục lục",
}


def load_urls():
    url_file = "../data/drug_urls.json"

    with open(url_file, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_drug_name(text):
    match = re.search(
        r"^#\s+(.+)$",
        text,
        re.MULTILINE
    )

    if match:
        return match.group(1).strip()

    return "Unknown"


def split_by_sections(text):
    pattern = r"(?=^#{2,3}\s+.+$)"

    sections = re.split(
        pattern,
        text,
        flags=re.MULTILINE
    )

    results = []

    for section in sections:

        section = section.strip()

        if not section:
            continue

        if not re.match(r"^#{2,3}\s+", section):
            continue

        results.append(section)

    return results


def recursive_split(text):
    if len(text) <= CHUNK_SIZE:
        return [text]

    separators = [
        "\n\n",
        "\n",
        ". ",
        "; ",
        ", ",
        " "
    ]

    parts = [text]

    for separator in separators:

        if all(
            len(part) <= CHUNK_SIZE
            for part in parts
        ):
            break

        new_parts = []

        for part in parts:

            if len(part) <= CHUNK_SIZE:
                new_parts.append(part)
            else:
                new_parts.extend(
                    part.split(separator)
                )

        parts = new_parts

    chunks = []
    current = ""

    for part in parts:

        part = part.strip()

        if not part:
            continue

        if len(part) <= CHUNK_SIZE:

            if (
                current
                and len(current) + len(part) + 1
                > CHUNK_SIZE
            ):
                chunks.append(current.strip())
                current = ""

            current += (
                " " + part
                if current
                else part
            )

        else:

            start = 0

            while start < len(part):

                end = start + CHUNK_SIZE

                chunk = part[start:end].strip()

                if chunk:
                    chunks.append(chunk)

                start = end - CHUNK_OVERLAP

    if current.strip():
        chunks.append(current.strip())

    return chunks


def build_chunk_text(
    drug_name,
    section,
    content
):
    return (
        f"Thuốc: {drug_name}\n"
        f"Phần: {section}\n\n"
        f"{content}"
    )


def process_file(
    filepath,
    drug_id,
    source_url
):

    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as f:
        text = f.read()

    drug_name = extract_drug_name(text)

    sections = split_by_sections(text)

    results = []

    for section in sections:

        lines = section.splitlines()

        if not lines:
            continue

        heading = lines[0].strip()

        section_name = re.sub(
            r"^#{2,3}\s*",
            "",
            heading
        ).strip()

        if section_name in SKIP_SECTIONS:
            continue

        content = "\n".join(
            lines[1:]
        ).strip()

        if not content:
            continue

        content_chunks = recursive_split(
            content
        )

        for chunk_index, content_chunk in enumerate(
            content_chunks
        ):

            final_text = build_chunk_text(
                drug_name=drug_name,
                section=section_name,
                content=content_chunk
            )

            results.append(
                {
                    "text": final_text,
                    "metadata": {
                        "drug_id": drug_id,
                        "drug_name": drug_name,
                        "section": section_name,
                        "source_url": source_url,
                        "chunk_index": chunk_index
                    }
                }
            )

    return results


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    drug_urls = load_urls()

    files = sorted(
        filename
        for filename in os.listdir(PROCESSED_DIR)
        if filename.lower().endswith(".md")
    )

    print(f"Processed files: {len(files)}")
    print(f"URLs:            {len(drug_urls)}")

    all_chunks = []

    for index, filename in enumerate(files):

        drug_id = os.path.splitext(
            filename
        )[0]

        filepath = os.path.join(
            PROCESSED_DIR,
            filename
        )

        source_url = ""

        if index < len(drug_urls):
            source_url = drug_urls[index]

        chunks = process_file(
            filepath=filepath,
            drug_id=drug_id,
            source_url=source_url
        )

        all_chunks.extend(chunks)

        print(
            f"{index + 1:03d}/{len(files)} "
            f"{filename} -> "
            f"{len(chunks)} chunks"
        )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for chunk in all_chunks:

            f.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False
                )
                + "\n"
            )

    print("\nCHUNKING FINISHED")
    print(f"Documents: {len(files)}")
    print(f"Chunks:    {len(all_chunks)}")
    print(f"Output:    {OUTPUT_FILE}")


if __name__ == "__main__":
    main()