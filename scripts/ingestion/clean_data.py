import os
import re


RAW_DIR = "../data/raw"
CLEAN_DIR = "../data/processed"


STOP_SECTIONS = [
    "## Hỗ trợ khách hàng",
    "## Sản phẩm cùng hoạt chất",
    "## Thuốc liên quan",
    "### Chọn hình thức liên hệ",
    "### Gửi đơn thuốc",
    "### Bình luận",
]


REMOVE_EXACT_LINES = {
    "Tư vấn mua hàng",
    "Gửi đơn thuốc",
    "Sản xuất",
    "Đăng ký",
}


def remove_markdown_links(text):

    pattern = r"\[([^\]]+)\]\([^)]+\)"

    return re.sub(pattern, r"\1", text)


def remove_html_tags(text):

    return re.sub(r"<[^>]+>", "", text)


def clean_line(line):
    line = remove_markdown_links(line)

    line = remove_html_tags(line)

    line = line.replace("\\", "")

    line = re.sub(r"[ \t]+", " ", line)

    return line.strip()


def clean_markdown(text):
    lines = text.splitlines()

    cleaned_lines = []

    for line in lines:

        stripped = line.strip()

        if stripped in STOP_SECTIONS:
            break

        if re.match(r"!\[.*\]\(.*\)", stripped):
            continue

        if stripped in REMOVE_EXACT_LINES:
            continue

        cleaned_line = clean_line(line)

        cleaned_lines.append(cleaned_line)

    result = []

    previous_blank = False

    for line in cleaned_lines:

        if line == "":
            if previous_blank:
                continue

            previous_blank = True
            result.append(line)

        else:
            previous_blank = False
            result.append(line)

    cleaned_text = "\n".join(result)

    cleaned_text = cleaned_text.strip()

    return cleaned_text


def clean_file(filename):
    input_path = os.path.join(
        RAW_DIR,
        filename
    )

    output_path = os.path.join(
        CLEAN_DIR,
        filename
    )

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as f:
        raw_text = f.read()

    cleaned_text = clean_markdown(raw_text)

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(cleaned_text)

    print(f"Cleaned: {filename}")


def main():

    os.makedirs(
        CLEAN_DIR,
        exist_ok=True
    )

    files = sorted(
        filename
        for filename in os.listdir(RAW_DIR)
        if filename.lower().endswith(".md")
    )

    print("=" * 60)
    print("CLEANING MEDICINE DATA")
    print("=" * 60)

    print(f"Raw files found: {len(files)}")
    print()

    for filename in files:

        input_path = os.path.join(
            RAW_DIR,
            filename
        )

        if not os.path.isfile(input_path):
            continue

        clean_file(filename)

    cleaned_files = [
        filename
        for filename in os.listdir(CLEAN_DIR)
        if filename.lower().endswith(".md")
    ]

    print()
    print("=" * 60)
    print("CLEANING FINISHED")
    print("=" * 60)

    print(f"Raw files:     {len(files)}")
    print(f"Cleaned files: {len(cleaned_files)}")
    print(f"Output:        {CLEAN_DIR}")

    if len(files) == len(cleaned_files):
        print("Status:        SUCCESS - all files cleaned")
    else:
        print("Status:        WARNING - file count mismatch")


if __name__ == "__main__":
    main()