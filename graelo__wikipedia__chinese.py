from datasets import load_dataset
import pangu  # Paranoid text spacing for good readability
import re
import json
from tqdm import tqdm
from pathlib import Path


def gen_chinese():
    # For Chinese, we use simple regex to sent_tokenize the given text
    # Reference: https://stackoverflow.com/a/45274695/7420769
    def zng(paragraph):
        for sent in re.findall(u'[^!?。\.\!\?]+[!?。\.\!\?]?', paragraph, flags=re.U):
            yield sent

    ################################################################
    # prepare chinese data
    ################################################################
    wikipedia_zh = load_dataset("graelo/wikipedia", "20230601.zh")

    # Number of document is 1357881
    assert len(wikipedia_zh['train']) == 1357881

    SENT_COUNT = 5
    OVERLAP = 1
    counter = 0
    output_file = Path(f"./graelo__wikipedia__chinese__{counter:07}.jsonl").open('w')

    for doc in tqdm(wikipedia_zh['train']):
        url, title, text = doc["url"], doc["title"], doc["text"]

        # Use a sliding window to obtain chunks from wiki text
        lines = [line.strip() for line in zng(text) if len(line.strip()) != 0]
        for i in range(0, len(lines), SENT_COUNT - OVERLAP):
            if counter > 0 and counter % 1000000 == 0:
                output_file.close()
                output_file = Path(f"./graelo__wikipedia__chinese__{counter}.jsonl").open('w')

            txt = ''.join(lines[i:i + SENT_COUNT])
            txt = txt.replace('\n', ' ')
            txt = re.sub(' {2,}', ' ', txt)
            txt = pangu.spacing_text(txt)
            print(json.dumps(
                {"url": url, "title": title, "text": txt}
            ), file=output_file)
            counter += 1


if __name__ == '__main__':
    gen_chinese()
