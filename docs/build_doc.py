"""Convert the Practical 1 prompt document from Markdown to DOCX and PDF-ready HTML.

Requires:  pip install pypandoc   and   brew install pandoc
Usage:     python docs/build_doc.py
"""
import os
import pypandoc

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "practical-1-prompt.md")

TARGETS = {
    "docx": os.path.join(HERE, "Practical-1-Prompt.docx"),
    "html": os.path.join(HERE, "Practical-1-Prompt.html"),
}

EXTRA_ARGS = ["--standalone", "--toc", "--toc-depth=2"]


def main() -> None:
    with open(SOURCE, encoding="utf-8") as fh:
        content = fh.read()

    for fmt, outfile in TARGETS.items():
        pypandoc.convert_text(
            content,
            fmt,
            format="gfm",
            outputfile=outfile,
            extra_args=EXTRA_ARGS,
        )
        print(f"wrote {outfile} ({os.path.getsize(outfile):,} bytes)")


if __name__ == "__main__":
    main()
