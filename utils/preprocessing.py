# utils/preprocessing.py

from bs4 import BeautifulSoup
import re


def clean_html(html_text: str) -> str:
    """
    Extract visible text from raw HTML.
    - Removes <script>, <style>, <meta>, and <noscript> tags
    - Collapses whitespace
    - Returns a clean string suitable for tokenization
    """
    soup = BeautifulSoup(html_text, "html.parser")

    # Remove non-content tags
    for tag in soup(["script", "style", "meta", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)

    # Collapse multiple whitespace / newlines into a single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text
