import sys
from urllib.request import urlopen
from html.parser import HTMLParser

class LinkParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for name, value in attrs:
                if name == 'href':
                    print(value)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 extract_links.py <URL>")
        sys.exit(1)

    url = sys.argv[1]
    try:
        with urlopen(url) as response:
            html_content = response.read().decode('utf-8')
            parser = LinkParser()
            parser.feed(html_content)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
