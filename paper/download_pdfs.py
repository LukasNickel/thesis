from tqdm import tqdm
import requests
import argparse
import re
import logging
import os
import shutil
import urllib.request
from contextlib import closing


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

def parse_bibtex_file(filename, output_folder):
    with open(filename, 'r') as f:
        content = f.read()
    logger.info(
            f'Collecting papers from {filename}, saving to {output_folder}')
    for ref in tqdm(content.split('@')[1:]):
        logger.info('\n\n')
        url, name = find_url(ref)
        if url:
            logger.debug(f'Found url {url} for document {name}')
            try:
                logger.debug(f'Requesting from {url}')
                pdf_filename = (
                        output_folder
                        + '/'
                        + name.replace(' ', '').lower()
                        + '.pdf')
                logger.debug(f'Saving to {pdf_filename}')
                if os.path.isfile(pdf_filename):
                    logger.info(
                            f'File {pdf_filename} already exists. Skipping.')
                    continue
                protocoll = url.split(':', 1)[0] or 'https'
                logger.debug('Protocoll: %s', protocoll)
                if protocoll == 'ftp':
                    with closing(urllib.request.urlopen(url)) as r:
                        with open(pdf_filename, 'wb') as f:
                            shutil.copyfileobj(r, f)

                else:
                    r = requests.get(url, stream=True, headers=headers)
                    total_size = int(r.headers.get('content_length', 0))/(32*1024)
                    block_size = 1024
                    t = tqdm(total=total_size, unit='iB', unit_scale=True)
                    with open(pdf_filename, 'wb') as f:
                        for data in r.iter_content(block_size):
                            t.update(len(data))
                            f.write(data)
            except Exception as e:
                logger.error("ERROR. Skipping this one", exc_info=True)
        else:
            continue
#            logger.info('Did not find any urls for bibtex entry \n%s', ref)


def parse_bibtex_entry(content):
    result = {}
    content = content.replace('{', '')
    content = content.replace('}', '')
    content = content.replace('"', '')
    for line in content.splitlines():
        if '=' not in line:
            continue
        attr, value = line.strip().split('=', 1)
        if value:
            result[attr.lower().strip()] = value.rsplit(',', 1)[0]
    return result


def find_url(text):
    ref_type, ref_content = text.split('{', 1)
    allowed_types = [
            'article',
            'online',
            'book',
            'manual']
    if ref_type.lower() not in allowed_types:
        logger.info(f'{ref_type.lower()} not in allowed types. Skipping this entry.')
        return None, None
    ref_content = parse_bibtex_entry(ref_content)
    if 'url' in ref_content.keys():
        url = ref_content['url'].strip()
    elif 'doi' in ref_content.keys():
        url = 'https://doi.org/' + ref_content['doi'].strip()
    elif 'eprint' in ref_content.keys():
        url = ref_content['eprint'].strip()
    else:
        return None, None

    if url.startswith('ftp'):
        return url, ref_content['title'] or 'dummy'

    r = requests.head(url, allow_redirects=True, headers=headers)
    resolved_url = r.url
    logger.debug('resolved url: %s', resolved_url)
    if resolved_url.endswith('.pdf'):
        return resolved_url, ref_content['title'] or 'dummy'

    journal = re.search('(?:https?://)?(.*?)/', resolved_url).groups()[0]
    logger.debug('journal: %s', journal)
    if journal == 'www.worldscientific.com':
        resolved_url = re.sub('worldscibooks', 'doi/pdf', resolved_url)
        resolved_url += '?download=True'
    elif journal == 'www.sciencedirect.com':
        resolved_url += '/pdfft?isDTMRedir=true'

    elif journal == 'link.springer.com':
        resolved_url = re.sub('article', 'content/pdf', resolved_url)
        resolved_url += '.pdf'
    elif journal == 'github.com':
        return None, None
    elif journal == 'arxiv.org':
        resolved_url = re.sub('abs', 'pdf', resolved_url)
        resolved_url += '.pdf'
    elif journal == 'journals.aps.org':
        resolved_url = re.sub('abstract', 'pdf', resolved_url)
    else:
        logger.error(
                f'Unsupported journal {journal}. Need to implement this one')
        return None, None

    return resolved_url, ref_content['title'] or 'dummy'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Paths for bibtex files and output')
    parser.add_argument('bibtex_files', type=str, nargs='+')
    parser.add_argument('--output_folder', metavar='O', default='.')
    args = parser.parse_args()

    for file in args.bibtex_files:
        print(file)
        parse_bibtex_file(file, args.output_folder)
