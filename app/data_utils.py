from csv import DictReader, DictWriter


def convert_fields(iterable: iter, **conversions):
    for item in iterable:
        for key in item.keys() & conversions:
            item[key] = conversions[key](item[key])
        yield item


def read_csv(filename: str, fieldnames: iter):
    with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
        reader = DictReader(csvfile, fieldnames, delimiter=';')
        return [row for row in reader]


def write_csv(data: list[dict], filename: str, fieldnames: iter):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = DictWriter(csvfile, fieldnames, delimiter=';', extrasaction='ignore')
        writer.writerows(data)
