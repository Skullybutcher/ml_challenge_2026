from pathlib import Path
import gzip
import shutil
import tempfile

root = Path(__file__).resolve().parent
parts = sorted((root / 'matching_results_parts').glob('matching_results.tsv.gz.part*'))
destination = root / 'matching_results.tsv'
if not parts:
    raise FileNotFoundError('No matching_results gzip parts found')
if destination.exists():
    raise FileExistsError(f'Refusing to overwrite {destination}')
archive_tmp = None
output_tmp = None
try:
    with tempfile.NamedTemporaryFile(prefix='matching_results_', suffix='.gz.tmp', dir=root, delete=False) as archive:
        archive_tmp = Path(archive.name)
        for part in parts:
            with part.open('rb') as source:
                shutil.copyfileobj(source, archive, length=4 * 1024 * 1024)
    with tempfile.NamedTemporaryFile(prefix='matching_results_', suffix='.tsv.tmp', dir=root, delete=False) as output:
        output_tmp = Path(output.name)
        with gzip.open(archive_tmp, 'rb') as source:
            shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    output_tmp.replace(destination)
    output_tmp = None
    print(f'Restored {destination.name}: {destination.stat().st_size} bytes')
finally:
    for path in (archive_tmp, output_tmp):
        if path is not None and path.exists():
            path.unlink()
