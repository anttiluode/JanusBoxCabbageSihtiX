"""Build index.html: one offline page with the JS core, trained weights and sample crops embedded."""
import base64, io, json
from pathlib import Path
import numpy as np
from PIL import Image
from jbx.data import gray

ROOT = Path(__file__).parent
SAMPLES = {'astronaut (held out)': ('astronaut.png', 0, 128), 'coffee (held out)': ('coffee.png', 100, 200),
           'chelsea (held out)': ('chelsea.png', 20, 60), 'grace hopper (held out)': ('grace_hopper.jpg', 120, 150)}


def png(name, i, j):
    im = gray(name)[i:i + 256, j:j + 256]
    b = io.BytesIO(); Image.fromarray(np.uint8(np.clip(im, 0, 1) * 255 + 0.5)).save(b, 'PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()


def main():
    weights = json.loads((ROOT / 'results/weights.json').read_text())
    receipt = json.loads((ROOT / 'results/receipt.json').read_text())
    data = {'weights': weights, 'receipt': {'writers': receipt['writers']},
            'samples': {k: png(*v) for k, v in SAMPLES.items()}}
    html = (ROOT / 'web/template.html').read_text()
    html = html.replace('__JBX_JS__', (ROOT / 'web/jbx.js').read_text()).replace('__DATA__', json.dumps(data, separators=(',', ':')))
    (ROOT / 'index.html').write_text(html)
    print('index.html', len(html) // 1024, 'KB')


if __name__ == '__main__':
    main()
