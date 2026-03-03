import pathlib
p = pathlib.Path('/workspace/karna-ocr/api.py')
t = p.read_text()
t = t.replace('karna-ocr', 'kaeva-brain')
p.write_text(t)
print('done')
