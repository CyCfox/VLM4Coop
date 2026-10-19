from pathlib import Path
p = Path('VLM4Coop/src/vlm4coop/training.py')
s = p.read_text().replace('''    start = time.monotonic()
    try:
        train_runner''', '''    from vlm4coop.provenance import capture
    capture(output)
    start = time.monotonic()
    try:
        train_runner''')
p.write_text(s)
p = Path('VLM4Coop/src/vlm4coop/pure_vlm.py')
s = p.read_text().replace('''    teacher = runner = None''', '''    from vlm4coop.provenance import capture
    capture(output)
    teacher = runner = None''')
p.write_text(s)
