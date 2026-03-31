import subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent
SEP = ";" if sys.platform == "win32" else ":"

subprocess.run([
    sys.executable, "-m", "PyInstaller",
    "--onefile",
    "--name", "word2vec_messages",
    f"--add-data={HERE / 'index.html'}{SEP}.",
    "--hidden-import=gensim",
    "--hidden-import=gensim.models.word2vec",
    "--hidden-import = scipy.special.cython_special",
    "--hidden-import=jieba",
    "--hidden-import = jieba.analyse",
    "app.py" 
], check=True)

